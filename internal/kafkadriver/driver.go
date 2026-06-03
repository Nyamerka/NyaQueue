package kafkadriver

import (
	"context"
	"errors"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
	"github.com/samber/oops"
	"github.com/twmb/franz-go/pkg/kerr"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/kmsg"
)

var ErrTopicNotFound = oops.Errorf("kafka: topic not found")

type Message struct {
	Key      []byte
	Value    []byte
	Offset   int64
	Priority uint8
}

type KafkaHarness struct {
	brokers        []string
	producer       *kgo.Client
	consumers      *xsync.MapOf[readerKey, *kgo.Client]
	mu             sync.Mutex
	loggedFirstErr atomic.Bool

	inflightProduces atomic.Int64
	successCount     atomic.Int64
	failureCount     atomic.Int64
}

type readerKey struct {
	topic     string
	group     string
	partition int
}

func New(brokers []string) *KafkaHarness {
	return &KafkaHarness{
		brokers:   brokers,
		consumers: xsync.NewMapOf[readerKey, *kgo.Client](),
	}
}

func (h *KafkaHarness) ensureProducer() (*kgo.Client, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.producer != nil {
		return h.producer, nil
	}
	cl, err := kgo.NewClient(
		kgo.SeedBrokers(h.brokers...),
		kgo.DisableIdempotentWrite(),
		kgo.RequiredAcks(kgo.LeaderAck()),
		kgo.ProducerLinger(0),
		kgo.ProducerBatchMaxBytes(16<<20),
		kgo.MaxBufferedRecords(1<<20),
		kgo.ProducerBatchCompression(kgo.Lz4Compression()),
		kgo.RetryBackoffFn(func(attempt int) time.Duration {
			d := time.Duration(attempt) * 250 * time.Millisecond
			if d > 2*time.Second {
				d = 2 * time.Second
			}
			return d
		}),
		kgo.RequestRetries(3),
		kgo.RetryTimeout(2*time.Second),
		kgo.RecordDeliveryTimeout(5*time.Second),
	)
	if err != nil {
		log.Printf("[kafka-producer] ensureProducer failed: %+v (brokers=%v)", err, h.brokers)
		return nil, oops.Wrapf(err, "create franz-go producer")
	}
	h.producer = cl
	return cl, nil
}

func (h *KafkaHarness) resetProducer() {
	h.mu.Lock()
	if h.producer != nil {
		h.producer.Close()
		h.producer = nil
	}
	h.mu.Unlock()
}

func (h *KafkaHarness) CreateTopic(ctx context.Context, name string, partitions int) error {
	cl, err := kgo.NewClient(kgo.SeedBrokers(h.brokers...))
	if err != nil {
		return oops.Wrapf(err, "dial admin client")
	}
	defer cl.Close()

	req := kmsg.NewPtrCreateTopicsRequest()
	topic := kmsg.NewCreateTopicsRequestTopic()
	topic.Topic = name
	topic.NumPartitions = int32(partitions)
	topic.ReplicationFactor = 1
	req.Topics = append(req.Topics, topic)

	resp, err := req.RequestWith(ctx, cl)
	if err != nil {
		return oops.Wrapf(err, "create topic %q", name)
	}
	for _, t := range resp.Topics {
		if errCode := kerr.ErrorForCode(t.ErrorCode); errCode != nil {
			return oops.Wrapf(errCode, "create topic %q", name)
		}
	}
	return nil
}

func (h *KafkaHarness) DeleteTopic(ctx context.Context, name string) error {
	cl, err := kgo.NewClient(kgo.SeedBrokers(h.brokers...))
	if err != nil {
		return oops.Wrapf(err, "dial admin client")
	}
	defer cl.Close()

	req := kmsg.NewPtrDeleteTopicsRequest()
	req.TopicNames = []string{name}
	reqTopic := kmsg.NewDeleteTopicsRequestTopic()
	reqTopic.Topic = kmsg.StringPtr(name)
	req.Topics = append(req.Topics, reqTopic)

	resp, err := req.RequestWith(ctx, cl)
	if err != nil {
		return oops.Wrapf(err, "delete topic %q", name)
	}
	for _, t := range resp.Topics {
		if errCode := kerr.ErrorForCode(t.ErrorCode); errCode != nil {
			if errors.Is(errCode, kerr.UnknownTopicOrPartition) {
				return ErrTopicNotFound
			}
			return oops.Wrapf(errCode, "delete topic %q", name)
		}
	}

	h.consumers.Range(func(k readerKey, c *kgo.Client) bool {
		if k.topic == name {
			c.Close()
			h.consumers.Delete(k)
		}
		return true
	})

	return nil
}

func (h *KafkaHarness) Produce(ctx context.Context, topic string, key, value []byte) error {
	cl, err := h.ensureProducer()
	if err != nil {
		return oops.Wrapf(err, "produce topic=%q ensure producer", topic)
	}
	rec := &kgo.Record{
		Topic: topic,
		Key:   key,
		Value: value,
	}

	errCh := make(chan error, 1)
	cl.Produce(ctx, rec, func(_ *kgo.Record, err error) {
		if err != nil {
			if !h.loggedFirstErr.Swap(true) {
				log.Printf("[kafka-produce] first error: %+v", err)
			}
			h.failureCount.Add(1)
		} else {
			h.successCount.Add(1)
		}
		errCh <- err
	})
	return <-errCh
}

func (h *KafkaHarness) ProduceAsync(ctx context.Context, topic string, key, value []byte) {
	cl, err := h.ensureProducer()
	if err != nil {
		if !h.loggedFirstErr.Swap(true) {
			log.Printf("[kafka-produce-async] ensure producer: %+v", err)
		}
		h.failureCount.Add(1)
		return
	}
	rec := &kgo.Record{
		Topic: topic,
		Key:   key,
		Value: value,
	}
	h.inflightProduces.Add(1)
	cl.Produce(ctx, rec, func(_ *kgo.Record, err error) {
		h.inflightProduces.Add(-1)
		if err != nil {
			if !h.loggedFirstErr.Swap(true) {
				log.Printf("[kafka-produce-async] first error: %+v", err)
			}
			h.failureCount.Add(1)
		} else {
			h.successCount.Add(1)
		}
	})
}

func (h *KafkaHarness) ProduceBatch(ctx context.Context, topic string, msgs []kgo.Record) error {
	cl, err := h.ensureProducer()
	if err != nil {
		return oops.Wrapf(err, "produce-batch topic=%q ensure producer", topic)
	}
	for i := range msgs {
		rec := &kgo.Record{
			Topic: topic,
			Key:   msgs[i].Key,
			Value: msgs[i].Value,
		}
		h.inflightProduces.Add(1)
		cl.Produce(ctx, rec, func(_ *kgo.Record, err error) {
			h.inflightProduces.Add(-1)
			if err != nil {
				h.failureCount.Add(1)
				if !h.loggedFirstErr.Swap(true) {
					log.Printf("[kafka-produce-batch] first error: %+v", err)
				}
			} else {
				h.successCount.Add(1)
			}
		})
	}
	return nil
}

func (h *KafkaHarness) Flush(ctx context.Context) error {
	h.mu.Lock()
	cl := h.producer
	h.mu.Unlock()
	if cl == nil {
		return nil
	}
	return cl.Flush(ctx)
}

func (h *KafkaHarness) SuccessCount() int64  { return h.successCount.Load() }
func (h *KafkaHarness) FailureCount() int64  { return h.failureCount.Load() }
func (h *KafkaHarness) InflightCount() int64 { return h.inflightProduces.Load() }

func (h *KafkaHarness) Consume(ctx context.Context, topic, group string, partition int, maxBytes int) ([]Message, error) {
	cl := h.consumerFor(topic, group, partition)
	fetches := cl.PollFetches(ctx)
	if errs := fetches.Errors(); len(errs) > 0 {
		return nil, oops.Wrapf(errs[0].Err, "kafka fetch topic=%q part=%d", topic, partition)
	}

	var msgs []Message
	fetches.EachRecord(func(rec *kgo.Record) {
		msgs = append(msgs, Message{
			Key:    rec.Key,
			Value:  rec.Value,
			Offset: rec.Offset,
		})
	})

	return msgs, nil
}

func (h *KafkaHarness) consumerFor(topic, group string, partition int) *kgo.Client {
	rk := readerKey{topic: topic, group: group, partition: partition}
	cl, _ := h.consumers.LoadOrCompute(rk, func() *kgo.Client {
		opts := []kgo.Opt{
			kgo.SeedBrokers(h.brokers...),
			kgo.FetchMaxWait(10 * time.Millisecond),
			kgo.FetchMinBytes(4 << 10),
			kgo.FetchMaxBytes(int32(50 << 20)),
			kgo.ConsumePartitions(map[string]map[int32]kgo.Offset{
				topic: {int32(partition): kgo.NewOffset().AtStart()},
			}),
		}
		cl, err := kgo.NewClient(opts...)
		if err != nil {
			panic("kafkadriver: failed to create consumer: " + err.Error())
		}
		return cl
	})
	return cl
}

func (h *KafkaHarness) Close() error {
	h.mu.Lock()
	if h.producer != nil {
		h.producer.Close()
		h.producer = nil
	}
	h.mu.Unlock()
	h.consumers.Range(func(_ readerKey, c *kgo.Client) bool {
		c.Close()
		return true
	})
	return nil
}
