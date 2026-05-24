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
		kgo.RequiredAcks(kgo.LeaderAck()),
		kgo.ProducerLinger(5*time.Millisecond),
		kgo.ProducerBatchMaxBytes(1<<20),
		kgo.ProducerBatchCompression(kgo.Lz4Compression()),
		kgo.RetryBackoffFn(func(attempt int) time.Duration {
			d := time.Duration(attempt) * 250 * time.Millisecond
			if d > 5*time.Second {
				d = 5 * time.Second
			}
			return d
		}),
		kgo.RequestRetries(20),
		kgo.RetryTimeout(30*time.Second),
	)
	if err != nil {
		return nil, oops.Wrapf(err, "create franz-go producer")
	}
	h.producer = cl
	return cl, nil
}

// resetProducer closes the current producer so the next call recreates it.
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

const produceMaxAttempts = 3

func (h *KafkaHarness) Produce(ctx context.Context, topic string, key, value []byte) error {
	rec := &kgo.Record{
		Topic: topic,
		Key:   key,
		Value: value,
	}
	var lastErr error
	for attempt := 0; attempt < produceMaxAttempts; attempt++ {
		cl, err := h.ensureProducer()
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
			continue
		}
		result := cl.ProduceSync(ctx, rec)
		if err := result.FirstErr(); err != nil {
			if !h.loggedFirstErr.Swap(true) {
				log.Printf("[kafka-produce] first error: %v", err)
			}
			lastErr = err
			h.resetProducer()
			continue
		}
		return nil
	}
	return oops.Wrapf(lastErr, "produce topic=%q exhausted %d attempts", topic, produceMaxAttempts)
}

func (h *KafkaHarness) ProduceBatch(ctx context.Context, topic string, msgs []kgo.Record) error {
	recs := make([]*kgo.Record, len(msgs))
	for i := range msgs {
		recs[i] = &kgo.Record{
			Topic: topic,
			Key:   msgs[i].Key,
			Value: msgs[i].Value,
		}
	}
	var lastErr error
	for attempt := 0; attempt < produceMaxAttempts; attempt++ {
		cl, err := h.ensureProducer()
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
			continue
		}
		results := cl.ProduceSync(ctx, recs...)
		if err := results.FirstErr(); err != nil {
			if !h.loggedFirstErr.Swap(true) {
				log.Printf("[kafka-produce-batch] first error: %v", err)
			}
			lastErr = err
			h.resetProducer()
			continue
		}
		return nil
	}
	return oops.Wrapf(lastErr, "produce-batch topic=%q exhausted %d attempts", topic, produceMaxAttempts)
}

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
			kgo.FetchMaxBytes(int32(1 << 20)),
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
