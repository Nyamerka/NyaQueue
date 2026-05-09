package kafkadriver

import (
	"context"
	"errors"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/samber/oops"
	"github.com/segmentio/kafka-go"
)

var ErrTopicNotFound = oops.Errorf("kafka: topic not found")

type Message struct {
	Key      []byte
	Value    []byte
	Offset   int64
	Priority uint8
}

type KafkaHarness struct {
	brokers []string

	mu      sync.RWMutex
	writers map[string]*kafka.Writer
	readers map[readerKey]*kafka.Reader
}

type readerKey struct {
	topic     string
	group     string
	partition int
}

func New(brokers []string) *KafkaHarness {
	return &KafkaHarness{
		brokers: brokers,
		writers: make(map[string]*kafka.Writer),
		readers: make(map[readerKey]*kafka.Reader),
	}
}

func (h *KafkaHarness) CreateTopic(ctx context.Context, name string, partitions int) error {
	conn, err := kafka.DialContext(ctx, "tcp", h.brokers[0])
	if err != nil {
		return oops.Wrapf(err, "dial")
	}
	defer conn.Close()

	controller, err := conn.Controller()
	if err != nil {
		return oops.Wrapf(err, "controller")
	}

	controllerConn, err := kafka.Dial("tcp", net.JoinHostPort(controller.Host, strconv.Itoa(controller.Port)))
	if err != nil {
		return oops.Wrapf(err, "dial controller")
	}
	defer controllerConn.Close()

	return controllerConn.CreateTopics(kafka.TopicConfig{
		Topic:             name,
		NumPartitions:     partitions,
		ReplicationFactor: 1,
	})
}

// DeleteTopic removes a topic from Kafka. Returns ErrTopicNotFound when the
// topic does not exist so callers can skip first-run cleanup.
func (h *KafkaHarness) DeleteTopic(ctx context.Context, name string) error {
	conn, err := kafka.DialContext(ctx, "tcp", h.brokers[0])
	if err != nil {
		return oops.Wrapf(err, "dial")
	}
	defer conn.Close()

	controller, err := conn.Controller()
	if err != nil {
		return oops.Wrapf(err, "controller")
	}

	controllerConn, err := kafka.Dial("tcp", net.JoinHostPort(controller.Host, strconv.Itoa(controller.Port)))
	if err != nil {
		return oops.Wrapf(err, "dial controller")
	}
	defer controllerConn.Close()

	if err := controllerConn.DeleteTopics(name); err != nil {
		var kerr kafka.Error
		if errors.As(err, &kerr) && kerr == kafka.UnknownTopicOrPartition {
			return ErrTopicNotFound
		}
		return oops.Wrapf(err, "delete kafka topic %q", name)
	}

	h.mu.Lock()
	if w, ok := h.writers[name]; ok {
		_ = w.Close()
		delete(h.writers, name)
	}
	for k, r := range h.readers {
		if k.topic == name {
			_ = r.Close()
			delete(h.readers, k)
		}
	}
	h.mu.Unlock()

	return nil
}

func (h *KafkaHarness) Produce(ctx context.Context, topic string, key, value []byte) error {
	return h.ProduceBatch(ctx, topic, []kafka.Message{{Key: key, Value: value}})
}

func (h *KafkaHarness) ProduceBatch(ctx context.Context, topic string, msgs []kafka.Message) error {
	return h.writerFor(topic).WriteMessages(ctx, msgs...)
}

func (h *KafkaHarness) writerFor(topic string) *kafka.Writer {
	h.mu.RLock()
	if w, ok := h.writers[topic]; ok {
		h.mu.RUnlock()
		return w
	}
	h.mu.RUnlock()

	h.mu.Lock()
	defer h.mu.Unlock()
	if w, ok := h.writers[topic]; ok {
		return w
	}
	w := &kafka.Writer{
		Addr:         kafka.TCP(h.brokers...),
		Topic:        topic,
		Balancer:     &kafka.RoundRobin{},
		BatchTimeout: 5 * time.Millisecond,
		BatchSize:    16,
		RequiredAcks: kafka.RequireAll,
	}
	h.writers[topic] = w
	return w
}

func (h *KafkaHarness) Consume(ctx context.Context, topic, group string, partition int, maxBytes int) ([]Message, error) {
	r := h.readerFor(topic, group, partition, maxBytes)

	msg, err := r.ReadMessage(ctx)
	if err != nil {
		return nil, err
	}
	return []Message{{
		Key:    msg.Key,
		Value:  msg.Value,
		Offset: msg.Offset,
	}}, nil
}

func (h *KafkaHarness) readerFor(topic, group string, partition, maxBytes int) *kafka.Reader {
	rk := readerKey{topic: topic, group: group, partition: partition}

	h.mu.RLock()
	if r, ok := h.readers[rk]; ok {
		h.mu.RUnlock()
		return r
	}
	h.mu.RUnlock()

	h.mu.Lock()
	defer h.mu.Unlock()
	if r, ok := h.readers[rk]; ok {
		return r
	}
	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers:     h.brokers,
		Topic:       topic,
		Partition:   partition,
		MaxBytes:    maxBytes,
		MaxWait:     10 * time.Millisecond,
		StartOffset: kafka.FirstOffset,
	})
	h.readers[rk] = r
	return r
}

func (h *KafkaHarness) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	for _, w := range h.writers {
		w.Close()
	}
	for _, r := range h.readers {
		r.Close()
	}
	return nil
}
