package kafkadriver

import (
	"context"
	"net"
	"strconv"
	"time"

	"github.com/samber/oops"
	"github.com/segmentio/kafka-go"
)

// Message mirrors the minimal fields needed by experiments.
type Message struct {
	Key      []byte
	Value    []byte
	Offset   int64
	Priority uint8
}

// KafkaHarness wraps segmentio/kafka-go for comparison experiments.
type KafkaHarness struct {
	brokers []string
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

func (h *KafkaHarness) Produce(ctx context.Context, topic string, key, value []byte) error {
	return h.ProduceBatch(ctx, topic, []kafka.Message{{Key: key, Value: value}})
}

func (h *KafkaHarness) ProduceBatch(ctx context.Context, topic string, msgs []kafka.Message) error {
	w, ok := h.writers[topic]
	if !ok {
		w = &kafka.Writer{
			Addr:         kafka.TCP(h.brokers...),
			Topic:        topic,
			Balancer:     &kafka.RoundRobin{},
			BatchTimeout: 10 * time.Millisecond,
			BatchSize:    100,
			RequiredAcks: kafka.RequireOne,
		}
		h.writers[topic] = w
	}

	return w.WriteMessages(ctx, msgs...)
}

func (h *KafkaHarness) Consume(ctx context.Context, topic, group string, partition int, maxBytes int) ([]Message, error) {
	rk := readerKey{topic: topic, group: group, partition: partition}
	r, ok := h.readers[rk]
	if !ok {
		r = kafka.NewReader(kafka.ReaderConfig{
			Brokers:   h.brokers,
			Topic:     topic,
			GroupID:   group,
			Partition: partition,
			MaxBytes:  maxBytes,
			MaxWait:   10 * time.Millisecond,
		})
		h.readers[rk] = r
	}

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

func (h *KafkaHarness) Close() error {
	for _, w := range h.writers {
		w.Close()
	}
	for _, r := range h.readers {
		r.Close()
	}
	return nil
}
