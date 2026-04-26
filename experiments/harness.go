package experiments

import (
	"context"
	"errors"

	kafka "github.com/segmentio/kafka-go"

	"github.com/Nyamerka/NyaQueue/internal/app"
	"github.com/Nyamerka/NyaQueue/internal/kafkadriver"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
	"github.com/samber/oops"
)

var ErrNoMessage = errors.New("no message available")

const grpcMaxFetchBytes = 1 << 20

// Mode selects how the experiment communicates with the broker.
type Mode int

const (
	ModeInProcess Mode = iota
	ModeGRPC
	ModeKafka
)

func (m Mode) String() string {
	switch m {
	case ModeInProcess:
		return "inprocess"
	case ModeGRPC:
		return "grpc"
	case ModeKafka:
		return "kafka"
	default:
		return "unknown"
	}
}

// Harness abstracts the target system (NyaQueue in-process, NyaQueue gRPC, or Kafka).
type Harness struct {
	mode Mode
	app  *app.BrokerApp
	grpc *transport.Client
	kfk  *kafkadriver.KafkaHarness
}

// HarnessConfig describes how to create a harness.
type HarnessConfig struct {
	Mode         Mode
	BrokerConfig broker.Config
	DataDir      string
	Algorithm    AlgorithmConfig
	KafkaBrokers []string
}

// NewHarness creates and starts the target system.
func NewHarness(ctx context.Context, cfg HarnessConfig) (*Harness, error) {
	h := &Harness{mode: cfg.Mode}

	switch cfg.Mode {
	case ModeInProcess:
		a, err := app.New(cfg.BrokerConfig, cfg.DataDir,
			app.WithBalancerFactory(cfg.Algorithm.NewBalancer),
		)
		if err != nil {
			return nil, oops.Wrapf(err, "app new")
		}
		if err := a.Start(); err != nil {
			return nil, oops.Wrapf(err, "app start")
		}
		h.app = a

	case ModeGRPC:
		a, err := app.New(cfg.BrokerConfig, cfg.DataDir,
			app.WithBalancerFactory(cfg.Algorithm.NewBalancer),
			app.WithGRPC(":0"),
		)
		if err != nil {
			return nil, oops.Wrapf(err, "app new")
		}
		if err := a.Start(); err != nil {
			return nil, oops.Wrapf(err, "app start")
		}
		h.app = a

		client, err := transport.NewClient(a.Addr())
		if err != nil {
			a.Stop()
			return nil, oops.Wrapf(err, "grpc client")
		}
		h.grpc = client

	case ModeKafka:
		h.kfk = kafkadriver.New(cfg.KafkaBrokers)

	default:
		return nil, oops.Errorf("unknown mode: %d", cfg.Mode)
	}

	return h, nil
}

// Broker returns the in-process broker (nil for Kafka mode).
func (h *Harness) Broker() *broker.Broker {
	if h.app != nil {
		return h.app.Broker()
	}
	return nil
}

func (h *Harness) CreateTopic(ctx context.Context, topic string, cfg broker.TopicConfig) error {
	switch h.mode {
	case ModeInProcess, ModeGRPC:
		brk := h.Broker()
		if brk == nil {
			return oops.Errorf("broker not initialised")
		}
		return brk.CreateTopic(topic, cfg)
	case ModeKafka:
		return h.kfk.CreateTopic(ctx, topic, cfg.NumPartitions)
	}
	return oops.Errorf("unsupported mode")
}

// Publish sends a message. For Kafka mode, priority is ignored.
func (h *Harness) Publish(ctx context.Context, topic string, key, value []byte, priority uint8) error {
	switch h.mode {
	case ModeInProcess:
		msg := broker.NewMessage(priority, key, value)
		_, _, err := h.app.Broker().Publish(topic, msg)
		return err
	case ModeGRPC:
		_, _, err := h.grpc.Produce(ctx, topic, key, value, uint32(priority))
		return err
	case ModeKafka:
		return h.kfk.Produce(ctx, topic, key, value)
	}
	return oops.Errorf("unsupported mode")
}

type BatchItem struct {
	Key      []byte
	Value    []byte
	Priority uint8
}

func (h *Harness) PublishBatch(ctx context.Context, topic string, items []BatchItem) (int, error) {
	if len(items) == 0 {
		return 0, nil
	}
	switch h.mode {
	case ModeInProcess:
		msgs := make([]*broker.Message, len(items))
		for i, it := range items {
			msgs[i] = broker.NewMessage(it.Priority, it.Key, it.Value)
		}

		var (
			results = h.app.Broker().PublishBatch(topic, msgs)
			ok = 0
			firstErr error
		)
		
		for _, r := range results {
			if r.Err == nil {
				ok++
			} else if firstErr == nil {
				firstErr = r.Err
			}
		}
		return ok, firstErr

	case ModeGRPC:
		pbmsgs := make([]*pb.ProduceMessage, len(items))
		for i, it := range items {
			pbmsgs[i] = &pb.ProduceMessage{
				Key:      it.Key,
				Value:    it.Value,
				Priority: uint32(it.Priority),
			}
		}
		results, err := h.grpc.ProduceBatch(ctx, topic, pbmsgs)
		if err != nil {
			return 0, err
		}
		return len(results), nil

	case ModeKafka:
		kmsgs := make([]kafka.Message, len(items))
		for i, it := range items {
			kmsgs[i] = kafka.Message{Key: it.Key, Value: it.Value}
		}
		if err := h.kfk.ProduceBatch(ctx, topic, kmsgs); err != nil {
			return 0, err
		}
		return len(items), nil
	}

	return 0, oops.Errorf("unsupported mode")
}

type ConsumedMessage struct {
	Value    []byte
	Offset   int64
	Priority uint8 // 0 = highest, 9 = lowest; 0 when not available (Kafka)
}

func (h *Harness) Consume(ctx context.Context, topic, group string, partition int) (*ConsumedMessage, error) {
	switch h.mode {
	case ModeInProcess:
		brk := h.app.Broker()
		msg, nextOffset, err := brk.Consume(topic, group, partition)
		if err != nil {
			if errors.Is(err, broker.ErrNoMessages) {
				return nil, ErrNoMessage
			}
			return nil, oops.Wrapf(err, "consume")
		}
		if err := brk.Commit(group, topic, partition, int64(nextOffset)); err != nil {
			return nil, oops.Wrapf(err, "commit")
		}
		return &ConsumedMessage{
			Value:    msg.Value,
			Offset:   int64(nextOffset - 1),
			Priority: msg.Header.Priority,
		}, nil

	case ModeGRPC:
		msgs, err := h.grpc.Consume(ctx, topic, group, int32(partition), grpcMaxFetchBytes)
		if err != nil {
			if errors.Is(err, broker.ErrNoMessages) {
				return nil, ErrNoMessage
			}
			return nil, oops.Wrapf(err, "grpc consume")
		}
		if len(msgs) == 0 {
			return nil, ErrNoMessage
		}
		env := msgs[0]
		if err := h.grpc.Commit(ctx, topic, group, int32(partition), env.Offset+1); err != nil {
			return nil, oops.Wrapf(err, "grpc commit")
		}
		return &ConsumedMessage{
			Value:    env.Value,
			Offset:   env.Offset,
			Priority: uint8(env.Priority),
		}, nil

	case ModeKafka:
		msgs, err := h.kfk.Consume(ctx, topic, group, partition, grpcMaxFetchBytes)
		if err != nil {
			return nil, oops.Wrapf(err, "kafka consume")
		}
		if len(msgs) == 0 {
			return nil, ErrNoMessage
		}
		// Kafka doesn't carry priority natively — priority is 0 (unknown).
		return &ConsumedMessage{Value: msgs[0].Value, Offset: msgs[0].Offset}, nil
	}

	return nil, oops.Errorf("unsupported mode")
}

// Close stops all resources.
func (h *Harness) Close() error {
	if h.grpc != nil {
		h.grpc.Close()
	}
	if h.app != nil {
		h.app.Stop()
	}
	if h.kfk != nil {
		h.kfk.Close()
	}
	return nil
}