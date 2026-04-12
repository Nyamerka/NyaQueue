package experiments

import (
	"context"

	"github.com/Nyamerka/NyaQueue/internal/app"
	"github.com/Nyamerka/NyaQueue/internal/kafkadriver"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
	"github.com/samber/oops"
)

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
