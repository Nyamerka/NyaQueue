package app

import (
	"fmt"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

// BrokerApp wires together the broker, offset store, transport, and pluggable
// balancer/scheduler. It is the single reusable entry point used by cmd/broker,
// experiments, and tests.
type BrokerApp struct {
	cfg     broker.Config
	dataDir string

	balancerFactory func() broker.Balancer
	schedulers      map[string]broker.Scheduler
	grpcAddr        string

	broker      *broker.Broker
	offsetStore *broker.OffsetStore
	server      *transport.Server
}

// New creates a BrokerApp. Call Start() to begin serving.
func New(cfg broker.Config, dataDir string, opts ...Option) (*BrokerApp, error) {
	a := &BrokerApp{
		cfg:        cfg,
		dataDir:    dataDir,
		schedulers: make(map[string]broker.Scheduler),
		balancerFactory: func() broker.Balancer {
			return balancer.NewRoundRobin()
		},
	}
	for _, opt := range opts {
		opt(a)
	}

	offsetStore, err := broker.NewOffsetStore(dataDir)
	if err != nil {
		return nil, fmt.Errorf("offset store: %w", err)
	}
	a.offsetStore = offsetStore

	bal := a.balancerFactory()
	a.broker = broker.New(cfg, dataDir, bal, offsetStore)

	for topic, sched := range a.schedulers {
		a.broker.SetScheduler(topic, sched)
	}

	return a, nil
}

// Start begins the broker loops and optionally starts the gRPC server.
func (a *BrokerApp) Start() error {
	a.broker.Start()

	if a.grpcAddr != "" {
		a.server = transport.NewServer(a.broker)
		if err := a.server.Start(a.grpcAddr); err != nil {
			return fmt.Errorf("grpc start: %w", err)
		}
	}
	return nil
}

// Stop gracefully shuts down the gRPC server and broker.
func (a *BrokerApp) Stop() {
	if a.server != nil {
		a.server.Stop()
	}
	a.broker.Stop()
}

// Broker returns the underlying broker for direct in-process access.
func (a *BrokerApp) Broker() *broker.Broker {
	return a.broker
}

// Addr returns the gRPC listen address, or empty string if gRPC is not enabled.
func (a *BrokerApp) Addr() string {
	if a.server != nil {
		return a.server.Addr()
	}
	return ""
}
