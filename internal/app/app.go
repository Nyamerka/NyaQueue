package app

import (
	"time"

	"github.com/samber/oops"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/optimizer"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

type loadPredictorConfig struct {
	window   int
	horizon  int
	interval time.Duration
}

type backpressureConfig struct {
	threshold float64
	horizon   int
}

type optimizerConfig struct {
	params   []optimizer.TunableParam
	interval time.Duration
}

// BrokerApp wires together the broker, offset store, transport, and pluggable
// balancer/scheduler. It is the single reusable entry point used by cmd/broker,
// experiments, and tests.
type BrokerApp struct {
	cfg     broker.Config
	dataDir string

	balancerFactory func() broker.Balancer
	schedulers      map[string]broker.Scheduler
	grpcAddr        string
	metricsInterval time.Duration

	loadPredictorCfg *loadPredictorConfig
	backpressureCfg  *backpressureConfig
	optimizerCfg     *optimizerConfig

	broker        *broker.Broker
	offsetStore   *broker.OffsetStore
	server        *transport.Server
	loadPredictor *broker.LoadPredictor
	opt           *optimizer.Optimizer
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
		return nil, oops.Wrapf(err, "offset store")
	}
	a.offsetStore = offsetStore

	bal := a.balancerFactory()
	a.broker = broker.New(cfg, dataDir, bal, offsetStore)

	for topic, sched := range a.schedulers {
		a.broker.SetScheduler(topic, sched)
	}

	if a.loadPredictorCfg != nil {
		lp := broker.NewLoadPredictor(
			a.loadPredictorCfg.window,
			a.loadPredictorCfg.horizon,
			a.loadPredictorCfg.interval,
		)
		a.loadPredictor = lp

		bpCfg := a.backpressureCfg
		threshold := 0.85
		horizon := 3
		if bpCfg != nil {
			threshold = bpCfg.threshold
			horizon = bpCfg.horizon
		}
		bp := broker.NewBackpressureController(lp, threshold, horizon)
		a.broker.SetBackpressure(bp)
	}

	if a.optimizerCfg != nil {
		a.opt = optimizer.NewOptimizer(
			a.broker,
			a.optimizerCfg.params,
			a.optimizerCfg.interval,
		)
	}

	return a, nil
}

// Start begins the broker loops and optionally starts the gRPC server.
func (a *BrokerApp) Start() error {
	a.broker.Start()

	if a.loadPredictor != nil {
		a.loadPredictor.Start()
	}

	if a.opt != nil {
		a.opt.Start()
	}

	if a.grpcAddr != "" {
		a.server = transport.NewServer(a.broker)
		if err := a.server.Start(a.grpcAddr); err != nil {
			return oops.Wrapf(err, "grpc start")
		}
	}
	return nil
}

// Stop gracefully shuts down all components.
func (a *BrokerApp) Stop() {
	if a.server != nil {
		a.server.Stop()
	}
	if a.opt != nil {
		a.opt.Stop()
	}
	if a.loadPredictor != nil {
		a.loadPredictor.Stop()
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
