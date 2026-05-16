package app

import (
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/samber/oops"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/optimizer"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
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
	params []optimizer.TunableParam
	optCfg optimizer.OptimizerConfig
	pilot  []optimizer.PilotData
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
	httpAddr        string
	metricsInterval time.Duration

	loadPredictorCfg *loadPredictorConfig
	backpressureCfg  *backpressureConfig
	optimizerCfg     *optimizerConfig

	adminAddr string

	broker        *broker.Broker
	offsetStore   *broker.OffsetStore
	server        *transport.Server
	httpServer    *transport.HTTPServer
	adminServer   *transport.AdminServer
	loadPredictor *broker.LoadPredictor
	opt           *optimizer.Optimizer
	ready         atomic.Bool
	promRegistry  *prometheus.Registry
}

// New creates a BrokerApp. Call Start() to begin serving.
func New(cfg broker.Config, dataDir string, opts ...Option) (*BrokerApp, error) {
	a := &BrokerApp{
		cfg:        cfg,
		dataDir:    dataDir,
		schedulers: make(map[string]broker.Scheduler),
		balancerFactory: func() broker.Balancer {
			return balancer.NewPowerOfTwoChoices()
		},
	}
	for _, opt := range opts {
		opt(a)
	}

	var commitInterval time.Duration
	if cfg.SyncPolicy != broker.SyncEveryWrite {
		interval := cfg.FlushIntervalMs
		if interval <= 0 {
			interval = 100
		}
		commitInterval = time.Duration(interval) * time.Millisecond
	}
	offsetStore, err := broker.NewOffsetStore(dataDir, commitInterval)
	if err != nil {
		return nil, oops.Wrapf(err, "offset store")
	}
	a.offsetStore = offsetStore

	bal := a.balancerFactory()
	a.broker = broker.New(cfg, dataDir, bal, offsetStore)

	a.broker.SetSchedulerFactory(func(tc broker.TopicConfig) broker.Scheduler {
		switch tc.ScheduleMode {
		case broker.ModeStrictPriority:
			return scheduler.NewStrictPriority()
		case broker.ModeDQNAdaptive:
			return scheduler.NewDQNScheduler()
		default:
			return scheduler.NewFIFO()
		}
	})

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
		if bpCfg != nil {
			threshold = bpCfg.threshold
		}
		bp := broker.NewBackpressureController(threshold)
		a.broker.SetBackpressure(bp)
	}

	if a.optimizerCfg != nil {
		a.opt = optimizer.NewOptimizer(
			a.broker,
			a.optimizerCfg.params,
			a.optimizerCfg.optCfg,
			a.optimizerCfg.pilot...,
		)
	}

	return a, nil
}

// Start begins the broker loops and optionally starts the gRPC server.
func (a *BrokerApp) Start() error {
	a.broker.Start()
	a.promRegistry = prometheus.NewRegistry()
	a.promRegistry.MustRegister(prometheus.NewGoCollector(), prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	a.broker.MetricsCollector().RegisterPrometheus(a.promRegistry)

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

	if a.httpAddr != "" {
		a.httpServer = transport.NewHTTPServer(a.broker)
		if err := a.httpServer.Start(a.httpAddr); err != nil {
			return oops.Wrapf(err, "http start")
		}
	}

	if a.adminAddr != "" {
		a.adminServer = transport.NewAdminServer(&a.ready, a.promRegistry)
		if err := a.adminServer.Start(a.adminAddr); err != nil {
			return oops.Wrapf(err, "admin start")
		}
	}

	a.ready.Store(true)
	return nil
}

// Stop gracefully shuts down all components.
func (a *BrokerApp) Stop() {
	a.ready.Store(false)
	if a.adminServer != nil {
		a.adminServer.Stop()
	}
	if a.httpServer != nil {
		a.httpServer.Stop()
	}
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

// HTTPAddr returns the HTTP listen address, or empty string if HTTP is not enabled.
func (a *BrokerApp) HTTPAddr() string {
	if a.httpServer != nil {
		return a.httpServer.Addr()
	}
	return ""
}
