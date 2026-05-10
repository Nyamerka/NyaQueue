package app

import (
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/optimizer"
)

// Option configures a BrokerApp before it starts.
type Option func(*BrokerApp)

// WithBalancer sets the partition balancer (default: RoundRobin).
func WithBalancer(b broker.Balancer) Option {
	return func(a *BrokerApp) {
		a.balancerFactory = func() broker.Balancer { return b }
	}
}

// WithBalancerFactory sets a factory that creates the balancer lazily.
func WithBalancerFactory(fn func() broker.Balancer) Option {
	return func(a *BrokerApp) {
		a.balancerFactory = fn
	}
}

// WithDefaultBalancer uses RoundRobin.
func WithDefaultBalancer() Option {
	return func(a *BrokerApp) {
		a.balancerFactory = func() broker.Balancer {
			return balancer.NewRoundRobin()
		}
	}
}

// WithScheduler attaches a scheduler for a specific topic.
func WithScheduler(topic string, s broker.Scheduler) Option {
	return func(a *BrokerApp) {
		a.schedulers[topic] = s
	}
}

// WithGRPC enables gRPC transport on the given address.
func WithGRPC(addr string) Option {
	return func(a *BrokerApp) {
		a.grpcAddr = addr
	}
}

// WithHTTP enables HTTP REST transport on the given address.
func WithHTTP(addr string) Option {
	return func(a *BrokerApp) {
		a.httpAddr = addr
	}
}

// WithAdmin enables the admin server (/healthz, /readyz, /debug/pprof/*, /metrics)
// on a separate port, keeping pprof and diagnostics away from client traffic.
func WithAdmin(addr string) Option {
	return func(a *BrokerApp) {
		a.adminAddr = addr
	}
}

// WithLoadPredictor enables the AR(p) load predictor that feeds predicted
// partition loads into the DQN balancer and backpressure controller.
func WithLoadPredictor(window, horizon int, interval time.Duration) Option {
	return func(a *BrokerApp) {
		a.loadPredictorCfg = &loadPredictorConfig{
			window:   window,
			horizon:  horizon,
			interval: interval,
		}
	}
}

// WithBackpressure enables predictive backpressure control.
// threshold is the load level at which producers are throttled (default 0.85).
// horizon is how many steps ahead to look in the prediction (default 3).
func WithBackpressure(threshold float64, horizon int) Option {
	return func(a *BrokerApp) {
		a.backpressureCfg = &backpressureConfig{
			threshold: threshold,
			horizon:   horizon,
		}
	}
}

// WithOptimizer enables the DDPG auto-configuration loop that tunes
// broker parameters online based on live metrics.
func WithOptimizer(params []optimizer.TunableParam, optCfg ...optimizer.OptimizerConfig) Option {
	return func(a *BrokerApp) {
		cfg := optimizer.DefaultOptimizerConfig()
		if len(optCfg) > 0 {
			cfg = optCfg[0]
		}
		a.optimizerCfg = &optimizerConfig{
			params: params,
			optCfg: cfg,
		}
	}
}

// WithLassoPilot provides pilot-run data for Lasso-based parameter selection.
// When set, DDPG tunes only the subset of parameters that Lasso identifies as
// significant at the given alpha. Must be called after WithOptimizer.
func WithLassoPilot(pilot ...optimizer.PilotData) Option {
	return func(a *BrokerApp) {
		if a.optimizerCfg != nil {
			a.optimizerCfg.pilot = pilot
		}
	}
}

// WithMetricsInterval sets how often the broker collects metrics and
// pushes them to the balancer/scheduler (default 100ms).
func WithMetricsInterval(d time.Duration) Option {
	return func(a *BrokerApp) {
		a.metricsInterval = d
	}
}
