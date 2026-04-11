package app

import (
	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
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

// WithDefaultBalancer uses RoundRobin.
func WithDefaultBalancer() Option {
	return func(a *BrokerApp) {
		a.balancerFactory = func() broker.Balancer {
			return balancer.NewRoundRobin()
		}
	}
}
