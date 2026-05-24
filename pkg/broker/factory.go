package broker

import (
	"github.com/samber/oops"
)

// BalancerKind enumerates the supported balancer algorithms for the factory.
type BalancerKind int

const (
	KindRoundRobin BalancerKind = iota
	KindPowerOfTwoChoices
	KindPSA
	KindDQN
)

// BalancerFactory is a function registered per BalancerKind that creates a
// new Balancer for the given partition count k. Registered by the balancer
// package at init time so that the broker package stays free of circular
// imports.
type BalancerFactory func(k int) Balancer

var factories = map[BalancerKind]BalancerFactory{}

// RegisterBalancerFactory registers a factory for the given kind. Intended
// to be called from an init() in the balancer package.
func RegisterBalancerFactory(kind BalancerKind, f BalancerFactory) {
	factories[kind] = f
}

// BalancerFromKind creates a new Balancer of the given kind for k partitions.
func BalancerFromKind(kind BalancerKind, k int) (Balancer, error) {
	f, ok := factories[kind]
	if !ok {
		return nil, oops.Errorf("unknown balancer kind: %d", kind)
	}
	return f(k), nil
}
