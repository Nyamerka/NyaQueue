package benchmarks

import (
	"crypto/rand"
	"math/big"
	"time"
)

// Scenario defines a load pattern for benchmarking.
type Scenario struct {
	Name        string
	Duration    time.Duration
	Producers   int
	Consumers   int
	MsgSize     int           // bytes per message
	RatePerSec  int           // target messages/second per producer
	BurstEvery  time.Duration // inject burst at this interval (0 = no bursts)
	BurstFactor int           // burst multiplier
	SkewRatio   float64       // 0 = uniform, 1 = all traffic to one partition
	Priorities  [10]float64   // probability distribution over priority levels
}

// Uniform generates steady, evenly distributed load.
func Uniform() Scenario {
	return Scenario{
		Name:       "uniform",
		Duration:   30 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 5000,
		Priorities: uniformPriorities(),
	}
}

// Skewed sends most traffic to a single partition (tests balancer effectiveness).
func Skewed() Scenario {
	return Scenario{
		Name:       "skewed",
		Duration:   30 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 5000,
		SkewRatio:  0.8,
		Priorities: uniformPriorities(),
	}
}

// Bursty injects periodic traffic spikes (tests backpressure).
func Bursty() Scenario {
	return Scenario{
		Name:        "bursty",
		Duration:    60 * time.Second,
		Producers:   4,
		Consumers:   4,
		MsgSize:     256,
		RatePerSec:  3000,
		BurstEvery:  10 * time.Second,
		BurstFactor: 5,
		Priorities:  uniformPriorities(),
	}
}

// GrowingLoad starts slow and ramps up (tests adaptive behaviour).
func GrowingLoad() Scenario {
	return Scenario{
		Name:       "growing",
		Duration:   60 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 1000, // starting rate; grows linearly
		Priorities: uniformPriorities(),
	}
}

// MixedPriority sends messages with a skewed priority distribution.
func MixedPriority() Scenario {
	return Scenario{
		Name:       "mixed_priority",
		Duration:   30 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 5000,
		Priorities: [10]float64{0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01},
	}
}

// AllScenarios returns the full set of benchmark scenarios.
func AllScenarios() []Scenario {
	return []Scenario{
		Uniform(),
		Skewed(),
		Bursty(),
		GrowingLoad(),
		MixedPriority(),
	}
}

// GenerateMessage creates a random message payload of the given size.
func GenerateMessage(size int) []byte {
	buf := make([]byte, size)
	_, _ = rand.Read(buf)
	return buf
}

// SamplePriority returns a priority level sampled from the scenario's distribution.
func (s *Scenario) SamplePriority() uint8 {
	n, _ := rand.Int(rand.Reader, big.NewInt(1_000_000))
	r := float64(n.Int64()) / 1_000_000.0
	cumulative := 0.0
	for i, p := range s.Priorities {
		cumulative += p
		if r <= cumulative {
			return uint8(i)
		}
	}
	return 0
}

func uniformPriorities() [10]float64 {
	var p [10]float64
	for i := range p {
		p[i] = 0.1
	}
	return p
}
