package benchmarks

import (
	"crypto/rand"
	"math/big"
	"time"
)

// Scenario defines a load pattern for benchmarking.
type Scenario struct {
	Name          string
	Duration      time.Duration
	NumPartitions int           // explicit partition count; 0 = use Producers
	Producers     int
	Consumers     int
	MsgSize       int           // bytes per message
	RatePerSec    int           // target messages/second per producer
	BurstEvery    time.Duration // inject burst at this interval (0 = no bursts)
	BurstFactor   int           // burst multiplier
	SkewRatio     float64       // fraction of messages using a fixed hot key (0 = uniform)
	Priorities    [10]float64   // probability distribution over priority levels
}

// Uniform generates steady, evenly distributed load at a rate both NyaQueue
// and Kafka can sustain — suitable for apples-to-apples latency comparison.
func Uniform() Scenario {
	return Scenario{
		Name:       "uniform",
		Duration:   30 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 1200,
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
		RatePerSec: 1200,
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
		RatePerSec:  1200,
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
		RatePerSec: 600,
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
		RatePerSec: 1200,
		Priorities: [10]float64{0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01},
	}
}

// Overload sends at maximum throughput with no rate limiting.
func Overload() Scenario {
	return Scenario{
		Name:       "overload",
		Duration:   30 * time.Second,
		Producers:  4,
		Consumers:  4,
		MsgSize:    256,
		RatePerSec: 0, // unlimited
		Priorities: [10]float64{0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01},
	}
}

// SkewedK16 is Skewed with 16 partitions and proportionally higher load so
// per-partition utilisation stays above 0.5 — the regime where DQN load
// balancing is theoretically non-trivial (K ≥ 16, ρ > 0.5).
func SkewedK16() Scenario {
	return Scenario{
		Name:          "skewed_k16",
		Duration:      30 * time.Second,
		NumPartitions: 16,
		Producers:     16,
		Consumers:     16,
		MsgSize:       256,
		RatePerSec:    4800, // 16×300 msg/s — same per-producer rate as Skewed
		SkewRatio:     0.8,
		Priorities:    uniformPriorities(),
	}
}

// OverloadK16 sends at maximum throughput across 16 partitions with a skewed
// priority distribution. At saturation queue depths diverge per partition,
// giving the DQN balancer a meaningful signal to act on.
func OverloadK16() Scenario {
	return Scenario{
		Name:          "overload_k16",
		Duration:      30 * time.Second,
		NumPartitions: 16,
		Producers:     16,
		Consumers:     16,
		MsgSize:       256,
		RatePerSec:    0, // unlimited
		Priorities:    [10]float64{0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01},
	}
}

type Scenarios []Scenario

// AllScenarios returns the full set of benchmark scenarios.
func AllScenarios() Scenarios {
	return Scenarios{
		Uniform(),
		Skewed(),
		Bursty(),
		GrowingLoad(),
		MixedPriority(),
		Overload(),
		SkewedK16(),
		OverloadK16(),
	}
}

func (ss Scenarios) FindByName(name string) (bool, Scenario) {
	for i := range ss {
		if ss[i].Name == name {
			return true, ss[i]
		}
	}
	return false, Scenario{}
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
