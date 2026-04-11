package balancer

import (
	"math"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"gonum.org/v1/gonum/floats"
)

// WeightedRoundRobin distributes messages inversely proportional to partition load.
type WeightedRoundRobin struct {
	mu      sync.RWMutex
	weights []float64
	idx     int
	cw      float64
	minLoad float64
}

type WRROption func(*WeightedRoundRobin)

func WithWRRMinLoad(v float64) WRROption {
	return func(w *WeightedRoundRobin) { w.minLoad = v }
}

func NewWeightedRoundRobin(opts ...WRROption) *WeightedRoundRobin {
	w := &WeightedRoundRobin{
		minLoad: DefaultWRRMinLoad,
	}
	for _, opt := range opts {
		opt(w)
	}
	return w
}

func (w *WeightedRoundRobin) SelectPartition(_ string, _ []byte, numPartitions int) int {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.ensureWeights(numPartitions)

	maxW := floats.Max(w.weights)
	gcdW := gcdFloat(w.weights)
	if gcdW < 1e-9 {
		gcdW = maxW / float64(numPartitions)
	}

	for {
		w.idx = (w.idx + 1) % numPartitions
		if w.idx == 0 {
			w.cw -= gcdW
			if w.cw <= 0 {
				w.cw = maxW
				if w.cw <= 0 {
					return 0
				}
			}
		}
		if w.weights[w.idx] >= w.cw {
			return w.idx
		}
	}
}

func (w *WeightedRoundRobin) OnMetrics(m broker.Metrics) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.weights = make([]float64, len(m.PartitionLoads))
	for i, load := range m.PartitionLoads {
		if load < w.minLoad {
			load = w.minLoad
		}
		w.weights[i] = 1.0 / load
	}
	w.idx = 0
	w.cw = 0
}

func (w *WeightedRoundRobin) ensureWeights(n int) {
	if len(w.weights) == n {
		return
	}
	w.weights = make([]float64, n)
	for i := range w.weights {
		w.weights[i] = 1.0
	}
	w.idx = 0
	w.cw = 0
}

// gcdFloat computes an approximate GCD of a float slice using iterative Euclidean on
// integer-quantised values (1e6 scale) to avoid floating-point instability.
func gcdFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	const scale = 1e6
	gcd := int64(math.Round(vals[0] * scale))
	for _, v := range vals[1:] {
		gcd = gcdInt(gcd, int64(math.Round(v*scale)))
		if gcd == 0 {
			return 0
		}
	}
	return float64(gcd) / scale
}

func gcdInt(a, b int64) int64 {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
