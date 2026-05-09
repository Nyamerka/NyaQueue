package balancer

import (
	"math"
	"math/bits"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"gonum.org/v1/gonum/floats"
)

// --- RoundRobin ---

type RoundRobin struct {
	counter uint64
}

func NewRoundRobin() *RoundRobin {
	return &RoundRobin{}
}

func (rr *RoundRobin) SelectPartition(_ string, _ []byte, numPartitions int) int {
	n := atomic.AddUint64(&rr.counter, 1)
	return int(n % uint64(numPartitions))
}

func (rr *RoundRobin) OnMetrics(_ broker.Metrics) {}

// --- WeightedRoundRobin ---

// WeightedRoundRobin distributes messages inversely proportional to partition load.
type WeightedRoundRobin struct {
	mu         sync.RWMutex
	weights    []float64
	idx        int
	cw         float64
	minLoad    float64
	lastUpdate time.Time
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

const wrrUpdateThrottle = 500 * time.Millisecond

func (w *WeightedRoundRobin) OnMetrics(m broker.Metrics) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if time.Since(w.lastUpdate) < wrrUpdateThrottle {
		return
	}
	w.lastUpdate = time.Now()

	resized := len(m.PartitionLoads) != len(w.weights)

	if resized {
		w.weights = make([]float64, len(m.PartitionLoads))
	}
	for i, load := range m.PartitionLoads {
		if load < w.minLoad {
			load = w.minLoad
		}
		w.weights[i] = 1.0 / load
	}

	if resized {
		w.idx = 0
		w.cw = 0
	}
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

// gcdFloat computes an approximate GCD of a float slice by quantizing to
// integers (1e6 scale) and using math/big.Int.GCD.
func gcdFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	const scale = 1e6
	result := uint64(math.Round(math.Abs(vals[0] * scale)))
	for _, v := range vals[1:] {
		next := uint64(math.Round(math.Abs(v * scale)))
		result = gcdBits(result, next)
	}
	return float64(result) / scale
}

func gcdBits(a, b uint64) uint64 {
	if a == 0 {
		return b
	}
	if b == 0 {
		return a
	}

	shift := bits.TrailingZeros64(a | b)
	a >>= bits.TrailingZeros64(a)

	for b != 0 {
		b >>= bits.TrailingZeros64(b)
		if a > b {
			a, b = b, a
		}
		b -= a
	}

	return a << uint(shift)
}
