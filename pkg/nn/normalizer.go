package nn

import (
	"math"

	"github.com/puzpuzpuz/xsync/v3"
)

// RunningNormalizer computes online mean and variance using Welford's algorithm
// and normalizes vectors to zero mean and unit variance.
type RunningNormalizer struct {
	mu    *xsync.RBMutex
	count int64
	mean  []float64
	m2    []float64
	size  int
}

func NewRunningNormalizer(size int) *RunningNormalizer {
	return &RunningNormalizer{
		mu:   xsync.NewRBMutex(),
		mean: make([]float64, size),
		m2:   make([]float64, size),
		size: size,
	}
}

// Observe updates the running statistics with a new observation.
func (rn *RunningNormalizer) Observe(x []float64) {
	if len(x) != rn.size {
		return
	}
	rn.mu.Lock()
	defer rn.mu.Unlock()

	rn.count++
	n := float64(rn.count)
	for i := 0; i < rn.size; i++ {
		delta := x[i] - rn.mean[i]
		rn.mean[i] += delta / n
		delta2 := x[i] - rn.mean[i]
		rn.m2[i] += delta * delta2
	}
}

// Normalize returns (x - mean) / (std + ε) without modifying x.
func (rn *RunningNormalizer) Normalize(x []float64) []float64 {
	out := make([]float64, len(x))
	copy(out, x)
	rn.NormalizeInPlace(out)
	return out
}

// NormalizeInPlace normalizes x in-place: x[i] = (x[i] - mean[i]) / (std[i] + ε).
func (rn *RunningNormalizer) NormalizeInPlace(x []float64) {
	rt := rn.mu.RLock()
	defer rn.mu.RUnlock(rt)

	if rn.count < 2 {
		for i := range x {
			x[i] = 0
		}
		return
	}

	n := float64(rn.count)
	for i := 0; i < rn.size && i < len(x); i++ {
		variance := rn.m2[i] / (n - 1)
		std := math.Sqrt(variance)
		x[i] = (x[i] - rn.mean[i]) / (std + 1e-8)
	}
}
