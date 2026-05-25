package nn

import (
	"math"
	"sync/atomic"
)

type normSnapshot struct {
	count int64
	mean  []float64
	m2    []float64
}

// RunningNormalizer computes online mean and variance using Welford's algorithm
// and normalizes vectors to zero mean and unit variance.
//
// Observe and Reset must be called from a single goroutine (the policy loop).
// NormalizeInPlace and NormalizeBatch may be called from any goroutine.
type RunningNormalizer struct {
	snap atomic.Pointer[normSnapshot]
	size int
}

func NewRunningNormalizer(size int) *RunningNormalizer {
	rn := &RunningNormalizer{size: size}
	rn.snap.Store(&normSnapshot{
		mean: make([]float64, size),
		m2:   make([]float64, size),
	})
	return rn
}

// Must be called from the single-writer goroutine only.
func (rn *RunningNormalizer) Observe(x []float64) {
	if len(x) != rn.size {
		return
	}
	old := rn.snap.Load()

	newMean := make([]float64, rn.size)
	newM2 := make([]float64, rn.size)
	copy(newMean, old.mean)
	copy(newM2, old.m2)

	count := old.count + 1
	n := float64(count)
	for i := 0; i < rn.size; i++ {
		delta := x[i] - newMean[i]
		newMean[i] += delta / n
		delta2 := x[i] - newMean[i]
		newM2[i] += delta * delta2
	}

	rn.snap.Store(&normSnapshot{
		count: count,
		mean:  newMean,
		m2:    newM2,
	})
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
	s := rn.snap.Load()

	if s.count < 2 {
		for i := range x {
			x[i] = 0
		}
		return
	}

	n := float64(s.count)
	for i := 0; i < rn.size && i < len(x); i++ {
		variance := s.m2[i] / (n - 1)
		std := math.Sqrt(variance)
		x[i] = (x[i] - s.mean[i]) / (std + 1e-8)
	}
}

// NormalizeBatch normalizes a flat buffer of concatenated state vectors in-place.
func (rn *RunningNormalizer) NormalizeBatch(buf []float64, batchSize int) {
	s := rn.snap.Load()

	if s.count < 2 {
		for i := range buf {
			buf[i] = 0
		}
		return
	}

	n := float64(s.count)
	for b := 0; b < batchSize; b++ {
		offset := b * rn.size
		for i := 0; i < rn.size && offset+i < len(buf); i++ {
			variance := s.m2[i] / (n - 1)
			std := math.Sqrt(variance)
			buf[offset+i] = (buf[offset+i] - s.mean[i]) / (std + 1e-8)
		}
	}
}

// Must be called from the single-writer goroutine only.
func (rn *RunningNormalizer) Reset() {
	rn.snap.Store(&normSnapshot{
		mean: make([]float64, rn.size),
		m2:   make([]float64, rn.size),
	})
}
