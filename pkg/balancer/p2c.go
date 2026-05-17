package balancer

import (
	"math/rand/v2"
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultP2CAlpha          = 0.3
	defaultP2CLoadWeight     = 1.0
	defaultP2CDepthWeight    = 1.0
	defaultP2CInflightWeight = 2.0
)

type P2COption func(*PowerOfTwoChoices)

func WithP2CLoadWeight(w float64) P2COption {
	return func(p *PowerOfTwoChoices) { p.loadWeight = w }
}

func WithP2CDepthWeight(w float64) P2COption {
	return func(p *PowerOfTwoChoices) { p.depthWeight = w }
}

func WithP2CAlpha(a float64) P2COption {
	return func(p *PowerOfTwoChoices) { p.alpha = a }
}

type PowerOfTwoChoices struct {
	loadEWMA  atomic.Pointer[[]float64]
	depthNorm atomic.Pointer[[]float64]
	inflight  atomic.Pointer[[]atomic.Int64]

	alpha          float64
	loadWeight     float64
	depthWeight    float64
	inflightWeight float64
}

func WithP2CInflightWeight(w float64) P2COption {
	return func(p *PowerOfTwoChoices) { p.inflightWeight = w }
}

func NewPowerOfTwoChoices(opts ...P2COption) *PowerOfTwoChoices {
	p := &PowerOfTwoChoices{
		alpha:          defaultP2CAlpha,
		loadWeight:     defaultP2CLoadWeight,
		depthWeight:    defaultP2CDepthWeight,
		inflightWeight: defaultP2CInflightWeight,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *PowerOfTwoChoices) ensureInflight(n int) *[]atomic.Int64 {
	for {
		ptr := p.inflight.Load()
		if ptr != nil && len(*ptr) >= n {
			return ptr
		}
		arr := make([]atomic.Int64, n)
		if p.inflight.CompareAndSwap(ptr, &arr) {
			return &arr
		}
	}
}

func (p *PowerOfTwoChoices) SelectPartition(_ string, _ []byte, numPartitions int) int {
	if numPartitions <= 1 {
		inf := p.ensureInflight(1)
		(*inf)[0].Add(1)
		return 0
	}

	inf := p.ensureInflight(numPartitions)

	a := rand.IntN(numPartitions)
	b := rand.IntN(numPartitions - 1)
	if b >= a {
		b++
	}

	sa := p.score(a, inf)
	sb := p.score(b, inf)

	var chosen int
	if sa < sb {
		chosen = a
	} else if sb < sa {
		chosen = b
	} else if rand.IntN(2) == 0 {
		chosen = a
	} else {
		chosen = b
	}

	(*inf)[chosen].Add(1)
	return chosen
}

func (p *PowerOfTwoChoices) score(idx int, inf *[]atomic.Int64) float64 {
	var s float64
	if lp := p.loadEWMA.Load(); lp != nil && idx < len(*lp) {
		s += p.loadWeight * (*lp)[idx]
	}
	if dp := p.depthNorm.Load(); dp != nil && idx < len(*dp) {
		s += p.depthWeight * (*dp)[idx]
	}
	if inf != nil && idx < len(*inf) {
		s += p.inflightWeight * float64((*inf)[idx].Load())
	}
	return s
}

func (p *PowerOfTwoChoices) OnPublishComplete(partition int) {
	if ptr := p.inflight.Load(); ptr != nil && partition < len(*ptr) {
		(*ptr)[partition].Add(-1)
	}
}

func (p *PowerOfTwoChoices) OnMetrics(m broker.Metrics) {
	if len(m.PartitionLoads) > 0 {
		p.updateLoadEWMA(m.PartitionLoads)
	}
	if len(m.QueueDepth) > 0 {
		p.updateDepthNorm(m.QueueDepth)
	}
}

func (p *PowerOfTwoChoices) updateLoadEWMA(sample []float64) {
	prev := p.loadEWMA.Load()
	ewma := make([]float64, len(sample))

	if prev == nil || len(*prev) != len(sample) {
		copy(ewma, sample)
	} else {
		old := *prev
		for i := range sample {
			ewma[i] = p.alpha*sample[i] + (1-p.alpha)*old[i]
		}
	}
	p.loadEWMA.Store(&ewma)
}

func (p *PowerOfTwoChoices) updateDepthNorm(depths []int) {
	maxD := 0
	for _, d := range depths {
		if d > maxD {
			maxD = d
		}
	}
	norm := make([]float64, len(depths))
	if maxD > 0 {
		inv := 1.0 / float64(maxD)
		for i, d := range depths {
			norm[i] = float64(d) * inv
		}
	}
	p.depthNorm.Store(&norm)
}
