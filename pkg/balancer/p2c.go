package balancer

import (
	"math/rand/v2"
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultP2CAlpha       = 0.3
	defaultP2CLoadWeight  = 1.0
	defaultP2CDepthWeight = 1.0
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

	alpha       float64
	loadWeight  float64
	depthWeight float64
}

func NewPowerOfTwoChoices(opts ...P2COption) *PowerOfTwoChoices {
	p := &PowerOfTwoChoices{
		alpha:       defaultP2CAlpha,
		loadWeight:  defaultP2CLoadWeight,
		depthWeight: defaultP2CDepthWeight,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *PowerOfTwoChoices) SelectPartition(_ string, _ []byte, numPartitions int) int {
	if numPartitions <= 1 {
		return 0
	}

	a := rand.IntN(numPartitions)
	b := rand.IntN(numPartitions - 1)
	if b >= a {
		b++
	}

	sa := p.score(a)
	sb := p.score(b)

	if sa < sb {
		return a
	}
	if sb < sa {
		return b
	}
	if rand.IntN(2) == 0 {
		return a
	}
	return b
}

func (p *PowerOfTwoChoices) score(idx int) float64 {
	var s float64
	if lp := p.loadEWMA.Load(); lp != nil && idx < len(*lp) {
		s += p.loadWeight * (*lp)[idx]
	}
	if dp := p.depthNorm.Load(); dp != nil && idx < len(*dp) {
		s += p.depthWeight * (*dp)[idx]
	}
	return s
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
