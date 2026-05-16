package balancer

import (
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultRRSkipThreshold = 0.9
	defaultRRSkipBudget    = 2
)

type RROption func(*RoundRobin)

func WithRRSkipThreshold(t float64) RROption {
	return func(rr *RoundRobin) { rr.skipThreshold = t }
}

func WithRRSkipBudget(b int) RROption {
	return func(rr *RoundRobin) { rr.skipBudget = b }
}

type RoundRobin struct {
	counter atomic.Uint64
	_pad    [120]byte // cache-line isolation for counter

	loads         atomic.Pointer[[]float64]
	skipThreshold float64
	skipBudget    int
}

func NewRoundRobin(opts ...RROption) *RoundRobin {
	rr := &RoundRobin{
		skipThreshold: defaultRRSkipThreshold,
		skipBudget:    defaultRRSkipBudget,
	}
	for _, o := range opts {
		o(rr)
	}
	return rr
}

func (rr *RoundRobin) SelectPartition(_ string, _ []byte, numPartitions int) int {
	n := rr.counter.Add(1)
	idx := int(n % uint64(numPartitions))

	if rr.skipBudget <= 0 {
		return idx
	}

	ld := rr.loads.Load()
	if ld == nil {
		return idx
	}

	for tries := 0; tries < rr.skipBudget; tries++ {
		if idx < len(*ld) && (*ld)[idx] > rr.skipThreshold {
			idx = (idx + 1) % numPartitions
		} else {
			break
		}
	}
	return idx
}

func (rr *RoundRobin) OnMetrics(m broker.Metrics) {
	if len(m.PartitionLoads) == 0 {
		return
	}
	cp := make([]float64, len(m.PartitionLoads))
	copy(cp, m.PartitionLoads)
	rr.loads.Store(&cp)
}
