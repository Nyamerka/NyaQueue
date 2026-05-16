package balancer

import (
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

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
