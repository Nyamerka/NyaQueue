package balancer

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

// PowerOfTwoChoices implements the "Power of Two Random Choices" load-balancing
// algorithm. On each call it picks two random partitions and returns the one
// with lower observed load. The hot path is lock-free: loads are read via
// atomic.Pointer and only written during OnMetrics (called every ~100ms).
type PowerOfTwoChoices struct {
	loads atomic.Pointer[[]float64]

	rngMu sync.Mutex
	rng   *rand.Rand
}

func NewPowerOfTwoChoices() *PowerOfTwoChoices {
	p := &PowerOfTwoChoices{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	return p
}

func (p *PowerOfTwoChoices) SelectPartition(_ string, _ []byte, numPartitions int) int {
	if numPartitions <= 1 {
		return 0
	}

	loadsPtr := p.loads.Load()

	p.rngMu.Lock()
	a := p.rng.Intn(numPartitions)
	b := p.rng.Intn(numPartitions - 1)
	p.rngMu.Unlock()
	if b >= a {
		b++
	}

	if loadsPtr == nil || len(*loadsPtr) < numPartitions {
		return a
	}
	loads := *loadsPtr
	if loads[a] <= loads[b] {
		return a
	}
	return b
}

func (p *PowerOfTwoChoices) OnMetrics(m broker.Metrics) {
	if len(m.PartitionLoads) == 0 {
		return
	}
	snapshot := make([]float64, len(m.PartitionLoads))
	copy(snapshot, m.PartitionLoads)
	p.loads.Store(&snapshot)
}
