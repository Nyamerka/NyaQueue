package nn

import (
	"math/rand/v2"
	"sync"
	"sync/atomic"
)

type Transition struct {
	State     []float64
	Action    []float64
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores experience transitions for DQN/DDPG training.
type ReplayBuffer struct {
	slots    []atomic.Pointer[Transition]
	capacity int64
	writeIdx atomic.Int64

	rngMu sync.Mutex
	rng   *rand.Rand
}

func NewReplayBuffer(capacity int) *ReplayBuffer {
	rb := &ReplayBuffer{
		slots:    make([]atomic.Pointer[Transition], capacity),
		capacity: int64(capacity),
		rng:      rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
	}
	return rb
}

func NewReplayBufferWithSeed(capacity int, seed int64) *ReplayBuffer {
	rb := &ReplayBuffer{
		slots:    make([]atomic.Pointer[Transition], capacity),
		capacity: int64(capacity),
		rng:      rand.New(rand.NewPCG(uint64(seed), uint64(seed)^0xDEAD)),
	}
	return rb
}

func (rb *ReplayBuffer) Push(t Transition) {
	idx := rb.writeIdx.Add(1) - 1
	slot := idx % rb.capacity

	cp := &Transition{
		State:     append([]float64(nil), t.State...),
		Action:    append([]float64(nil), t.Action...),
		Reward:    t.Reward,
		NextState: append([]float64(nil), t.NextState...),
		Done:      t.Done,
	}

	rb.slots[slot].Store(cp)
}

// Sample returns up to batchSize random transitions.
func (rb *ReplayBuffer) Sample(batchSize int) []Transition {
	n := rb.Len()
	if n == 0 {
		return nil
	}
	if batchSize > n {
		batchSize = n
	}

	batch := make([]Transition, batchSize)
	rb.SampleInto(batch)
	return batch[:batchSize]
}

func (rb *ReplayBuffer) SampleInto(dst []Transition) int {
	n := rb.Len()
	if n == 0 {
		return 0
	}
	count := len(dst)
	if count > n {
		count = n
	}

	rb.rngMu.Lock()
	for i := 0; i < count; i++ {
		idx := rb.rng.IntN(n)
		ptr := rb.slots[idx].Load()
		if ptr != nil {
			dst[i] = *ptr
		}
	}
	rb.rngMu.Unlock()
	return count
}

func (rb *ReplayBuffer) Len() int {
	w := rb.writeIdx.Load()
	if w >= rb.capacity {
		return int(rb.capacity)
	}
	return int(w)
}
