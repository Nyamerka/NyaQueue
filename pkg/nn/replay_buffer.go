package nn

import (
	"math/rand"
	"sync"
)

type Transition struct {
	State     []float64
	Action    []float64
	Reward    float64
	NextState []float64
	Done      bool
}

type ReplayBuffer struct {
	mu       sync.Mutex
	buf      []Transition
	capacity int
	pos      int
	full     bool
}

func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		buf:      make([]Transition, capacity),
		capacity: capacity,
	}
}

func (rb *ReplayBuffer) Push(t Transition) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.buf[rb.pos] = t
	rb.pos++
	if rb.pos >= rb.capacity {
		rb.pos = 0
		rb.full = true
	}
}

func (rb *ReplayBuffer) Sample(batchSize int) []Transition {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	n := rb.len()
	if batchSize > n {
		batchSize = n
	}

	batch := make([]Transition, batchSize)
	for i := range batch {
		idx := rand.Intn(n)
		batch[i] = rb.buf[idx]
	}
	return batch
}

func (rb *ReplayBuffer) Len() int {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	return rb.len()
}

func (rb *ReplayBuffer) len() int {
	if rb.full {
		return rb.capacity
	}
	return rb.pos
}
