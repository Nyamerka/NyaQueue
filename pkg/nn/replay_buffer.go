package nn

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

type Transition struct {
	State     []float64
	Action    []float64
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores experience transitions for DQN/DDPG training.
// Uses an atomic monotonic counter for lock-free Len() and slot assignment.
// A mutex protects concurrent buf reads/writes (Push vs Sample) and the
// per-instance RNG. Deep copies are made on Push to prevent slice aliasing.
type ReplayBuffer struct {
	mu       sync.Mutex
	buf      []Transition
	capacity int64
	writeIdx atomic.Int64
	rng      *rand.Rand
}

func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		buf:      make([]Transition, capacity),
		capacity: int64(capacity),
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func NewReplayBufferWithSeed(capacity int, seed int64) *ReplayBuffer {
	return &ReplayBuffer{
		buf:      make([]Transition, capacity),
		capacity: int64(capacity),
		rng:      rand.New(rand.NewSource(seed)),
	}
}

// Push stores a transition with deep-copied slices to prevent aliasing.
// The deep copy is done before acquiring the lock to minimize hold time.
func (rb *ReplayBuffer) Push(t Transition) {
	idx := rb.writeIdx.Add(1) - 1
	slot := idx % rb.capacity

	cp := Transition{
		State:     append([]float64(nil), t.State...),
		Action:    append([]float64(nil), t.Action...),
		Reward:    t.Reward,
		NextState: append([]float64(nil), t.NextState...),
		Done:      t.Done,
	}

	rb.mu.Lock()
	rb.buf[slot] = cp
	rb.mu.Unlock()
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

// SampleInto fills dst with random transitions from the buffer without
// allocating a new slice. Returns the number of transitions written.
func (rb *ReplayBuffer) SampleInto(dst []Transition) int {
	n := rb.Len()
	if n == 0 {
		return 0
	}
	count := len(dst)
	if count > n {
		count = n
	}

	rb.mu.Lock()
	for i := 0; i < count; i++ {
		idx := rb.rng.Intn(n)
		dst[i] = rb.buf[idx]
	}
	rb.mu.Unlock()
	return count
}

// Len returns the number of stored transitions (lock-free via atomic).
func (rb *ReplayBuffer) Len() int {
	w := rb.writeIdx.Load()
	if w >= rb.capacity {
		return int(rb.capacity)
	}
	return int(w)
}
