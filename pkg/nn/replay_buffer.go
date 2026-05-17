package nn

import (
	"math"
	"math/rand/v2"
	"sync/atomic"

	"github.com/puzpuzpuz/xsync/v3"
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
}

func NewReplayBuffer(capacity int) *ReplayBuffer {
	rb := &ReplayBuffer{
		slots:    make([]atomic.Pointer[Transition], capacity),
		capacity: int64(capacity),
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

	for i := 0; i < count; i++ {
		idx := rand.IntN(n)
		ptr := rb.slots[idx].Load()
		if ptr != nil {
			dst[i] = *ptr
		}
	}
	return count
}

func (rb *ReplayBuffer) Len() int {
	w := rb.writeIdx.Load()
	if w >= rb.capacity {
		return int(rb.capacity)
	}
	return int(w)
}

// PrioritizedReplayBuffer samples transitions proportional to priority^alpha.
// Higher TD-error transitions are sampled more frequently.
type PrioritizedReplayBuffer struct {
	mu          *xsync.RBMutex
	transitions []Transition
	priorities  []float64
	capacity    int
	size        int
	writeIdx    int
	alpha       float64
	defaultPrio float64
	maxPriority float64
}

func NewPrioritizedReplayBuffer(capacity int, alpha float64) *PrioritizedReplayBuffer {
	return &PrioritizedReplayBuffer{
		mu:          xsync.NewRBMutex(),
		transitions: make([]Transition, capacity),
		priorities:  make([]float64, capacity),
		capacity:    capacity,
		alpha:       alpha,
		defaultPrio: 1.0,
		maxPriority: 1.0,
	}
}

func (b *PrioritizedReplayBuffer) Push(t Transition) {
	b.mu.Lock()
	defer b.mu.Unlock()

	cp := Transition{
		State:     append([]float64(nil), t.State...),
		Action:    append([]float64(nil), t.Action...),
		Reward:    t.Reward,
		NextState: append([]float64(nil), t.NextState...),
		Done:      t.Done,
	}

	idx := b.writeIdx % b.capacity
	b.transitions[idx] = cp
	b.priorities[idx] = b.maxPriority
	b.writeIdx++
	if b.size < b.capacity {
		b.size++
	}
}

// Sample returns a batch of transitions sampled proportional to priority^alpha,
// together with their buffer indices and importance-sampling weights.
// IS-weights are computed as w_i = (N * P(i))^(-beta) / max(w) and normalized so max=1.0.
func (b *PrioritizedReplayBuffer) Sample(batchSize int, beta float64) ([]Transition, []int, []float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.size == 0 {
		return nil, nil, nil
	}
	if batchSize > b.size {
		batchSize = b.size
	}

	probs := make([]float64, b.size)
	var total float64
	for i := 0; i < b.size; i++ {
		p := math.Pow(b.priorities[i], b.alpha)
		probs[i] = p
		total += p
	}

	if total == 0 {
		total = 1.0
	}

	batch := make([]Transition, batchSize)
	indices := make([]int, batchSize)
	weights := make([]float64, batchSize)
	n := float64(b.size)

	var maxWeight float64
	for i := 0; i < batchSize; i++ {
		r := rand.Float64() * total
		var cum float64
		idx := 0
		for j := 0; j < b.size; j++ {
			cum += probs[j]
			if cum >= r {
				idx = j
				break
			}
			idx = j
		}
		batch[i] = b.transitions[idx]
		indices[i] = idx

		prob := probs[idx] / total
		w := math.Pow(n*prob, -beta)
		weights[i] = w
		if w > maxWeight {
			maxWeight = w
		}
	}

	if maxWeight > 0 {
		for i := range weights {
			weights[i] /= maxWeight
		}
	}

	return batch, indices, weights
}

func (b *PrioritizedReplayBuffer) UpdatePriority(idx int, tdError float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if idx < 0 || idx >= b.size {
		return
	}
	p := math.Abs(tdError) + 1e-6
	b.priorities[idx] = p
	if p > b.maxPriority {
		b.maxPriority = p
	}
}

func (b *PrioritizedReplayBuffer) Cap() int {
	rt := b.mu.RLock()
	defer b.mu.RUnlock(rt)
	return b.capacity
}

func (b *PrioritizedReplayBuffer) Len() int {
	rt := b.mu.RLock()
	defer b.mu.RUnlock(rt)
	return b.size
}

// BetaSchedule linearly anneals beta from start to end over a given number of steps.
type BetaSchedule struct {
	start   float64
	end     float64
	steps   int
	current int
}

func NewBetaSchedule(start, end float64, steps int) *BetaSchedule {
	return &BetaSchedule{start: start, end: end, steps: steps}
}

func (b *BetaSchedule) Next() float64 {
	if b.steps <= 0 {
		return b.end
	}
	frac := float64(b.current) / float64(b.steps)
	if frac > 1.0 {
		frac = 1.0
	}
	b.current++
	return b.start + (b.end-b.start)*frac
}
