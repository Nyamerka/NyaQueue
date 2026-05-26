package nn

import (
	"math"
	"math/rand/v2"
	"sync/atomic"
)

type Transition struct {
	State     []float64
	Action    []float64
	Reward    float64
	NextState []float64
	Done      bool
}

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

// SumTree is a segment tree that supports O(log N) priority-weighted sampling
// and O(log N) priority updates (Schaul et al., ICML 2016).
type SumTree struct {
	capacity int
	tree     []float64
	data     []Transition
	size     int
	writeIdx int
}

func NewSumTree(capacity int) *SumTree {
	return &SumTree{
		capacity: capacity,
		tree:     make([]float64, 2*capacity),
		data:     make([]Transition, capacity),
	}
}

func (st *SumTree) Update(idx int, priority float64) {
	treeIdx := idx + st.capacity
	delta := priority - st.tree[treeIdx]
	st.tree[treeIdx] = priority
	for treeIdx > 1 {
		treeIdx /= 2
		st.tree[treeIdx] += delta
	}
}

func (st *SumTree) Add(transition Transition, priority float64) int {
	idx := st.writeIdx % st.capacity
	st.data[idx] = transition
	st.Update(idx, priority)
	st.writeIdx++
	if st.size < st.capacity {
		st.size++
	}
	return idx
}

func (st *SumTree) Sample(prefixSum float64) int {
	idx := 1
	for idx < st.capacity {
		left := 2 * idx
		if st.tree[left] > prefixSum {
			idx = left
		} else {
			prefixSum -= st.tree[left]
			idx = left + 1
		}
	}
	dataIdx := idx - st.capacity
	if dataIdx >= st.size {
		dataIdx = st.size - 1
	}
	if dataIdx < 0 {
		dataIdx = 0
	}
	return dataIdx
}

func (st *SumTree) Total() float64 { return st.tree[1] }

func (st *SumTree) Priority(idx int) float64 {
	return st.tree[idx+st.capacity]
}

// PrioritizedReplayBuffer samples transitions proportional to priority^alpha
// using a SumTree for O(log N) sampling (vs O(N) linear scan).
//
// Not safe for concurrent use; designed for the single-writer trainLoop pattern
// where Push and Sample are called from the same goroutine.
type PrioritizedReplayBuffer struct {
	sumTree     *SumTree
	alpha       float64
	maxPriority float64
	atomicLen   atomic.Int64
}

func NewPrioritizedReplayBuffer(capacity int, alpha float64) *PrioritizedReplayBuffer {
	return &PrioritizedReplayBuffer{
		sumTree:     NewSumTree(capacity),
		alpha:       alpha,
		maxPriority: 1.0,
	}
}

func (b *PrioritizedReplayBuffer) Push(t Transition) {
	cp := Transition{
		State:     append([]float64(nil), t.State...),
		Action:    append([]float64(nil), t.Action...),
		Reward:    t.Reward,
		NextState: append([]float64(nil), t.NextState...),
		Done:      t.Done,
	}

	priority := math.Pow(b.maxPriority, b.alpha)
	b.sumTree.Add(cp, priority)
	b.atomicLen.Store(int64(b.sumTree.size))
}

// Sample returns a batch of transitions sampled proportional to priority^alpha
// using stratified sampling over the SumTree. O(B * log N) total.
// IS-weights are computed as w_i = (N * P(i))^(-beta) / max(w), normalized so max=1.0.
func (b *PrioritizedReplayBuffer) Sample(batchSize int, beta float64) ([]Transition, []int, []float64) {
	n := b.sumTree.size
	if n == 0 {
		return nil, nil, nil
	}
	if batchSize > n {
		batchSize = n
	}

	batch := make([]Transition, batchSize)
	indices := make([]int, batchSize)
	weights := make([]float64, batchSize)

	total := b.sumTree.Total()
	if total == 0 {
		total = 1.0
	}

	segment := total / float64(batchSize)
	nf := float64(n)

	var maxWeight float64
	for i := 0; i < batchSize; i++ {
		lo := segment * float64(i)
		s := lo + rand.Float64()*segment

		idx := b.sumTree.Sample(s)
		batch[i] = b.sumTree.data[idx]
		indices[i] = idx

		prob := b.sumTree.Priority(idx) / total
		if prob < 1e-10 {
			prob = 1e-10
		}
		w := math.Pow(nf*prob, -beta)
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
	if idx < 0 || idx >= b.sumTree.size {
		return
	}
	p := math.Abs(tdError) + 1e-6
	if p > b.maxPriority {
		b.maxPriority = p
	}
	b.sumTree.Update(idx, math.Pow(p, b.alpha))
}

// ToBuffer exports all transitions for warmup caching.
func (b *PrioritizedReplayBuffer) ToBuffer() []Transition {
	n := b.sumTree.size
	if n == 0 {
		return nil
	}
	buf := make([]Transition, n)
	for i := 0; i < n; i++ {
		t := b.sumTree.data[i]
		buf[i] = Transition{
			State:     append([]float64(nil), t.State...),
			Action:    append([]float64(nil), t.Action...),
			Reward:    t.Reward,
			NextState: append([]float64(nil), t.NextState...),
			Done:      t.Done,
		}
	}
	return buf
}

// LoadFromBuffer imports previously cached transitions.
func (b *PrioritizedReplayBuffer) LoadFromBuffer(buf []Transition) {
	for _, t := range buf {
		b.Push(t)
	}
}

func (b *PrioritizedReplayBuffer) Cap() int {
	return b.sumTree.capacity
}

// Len is safe to call from any goroutine.
func (b *PrioritizedReplayBuffer) Len() int {
	return int(b.atomicLen.Load())
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
