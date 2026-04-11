package balancer

import (
	"hash/fnv"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

// PSA implements the Partition Selection Algorithm from Paper 2 (DMSCO).
//
// Algorithm:
//  1. If key is bound to a partition -> route there (ordering guarantee)
//  2. If free partitions exist (m > 0) -> hash(key) % (m+1), bind
//  3. No free partitions -> hash(key) % n
//  4. Background: if partition becomes empty -> release bindings, m++
type PSA struct {
	mu       sync.Mutex
	bindings map[string]int   // key -> partition
	free     map[int]struct{} // set of unbound partitions
	loads    []float64
}

func NewPSA(numPartitions int) *PSA {
	free := make(map[int]struct{}, numPartitions)
	for i := 0; i < numPartitions; i++ {
		free[i] = struct{}{}
	}
	return &PSA{
		bindings: make(map[string]int),
		free:     free,
	}
}

func (p *PSA) SelectPartition(_ string, key []byte, numPartitions int) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	keyStr := string(key)

	if part, bound := p.bindings[keyStr]; bound {
		return part
	}

	freeList := p.freeSlice()
	if len(freeList) > 0 {
		h := hashKey(key)
		idx := h % uint64(len(freeList)+1)
		if int(idx) < len(freeList) {
			part := freeList[idx]
			p.bindings[keyStr] = part
			delete(p.free, part)
			return part
		}
	}

	h := hashKey(key)
	return int(h % uint64(numPartitions))
}

func (p *PSA) OnMetrics(m broker.Metrics) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.loads = m.PartitionLoads

	// Release bindings for empty partitions.
	for i, depth := range m.QueueDepth {
		if depth == 0 {
			for k, part := range p.bindings {
				if part == i {
					delete(p.bindings, k)
				}
			}
			p.free[i] = struct{}{}
		}
	}
}

func (p *PSA) freeSlice() []int {
	s := make([]int, 0, len(p.free))
	for id := range p.free {
		s = append(s, id)
	}
	return s
}

func hashKey(key []byte) uint64 {
	h := fnv.New64a()
	h.Write(key)
	return h.Sum64()
}
