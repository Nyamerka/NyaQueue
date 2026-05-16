package balancer

import (
	"hash/fnv"
	"math"
	"sync"
	"sync/atomic"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/puzpuzpuz/xsync/v3"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultPSAMaxBindings = 100_000
)

// PSA implements the Partition Selection Algorithm from Paper 2 (DMSCO).
//
// Algorithm:
//  1. If key is bound to a partition -> route there (ordering guarantee)
//  2. If free partitions exist (m > 0) -> hash(key) % (m+1), bind
//  3. No free partitions -> hash(key) % n
//  4. Background: if partition becomes empty -> release bindings, m++
//
// Uses an LRU cache for bindings to prevent unbounded memory growth.
// SelectPartition uses a two-phase locking strategy: RLock for cache hits,
// full Lock only for new bindings.
type PSA struct {
	mu              sync.RWMutex
	bindings        *lru.Cache[string, int]
	partitionToKeys map[int]map[string]struct{}
	free            *xsync.MapOf[int, struct{}]
	loads           []float64

	evictionCount atomic.Int64
}

func NewPSA(numPartitions int) *PSA {
	p := &PSA{
		partitionToKeys: make(map[int]map[string]struct{}, numPartitions),
		free:            xsync.NewMapOf[int, struct{}](),
	}

	cache, _ := lru.NewWithEvict[string, int](defaultPSAMaxBindings, func(key string, partID int) {
		p.evictionCount.Add(1)
		p.removeFromReverseIndex(key, partID)
	})
	p.bindings = cache

	for i := 0; i < numPartitions; i++ {
		p.free.Store(i, struct{}{})
	}
	return p
}

func (p *PSA) removeFromReverseIndex(key string, partID int) {
	if keys, ok := p.partitionToKeys[partID]; ok {
		delete(keys, key)
		if len(keys) == 0 {
			delete(p.partitionToKeys, partID)
			p.free.Store(partID, struct{}{})
		}
	}
}

func (p *PSA) SelectPartition(_ string, key []byte, numPartitions int) int {
	keyStr := string(key)

	// Fast path: RLock for already-bound keys.
	// Uses Peek (not Get) because Get mutates the LRU internal list,
	// which is not safe under a read lock with concurrent readers.
	p.mu.RLock()
	if part, ok := p.bindings.Peek(keyStr); ok {
		p.mu.RUnlock()
		return part
	}
	p.mu.RUnlock()

	// Slow path: need to create a new binding.
	p.mu.Lock()
	defer p.mu.Unlock()

	// Double-check after acquiring write lock.
	if part, ok := p.bindings.Get(keyStr); ok {
		return part
	}

	freeList := p.collectFree()
	if len(freeList) > 0 {
		part := p.leastLoadedFree(freeList)
		p.bind(keyStr, part)
		p.free.Delete(part)
		return part
	}

	h := hashKey(key)
	return int(h % uint64(numPartitions))
}

func (p *PSA) bind(key string, part int) {
	p.bindings.Add(key, part)
	if p.partitionToKeys[part] == nil {
		p.partitionToKeys[part] = make(map[string]struct{})
	}
	p.partitionToKeys[part][key] = struct{}{}
}

func (p *PSA) leastLoadedFree(freeList []int) int {
	if len(p.loads) == 0 {
		return freeList[0]
	}
	best := freeList[0]
	bestLoad := math.MaxFloat64
	for _, id := range freeList {
		if id < len(p.loads) && p.loads[id] < bestLoad {
			bestLoad = p.loads[id]
			best = id
		}
	}
	return best
}

func (p *PSA) OnMetrics(m broker.Metrics) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if cap(p.loads) < len(m.PartitionLoads) {
		p.loads = make([]float64, len(m.PartitionLoads))
	}
	p.loads = p.loads[:len(m.PartitionLoads)]
	copy(p.loads, m.PartitionLoads)

	for i, depth := range m.QueueDepth {
		if depth == 0 {
			p.releasePartition(i)
		}
	}

	if len(m.PartitionLoads) > 1 {
		avg := avgLoad(m.PartitionLoads)
		for i, load := range m.PartitionLoads {
			if avg > 0 && load > PSARebalanceLoadFactor*avg {
				p.releasePartition(i)
			}
		}
	}
}

// releasePartition uses the reverse index for O(|keys for partition|) cleanup.
// Collects keys first to avoid mutating partitionToKeys during eviction callback iteration.
func (p *PSA) releasePartition(id int) {
	keys, ok := p.partitionToKeys[id]
	if !ok {
		p.free.Store(id, struct{}{})
		return
	}
	toRemove := make([]string, 0, len(keys))
	for k := range keys {
		toRemove = append(toRemove, k)
	}
	delete(p.partitionToKeys, id)
	for _, k := range toRemove {
		p.bindings.Remove(k)
	}
	p.free.Store(id, struct{}{})
}

func (p *PSA) EvictionCount() int64 {
	return p.evictionCount.Load()
}

func avgLoad(loads []float64) float64 {
	if len(loads) == 0 {
		return 0
	}
	var sum float64
	for _, v := range loads {
		sum += v
	}
	return sum / float64(len(loads))
}

func (p *PSA) collectFree() []int {
	var out []int
	p.free.Range(func(id int, _ struct{}) bool {
		out = append(out, id)
		return true
	})
	return out
}

func hashKey(key []byte) uint64 {
	h := fnv.New64a()
	h.Write(key)
	return h.Sum64()
}
