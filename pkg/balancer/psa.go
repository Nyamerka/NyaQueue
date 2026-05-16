package balancer

import (
	"hash/fnv"
	"math"
	"sync/atomic"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/puzpuzpuz/xsync/v3"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultPSAMaxBindings = 100_000
)

type psaShard struct {
	mu   xsync.RBMutex
	keys map[string]struct{}
}

type PSA struct {
	bindings *lru.Cache[string, int]
	shards   []psaShard
	free     *xsync.MapOf[int, struct{}]
	loads    atomic.Pointer[[]float64]

	numPartitions int
	evictionCount atomic.Int64
}

func NewPSA(numPartitions int) *PSA {
	p := &PSA{
		shards:        make([]psaShard, numPartitions),
		free:          xsync.NewMapOf[int, struct{}](),
		numPartitions: numPartitions,
	}
	for i := range p.shards {
		p.shards[i].keys = make(map[string]struct{})
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
	if partID < 0 || partID >= len(p.shards) {
		return
	}
	sh := &p.shards[partID]
	sh.mu.Lock()
	delete(sh.keys, key)
	if len(sh.keys) == 0 {
		p.free.Store(partID, struct{}{})
	}
	sh.mu.Unlock()
}

func (p *PSA) SelectPartition(_ string, key []byte, numPartitions int) int {
	keyStr := string(key)

	if part, ok := p.bindings.Peek(keyStr); ok {
		return part
	}

	// Double-check via Get (promotes in LRU).
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
	// Add first — may trigger eviction which locks a (potentially different) shard.
	p.bindings.Add(key, part)
	sh := &p.shards[part]
	sh.mu.Lock()
	sh.keys[key] = struct{}{}
	sh.mu.Unlock()
}

func (p *PSA) leastLoadedFree(freeList []int) int {
	loads := p.loads.Load()
	if loads == nil || len(*loads) == 0 {
		return freeList[0]
	}
	ld := *loads
	best := freeList[0]
	bestLoad := math.MaxFloat64
	for _, id := range freeList {
		if id < len(ld) && ld[id] < bestLoad {
			bestLoad = ld[id]
			best = id
		}
	}
	return best
}

func (p *PSA) OnMetrics(m broker.Metrics) {
	cp := make([]float64, len(m.PartitionLoads))
	copy(cp, m.PartitionLoads)
	p.loads.Store(&cp)

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

func (p *PSA) releasePartition(id int) {
	if id < 0 || id >= len(p.shards) {
		return
	}
	sh := &p.shards[id]
	sh.mu.Lock()
	if len(sh.keys) == 0 {
		sh.mu.Unlock()
		p.free.Store(id, struct{}{})
		return
	}
	toRemove := make([]string, 0, len(sh.keys))
	for k := range sh.keys {
		toRemove = append(toRemove, k)
	}
	clear(sh.keys)
	sh.mu.Unlock()

	for _, k := range toRemove {
		// Only remove from cache if it still points to this partition,
		// to avoid invalidating a concurrent re-bind to a different partition.
		if cur, ok := p.bindings.Peek(k); ok && cur == id {
			p.bindings.Remove(k)
		}
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
