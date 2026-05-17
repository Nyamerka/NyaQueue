package balancer

import (
	"hash/fnv"
	"math"
	"math/rand/v2"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/puzpuzpuz/xsync/v3"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

const (
	defaultPSAMaxBindings   = 100_000
	psaGracePeriod          = 500 * time.Millisecond
	psaRebalanceStealFactor = 1.5
)

type psaShard struct {
	mu           xsync.RBMutex
	keys         map[string]struct{}
	lastBindTime atomic.Int64
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

	if part, ok := p.bindings.Get(keyStr); ok {
		return part
	}

	if part, ok := p.claimFreePartition(); ok {
		p.bind(keyStr, part)
		return part
	}

	h := hashKey(key)
	part := int(h % uint64(numPartitions))
	p.bind(keyStr, part)
	return part
}

// claimFreePartition atomically claims the least-loaded free partition.
// Uses LoadAndDelete for atomic ownership transfer — first goroutine to
// successfully delete a partition from the free set owns it.
// A random skip offset prevents hash-order bias that starves some partitions.
func (p *PSA) claimFreePartition() (int, bool) {
	loads := p.loads.Load()
	skip := rand.IntN(p.numPartitions)
	seen := 0
	bestPart := -1
	bestLoad := math.MaxFloat64

	p.free.Range(func(id int, _ struct{}) bool {
		if seen < skip {
			seen++
			return true
		}
		load := math.MaxFloat64
		if loads != nil && id < len(*loads) {
			load = (*loads)[id]
		}
		if load < bestLoad {
			bestLoad = load
			bestPart = id
		}
		return true
	})

	if bestPart < 0 {
		p.free.Range(func(id int, _ struct{}) bool {
			load := math.MaxFloat64
			if loads != nil && id < len(*loads) {
				load = (*loads)[id]
			}
			if load < bestLoad {
				bestLoad = load
				bestPart = id
			}
			return true
		})
	}

	if bestPart < 0 {
		return 0, false
	}

	if _, loaded := p.free.LoadAndDelete(bestPart); loaded {
		return bestPart, true
	}
	return 0, false
}

func (p *PSA) bind(key string, part int) {
	p.bindings.Add(key, part)
	sh := &p.shards[part]
	sh.mu.Lock()
	sh.keys[key] = struct{}{}
	sh.mu.Unlock()
	sh.lastBindTime.Store(time.Now().UnixNano())
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

		// Periodic rebalance: steal bindings from overloaded partitions
		// and migrate to idle ones (load == 0 with active neighbors > avg*1.5).
		for i, load := range m.PartitionLoads {
			if load > 0 {
				continue
			}
			if i < len(m.QueueDepth) && m.QueueDepth[i] > 0 {
				continue
			}
			for j, jLoad := range m.PartitionLoads {
				if j == i || avg == 0 {
					continue
				}
				if jLoad > psaRebalanceStealFactor*avg {
					p.stealOneBinding(j, i)
					break
				}
			}
		}
	}
}

func (p *PSA) releasePartition(id int) {
	if id < 0 || id >= len(p.shards) {
		return
	}
	sh := &p.shards[id]

	lastBind := sh.lastBindTime.Load()
	if lastBind > 0 && time.Since(time.Unix(0, lastBind)) < psaGracePeriod {
		return
	}

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
		if cur, ok := p.bindings.Peek(k); ok && cur == id {
			p.bindings.Remove(k)
		}
	}
	p.free.Store(id, struct{}{})
}

// stealOneBinding moves one key binding from overloaded src to idle dst.
func (p *PSA) stealOneBinding(src, dst int) {
	if src < 0 || src >= len(p.shards) || dst < 0 || dst >= len(p.shards) {
		return
	}
	sh := &p.shards[src]
	sh.mu.Lock()
	var victim string
	for k := range sh.keys {
		victim = k
		break
	}
	if victim == "" {
		sh.mu.Unlock()
		return
	}
	delete(sh.keys, victim)
	sh.mu.Unlock()

	if cur, ok := p.bindings.Peek(victim); ok && cur == src {
		p.bindings.Remove(victim)
	}
	p.bind(victim, dst)
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

func hashKey(key []byte) uint64 {
	h := fnv.New64a()
	h.Write(key)
	return h.Sum64()
}
