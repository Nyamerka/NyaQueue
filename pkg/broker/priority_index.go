package broker

import (
	"math/bits"
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
)

const MaxPriority = 10

type PendingEntry struct {
	WalOffset int64
	ArrivedAt time.Time
}

type entryRing struct {
	buf  []PendingEntry
	head int
	size int
	cap  int
}

func (r *entryRing) pushBack(e PendingEntry) (evicted PendingEntry, didEvict bool) {
	if r.cap > 0 && r.size >= r.cap {
		evicted = r.buf[r.head]
		didEvict = true
		r.buf[r.head] = e
		r.head = (r.head + 1) % r.cap
		return
	}
	if r.size == len(r.buf) {
		r.grow()
	}
	idx := (r.head + r.size) % len(r.buf)
	r.buf[idx] = e
	r.size++
	return
}

func (r *entryRing) grow() {
	newCap := max(len(r.buf)*2, 16)
	if r.cap > 0 && newCap > r.cap {
		newCap = r.cap
	}
	newBuf := make([]PendingEntry, newCap)
	for i := range r.size {
		newBuf[i] = r.buf[(r.head+i)%len(r.buf)]
	}
	r.buf = newBuf
	r.head = 0
}

func (r *entryRing) popFront() (PendingEntry, bool) {
	if r.size == 0 {
		return PendingEntry{}, false
	}
	e := r.buf[r.head]
	r.buf[r.head] = PendingEntry{}
	r.head = (r.head + 1) % len(r.buf)
	r.size--
	return e, true
}

func (r *entryRing) peekFront() (PendingEntry, bool) {
	if r.size == 0 {
		return PendingEntry{}, false
	}
	return r.buf[r.head], true
}

func (r *entryRing) linearize() []PendingEntry {
	out := make([]PendingEntry, r.size)
	for i := range r.size {
		out[i] = r.buf[(r.head+i)%len(r.buf)]
	}
	return out
}

func (r *entryRing) resetFrom(entries []PendingEntry) {
	if r.cap > 0 && len(entries) > r.cap {
		entries = entries[len(entries)-r.cap:]
	}
	need := len(entries)
	if need > len(r.buf) {
		r.buf = make([]PendingEntry, need)
	}
	copy(r.buf, entries)
	for i := need; i < len(r.buf); i++ {
		r.buf[i] = PendingEntry{}
	}
	r.head = 0
	r.size = need
}

type priorityLevel struct {
	mu    xsync.RBMutex
	ring  entryRing
	count atomic.Int32
}

type PriorityIndex struct {
	levels      [MaxPriority]priorityLevel
	total       atomic.Int64
	nonEmpty    atomic.Uint64 // bitmask: bit i set iff levels[i].count > 0
	maxPerLevel int
}

type PriorityIndexOption func(*PriorityIndex)

func WithMaxPerLevel(n int) PriorityIndexOption {
	return func(pi *PriorityIndex) { pi.maxPerLevel = n }
}

func NewPriorityIndex(opts ...PriorityIndexOption) *PriorityIndex {
	pi := &PriorityIndex{}
	for _, opt := range opts {
		opt(pi)
	}
	if pi.maxPerLevel > 0 {
		for i := range pi.levels {
			pi.levels[i].ring = entryRing{cap: pi.maxPerLevel}
		}
	}
	return pi
}

func (pi *PriorityIndex) Add(priority int, offset int64, arrivedAt time.Time) bool {
	if priority < 0 {
		priority = 0
	}
	if priority >= MaxPriority {
		priority = MaxPriority - 1
	}

	lv := &pi.levels[priority]
	lv.mu.Lock()
	_, evicted := lv.ring.pushBack(PendingEntry{
		WalOffset: offset,
		ArrivedAt: arrivedAt,
	})
	if evicted {
		lv.mu.Unlock()
		return false
	}
	lv.count.Add(1)
	lv.mu.Unlock()
	pi.nonEmpty.Or(1 << uint(priority))
	pi.total.Add(1)
	return true
}

func (pi *PriorityIndex) PopHighest() (PendingEntry, bool) {
	mask := pi.nonEmpty.Load()
	if mask == 0 {
		return PendingEntry{}, false
	}
	for p := MaxPriority - 1; p >= 0; p-- {
		if mask&(1<<uint(p)) == 0 {
			continue
		}
		if entry, ok := pi.popFromLevel(p); ok {
			return entry, true
		}
	}
	return PendingEntry{}, false
}

func (pi *PriorityIndex) PopWithThreshold(threshold int) (PendingEntry, bool) {
	mask := pi.nonEmpty.Load()
	if mask == 0 {
		return PendingEntry{}, false
	}

	aboveMask := mask & (^uint64(0) << uint(threshold))
	if aboveMask != 0 {
		top := bits.Len64(aboveMask) - 1
		if entry, ok := pi.popFromLevel(top); ok {
			return entry, true
		}
	}

	belowMask := mask & ((1 << uint(threshold)) - 1)
	if belowMask != 0 {
		top := bits.Len64(belowMask) - 1
		if entry, ok := pi.popFromLevel(top); ok {
			return entry, true
		}
	}

	return PendingEntry{}, false
}

func (pi *PriorityIndex) popFromLevel(p int) (PendingEntry, bool) {
	lv := &pi.levels[p]
	lv.mu.Lock()
	entry, ok := lv.ring.popFront()
	if ok {
		cnt := lv.count.Add(-1)
		lv.mu.Unlock()
		if cnt <= 0 {
			pi.nonEmpty.And(^(1 << uint(p)))
		}
		pi.total.Add(-1)
		return entry, true
	}
	lv.mu.Unlock()
	pi.nonEmpty.And(^(1 << uint(p)))
	return PendingEntry{}, false
}

func (pi *PriorityIndex) Len() int {
	return int(pi.total.Load())
}

func (pi *PriorityIndex) LevelDistribution() [MaxPriority]int {
	var dist [MaxPriority]int
	for i := range pi.levels {
		dist[i] = int(pi.levels[i].count.Load())
	}
	return dist
}

type DeficitState struct {
	Weight          [MaxPriority]float64
	deficit         [MaxPriority]float64
	tick            int64
	topReserveEvery int64
}

func NewDeficitState(topReserveEvery int64) *DeficitState {
	ds := &DeficitState{topReserveEvery: topReserveEvery}
	for i := range ds.Weight {
		ds.Weight[i] = 1.0 / float64(MaxPriority)
	}
	return ds
}

func (pi *PriorityIndex) PopWithBudget(ds *DeficitState) (PendingEntry, bool) {
	mask := pi.nonEmpty.Load()
	if mask == 0 {
		return PendingEntry{}, false
	}

	ds.tick++

	if ds.topReserveEvery > 0 && ds.tick%ds.topReserveEvery == 0 {
		top := bits.Len64(mask) - 1
		if e, ok := pi.popFromLevel(top); ok {
			return e, true
		}
	}

	for p := MaxPriority - 1; p >= 0; p-- {
		if mask&(1<<uint(p)) == 0 {
			continue
		}
		ds.deficit[p] += ds.Weight[p]
		if ds.deficit[p] >= 1.0 {
			if e, ok := pi.popFromLevel(p); ok {
				ds.deficit[p] -= 1.0
				return e, true
			}
		}
	}

	return pi.PopHighest()
}

func (pi *PriorityIndex) PromoteStale(ttl time.Duration) {
	now := time.Now()
	for p := MaxPriority - 2; p >= 0; p-- {
		lv := &pi.levels[p]
		lv.mu.Lock()

		all := lv.ring.linearize()
		var stale, fresh []PendingEntry
		for _, e := range all {
			if now.Sub(e.ArrivedAt) > ttl {
				stale = append(stale, e)
			} else {
				fresh = append(fresh, e)
			}
		}
		lv.ring.resetFrom(fresh)
		lv.count.Store(int32(len(fresh)))
		if len(fresh) == 0 {
			pi.nonEmpty.And(^(1 << uint(p)))
		}
		lv.mu.Unlock()

		if len(stale) > 0 {
			up := &pi.levels[p+1]
			up.mu.Lock()
			for _, e := range stale {
				_, evicted := up.ring.pushBack(e)
				if evicted {
					pi.total.Add(-1)
				}
			}
			up.count.Store(int32(up.ring.size))
			if up.ring.size > 0 {
				pi.nonEmpty.Or(1 << uint(p+1))
			}
			up.mu.Unlock()
		}
	}
}
