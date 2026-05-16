package broker

import (
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
)

const MaxPriority = 10

type PendingEntry struct {
	WalOffset int64
	ArrivedAt time.Time
}

// entryRing is a circular buffer of PendingEntry with optional fixed capacity.
// When cap > 0, the buffer never grows beyond cap; pushBack evicts the oldest
// entry once full. When cap == 0, the buffer grows dynamically.
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

// linearize returns a contiguous copy of all entries, oldest first.
func (r *entryRing) linearize() []PendingEntry {
	out := make([]PendingEntry, r.size)
	for i := range r.size {
		out[i] = r.buf[(r.head+i)%len(r.buf)]
	}
	return out
}

// resetFrom replaces ring contents. Entries exceeding cap are silently truncated
// (keeping the newest).
func (r *entryRing) resetFrom(entries []PendingEntry) {
	if r.cap > 0 && len(entries) > r.cap {
		entries = entries[len(entries)-r.cap:]
	}
	need := len(entries)
	if need > len(r.buf) {
		r.buf = make([]PendingEntry, need)
	}
	copy(r.buf, entries)
	// Zero out trailing slots so stale pointers don't prevent GC.
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
			pi.levels[i].ring = entryRing{
				buf: make([]PendingEntry, pi.maxPerLevel),
				cap: pi.maxPerLevel,
			}
		}
	}
	return pi
}

// Add inserts an entry into the priority level. Returns true if the entry was
// added to a free slot, false if the level was at capacity and the oldest entry
// was evicted to make room.
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
	pi.total.Add(1)
	return true
}

func (pi *PriorityIndex) PopHighest() (PendingEntry, bool) {
	for p := MaxPriority - 1; p >= 0; p-- {
		lv := &pi.levels[p]
		lv.mu.Lock()
		entry, ok := lv.ring.popFront()
		if ok {
			lv.count.Add(-1)
			lv.mu.Unlock()
			pi.total.Add(-1)
			return entry, true
		}
		lv.mu.Unlock()
	}
	return PendingEntry{}, false
}

func (pi *PriorityIndex) PopWithThreshold(threshold int) (PendingEntry, bool) {
	for p := MaxPriority - 1; p >= threshold; p-- {
		lv := &pi.levels[p]
		lv.mu.Lock()
		entry, ok := lv.ring.popFront()
		if ok {
			lv.count.Add(-1)
			lv.mu.Unlock()
			pi.total.Add(-1)
			return entry, true
		}
		lv.mu.Unlock()
	}

	bestPriority := -1
	var bestTime time.Time
	for p := 0; p < threshold && p < MaxPriority; p++ {
		lv := &pi.levels[p]
		lv.mu.Lock()
		e, ok := lv.ring.peekFront()
		if ok {
			if bestPriority == -1 || e.ArrivedAt.Before(bestTime) {
				if bestPriority >= 0 {
					pi.levels[bestPriority].mu.Unlock()
				}
				bestPriority = p
				bestTime = e.ArrivedAt
			} else {
				lv.mu.Unlock()
			}
		} else {
			lv.mu.Unlock()
		}
	}
	if bestPriority >= 0 {
		lv := &pi.levels[bestPriority]
		entry, _ := lv.ring.popFront()
		lv.count.Add(-1)
		lv.mu.Unlock()
		pi.total.Add(-1)
		return entry, true
	}
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
			up.mu.Unlock()
		}
	}
}
