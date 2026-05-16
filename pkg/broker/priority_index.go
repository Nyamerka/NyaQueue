package broker

import (
	"sync"
	"sync/atomic"
	"time"
)

const MaxPriority = 10

type PendingEntry struct {
	WalOffset int64
	ArrivedAt time.Time
}

type priorityLevel struct {
	mu      sync.Mutex
	entries []PendingEntry
	count   atomic.Int32
}

type PriorityIndex struct {
	levels [MaxPriority]priorityLevel
	total  atomic.Int64
}

func NewPriorityIndex() *PriorityIndex {
	return &PriorityIndex{}
}

func (pi *PriorityIndex) Add(priority int, offset int64, arrivedAt time.Time) {
	if priority < 0 {
		priority = 0
	}
	if priority >= MaxPriority {
		priority = MaxPriority - 1
	}

	lv := &pi.levels[priority]
	lv.mu.Lock()
	lv.entries = append(lv.entries, PendingEntry{
		WalOffset: offset,
		ArrivedAt: arrivedAt,
	})
	lv.count.Add(1)
	lv.mu.Unlock()
	pi.total.Add(1)
}

func (pi *PriorityIndex) PopHighest() (PendingEntry, bool) {
	for p := MaxPriority - 1; p >= 0; p-- {
		lv := &pi.levels[p]
		lv.mu.Lock()
		if len(lv.entries) > 0 {
			entry := lv.entries[0]
			lv.entries = lv.entries[1:]
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
		if len(lv.entries) > 0 {
			entry := lv.entries[0]
			lv.entries = lv.entries[1:]
			lv.count.Add(-1)
			lv.mu.Unlock()
			pi.total.Add(-1)
			return entry, true
		}
		lv.mu.Unlock()
	}

	// Below threshold: oldest entry across remaining levels (FIFO-like).
	// Hold lock on current-best level while scanning; acquire in ascending
	// order to prevent deadlock.
	bestPriority := -1
	var bestTime time.Time
	for p := 0; p < threshold && p < MaxPriority; p++ {
		lv := &pi.levels[p]
		lv.mu.Lock()
		if len(lv.entries) > 0 {
			e := lv.entries[0]
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
		entry := lv.entries[0]
		lv.entries = lv.entries[1:]
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

		var stale []PendingEntry
		fresh := lv.entries[:0]
		for _, e := range lv.entries {
			if now.Sub(e.ArrivedAt) > ttl {
				stale = append(stale, e)
			} else {
				fresh = append(fresh, e)
			}
		}
		lv.entries = fresh
		lv.count.Store(int32(len(fresh)))
		lv.mu.Unlock()

		if len(stale) > 0 {
			up := &pi.levels[p+1]
			up.mu.Lock()
			up.entries = append(up.entries, stale...)
			up.count.Store(int32(len(up.entries)))
			up.mu.Unlock()
		}
	}
}
