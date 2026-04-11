package broker

import (
	"sync"
	"time"
)

const MaxPriority = 10

type PendingEntry struct {
	WalOffset int64
	ArrivedAt time.Time
}

type PriorityIndex struct {
	mu     sync.Mutex
	levels [MaxPriority][]PendingEntry // 0=lowest, 9=highest
	count  int
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

	pi.mu.Lock()
	defer pi.mu.Unlock()

	pi.levels[priority] = append(pi.levels[priority], PendingEntry{
		WalOffset: offset,
		ArrivedAt: arrivedAt,
	})
	pi.count++
}

// PopHighest returns the offset of the highest-priority pending message (FIFO within level).
func (pi *PriorityIndex) PopHighest() (PendingEntry, bool) {
	pi.mu.Lock()
	defer pi.mu.Unlock()

	for p := MaxPriority - 1; p >= 0; p-- {
		if len(pi.levels[p]) > 0 {
			entry := pi.levels[p][0]
			pi.levels[p] = pi.levels[p][1:]
			pi.count--
			return entry, true
		}
	}
	return PendingEntry{}, false
}

// PopWithThreshold pops: priority >= threshold by priority order, below threshold by FIFO.
func (pi *PriorityIndex) PopWithThreshold(threshold int) (PendingEntry, bool) {
	pi.mu.Lock()
	defer pi.mu.Unlock()

	// First try high-priority entries
	for p := MaxPriority - 1; p >= threshold; p-- {
		if len(pi.levels[p]) > 0 {
			entry := pi.levels[p][0]
			pi.levels[p] = pi.levels[p][1:]
			pi.count--
			return entry, true
		}
	}

	// Below threshold: oldest entry across all remaining levels (FIFO-like)
	bestIdx := -1
	var bestEntry PendingEntry
	for p := 0; p < threshold && p < MaxPriority; p++ {
		if len(pi.levels[p]) > 0 {
			e := pi.levels[p][0]
			if bestIdx == -1 || e.ArrivedAt.Before(bestEntry.ArrivedAt) {
				bestIdx = p
				bestEntry = e
			}
		}
	}
	if bestIdx >= 0 {
		pi.levels[bestIdx] = pi.levels[bestIdx][1:]
		pi.count--
		return bestEntry, true
	}
	return PendingEntry{}, false
}

func (pi *PriorityIndex) Len() int {
	pi.mu.Lock()
	defer pi.mu.Unlock()
	return pi.count
}

func (pi *PriorityIndex) LevelDistribution() [MaxPriority]int {
	pi.mu.Lock()
	defer pi.mu.Unlock()

	var dist [MaxPriority]int
	for i := range pi.levels {
		dist[i] = len(pi.levels[i])
	}
	return dist
}

// PromoteStale bumps entries older than ttl to the next priority level (anti-starvation).
func (pi *PriorityIndex) PromoteStale(ttl time.Duration) {
	pi.mu.Lock()
	defer pi.mu.Unlock()

	now := time.Now()
	// Iterate from high to low to avoid cascading promotions within a single call.
	for p := MaxPriority - 2; p >= 0; p-- {
		fresh := pi.levels[p][:0]
		for _, e := range pi.levels[p] {
			if now.Sub(e.ArrivedAt) > ttl {
				pi.levels[p+1] = append(pi.levels[p+1], e)
			} else {
				fresh = append(fresh, e)
			}
		}
		pi.levels[p] = fresh
	}
}
