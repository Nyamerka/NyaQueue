package broker

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type PriorityIndexSuite struct {
	suite.Suite
}

func TestPriorityIndexSuite(t *testing.T) { suite.Run(t, new(PriorityIndexSuite)) }

func (s *PriorityIndexSuite) TestAddAndPopHighest() {
	tests := []struct {
		name string
		adds []struct {
			priority int
			offset   int64
		}
		wantOrder []int64
	}{
		{
			"single",
			[]struct {
				priority int
				offset   int64
			}{{5, 100}},
			[]int64{100},
		},
		{
			"highest_first",
			[]struct {
				priority int
				offset   int64
			}{{1, 10}, {9, 20}, {5, 30}},
			[]int64{20, 30, 10},
		},
		{
			"same_priority_fifo",
			[]struct {
				priority int
				offset   int64
			}{{3, 1}, {3, 2}, {3, 3}},
			[]int64{1, 2, 3},
		},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			pi := NewPriorityIndex()
			now := time.Now()

			for _, a := range tc.adds {
				pi.Add(a.priority, a.offset, now)
			}
			require.Equal(s.T(), len(tc.adds), pi.Len())

			for _, wantOff := range tc.wantOrder {
				entry, ok := pi.PopHighest()
				require.True(s.T(), ok)
				require.Equal(s.T(), wantOff, entry.WalOffset)
			}

			_, ok := pi.PopHighest()
			require.False(s.T(), ok)
			require.Equal(s.T(), 0, pi.Len())
		})
	}
}

func (s *PriorityIndexSuite) TestPopWithThreshold() {
	pi := NewPriorityIndex()
	now := time.Now()

	pi.Add(1, 10, now)
	pi.Add(5, 20, now.Add(time.Millisecond))
	pi.Add(8, 30, now.Add(2*time.Millisecond))

	// threshold=5: priority>=5 first by priority, below by FIFO
	entry, ok := pi.PopWithThreshold(5)
	require.True(s.T(), ok)
	require.Equal(s.T(), int64(30), entry.WalOffset) // priority 8

	entry, ok = pi.PopWithThreshold(5)
	require.True(s.T(), ok)
	require.Equal(s.T(), int64(20), entry.WalOffset) // priority 5

	entry, ok = pi.PopWithThreshold(5)
	require.True(s.T(), ok)
	require.Equal(s.T(), int64(10), entry.WalOffset) // priority 1 (below threshold, FIFO)
}

func (s *PriorityIndexSuite) TestLevelDistribution() {
	pi := NewPriorityIndex()
	now := time.Now()

	pi.Add(0, 1, now)
	pi.Add(0, 2, now)
	pi.Add(9, 3, now)

	dist := pi.LevelDistribution()
	require.Equal(s.T(), 2, dist[0])
	require.Equal(s.T(), 1, dist[9])
	for i := 1; i < 9; i++ {
		require.Equal(s.T(), 0, dist[i])
	}
}

func (s *PriorityIndexSuite) TestPromoteStale() {
	pi := NewPriorityIndex()

	staleTime := time.Now().Add(-10 * time.Second)
	freshTime := time.Now().Add(time.Hour)

	pi.Add(0, 10, staleTime)
	pi.Add(0, 20, freshTime)

	pi.PromoteStale(time.Second)

	dist := pi.LevelDistribution()
	require.Equal(s.T(), 1, dist[0], "one fresh entry stays at level 0")
	require.Equal(s.T(), 1, dist[1], "one stale entry promoted to level 1")
}

func (s *PriorityIndexSuite) TestClampPriority() {
	pi := NewPriorityIndex()
	now := time.Now()

	pi.Add(-1, 1, now)
	pi.Add(100, 2, now)

	entry, ok := pi.PopHighest()
	require.True(s.T(), ok)
	require.Equal(s.T(), int64(2), entry.WalOffset) // clamped to 9

	entry, ok = pi.PopHighest()
	require.True(s.T(), ok)
	require.Equal(s.T(), int64(1), entry.WalOffset) // clamped to 0
}

func (s *PriorityIndexSuite) TestConcurrentAddAndPop() {
	pi := NewPriorityIndex()
	var wg sync.WaitGroup
	now := time.Now()

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				pi.Add(id%MaxPriority, int64(id*1000+j), now)
			}
		}(i)
	}

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				pi.PopHighest()
			}
		}()
	}

	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				pi.LevelDistribution()
				pi.Len()
			}
		}()
	}

	wg.Wait()
}

func (s *PriorityIndexSuite) TestConcurrentPopWithThresholdAndAdd() {
	pi := NewPriorityIndex()
	var wg sync.WaitGroup
	now := time.Now()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 1000; j++ {
			pi.Add(j%MaxPriority, int64(j), now)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 500; j++ {
			pi.PopWithThreshold(j % MaxPriority)
		}
	}()

	wg.Wait()
}

func (s *PriorityIndexSuite) TestPriorityIndex_CapReject() {
	pi := NewPriorityIndex(WithMaxPerLevel(10))
	now := time.Now()

	accepted := 0
	for i := 0; i < 20; i++ {
		if pi.Add(3, int64(i), now.Add(time.Duration(i)*time.Millisecond)) {
			accepted++
		}
	}
	require.Equal(s.T(), 10, accepted, "first 10 should be accepted (fresh slots)")

	dist := pi.LevelDistribution()
	require.Equal(s.T(), 10, dist[3])
}

func (s *PriorityIndexSuite) TestPriorityIndex_RingEviction() {
	pi := NewPriorityIndex(WithMaxPerLevel(5))
	now := time.Now()

	for i := 0; i < 5; i++ {
		ok := pi.Add(0, int64(i+1), now.Add(time.Duration(i)*time.Millisecond))
		require.True(s.T(), ok)
	}

	// Add 3 more — should evict the 3 oldest (offsets 1, 2, 3).
	for i := 5; i < 8; i++ {
		ok := pi.Add(0, int64(i+1), now.Add(time.Duration(i)*time.Millisecond))
		require.False(s.T(), ok, "level at cap, expect eviction signal")
	}

	// Remaining entries should be offsets 4,5,6,7,8 (oldest 1,2,3 evicted).
	require.Equal(s.T(), 5, pi.Len())
	for _, wantOff := range []int64{4, 5, 6, 7, 8} {
		entry, ok := pi.PopHighest()
		require.True(s.T(), ok)
		require.Equal(s.T(), wantOff, entry.WalOffset)
	}
}
