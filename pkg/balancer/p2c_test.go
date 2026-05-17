package balancer

import (
	"math"
	"sync"
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type P2CSuite struct {
	suite.Suite
}

func TestP2CSuite(t *testing.T) { suite.Run(t, new(P2CSuite)) }

func (s *P2CSuite) TestRandomTiebreakChiSquared() {
	p := NewPowerOfTwoChoices()
	p.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.5, 0.5},
			QueueDepth:     []int{10, 10},
		},
	})

	const n = 1_000_000
	counts := [2]int{}
	for i := 0; i < n; i++ {
		idx := p.SelectPartition("t", nil, 2)
		counts[idx]++
	}

	expected := float64(n) / 2.0
	chi2 := math.Pow(float64(counts[0])-expected, 2)/expected +
		math.Pow(float64(counts[1])-expected, 2)/expected

	// chi-squared critical value for 1 df at p=0.001 is 10.828
	require.Less(s.T(), chi2, 10.828,
		"50/50 split within 0.1%% — got %d/%d", counts[0], counts[1])
}

func (s *P2CSuite) TestEWMASmoothing() {
	p := NewPowerOfTwoChoices(WithP2CAlpha(0.3), WithP2CDepthWeight(0))

	p.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.5, 0.5},
			QueueDepth:     []int{0, 0},
		},
	})

	// Single spike on partition 0.
	p.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{1.0, 0.5},
			QueueDepth:     []int{0, 0},
		},
	})

	ewma := p.loadEWMA.Load()
	require.NotNil(s.T(), ewma)
	// EWMA(0) = 0.3*1.0 + 0.7*0.5 = 0.65
	require.InDelta(s.T(), 0.65, (*ewma)[0], 0.001)
	require.InDelta(s.T(), 0.50, (*ewma)[1], 0.001)

	const n = 10_000
	counts := [2]int{}
	for i := 0; i < n; i++ {
		idx := p.SelectPartition("t", nil, 2)
		counts[idx]++
		p.OnPublishComplete(idx)
	}
	require.Greater(s.T(), counts[1], counts[0],
		"partition 1 (lower EWMA) should get more traffic after spike on 0")
}

func (s *P2CSuite) TestQueueDepthSignal() {
	p := NewPowerOfTwoChoices(WithP2CLoadWeight(0))
	p.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.5, 0.5, 0.5},
			QueueDepth:     []int{10, 0, 10},
		},
	})

	const n = 10_000
	counts := [3]int{}
	for i := 0; i < n; i++ {
		idx := p.SelectPartition("t", nil, 3)
		counts[idx]++
		p.OnPublishComplete(idx)
	}
	require.Greater(s.T(), counts[1], counts[0])
	require.Greater(s.T(), counts[1], counts[2])
}

func (s *P2CSuite) TestSinglePartition() {
	p := NewPowerOfTwoChoices()
	idx := p.SelectPartition("t", nil, 1)
	require.Equal(s.T(), 0, idx)
	p.OnPublishComplete(idx)
}

func (s *P2CSuite) TestNoMetrics() {
	p := NewPowerOfTwoChoices()
	idx := p.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), idx, 0)
	require.Less(s.T(), idx, 4)
	p.OnPublishComplete(idx)
}

func (s *P2CSuite) TestInflightTracking() {
	p := NewPowerOfTwoChoices(
		WithP2CLoadWeight(0),
		WithP2CDepthWeight(0),
		WithP2CInflightWeight(1.0),
	)

	for i := 0; i < 100; i++ {
		idx := p.SelectPartition("t", nil, 2)
		if idx == 0 {
			break
		}
		p.OnPublishComplete(idx)
	}

	inf := p.inflight.Load()
	require.NotNil(s.T(), inf)
	v0 := (*inf)[0].Load()
	require.GreaterOrEqual(s.T(), v0, int64(1), "partition 0 should have inflight > 0")

	p.OnPublishComplete(0)
	v0after := (*inf)[0].Load()
	require.Equal(s.T(), v0-1, v0after)
}

func (s *P2CSuite) TestInflightRaceInit() {
	const goroutines = 16
	const perGoroutine = 500

	p := NewPowerOfTwoChoices(
		WithP2CLoadWeight(0),
		WithP2CDepthWeight(0),
		WithP2CInflightWeight(1.0),
	)

	type result struct {
		partition int
	}
	results := make([][]result, goroutines)

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			local := make([]result, perGoroutine)
			for i := 0; i < perGoroutine; i++ {
				idx := p.SelectPartition("t", nil, 4)
				local[i] = result{partition: idx}
			}
			results[id] = local
		}(g)
	}
	wg.Wait()

	for _, local := range results {
		for _, r := range local {
			p.OnPublishComplete(r.partition)
		}
	}

	inf := p.inflight.Load()
	require.NotNil(s.T(), inf)
	for i := 0; i < 4; i++ {
		require.Equal(s.T(), int64(0), (*inf)[i].Load(),
			"partition %d inflight should be 0 after all completions", i)
	}
}
