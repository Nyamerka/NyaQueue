package balancer

import (
	"math"
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
		counts[p.SelectPartition("t", nil, 2)]++
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
		counts[p.SelectPartition("t", nil, 3)]++
	}
	require.Greater(s.T(), counts[1], counts[0])
	require.Greater(s.T(), counts[1], counts[2])
}

func (s *P2CSuite) TestSinglePartition() {
	p := NewPowerOfTwoChoices()
	require.Equal(s.T(), 0, p.SelectPartition("t", nil, 1))
}

func (s *P2CSuite) TestNoMetrics() {
	p := NewPowerOfTwoChoices()
	idx := p.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), idx, 0)
	require.Less(s.T(), idx, 4)
}
