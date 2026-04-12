package balancer

import (
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type RoundRobinSuite struct {
	suite.Suite
}

func TestRoundRobinSuite(t *testing.T) { suite.Run(t, new(RoundRobinSuite)) }

func (s *RoundRobinSuite) TestDistribution() {
	tests := []struct {
		name       string
		partitions int
		calls      int
	}{
		{"4_partitions", 4, 100},
		{"1_partition", 1, 10},
		{"8_partitions", 8, 80},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			rr := NewRoundRobin()
			counts := make([]int, tc.partitions)

			for i := 0; i < tc.calls; i++ {
				p := rr.SelectPartition("t", []byte("k"), tc.partitions)
				require.GreaterOrEqual(s.T(), p, 0)
				require.Less(s.T(), p, tc.partitions)
				counts[p]++
			}

			expected := tc.calls / tc.partitions
			for _, c := range counts {
				require.InDelta(s.T(), expected, c, 1, "partitions should be evenly distributed")
			}
		})
	}
}

func (s *RoundRobinSuite) TestOnMetricsNoop() {
	rr := NewRoundRobin()
	rr.OnMetrics(broker.Metrics{Throughput: 100})
	p := rr.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), p, 0)
}

// --- WRR ---

type WRRSuite struct {
	suite.Suite
}

func TestWRRSuite(t *testing.T) { suite.Run(t, new(WRRSuite)) }

func (s *WRRSuite) TestDefaultWeights() {
	wrr := NewWeightedRoundRobin()

	counts := make([]int, 4)
	for i := 0; i < 100; i++ {
		p := wrr.SelectPartition("t", nil, 4)
		counts[p]++
	}

	for _, c := range counts {
		require.Greater(s.T(), c, 0, "each partition should get at least one message")
	}
}

func (s *WRRSuite) TestSkewedWeights() {
	wrr := NewWeightedRoundRobin()
	wrr.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.01, 0.5, 0.5, 0.5},
	})

	counts := make([]int, 4)
	for i := 0; i < 1000; i++ {
		p := wrr.SelectPartition("t", nil, 4)
		counts[p]++
	}

	require.Greater(s.T(), counts[0], counts[1],
		"partition 0 (low load) should get more traffic")
}

func (s *WRRSuite) TestWithMinLoadOption() {
	wrr := NewWeightedRoundRobin(WithWRRMinLoad(0.1))
	wrr.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.001, 0.5},
	})

	p := wrr.SelectPartition("t", nil, 2)
	require.GreaterOrEqual(s.T(), p, 0)
	require.Less(s.T(), p, 2)
}

func (s *WRRSuite) TestGCDFloat() {
	tests := []struct {
		name   string
		vals   []float64
		expect float64
	}{
		{"equal_values", []float64{0.5, 0.5, 0.5}, 0.5},
		{"empty", []float64{}, 0},
		{"single", []float64{2.0}, 2.0},
		{"different", []float64{1.0, 2.0, 3.0}, 1.0},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			result := gcdFloat(tc.vals)
			require.InDelta(s.T(), tc.expect, result, 0.001)
		})
	}
}

func (s *WRRSuite) TestGCDFloatNegativeValues() {
	result := gcdFloat([]float64{-2.0, 4.0})
	require.InDelta(s.T(), 2.0, result, 0.001)
}

func (s *WRRSuite) TestGCDFloatFractional() {
	result := gcdFloat([]float64{0.5, 1.0, 1.5})
	require.InDelta(s.T(), 0.5, result, 0.001)
}

func (s *WRRSuite) TestEnsureWeightsResize() {
	wrr := NewWeightedRoundRobin()
	wrr.ensureWeights(3)
	require.Len(s.T(), wrr.weights, 3)
	wrr.ensureWeights(3)
	require.Len(s.T(), wrr.weights, 3)
	wrr.ensureWeights(5)
	require.Len(s.T(), wrr.weights, 5)
}
