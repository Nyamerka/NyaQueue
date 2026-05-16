package balancer

import (
	"testing"
	"unsafe"

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
	rr.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{Throughput: 100}})
	p := rr.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), p, 0)
}

func (s *RoundRobinSuite) TestCacheLinePadding() {
	require.GreaterOrEqual(s.T(), unsafe.Sizeof(RoundRobin{}), uintptr(128))
}

func (s *RoundRobinSuite) TestLoadAwareSkip() {
	rr := NewRoundRobin(WithRRSkipThreshold(0.9), WithRRSkipBudget(2))
	rr.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.95, 0.1, 0.95},
		},
	})

	hits := make(map[int]int)
	for i := 0; i < 300; i++ {
		p := rr.SelectPartition("t", nil, 3)
		hits[p]++
	}
	require.Greater(s.T(), hits[1], hits[0]+hits[2],
		"partition 1 (low load) should receive majority of traffic")
}

func (s *RoundRobinSuite) TestPureModeWhenBudgetZero() {
	rr := NewRoundRobin(WithRRSkipBudget(0))
	rr.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.99, 0.01, 0.99, 0.01},
		},
	})

	counts := make([]int, 4)
	const n = 400
	for i := 0; i < n; i++ {
		p := rr.SelectPartition("t", nil, 4)
		counts[p]++
	}
	for _, c := range counts {
		require.Equal(s.T(), n/4, c, "budget=0 must yield classic uniform RR")
	}
}
