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
	rr.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{Throughput: 100}})
	p := rr.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), p, 0)
}
