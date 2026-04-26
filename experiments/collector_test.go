package experiments

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPercentile(t *testing.T) {
	tests := []struct {
		name   string
		sorted []float64
		p      float64
		want   float64
	}{
		{"empty", nil, 0.5, 0},
		{"single", []float64{42}, 0.5, 42},
		{"single_p0", []float64{42}, 0, 42},
		{"single_p1", []float64{42}, 1, 42},
		{"two_p50", []float64{10, 20}, 0.5, 15},
		{"two_p0", []float64{10, 20}, 0, 10},
		{"two_p100", []float64{10, 20}, 1, 20},
		{"three_p50", []float64{1, 2, 3}, 0.5, 2},
		{"five_p95", []float64{1, 2, 3, 4, 5}, 0.95, 4.8},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := percentile(tc.sorted, tc.p)
			require.InDelta(t, tc.want, got, 0.001)
		})
	}
}

func TestSnapshotMinimalData(t *testing.T) {
	c := NewMetricsCollector()
	c.Start()
	c.RecordConsumeWithPriority(0, 1*time.Millisecond)
	c.Stop()

	result := c.Snapshot("sc", "alg", "sys", "mode")
	require.Equal(t, int64(1), result.Consumed)
	require.Greater(t, result.LatencyP50, time.Duration(0))
}

func TestSnapshotNoData(t *testing.T) {
	c := NewMetricsCollector()
	c.Start()
	c.Stop()

	result := c.Snapshot("sc", "alg", "sys", "mode")
	require.Equal(t, int64(0), result.Consumed)
	require.Equal(t, time.Duration(0), result.LatencyP50)
}

func TestSnapshotPriorityStats(t *testing.T) {
	c := NewMetricsCollector()
	c.Start()
	for i := 0; i < 100; i++ {
		c.RecordConsumeWithPriority(0, 1*time.Millisecond)
		c.RecordConsumeWithPriority(5, 10*time.Millisecond)
	}
	c.Stop()

	result := c.Snapshot("sc", "alg", "sys", "mode")
	require.Equal(t, int64(100), result.LatencyByPriority[0].Count)
	require.Equal(t, int64(100), result.LatencyByPriority[5].Count)
	require.Greater(t, result.LatencyByPriority[5].P50, result.LatencyByPriority[0].P50)
}
