package experiments

import (
	"context"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
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

func TestGRPCHarnessReadiness(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dir := t.TempDir()
	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         ModeGRPC,
		BrokerConfig: broker.DefaultConfig(),
		DataDir:      dir,
		Algorithm: AlgorithmConfig{
			Name:         "RR+FIFO",
			NewBalancer:  AllAlgorithms()[0].NewBalancer,
			NewScheduler: AllAlgorithms()[0].NewScheduler,
			Mode:         broker.ModeFIFO,
		},
	})
	require.NoError(t, err)
	defer h.Close()

	require.NotNil(t, h.grpc, "grpc client should be initialized")
	require.NotNil(t, h.Broker(), "local broker should be accessible")

	err = h.CreateTopic(ctx, "readiness-test", broker.DefaultTopicConfig())
	require.NoError(t, err)
	h.Broker().SetScheduler("readiness-test", AllAlgorithms()[0].NewScheduler())

	err = h.Publish(ctx, "readiness-test", []byte("k"), []byte("v"), 0)
	require.NoError(t, err)
}

func TestHTTPHarnessReadiness(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dir := t.TempDir()
	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         ModeHTTP,
		BrokerConfig: broker.DefaultConfig(),
		DataDir:      dir,
		Algorithm: AlgorithmConfig{
			Name:         "RR+FIFO",
			NewBalancer:  AllAlgorithms()[0].NewBalancer,
			NewScheduler: AllAlgorithms()[0].NewScheduler,
			Mode:         broker.ModeFIFO,
		},
	})
	require.NoError(t, err)
	defer h.Close()

	require.NotNil(t, h.httpClient, "http client should be initialized")
	require.NotNil(t, h.Broker(), "local broker should be accessible")

	cfg := broker.DefaultTopicConfig()
	cfg.NumPartitions = 1
	err = h.CreateTopic(ctx, "http-readiness", cfg)
	require.NoError(t, err)
	h.Broker().SetScheduler("http-readiness", AllAlgorithms()[0].NewScheduler())

	err = h.Publish(ctx, "http-readiness", []byte("k"), []byte("v"), 0)
	require.NoError(t, err)

	msgs, err := h.ConsumeBatch(ctx, "http-readiness", "g1", 0)
	require.NoError(t, err)
	require.Len(t, msgs, 1)
	require.Equal(t, []byte("v"), msgs[0].Value)
}

func TestEncodeDecodeLatency(t *testing.T) {
	value := encodeValue(256)
	require.Len(t, value, 256)

	latency, ok := decodeLatency(value)
	require.True(t, ok)
	require.Less(t, latency, 1*time.Second,
		"latency for in-process encode/decode should be sub-second")
	require.GreaterOrEqual(t, latency, time.Duration(0))
}

func TestDecodeLatencyTooShort(t *testing.T) {
	_, ok := decodeLatency([]byte{1, 2, 3})
	require.False(t, ok, "payload shorter than 8 bytes should fail")
}
