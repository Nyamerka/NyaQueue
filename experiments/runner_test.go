package experiments

import (
	"context"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
)

func TestRunnerRetryOnTopicAlreadyExists(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sc := benchmarks.Scenario{
		Name:       "retry_test",
		Duration:   500 * time.Millisecond,
		Producers:  1,
		MsgSize:    64,
		RatePerSec: 100,
		Priorities: [10]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}

	alg := AllAlgorithms()[0] // RR+FIFO

	r := &Runner{
		Scenarios:  []benchmarks.Scenario{sc},
		Algorithms: []AlgorithmConfig{alg},
		Modes:      []Mode{ModeInProcess},
		Duration:   500 * time.Millisecond,
	}

	results, err := r.RunAll(ctx)
	require.NoError(t, err)
	require.Len(t, results, 1)

	results2, err := r.RunAll(ctx)
	require.NoError(t, err)
	require.Len(t, results2, 1)
}

func TestRunnerCreateTopicRetryDeleteCreate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	dir := t.TempDir()
	store, err := broker.NewOffsetStore(dir)
	require.NoError(t, err)
	defer store.Close()

	cfg := broker.DefaultConfig()
	b := broker.New(cfg, dir, noopTestBalancer{}, store)
	defer b.Stop()

	tcfg := broker.DefaultTopicConfig()
	tcfg.NumPartitions = 1
	testTopic := expTopicName("test", "test", ModeInProcess)
	require.NoError(t, b.CreateTopic(testTopic, tcfg))

	err = b.CreateTopic(testTopic, tcfg)
	require.ErrorIs(t, err, broker.ErrTopicAlreadyExists)

	require.NoError(t, b.DeleteTopic(testTopic))
	require.NoError(t, b.CreateTopic(testTopic, tcfg))

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         ModeInProcess,
		BrokerConfig: cfg,
		DataDir:      t.TempDir(),
		Algorithm:    AllAlgorithms()[0],
	})
	require.NoError(t, err)
	defer h.Close()

	require.NoError(t, h.CreateTopic(ctx, "fresh-topic", tcfg))

	err = h.CreateTopic(ctx, "fresh-topic", tcfg)
	require.ErrorIs(t, err, broker.ErrTopicAlreadyExists)
}

func TestRunnerFailFastDisablesMode(t *testing.T) {
	sc1 := benchmarks.Scenario{
		Name:       "failfast_s1",
		Duration:   200 * time.Millisecond,
		Producers:  1,
		MsgSize:    64,
		RatePerSec: 100,
		Priorities: [10]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	sc2 := benchmarks.Scenario{
		Name:       "failfast_s2",
		Duration:   200 * time.Millisecond,
		Producers:  1,
		MsgSize:    64,
		RatePerSec: 100,
		Priorities: [10]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}

	r := &Runner{
		Scenarios:  []benchmarks.Scenario{sc1, sc2},
		Algorithms: AllAlgorithms(),
		Modes:      []Mode{ModeGRPC},
		BrokerAddr: "localhost:1", // intentionally unreachable
		Duration:   200 * time.Millisecond,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results, err := r.RunAll(ctx)
	require.NoError(t, err)
	require.Empty(t, results, "all runs should fail against unreachable broker")
}

type noopTestBalancer struct{}

func (noopTestBalancer) SelectPartition(_ string, _ []byte, n int) int { return 0 }
func (noopTestBalancer) OnMetrics(_ broker.Metrics)                    {}
