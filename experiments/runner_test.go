package experiments

import (
	"context"
	"encoding/binary"
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

func TestSkipWarmupHonored(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sc := benchmarks.Scenario{
		Name:       "skipwarmup_test",
		Duration:   300 * time.Millisecond,
		Producers:  1,
		MsgSize:    64,
		RatePerSec: 100,
		Priorities: [10]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}

	r := &Runner{
		Scenarios:  []benchmarks.Scenario{sc},
		Algorithms: []AlgorithmConfig{AllAlgorithms()[0]},
		Modes:      []Mode{ModeInProcess},
		Duration:   300 * time.Millisecond,
		SkipWarmup: true,
	}

	start := time.Now()
	results, err := r.RunAll(ctx)
	elapsed := time.Since(start)
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Less(t, elapsed, 10*time.Second,
		"with SkipWarmup=true, RunAll should not spend time in runWarmup")
}

func TestSetBalancerRPCInProcess(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:          ModeInProcess,
		BrokerConfig:  broker.DefaultConfig(),
		DataDir:       t.TempDir(),
		Algorithm:     AllAlgorithms()[0],
		NumPartitions: 4,
	})
	require.NoError(t, err)
	defer h.Close()

	brk := h.Broker()
	require.NotNil(t, brk, "in-process harness must expose broker")

	topicCfg := broker.DefaultTopicConfig()
	topicCfg.NumPartitions = 4
	require.NoError(t, h.CreateTopic(ctx, "setbal-test", topicCfg))

	algs := AllAlgorithms()
	for _, alg := range algs {
		if alg.UseOptimizer {
			continue
		}
		err := h.SwitchBalancer(alg, 4)
		require.NoError(t, err, "SwitchBalancer in-process should succeed for %s", alg.Name)

		msg := broker.NewMessage(0, []byte("k"), []byte("v"))
		_, _, pubErr := brk.Publish(ctx, "setbal-test", msg)
		require.NoError(t, pubErr,
			"Publish should succeed after switching to %s", alg.Name)
	}
}

func TestExperienceBasedWarmupExits(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sc := benchmarks.Scenario{
		Name:          "warmup_exit_test",
		Duration:      30 * time.Second,
		Producers:     2,
		NumPartitions: 4,
		MsgSize:       64,
		RatePerSec:    5000,
		Priorities:    [10]float64{0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0},
	}

	dqnAlg := AllAlgorithms()[3] // DQN+Priority

	r := &Runner{
		Scenarios:  []benchmarks.Scenario{sc},
		Algorithms: []AlgorithmConfig{dqnAlg},
		Modes:      []Mode{ModeInProcess},
		Duration:   2 * time.Second,
	}

	start := time.Now()
	results, err := r.RunAll(ctx)
	elapsed := time.Since(start)

	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Less(t, elapsed, dqnWarmupMaxDuration+30*time.Second,
		"warmup should exit early once dqnWarmupTargetExp is reached, not run for the full max duration")
}

func TestDecodeLatencyRejectsCorrupted(t *testing.T) {
	maxLatency := 60 * time.Second

	// timestamp = 0 → latency ≈ time.Since(epoch) >> maxLatency → rejected
	zeroTS := make([]byte, 8)
	_, _, ok := decodeLatency(zeroTS, maxLatency)
	require.False(t, ok, "timestamp=0 should be rejected as too old")

	// timestamp far in the future → negative latency → rejected
	futureTS := make([]byte, 8)
	binary.BigEndian.PutUint64(futureTS, uint64(time.Now().Add(10*time.Minute).UnixNano()))
	_, _, ok = decodeLatency(futureTS, maxLatency)
	require.False(t, ok, "future timestamp should be rejected (negative latency)")

	// valid recent timestamp → accepted
	recentTS := make([]byte, 8)
	binary.BigEndian.PutUint64(recentTS, uint64(time.Now().Add(-1*time.Millisecond).UnixNano()))
	latency, ts, ok := decodeLatency(recentTS, maxLatency)
	require.True(t, ok, "recent timestamp should be accepted")
	require.Greater(t, latency, time.Duration(0))
	require.Greater(t, ts, int64(0))

	// too short value → rejected
	_, _, ok = decodeLatency([]byte{1, 2, 3}, maxLatency)
	require.False(t, ok, "short value should be rejected")
}

func TestKafkaSmokeCheckFailsFast(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         ModeKafka,
		KafkaBrokers: []string{"localhost:1"},
	})
	if err != nil {
		t.Skipf("cannot create kafka harness: %v", err)
	}
	defer h.Close()

	start := time.Now()
	err = kafkaSmokeCheck(ctx, h, "smoke-test-unreachable")
	elapsed := time.Since(start)

	require.Error(t, err, "smoke check with unreachable broker must fail")
	require.Less(t, elapsed, 10*time.Second,
		"smoke check should fail within timeout, took %v", elapsed)
}

func TestRunnerDQNAlgorithmsGrouped(t *testing.T) {
	algs := AllAlgorithms()
	grouped := groupAlgorithmsByMatch(algs, "DQN")

	require.Len(t, grouped, len(algs))

	seenDQN := false
	for _, alg := range grouped {
		isDQN := alg.Name == "DQN+Priority" || alg.Name == "DQN+DQNSched"
		if isDQN {
			seenDQN = true
		}
		if seenDQN && !isDQN {
			t.Fatalf("non-DQN algorithm %q found after DQN algorithms — DQN must be grouped at end", alg.Name)
		}
	}
	require.True(t, seenDQN, "should contain DQN algorithms")

	// Verify DQN+Priority comes before DQN+DQNSched (original order preserved)
	var dqnOrder []string
	for _, alg := range grouped {
		if alg.Name == "DQN+Priority" || alg.Name == "DQN+DQNSched" {
			dqnOrder = append(dqnOrder, alg.Name)
		}
	}
	require.Equal(t, []string{"DQN+Priority", "DQN+DQNSched"}, dqnOrder,
		"DQN+Priority must come immediately before DQN+DQNSched")
}

type noopTestBalancer struct{}

func (noopTestBalancer) SelectPartition(_ string, _ []byte, n int) int { return 0 }
func (noopTestBalancer) OnMetrics(_ broker.Metrics)                    {}
