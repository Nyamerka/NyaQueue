package balancer

import (
	"runtime"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type DQNSuite struct {
	suite.Suite
}

func TestDQNSuite(t *testing.T) { suite.Run(t, new(DQNSuite)) }

func (s *DQNSuite) TestSelectPartition() {
	tests := []struct {
		name       string
		partitions int
		opts       []DQNOption
	}{
		{"default", 4, nil},
		{"high_epsilon", 4, []DQNOption{WithDQNEpsilon(1.0)}},
		{"custom_hidden", 4, []DQNOption{WithDQNHiddenSize(32)}},
		{"custom_lr", 4, []DQNOption{WithDQNLearningRate(0.01)}},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			dqn := NewDQNBalancer(tc.partitions, tc.opts...)
			defer dqn.Stop()
			p := dqn.SelectPartition("t", []byte("k"), tc.partitions)
			require.GreaterOrEqual(s.T(), p, 0)
			require.Less(s.T(), p, tc.partitions)
		})
	}
}

func (s *DQNSuite) TestFallbackWatchdog() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(0.0), WithDQNLoadThreshold(0))
	defer dqn.Stop()
	dqn.SetBaseThroughput(1000)

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
		Throughput:     500,
	})

	require.True(s.T(), dqn.IsFallbackActive())

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
		Throughput:     900,
	})

	require.False(s.T(), dqn.IsFallbackActive())
}

func (s *DQNSuite) TestProactiveWatchdog() {
	dqn := NewDQNBalancer(4, WithDQNLoadThreshold(0.7))
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.3, 0.3, 0.3, 0.3},
	})
	require.False(s.T(), dqn.IsFallbackActive(), "low load should not trigger fallback")

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.8, 0.9, 0.7, 0.85},
	})
	require.True(s.T(), dqn.IsFallbackActive(), "high mean load should trigger proactive fallback")

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.3, 0.3, 0.3, 0.3},
	})
	require.False(s.T(), dqn.IsFallbackActive(), "load recovery should deactivate fallback")
}

func (s *DQNSuite) TestOnMetricsTrains() {
	dqn := NewDQNBalancer(4, WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.SelectPartition("t", []byte("k"), 4)
	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8},
	})

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() > 0
	}, time.Second, 5*time.Millisecond)
}

func (s *DQNSuite) TestComputeRewardEmpty() {
	reward := computeReward(broker.Metrics{})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestComputeRewardPerfectBalance() {
	reward := computeReward(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
	})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestComputeRewardImbalance() {
	reward := computeReward(broker.Metrics{
		PartitionLoads: []float64{0.1, 0.9, 0.1, 0.9},
	})
	require.Less(s.T(), reward, 0.0, "imbalanced loads yield negative reward")
}

func (s *DQNSuite) TestSetPredictedLoads() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()
	dqn.SetPredictedLoads([]float64{0.1, 0.2, 0.3, 0.4})

	dqn.stateMu.Lock()
	loads := dqn.predictedLoads
	dqn.stateMu.Unlock()
	require.Equal(s.T(), []float64{0.1, 0.2, 0.3, 0.4}, loads)
}

func (s *DQNSuite) TestBuildStateSize() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.stateMu.Lock()
	state := dqn.buildStateLocked()
	dqn.stateMu.Unlock()
	require.Len(s.T(), state, 4*2+2)
}

func (s *DQNSuite) TestForwardOutputSize() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.stateMu.Lock()
	state := dqn.buildStateLocked()
	dqn.stateMu.Unlock()

	dqn.weightsMu.RLock()
	q := dqn.forward(state)
	dqn.weightsMu.RUnlock()
	require.Len(s.T(), q, 4)
}

func (s *DQNSuite) TestForwardDeterministic() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(0))
	defer dqn.Stop()
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0}

	dqn.weightsMu.RLock()
	q1 := dqn.forward(state)
	q2 := dqn.forward(state)
	dqn.weightsMu.RUnlock()

	require.InDeltaSlice(s.T(), q1, q2, 1e-12)
}

func (s *DQNSuite) TestAllOptions() {
	dqn := NewDQNBalancer(4,
		WithDQNEpsilon(0.1),
		WithDQNGamma(0.9),
		WithDQNLearningRate(0.01),
		WithDQNHiddenSize(32),
		WithDQNBatchSize(16),
		WithDQNMinReplay(32),
		WithDQNFallbackRatio(0.5),
		WithDQNWeightInit(0.05),
		WithDQNReplayBufSize(1000),
	)
	defer dqn.Stop()
	require.Equal(s.T(), 0.1, dqn.epsilon)
	require.Equal(s.T(), 0.9, dqn.gamma)
	require.Equal(s.T(), 0.01, dqn.lr)
	require.Equal(s.T(), 32, dqn.hiddenSize)
	require.Equal(s.T(), 16, dqn.batchSize)
	require.Equal(s.T(), 32, dqn.minReplay)
	require.Equal(s.T(), 0.5, dqn.fallbackRatio)
	require.Equal(s.T(), 0.05, dqn.weightInit)
}

func (s *DQNSuite) TestTrainStepWithEnoughData() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()
	for i := 0; i < 5; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		dqn.OnMetrics(broker.Metrics{
			PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
		})
	}
	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() >= 2
	}, time.Second, 5*time.Millisecond)
}

func (s *DQNSuite) TestTrainStepChangesWeights() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNLearningRate(0.1))
	defer dqn.Stop()

	// Get initial forward output.
	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0}
	dqn.weightsMu.RLock()
	qBefore := dqn.forward(state)
	dqn.weightsMu.RUnlock()

	// Fill replay buffer and train.
	for i := 0; i < 10; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		dqn.OnMetrics(broker.Metrics{
			PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
		})
	}

	// Allow training to happen.
	time.Sleep(200 * time.Millisecond)

	dqn.weightsMu.RLock()
	qAfter := dqn.forward(state)
	dqn.weightsMu.RUnlock()

	changed := false
	for i := range qBefore {
		if qAfter[i] != qBefore[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "weights should change after training step")
}

func (s *DQNSuite) TestAsyncTrainingDoesNotBlockInference() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	for i := 0; i < 100; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		dqn.OnMetrics(broker.Metrics{
			PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
		})
	}

	start := time.Now()
	for i := 0; i < 1000; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		runtime.Gosched()
	}
	elapsed := time.Since(start)

	require.Less(s.T(), elapsed, 5*time.Second,
		"1000 inference calls should complete without excessive contention")
}
