package balancer

import (
	"testing"

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
			p := dqn.SelectPartition("t", []byte("k"), tc.partitions)
			require.GreaterOrEqual(s.T(), p, 0)
			require.Less(s.T(), p, tc.partitions)
		})
	}
}

func (s *DQNSuite) TestFallbackWatchdog() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(0.0))
	dqn.SetBaseThroughput(1000)

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
		Throughput:     500, // below 0.8 * 1000
	})

	require.True(s.T(), dqn.IsFallbackActive())

	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
		Throughput:     900, // above threshold
	})

	require.False(s.T(), dqn.IsFallbackActive())
}

func (s *DQNSuite) TestOnMetricsTrains() {
	dqn := NewDQNBalancer(4)

	dqn.SelectPartition("t", []byte("k"), 4)
	dqn.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8},
	})

	require.Greater(s.T(), dqn.replayBuffer.Len(), 0)
}

func (s *DQNSuite) TestComputeRewardEmpty() {
	dqn := NewDQNBalancer(4)
	reward := dqn.computeReward(broker.Metrics{})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestComputeRewardPerfectBalance() {
	dqn := NewDQNBalancer(4)
	reward := dqn.computeReward(broker.Metrics{
		PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
	})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestComputeRewardImbalance() {
	dqn := NewDQNBalancer(4)
	reward := dqn.computeReward(broker.Metrics{
		PartitionLoads: []float64{0.1, 0.9, 0.1, 0.9},
	})
	require.Less(s.T(), reward, 0.0, "imbalanced loads yield negative reward")
}

func (s *DQNSuite) TestSetPredictedLoads() {
	dqn := NewDQNBalancer(4)
	dqn.SetPredictedLoads([]float64{0.1, 0.2, 0.3, 0.4})
	require.Equal(s.T(), []float64{0.1, 0.2, 0.3, 0.4}, dqn.predictedLoads)
}

func (s *DQNSuite) TestBuildStateSize() {
	dqn := NewDQNBalancer(4)
	state := dqn.buildState()
	require.Len(s.T(), state, 4*2+2)
}

func (s *DQNSuite) TestForwardOutputSize() {
	dqn := NewDQNBalancer(4)
	state := dqn.buildState()
	q, hidden := dqn.forward(state)
	require.Len(s.T(), q, 4)
	require.Len(s.T(), hidden, dqn.hiddenSize)
}

func (s *DQNSuite) TestForwardDeterministic() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(0))
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0}
	q1, h1 := dqn.forward(state)
	q2, h2 := dqn.forward(state)
	require.InDeltaSlice(s.T(), q1, q2, 1e-12)
	require.InDeltaSlice(s.T(), h1, h2, 1e-12)
}

func (s *DQNSuite) TestUpdateWeightsChangesWeights() {
	dqn := NewDQNBalancer(4, WithDQNLearningRate(0.1))
	state := dqn.buildState()
	for i := range state {
		state[i] = 0.5
	}

	qBefore, hidden := dqn.forward(state)
	beforeCopy := make([]float64, len(qBefore))
	copy(beforeCopy, qBefore)

	dqn.updateWeights(state, 0, 1.0, hidden)

	qAfter, _ := dqn.forward(state)
	changed := false
	for i := range qBefore {
		if qAfter[i] != beforeCopy[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "weights should change after update")
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
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2))
	for i := 0; i < 5; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		dqn.OnMetrics(broker.Metrics{
			PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
		})
	}
	require.GreaterOrEqual(s.T(), dqn.replayBuffer.Len(), 2)
}
