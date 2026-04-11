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

// --- PSA ---

type PSASuite struct {
	suite.Suite
}

func TestPSASuite(t *testing.T) { suite.Run(t, new(PSASuite)) }

func (s *PSASuite) TestKeyBinding() {
	psa := NewPSA(4)

	p1 := psa.SelectPartition("t", []byte("key-a"), 4)
	p2 := psa.SelectPartition("t", []byte("key-a"), 4)
	require.Equal(s.T(), p1, p2, "same key must route to same partition")
}

func (s *PSASuite) TestDifferentKeys() {
	psa := NewPSA(4)

	results := make(map[int]bool)
	for i := 0; i < 100; i++ {
		key := []byte{byte(i)}
		p := psa.SelectPartition("t", key, 4)
		results[p] = true
	}

	require.Greater(s.T(), len(results), 1, "different keys should spread across partitions")
}

func (s *PSASuite) TestReleaseBindingsOnEmptyQueue() {
	psa := NewPSA(4)

	psa.SelectPartition("t", []byte("k1"), 4)
	psa.SelectPartition("t", []byte("k2"), 4)

	psa.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0, 0, 0, 0},
		QueueDepth:     []int{0, 0, 0, 0},
	})

	require.Len(s.T(), psa.bindings, 0, "bindings should be released for empty partitions")
	require.Len(s.T(), psa.free, 4, "all partitions should be free")
}

// --- DQN ---

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
	require.Equal(s.T(), 0.0, reward) // stddev = 0
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
	require.Len(s.T(), state, 4*2+2) // loads + predicted + rate + avgsize
}

func (s *DQNSuite) TestForwardOutputSize() {
	dqn := NewDQNBalancer(4)
	state := dqn.buildState()
	q := dqn.forward(state)
	require.Len(s.T(), q, 4)
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

// --- WRR edge cases ---

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

func (s *WRRSuite) TestGCDInt() {
	tests := []struct {
		name   string
		a, b   int64
		expect int64
	}{
		{"coprime", 7, 3, 1},
		{"equal", 5, 5, 5},
		{"one_zero", 6, 0, 6},
		{"both_zero", 0, 0, 0},
		{"negative", -12, 8, 4},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			require.Equal(s.T(), tc.expect, gcdInt(tc.a, tc.b))
		})
	}
}

func (s *WRRSuite) TestEnsureWeightsResize() {
	wrr := NewWeightedRoundRobin()
	wrr.ensureWeights(3)
	require.Len(s.T(), wrr.weights, 3)
	wrr.ensureWeights(3) // no-op
	require.Len(s.T(), wrr.weights, 3)
	wrr.ensureWeights(5) // resize
	require.Len(s.T(), wrr.weights, 5)
}

// --- PSA edge cases ---

func (s *PSASuite) TestNoFreePartitions() {
	psa := NewPSA(2)

	psa.SelectPartition("t", []byte("a"), 2)
	psa.SelectPartition("t", []byte("b"), 2)

	p := psa.SelectPartition("t", []byte("c"), 2)
	require.GreaterOrEqual(s.T(), p, 0)
	require.Less(s.T(), p, 2)
}

func (s *PSASuite) TestHashKeyDeterministic() {
	h1 := hashKey([]byte("test-key"))
	h2 := hashKey([]byte("test-key"))
	require.Equal(s.T(), h1, h2)
}

// --- RR edge case ---

func (s *RoundRobinSuite) TestOnMetricsNoop() {
	rr := NewRoundRobin()
	rr.OnMetrics(broker.Metrics{Throughput: 100})
	p := rr.SelectPartition("t", nil, 4)
	require.GreaterOrEqual(s.T(), p, 0)
}
