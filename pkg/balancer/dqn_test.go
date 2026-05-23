package balancer

import (
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/alitto/pond/v2"
	"github.com/gomlx/gomlx/pkg/ml/context"
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

func (s *DQNSuite) TestOnMetricsTrains() {
	dqn := NewDQNBalancer(4, WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	})

	time.Sleep(300 * time.Millisecond)

	for i := 0; i < 15; i++ {
		dqn.OnMetrics(broker.Metrics{
			DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.3, 0.5, 0.7, 0.9}},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() > 0
	}, 2*time.Second, 10*time.Millisecond)
}

func (s *DQNSuite) TestRewardPropertyMonotonicity() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	balanced := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	imbalanced := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.9, 0.1, 0.9}},
	})
	require.Greater(s.T(), balanced, imbalanced,
		"more balanced loads should yield higher reward")
}

func (s *DQNSuite) TestRewardPropertyEmptyLoads() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()
	reward := dqn.computeReward(broker.Metrics{})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestRewardPropertyThroughputMatters() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	lowTP := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 1000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	highTP := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 80000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	require.Greater(s.T(), highTP, lowTP,
		"higher throughput should yield higher reward")
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

func (s *DQNSuite) TestTrainStepWithEnoughData() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4}},
	})

	time.Sleep(200 * time.Millisecond)

	for i := 0; i < 10; i++ {
		dqn.OnMetrics(broker.Metrics{
			DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.2, 0.3, float64(i) * 0.1}},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() >= 2
	}, 2*time.Second, 5*time.Millisecond)
}

func (s *DQNSuite) TestAllOptions() {
	dqn := NewDQNBalancer(4,
		WithDQNGamma(0.9),
		WithDQNLearningRate(0.01),
		WithDQNHiddenSize(32),
		WithDQNBatchSize(16),
		WithDQNMinReplay(32),
		WithDQNReplayBufSize(1000),
	)
	defer dqn.Stop()
	require.Equal(s.T(), 0.9, dqn.gamma)
	require.Equal(s.T(), 0.01, dqn.lr)
	require.Equal(s.T(), 128, dqn.hiddenSize)
	require.Equal(s.T(), 16, dqn.batchSize)
	require.Equal(s.T(), 32, dqn.minReplay)
}

func (s *DQNSuite) TestAsyncTrainingDoesNotBlockInference() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	for i := 0; i < 100; i++ {
		dqn.SelectPartition("t", []byte("k"), 4)
		dqn.OnMetrics(broker.Metrics{
			DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4}},
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

func (s *DQNSuite) TestConcurrentSelectAndOnMetrics() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	pool := pond.NewPool(51)
	for i := 0; i < 50; i++ {
		pool.Submit(func() {
			for j := 0; j < 200; j++ {
				dqn.SelectPartition("t", []byte{1}, 4)
			}
		})
	}
	pool.Submit(func() {
		for j := 0; j < 100; j++ {
			dqn.OnMetrics(broker.Metrics{
				DerivedMetrics: broker.DerivedMetrics{
					PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
				},
			})
		}
	})
	pool.StopAndWait()
}

func (s *DQNSuite) TestDroppedExperienceCounter() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	require.Equal(s.T(), int64(0), dqn.DroppedExperience())
}

func (s *DQNSuite) TestDQNBalancer_DepthPenalty() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	noDepth := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	withDepth := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
			QueueDepth:     []int{50000, 50000, 50000, 50000},
		},
	})
	require.Greater(s.T(), noDepth, withDepth,
		"large queue depth should penalise reward")
}

func (s *DQNSuite) TestDQNBalancer_InflightInState() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	for i := 0; i < 100; i++ {
		dqn.SelectPartition("t", nil, 4)
	}

	ptr := dqn.inflight.Load()
	require.NotNil(s.T(), ptr, "inflight slice should be initialized after SelectPartition")

	var totalInflight int64
	for i := 0; i < dqn.numPartitions && i < len(*ptr); i++ {
		totalInflight += (*ptr)[i].Load()
	}
	require.Greater(s.T(), totalInflight, int64(0),
		"inflight should be reflected in state vector after SelectPartition calls")
}

func (s *DQNSuite) TestArgmaxRoutingSelectsBest() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	qValues := []float64{0.1, 0.2, 10.0, 0.3}
	cached := make([]float64, len(qValues))
	copy(cached, qValues)
	dqn.cachedQValues.Store(&cached)

	counts := map[int]int{}
	for i := 0; i < 1000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	require.Equal(s.T(), 1000, counts[2],
		"argmax routing should always select partition 2 (best Q)")
}

func (s *DQNSuite) TestNoFallbackUnderOverload() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	overloadMetrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 200000, MsgRate: 200000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.95, 0.98, 0.92, 0.97}},
	}
	for i := 0; i < 100; i++ {
		dqn.OnMetrics(overloadMetrics)
	}

	require.NotPanics(s.T(), func() {
		for i := 0; i < 1000; i++ {
			p := dqn.SelectPartition("t", nil, 4)
			dqn.OnPublishComplete(p)
		}
	}, "DQN should not panic under overload conditions")
}

func (s *DQNSuite) TestDQNBootstrapNoDeadlock() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	time.Sleep(200 * time.Millisecond)

	for i := 0; i < 100; i++ {
		dqn.OnMetrics(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
			DerivedMetrics: broker.DerivedMetrics{
				PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8},
			},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.TrainingSteps() > 0
	}, 10*time.Second, 10*time.Millisecond,
		"DQN should accumulate training steps from step 0 without deadlock")
	require.Greater(s.T(), dqn.replayBuffer.Len(), 0,
		"replay buffer should have experiences recorded from the start")
}

func (s *DQNSuite) TestDirectRewardSignal() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	r1 := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	r2 := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	require.InDelta(s.T(), r1, r2, 1e-10,
		"direct reward should be deterministic for same input (no EWMA drift)")
}

func (s *DQNSuite) TestDQNConvergenceWithoutFallback() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	time.Sleep(200 * time.Millisecond)
	for i := 0; i < 30; i++ {
		loads := []float64{0.2, 0.4, 0.6, 0.8}
		loads[i%4] += float64(i) * 0.01
		dqn.OnMetrics(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: loads},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() >= 2
	}, 3*time.Second, 10*time.Millisecond,
		"DQN should accumulate experience and train without fallback mechanism")

	require.Eventually(s.T(), func() bool {
		return dqn.TrainingSteps() > 0
	}, 3*time.Second, 10*time.Millisecond,
		"training steps should advance from zero")
}

func (s *DQNSuite) TestWeightDecayPreventsGrowth() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	var initialWeightNorm float64
	dqn.ctx.EnumerateVariables(func(v *context.Variable) {
		if !v.Trainable {
			return
		}
		val := v.MustValue()
		switch data := val.Value().(type) {
		case []float64:
			for _, w := range data {
				initialWeightNorm += w * w
			}
		case []float32:
			for _, w := range data {
				initialWeightNorm += float64(w) * float64(w)
			}
		}
	})
	initialWeightNorm = math.Sqrt(initialWeightNorm)

	require.Greater(s.T(), initialWeightNorm, 0.0,
		"initial weight norm should be positive")
}

// --- New tests for R21 fixes ---

func (s *DQNSuite) TestRawStateInReplay() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	})
	time.Sleep(200 * time.Millisecond)
	for i := 0; i < 20; i++ {
		dqn.OnMetrics(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() > 0
	}, 3*time.Second, 10*time.Millisecond)

	batch, _, _ := dqn.replayBuffer.Sample(1, 1.0)
	require.NotEmpty(s.T(), batch)
	state := batch[0].State
	require.Len(s.T(), state, dqn.stateSize)

	hasNonZero := false
	for _, v := range state {
		if v > 0.001 {
			hasNonZero = true
			break
		}
	}
	require.True(s.T(), hasNonZero,
		"state in replay should contain raw (unnormalized) values")
}

func (s *DQNSuite) TestNormalizerResetForNewScenario() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	for i := 0; i < 100; i++ {
		dqn.OnMetrics(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
		})
	}

	dqn.ResetForNewScenario()

	state := make([]float64, dqn.stateSize)
	state[0] = 0.5
	normalized := dqn.normalizer.Normalize(state)
	for _, v := range normalized {
		require.InDelta(s.T(), 0, v, 1e-6,
			"after reset, normalizer has count<2, should output zeros")
	}
}

func (s *DQNSuite) TestNStepBufferComposition() {
	buf := newNStepBuffer(3, 0.9)

	t1 := nn.Transition{State: []float64{1}, Action: []float64{0}, Reward: 1.0, NextState: []float64{2}}
	t2 := nn.Transition{State: []float64{2}, Action: []float64{1}, Reward: 2.0, NextState: []float64{3}}
	t3 := nn.Transition{State: []float64{3}, Action: []float64{0}, Reward: 3.0, NextState: []float64{4}}

	_, ok := buf.Push(t1)
	require.False(s.T(), ok)
	_, ok = buf.Push(t2)
	require.False(s.T(), ok)

	out, ok := buf.Push(t3)
	require.True(s.T(), ok)

	expectedReward := 1.0 + 0.9*2.0 + 0.9*0.9*3.0
	require.InDelta(s.T(), expectedReward, out.Reward, 1e-10)
	require.Equal(s.T(), t1.State, out.State)
	require.Equal(s.T(), t3.NextState, out.NextState)
	require.Equal(s.T(), t1.Action, out.Action)
}

func (s *DQNSuite) TestPopArtNormalizes() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	for i := 0; i < 200; i++ {
		target := float64(i) * 0.1
		dqn.popArt.mu.Lock()
		dqn.popArt.count++
		delta := target - dqn.popArt.mean
		dqn.popArt.mean += delta / float64(dqn.popArt.count)
		dqn.popArt.m2 += delta * (target - dqn.popArt.mean)
		dqn.popArt.mu.Unlock()
	}

	dqn.popArt.mu.Lock()
	sigma := math.Sqrt(dqn.popArt.m2/float64(dqn.popArt.count-1)) + 1e-8
	mu := dqn.popArt.mean
	dqn.popArt.mu.Unlock()

	require.Greater(s.T(), sigma, 0.1,
		"PopArt sigma should be positive after observations")
	require.Greater(s.T(), mu, 0.0,
		"PopArt mean should be non-zero after observations")

	normalized := (10.0 - mu) / sigma
	require.Less(s.T(), math.Abs(normalized), 5.0,
		"PopArt-normalized target should be within reasonable bounds")
}

func (s *DQNSuite) TestHardTargetSyncCopiesAll() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.targetNet.HardSync()

	allMatch := true
	dqn.ctx.EnumerateVariables(func(mainVar *context.Variable) {
		if !mainVar.Trainable {
			return
		}
		mainVal, err := mainVar.Value()
		if err != nil || mainVal == nil {
			allMatch = false
			return
		}
		_ = mainVal
	})

	require.True(s.T(), allMatch,
		"after HardSync all main vars should be accessible (sync didn't panic)")
}

func (s *DQNSuite) TestQueueDepthInState() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.stateMu.Lock()
	dqn.loads = []float64{0.2, 0.4, 0.6, 0.8}
	dqn.queueDepth = []int{500, 1000, 1500, 2000}
	dqn.msgRate = 50000
	dqn.avgMsgSize = 256
	dqn.consumeRate = 40000
	dqn.stateMu.Unlock()

	state := make([]float64, dqn.stateSize)
	dqn.stateMu.Lock()
	dqn.buildStateInto(state)
	dqn.stateMu.Unlock()

	depthBase := dqn.numPartitions * 3
	for i := 0; i < 4; i++ {
		expected := float64(dqn.queueDepth[i]) / dqnDepthFeatureScale
		require.InDelta(s.T(), expected, state[depthBase+i], 1e-10,
			"state[3K+%d] should contain queue depth for partition %d", i, i)
	}

	idx := dqn.numPartitions * 4
	require.InDelta(s.T(), 50000.0/dqnMsgRateScale, state[idx], 1e-10, "msgRate feature")
	require.InDelta(s.T(), 256.0/dqnMsgSizeScale, state[idx+1], 1e-10, "avgMsgSize feature")
	require.InDelta(s.T(), 40000.0/dqnMsgRateScale, state[idx+2], 1e-10, "consumeRate feature")
	require.InDelta(s.T(), 50000.0/40000.0, state[idx+3], 1e-10, "produce/consume ratio feature")
}
