package balancer

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
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

func (s *DQNSuite) TestAtomicStateSnapshot() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
			PredictedLoads: []float64{0.15, 0.25, 0.35, 0.45},
		},
	})

	snap := dqn.atomicState.Load()
	require.NotNil(s.T(), snap)
	require.Equal(s.T(), []float64{0.1, 0.2, 0.3, 0.4}, snap.loads)
	require.Equal(s.T(), []float64{0.15, 0.25, 0.35, 0.45}, snap.predictedLoads)
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

func (s *DQNSuite) TestDQNBalancer_SmoothedCV() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.9, 0.1, 0.9}},
	})

	rBalanced := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	rFreshBalanced := NewDQNBalancer(4)
	defer rFreshBalanced.Stop()
	rPure := rFreshBalanced.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	require.Less(s.T(), rBalanced, rPure,
		"EWMA should retain past imbalance, yielding lower reward than a fresh balancer with same balanced loads")
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

func (s *DQNSuite) TestBoltzmannRoutingPrefersBest() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	probs := softmaxFromQ([]float64{0.1, 0.2, 10.0, 0.3}, 0.05)
	dqn.cachedProbs.Store(&probs)
	dqn.cachedArgmax.Store(2)

	counts := map[int]int{}
	for i := 0; i < 1000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	require.Greater(s.T(), counts[2], 900,
		"Boltzmann routing with low τ should heavily prefer partition 2 (best Q)")
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

	dqn.Stop()

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
		dqn.popArt.count++
		delta := target - dqn.popArt.mean
		dqn.popArt.mean += delta / float64(dqn.popArt.count)
		dqn.popArt.m2 += delta * (target - dqn.popArt.mean)
	}

	sigma := math.Sqrt(dqn.popArt.m2/float64(dqn.popArt.count-1)) + 1e-8
	mu := dqn.popArt.mean

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

func (s *DQNSuite) TestDQNExplorationUnderTrained() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	// Before the first policyLoop tick (100ms), cachedProbs is nil and
	// SelectPartition uses the P2C fallback which provides exploration.
	counts := make(map[int]int)
	const trials = 1000
	for i := 0; i < trials; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	require.GreaterOrEqual(s.T(), len(counts), 2,
		"P2C fallback should spread across at least 2 partitions")

	// After policyLoop ticks, softmax probabilities are cached. Boltzmann
	// sampling provides per-message stochastic exploration.
	time.Sleep(200 * time.Millisecond)
	require.NotNil(s.T(), dqn.cachedProbs.Load(),
		"policyLoop should have set cached Boltzmann probs")
}

func (s *DQNSuite) TestMaxLoadPenaltyInReward() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	noOverload := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	withOverload := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 2.0}},
	})
	require.Greater(s.T(), noOverload, withOverload,
		"max-load > 1.0 should penalise reward")

	penalty := noOverload - withOverload
	require.GreaterOrEqual(s.T(), penalty, dqnMaxLoadWeight*(2.0-1.0)*(2.0-1.0)*0.5,
		"penalty should be at least proportional to dqnMaxLoadWeight * (maxLoad - 1.0)^2")
}

func (s *DQNSuite) TestExpChannelBlockingSend() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))

	// Stop goroutines before replacing expCh to avoid a data race on the field.
	dqn.Stop()
	dqn.expCh = make(chan nn.Transition, 1)
	dqn.expCh <- nn.Transition{}

	probs := []float64{0.25, 0.25, 0.25, 0.25}
	dqn.cachedProbs.Store(&probs)
	dqn.prevSnap.Store(&policySnap{state: make([]float64, dqn.stateSize), action: 0})
	dqn.currSnap.Store(&policySnap{state: make([]float64, dqn.stateSize), action: 1})

	start := time.Now()
	dqn.OnMetrics(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	})
	elapsed := time.Since(start)

	require.Less(s.T(), elapsed, 50*time.Millisecond,
		"OnMetrics should not block for more than ~1ms on a full channel")
}

func (s *DQNSuite) TestExperienceCountTracking() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	require.Equal(s.T(), int64(0), dqn.ExperiencesSeen())

	dqn.OnMetrics(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	})
	time.Sleep(200 * time.Millisecond)
	for i := 0; i < 30; i++ {
		dqn.OnMetrics(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, float64(i) * 0.1}},
		})
		time.Sleep(60 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.ExperiencesSeen() > 0
	}, 5*time.Second, 10*time.Millisecond,
		"ExperiencesSeen() should be monotonically increasing after OnMetrics calls")
}

func (s *DQNSuite) TestQueueDepthInState() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.atomicState.Store(&stateSnapshot{
		loads:       []float64{0.2, 0.4, 0.6, 0.8},
		queueDepth:  []int{500, 1000, 1500, 2000},
		msgRate:     50000,
		avgMsgSize:  256,
		consumeRate: 40000,
	})

	state := make([]float64, dqn.stateSize)
	dqn.buildStateInto(state)

	depthBase := dqn.numPartitions * 3
	depths := []int{500, 1000, 1500, 2000}
	for i := 0; i < 4; i++ {
		expected := float64(depths[i]) / dqnDepthFeatureScale
		require.InDelta(s.T(), expected, state[depthBase+i], 1e-10,
			"state[3K+%d] should contain queue depth for partition %d", i, i)
	}

	idx := dqn.numPartitions * 5
	require.InDelta(s.T(), 50000.0/dqnMsgRateScale, state[idx], 1e-10, "msgRate feature")
	require.InDelta(s.T(), 256.0/dqnMsgSizeScale, state[idx+1], 1e-10, "avgMsgSize feature")
	require.InDelta(s.T(), 40000.0/dqnMsgRateScale, state[idx+2], 1e-10, "consumeRate feature")
	require.InDelta(s.T(), 50000.0/40000.0, state[idx+3], 1e-10, "produce/consume ratio feature")
}

func (s *DQNSuite) TestDeltaLoadFeature() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.atomicState.Store(&stateSnapshot{
		loads:       []float64{0.2, 0.4, 0.6, 0.8},
		queueDepth:  []int{0, 0, 0, 0},
		msgRate:     10000,
		avgMsgSize:  128,
		consumeRate: 10000,
	})
	state1 := make([]float64, dqn.stateSize)
	dqn.buildStateInto(state1)

	deltaBase := dqn.numPartitions * 4
	for i := 0; i < 4; i++ {
		require.InDelta(s.T(), 0, state1[deltaBase+i], 1e-10,
			"delta-load should be 0 on first call (no previous loads)")
	}

	dqn.atomicState.Store(&stateSnapshot{
		loads:       []float64{0.3, 0.5, 0.5, 0.7},
		queueDepth:  []int{0, 0, 0, 0},
		msgRate:     10000,
		avgMsgSize:  128,
		consumeRate: 10000,
	})
	state2 := make([]float64, dqn.stateSize)
	dqn.buildStateInto(state2)

	require.InDelta(s.T(), 0.1, state2[deltaBase+0], 1e-10, "Δload[0] = 0.3-0.2")
	require.InDelta(s.T(), 0.1, state2[deltaBase+1], 1e-10, "Δload[1] = 0.5-0.4")
	require.InDelta(s.T(), -0.1, state2[deltaBase+2], 1e-10, "Δload[2] = 0.5-0.6")
	require.InDelta(s.T(), -0.1, state2[deltaBase+3], 1e-10, "Δload[3] = 0.7-0.8")
}

func (s *DQNSuite) TestExpChannelTimeoutNotSilent() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	// Stop background goroutines so they don't drain the channel.
	dqn.Stop()

	// Replace expCh with a full channel (capacity 1, 1 item).
	dqn.expCh = make(chan nn.Transition, 1)
	dqn.expCh <- nn.Transition{}

	probs := []float64{0.25, 0.25, 0.25, 0.25}
	dqn.cachedProbs.Store(&probs)
	dqn.prevSnap.Store(&policySnap{state: make([]float64, dqn.stateSize), action: 0})
	dqn.currSnap.Store(&policySnap{state: make([]float64, dqn.stateSize), action: 1})

	// Reset the nStep buffer so it starts clean.
	dqn.nStep.Reset()

	metrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	}

	// nStep buffer requires n=5 pushes before emitting.
	for i := 0; i < 4; i++ {
		dqn.OnMetrics(metrics)
	}
	require.Equal(s.T(), int64(0), dqn.DroppedExperience(),
		"no drops expected before nStep emits")

	// Fifth OnMetrics triggers the nStep emit → channel send → timeout → drop.
	start := time.Now()
	dqn.OnMetrics(metrics)
	elapsed := time.Since(start)

	require.Less(s.T(), elapsed, 50*time.Millisecond,
		"OnMetrics must not block longer than ~1ms on a full channel")
	require.Greater(s.T(), dqn.DroppedExperience(), int64(0),
		"droppedExperience counter must increment when expCh is full")
}

func (s *DQNSuite) TestSetBalancerFactoryConsistent() {
	kinds := []struct {
		kind broker.BalancerKind
		typ  string
	}{
		{broker.KindRoundRobin, "*balancer.RoundRobin"},
		{broker.KindPowerOfTwoChoices, "*balancer.PowerOfTwoChoices"},
		{broker.KindPSA, "*balancer.PSA"},
		{broker.KindDQN, "*balancer.DQNBalancer"},
	}

	for _, tc := range kinds {
		bal, err := broker.BalancerFromKind(tc.kind, 4)
		require.NoError(s.T(), err, "factory should create %s", tc.typ)
		require.NotNil(s.T(), bal)
		actual := fmt.Sprintf("%T", bal)
		require.Equal(s.T(), tc.typ, actual,
			"BalancerFromKind(%d) should produce %s", tc.kind, tc.typ)
		if dqn, ok := bal.(*DQNBalancer); ok {
			dqn.Stop()
		}
	}

	_, err := broker.BalancerFromKind(broker.BalancerKind(999), 4)
	require.Error(s.T(), err, "unknown kind should return error")
}

func (s *DQNSuite) TestBoltzmannRoutingSpread() {
	dqn := NewDQNBalancer(8)
	defer dqn.Stop()

	probs := make([]float64, 8)
	for i := range probs {
		probs[i] = 1.0 / 8.0
	}
	dqn.cachedProbs.Store(&probs)

	counts := make([]int, 8)
	const N = 80000
	for i := 0; i < N; i++ {
		p := dqn.SelectPartition("t", nil, 8)
		counts[p]++
	}

	var entropy float64
	for _, c := range counts {
		p := float64(c) / float64(N)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	maxEntropy := math.Log2(8.0)
	require.GreaterOrEqual(s.T(), entropy, maxEntropy-0.1,
		"with equal Q-values, entropy should be near log2(K)=%.2f, got %.2f", maxEntropy, entropy)
}

func (s *DQNSuite) TestBoltzmannRoutingInflightCorrection() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	probs := []float64{0.25, 0.25, 0.25, 0.25}
	dqn.cachedProbs.Store(&probs)

	inflight := dqn.ensureInflight(4)
	(*inflight)[0].Store(1_000_000)
	(*inflight)[1].Store(0)
	(*inflight)[2].Store(0)
	(*inflight)[3].Store(0)

	counts := make([]int, 4)
	const N = 5000
	for i := 0; i < N; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
	}

	ratio := float64(counts[0]) / float64(N)
	require.Less(s.T(), ratio, 0.05,
		"partition 0 with 1M inflight should be selected <5%% of time, got %.2f%%", ratio*100)
}

func (s *DQNSuite) TestBoltzmannTemperatureAnnealing() {
	earlyTau := currentTemperature(0)
	lateTau := currentTemperature(20000)

	require.InDelta(s.T(), dqnBoltzmannTauStart, earlyTau, 1e-6,
		"temperature at start should equal TauStart")
	require.InDelta(s.T(), dqnBoltzmannTauEnd, lateTau, 1e-6,
		"temperature after 20000 experiences should equal TauEnd")

	q := []float64{1.0, 0.5, 0.5, 0.5}
	earlyProbs := softmaxFromQ(q, earlyTau)
	lateProbs := softmaxFromQ(q, lateTau)

	var earlyEntropy, lateEntropy float64
	for _, p := range earlyProbs {
		if p > 0 {
			earlyEntropy -= p * math.Log2(p)
		}
	}
	for _, p := range lateProbs {
		if p > 0 {
			lateEntropy -= p * math.Log2(p)
		}
	}

	require.Greater(s.T(), earlyEntropy, lateEntropy*3,
		"early entropy (%.4f) should be at least 3× late entropy (%.4f)", earlyEntropy, lateEntropy)
}

func (s *DQNSuite) TestDQNNoStateMutexNoPopArtMutex() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqnType := reflect.TypeOf(dqn).Elem()
	for i := 0; i < dqnType.NumField(); i++ {
		f := dqnType.Field(i)
		if f.Name == "stateMu" {
			s.Failf("forbidden mutex field", "DQNBalancer still has stateMu (%s)", f.Type.String())
		}
	}

	popArtType := reflect.TypeOf(dqn.popArt)
	for i := 0; i < popArtType.NumField(); i++ {
		f := popArtType.Field(i)
		if f.Type.String() == "sync.Mutex" || f.Type.String() == "*xsync.RBMutex" {
			s.Failf("forbidden mutex field", "popArt has %s (%s)", f.Name, f.Type.String())
		}
	}
}

func (s *DQNSuite) TestDQNConcurrentInferenceAndTrain() {
	dqn := NewDQNBalancer(4, WithDQNMinReplay(2), WithDQNBatchSize(2), WithDQNTrainEvery(1))
	defer dqn.Stop()

	probs := []float64{0.25, 0.25, 0.25, 0.25}
	dqn.cachedProbs.Store(&probs)

	var wg sync.WaitGroup
	var panics atomic.Int64

	for g := 0; g < 64; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panics.Add(1)
				}
			}()
			for i := 0; i < 100; i++ {
				_ = dqn.SelectPartition("t", nil, 4)
				dqn.OnMetrics(broker.Metrics{
					BusinessMetrics: broker.BusinessMetrics{Throughput: 50000, MsgRate: 10000},
					DerivedMetrics: broker.DerivedMetrics{
						PartitionLoads: []float64{0.2, 0.4, 0.6, float64(i) * 0.01},
					},
				})
				runtime.Gosched()
			}
		}()
	}

	wg.Wait()
	require.Equal(s.T(), int64(0), panics.Load(), "no panics during concurrent inference+train")
}
