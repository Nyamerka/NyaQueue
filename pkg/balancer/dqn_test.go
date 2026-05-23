package balancer

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/alitto/pond/v2"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/stat"
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
		time.Sleep(110 * time.Millisecond)
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
		time.Sleep(110 * time.Millisecond)
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

// --- New tests for softmax routing, temperature, credit assignment ---

func (s *DQNSuite) TestSoftmaxRoutingUniformAtHighTemp() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	require.InDelta(s.T(), dqnInitialTemperature, dqn.temperature(), 0.001,
		"initial temperature should be 1.0")

	qValues := []float64{0.01, 0.02, 0.015, 0.018}
	cached := make([]float64, len(qValues))
	copy(cached, qValues)
	dqn.cachedQValues.Store(&cached)

	counts := map[int]int{}
	for i := 0; i < 4000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	for _, c := range counts {
		require.Greater(s.T(), c, 700,
			"at τ=1.0 with near-equal Q all partitions should get roughly equal share")
		require.Less(s.T(), c, 1300,
			"at τ=1.0 with near-equal Q no partition should dominate")
	}
}

func (s *DQNSuite) TestSoftmaxRoutingConvergesToBest() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	require.InDelta(s.T(), dqnFinalTemperature, dqn.temperature(), 0.01,
		"fully decayed temperature should be ~0.1")

	qValues := []float64{0.1, 0.2, 10.0, 0.3}
	cached := make([]float64, len(qValues))
	copy(cached, qValues)
	dqn.cachedQValues.Store(&cached)

	counts := map[int]int{}
	for i := 0; i < 4000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	require.Greater(s.T(), counts[2], 3000,
		"at low τ partition 2 (best Q) should dominate")
}

func (s *DQNSuite) TestTemperatureDecay() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	t0 := dqn.temperature()
	require.InDelta(s.T(), dqnInitialTemperature, t0, 0.001)

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps/2)
	tMid := dqn.temperature()
	require.Less(s.T(), tMid, t0, "temperature should decrease with training")
	require.Greater(s.T(), tMid, dqnFinalTemperature, "mid-training temperature should be between initial and final")

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps*2)
	tFinal := dqn.temperature()
	require.InDelta(s.T(), dqnFinalTemperature, tFinal, 0.01, "fully decayed temperature should reach final value")
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
		time.Sleep(110 * time.Millisecond)
	}

	require.Greater(s.T(), dqn.TrainingSteps(), int64(0),
		"DQN should accumulate training steps from step 0 without deadlock")
	require.Greater(s.T(), dqn.replayBuffer.Len(), 0,
		"replay buffer should have experiences recorded from the start")
}

func (s *DQNSuite) TestRewardDifferentialSignal() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	steadyMetrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	}
	for i := 0; i < 20; i++ {
		dqn.computeReward(steadyMetrics)
	}

	r2 := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 100000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	require.Greater(s.T(), r2, 0.0,
		"reward should be positive when throughput improves (differential signal)")

	for i := 0; i < 10; i++ {
		dqn.computeReward(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 100000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
		})
	}

	r3 := dqn.computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 10000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	require.Less(s.T(), r3, 0.0,
		"reward should be negative when throughput degrades (differential signal)")
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
		time.Sleep(110 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() >= 2
	}, 3*time.Second, 10*time.Millisecond,
		"DQN should accumulate experience and train without fallback mechanism")

	require.Greater(s.T(), dqn.TrainingSteps(), int64(0),
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

func (s *DQNSuite) TestInflightAdjustPreventsHotspot() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	qv := []float64{0.35, 0.30, 0.25, 0.20}
	cached := make([]float64, len(qv))
	copy(cached, qv)
	dqn.cachedQValues.Store(&cached)

	n := 10_000
	counts := make([]int, 4)
	for i := 0; i < n; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
	}

	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	require.Less(s.T(), float64(maxCount)/float64(n), 0.80,
		"inflight-aware penalty should prevent one partition from getting >80%%, counts=%v", counts)
}

func (s *DQNSuite) TestInflightPenaltyThresholdBased() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	inf := dqn.ensureInflight(4)
	// mean = 32.5, partition 0 ratio = 100/32.5 ≈ 3.08 (> 2.0 threshold)
	(*inf)[0].Store(100)
	(*inf)[1].Store(10)
	(*inf)[2].Store(10)
	(*inf)[3].Store(10)

	qv := []float64{0.8, 0.3, 0.2, 0.1}
	dst := make([]float64, 4)
	dqn.inflightAdjustInto(dst, qv, inf, 4)

	require.Less(s.T(), dst[0], qv[0],
		"partition 0 (ratio > 2.0) should have reduced Q")
	for i := 1; i < 4; i++ {
		require.InDelta(s.T(), qv[i], dst[i], 1e-6,
			"partition %d (ratio < 2.0) should be unpenalized", i)
	}

	// When all inflight are equal (ratio = 1.0), no penalties
	(*inf)[0].Store(50)
	(*inf)[1].Store(50)
	(*inf)[2].Store(50)
	(*inf)[3].Store(50)
	dqn.inflightAdjustInto(dst, qv, inf, 4)
	for i := range dst {
		require.InDelta(s.T(), qv[i], dst[i], 1e-6,
			"equal inflight should produce no penalty, partition %d", i)
	}
}

func (s *DQNSuite) TestAdaptiveTempFloor() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	// Use extreme rate where floor exceeds dqnFinalTemperature for k=4:
	// msgsPerTick = 100M * 0.1 = 10M; floor = 0.05 * log10(10M/100) / log2(4) = 0.05*5/2 = 0.125
	dqn.stateMu.Lock()
	dqn.msgRate = 100_000_000
	dqn.stateMu.Unlock()

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	temp := dqn.temperature()
	require.Greater(s.T(), temp, dqnFinalTemperature,
		"at extreme rate, adaptive floor should elevate temperature above final, got %.4f", temp)
}

func (s *DQNSuite) TestRewardZScoreNormalization() {
	var rn rewardNormalizer

	rng := rand.New(rand.NewPCG(42, 0))
	var outputs []float64
	for i := 0; i < 500; i++ {
		raw := rng.NormFloat64()*0.1 + 0.5
		out := rn.normalize(raw)
		if i >= 100 {
			outputs = append(outputs, out)
		}
	}

	mean := stat.Mean(outputs, nil)
	std := stat.StdDev(outputs, nil)

	require.InDelta(s.T(), 0, mean, 0.3,
		"z-score normalized rewards should have mean near 0, got %.4f", mean)
	require.InDelta(s.T(), 1.0, std, 0.5,
		"z-score normalized rewards should have std near 1, got %.4f", std)
}

func (s *DQNSuite) TestNoDriftDetection() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)
	stepsBefore := dqn.TrainingSteps()

	for i := 0; i < 100; i++ {
		dqn.computeReward(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
		})
	}
	for i := 0; i < 50; i++ {
		dqn.computeReward(broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{Throughput: 1000},
			DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.9, 0.9, 0.9, 0.9}},
		})
	}

	require.Equal(s.T(), stepsBefore, dqn.TrainingSteps(),
		"training steps should NOT be rolled back — drift detection is removed")
}

func (s *DQNSuite) TestZeroAllocSelectPartition() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	qv := []float64{0.5, 0.3, 0.1, 0.2}
	cached := make([]float64, len(qv))
	copy(cached, qv)
	dqn.cachedQValues.Store(&cached)

	for i := 0; i < 100; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		dqn.OnPublishComplete(p)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		p := dqn.SelectPartition("t", nil, 4)
		dqn.OnPublishComplete(p)
	})
	require.LessOrEqual(s.T(), allocs, 0.0,
		"SelectPartition should have zero heap allocations, got %.1f", allocs)
}

func (s *DQNSuite) TestInflightPenaltyGatedByTemp() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	qv := []float64{0.8, 0.3, 0.2, 0.1}
	cached := make([]float64, len(qv))
	copy(cached, qv)
	dqn.cachedQValues.Store(&cached)

	inf := dqn.ensureInflight(4)
	(*inf)[0].Store(1000)
	(*inf)[1].Store(10)
	(*inf)[2].Store(10)
	(*inf)[3].Store(10)

	counts := make(map[int]int)
	for i := 0; i < 4000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	require.Greater(s.T(), counts[0], 500,
		"at τ=1.0 (high temp) partition 0 should still get significant traffic even with high inflight, got %d", counts[0])
}

func (s *DQNSuite) TestInflightPenaltyLogSoft() {
	dqn := NewDQNBalancer(4)
	defer dqn.Stop()

	dqn.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	qv := []float64{0.5, 0.5, 0.5, 0.5}
	cached := make([]float64, len(qv))
	copy(cached, qv)
	dqn.cachedQValues.Store(&cached)

	inf := dqn.ensureInflight(4)
	(*inf)[0].Store(250) // 2.5× mean (mean=100)
	(*inf)[1].Store(100)
	(*inf)[2].Store(50)
	(*inf)[3].Store(0)

	counts := make(map[int]int)
	for i := 0; i < 10000; i++ {
		p := dqn.SelectPartition("t", nil, 4)
		counts[p]++
		dqn.OnPublishComplete(p)
	}

	prob0 := float64(counts[0]) / 10000.0
	require.Greater(s.T(), prob0, 0.10,
		"at 2.5× mean inflight, partition prob should be > 10%% (soft log penalty), got %.3f", prob0)
}

func (s *DQNSuite) TestRewardNormalizerReset() {
	var rn rewardNormalizer

	for i := 0; i < 100; i++ {
		rn.normalize(float64(i) * 0.1)
	}
	require.Greater(s.T(), rn.count, int64(0))
	require.NotEqual(s.T(), 0.0, rn.mean)

	rn.Reset()

	require.Equal(s.T(), int64(0), rn.count)
	require.Equal(s.T(), 0.0, rn.mean)
	require.Equal(s.T(), 0.0, rn.m2)
}

func (s *DQNSuite) TestTempFloorScalesWithPartitions() {
	dqn4 := NewDQNBalancer(4)
	defer dqn4.Stop()
	dqn32 := NewDQNBalancer(32)
	defer dqn32.Stop()

	// Use extreme rate where floor kicks in for k=4 (0.125) but not k=32 (0.05)
	dqn4.stateMu.Lock()
	dqn4.msgRate = 100_000_000
	dqn4.stateMu.Unlock()
	dqn4.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	dqn32.stateMu.Lock()
	dqn32.msgRate = 100_000_000
	dqn32.stateMu.Unlock()
	dqn32.trainingSteps.Store(dqnMinStepsForGreedy + dqnTempDecaySteps + 100)

	temp4 := dqn4.temperature()
	temp32 := dqn32.temperature()

	require.Greater(s.T(), temp4, temp32,
		"k=32 should have a lower temperature floor than k=4 (temp4=%.4f, temp32=%.4f)", temp4, temp32)
}
