package balancer

import (
	"runtime"
	"sync"
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

	badMetrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 500},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	}
	for i := 0; i < fallbackEnterTicks; i++ {
		dqn.OnMetrics(badMetrics)
	}
	require.True(s.T(), dqn.IsFallbackActive())

	goodMetrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 900},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	}
	for i := 0; i < fallbackExitTicks; i++ {
		dqn.OnMetrics(goodMetrics)
	}
	require.False(s.T(), dqn.IsFallbackActive())
}

func (s *DQNSuite) TestProactiveWatchdog() {
	dqn := NewDQNBalancer(4, WithDQNLoadThreshold(0.7))
	defer dqn.Stop()

	lowLoad := broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.3, 0.3, 0.3, 0.3}},
	}
	dqn.OnMetrics(lowLoad)
	require.False(s.T(), dqn.IsFallbackActive(), "low load should not trigger fallback")

	highLoad := broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.8, 0.9, 0.7, 0.85}},
	}
	for i := 0; i < fallbackEnterTicks; i++ {
		dqn.OnMetrics(highLoad)
	}
	require.True(s.T(), dqn.IsFallbackActive(), "high mean load should trigger proactive fallback")

	for i := 0; i < fallbackExitTicks; i++ {
		dqn.OnMetrics(lowLoad)
	}
	require.False(s.T(), dqn.IsFallbackActive(), "load recovery should deactivate fallback")
}

func (s *DQNSuite) TestOnMetricsTrains() {
	dqn := NewDQNBalancer(4, WithDQNTrainEvery(1))
	defer dqn.Stop()

	dqn.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.2, 0.4, 0.6, 0.8}},
	})

	// Wait for policyLoop to produce snapshots.
	time.Sleep(300 * time.Millisecond)

	// Send more OnMetrics to trigger experience push.
	for i := 0; i < 5; i++ {
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
	balanced := computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	imbalanced := computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.9, 0.1, 0.9}},
	})
	require.Greater(s.T(), balanced, imbalanced,
		"more balanced loads should yield higher reward")
}

func (s *DQNSuite) TestRewardPropertyEmptyLoads() {
	reward := computeReward(broker.Metrics{})
	require.Equal(s.T(), 0.0, reward)
}

func (s *DQNSuite) TestRewardPropertyThroughputMatters() {
	lowTP := computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 1000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	highTP := computeReward(broker.Metrics{
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

	for i := 0; i < 5; i++ {
		dqn.OnMetrics(broker.Metrics{
			DerivedMetrics: broker.DerivedMetrics{PartitionLoads: []float64{0.1, 0.2, 0.3, float64(i) * 0.1}},
		})
		time.Sleep(110 * time.Millisecond)
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() >= 2
	}, time.Second, 5*time.Millisecond)
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

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				dqn.SelectPartition("t", []byte{1}, 4)
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 100; j++ {
			dqn.OnMetrics(broker.Metrics{
				DerivedMetrics: broker.DerivedMetrics{
					PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
				},
			})
		}
	}()
	wg.Wait()
}

func (s *DQNSuite) TestDroppedExperienceCounter() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(1.0))
	defer dqn.Stop()

	require.Equal(s.T(), int64(0), dqn.DroppedExperience())
}

func (s *DQNSuite) TestDQNBalancer_DepthPenalty() {
	noDepth := computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	})
	withDepth := computeReward(broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 50000},
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5},
			QueueDepth:     []int{50000, 50000, 50000, 50000},
		},
	})
	require.Greater(s.T(), noDepth, withDepth,
		"large queue depth should penalise reward")
}

func (s *DQNSuite) TestDQNBalancer_NoExploreInFallback() {
	dqn := NewDQNBalancer(4, WithDQNEpsilon(1.0), WithDQNLoadThreshold(0))
	defer dqn.Stop()
	dqn.SetBaseThroughput(1000)

	badMetrics := broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{Throughput: 500},
		DerivedMetrics:  broker.DerivedMetrics{PartitionLoads: []float64{0.5, 0.5, 0.5, 0.5}},
	}
	for i := 0; i < fallbackEnterTicks; i++ {
		dqn.OnMetrics(badMetrics)
	}
	require.True(s.T(), dqn.IsFallbackActive())
	require.True(s.T(), dqn.epsilonSuppressed.Load(),
		"epsilon should be suppressed during fallback")

	counts := map[int]int{}
	for i := 0; i < 1000; i++ {
		p := dqn.SelectPartition("t", []byte("k"), 4)
		counts[p]++
	}
	for _, c := range counts {
		require.Greater(s.T(), c, 200,
			"under fallback (RR) all partitions should get roughly equal share, not random")
	}
}
