package scheduler

import (
	"sync"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type DQNSchedSuite struct {
	suite.Suite
}

func TestDQNSchedSuite(t *testing.T) { suite.Run(t, new(DQNSchedSuite)) }

func (s *DQNSchedSuite) TestSelectsFromPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 5; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler()
	defer dqn.Stop()
	msg, _, err := dqn.Next(p, 0)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), msg)
}

func (s *DQNSchedSuite) TestFunctionalOptions() {
	tests := []struct {
		name string
		opts []DQNSchedOption
	}{
		{"default", nil},
		{"custom_gamma", []DQNSchedOption{WithDQNSchedGamma(0.95)}},
		{"custom_hidden", []DQNSchedOption{WithDQNSchedHiddenSize(32)}},
		{"custom_lr", []DQNSchedOption{WithDQNSchedLearningRate(0.01)}},
		{"custom_batch", []DQNSchedOption{WithDQNSchedBatchSize(16)}},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			dqn := NewDQNScheduler(tc.opts...)
			defer dqn.Stop()
			require.NotNil(s.T(), dqn)
		})
	}
}

func (s *DQNSchedSuite) TestOnMetrics() {
	dqn := NewDQNScheduler()

	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 3; i++ {
		_, _ = p.Append(broker.NewMessage(5, []byte("k"), []byte("v")))
	}

	_, _, _ = dqn.Next(p, 0)

	require.Eventually(s.T(), func() bool {
		return dqn.prevSnap.Load() != nil && dqn.currSnap.Load() != nil
	}, 2*time.Second, 10*time.Millisecond)

	dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: 10.0}})

	dqn.Stop()
	require.Greater(s.T(), dqn.replayBuffer.Len(), 0)
}

func (s *DQNSchedSuite) TestOnMetricsWithoutPriorState() {
	dqn := NewDQNScheduler()
	initial := dqn.replayBuffer.Len()
	dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: 5.0}})
	dqn.Stop()
	require.Equal(s.T(), initial, dqn.replayBuffer.Len(), "no push without prior state")
}

func (s *DQNSchedSuite) TestEnqueueNoop() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()
	require.NotPanics(s.T(), func() {
		dqn.Enqueue(&broker.Message{}, 0)
	})
}

func (s *DQNSchedSuite) TestReplayBufSizeOption() {
	dqn := NewDQNScheduler(WithDQNSchedReplayBufSize(500))
	defer dqn.Stop()
	require.NotNil(s.T(), dqn.replayBuffer)
}

func (s *DQNSchedSuite) TestNoPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	dqn := NewDQNScheduler()
	defer dqn.Stop()
	_, _, err = dqn.Next(p, 0)
	require.Error(s.T(), err)
}

func (s *DQNSchedSuite) TestAsyncTrainingDoesNotBlockInference() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 200; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler()
	defer dqn.Stop()

	_, _, _ = dqn.Next(p, 0)
	require.Eventually(s.T(), func() bool {
		return dqn.prevSnap.Load() != nil
	}, 2*time.Second, 10*time.Millisecond)

	for i := 0; i < 100; i++ {
		_, _, _ = dqn.Next(p, 0)
		dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: float64(i)}})
	}

	require.Eventually(s.T(), func() bool {
		return dqn.replayBuffer.Len() > 0
	}, time.Second, 5*time.Millisecond, "experience should be collected via async channel")
}

func (s *DQNSchedSuite) TestBellmanUsesCorrectNextState() {
	dqn := NewDQNScheduler()

	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 10; i++ {
		_, _ = p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
	}

	_, _, _ = dqn.Next(p, 0)

	require.Eventually(s.T(), func() bool {
		return dqn.prevSnap.Load() != nil && dqn.currSnap.Load() != nil
	}, 2*time.Second, 10*time.Millisecond)

	dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: 5.0}})
	dqn.Stop()

	require.Greater(s.T(), dqn.replayBuffer.Len(), 0)
}

func (s *DQNSchedSuite) TestConcurrentNextAndOnMetrics() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 500; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler()
	defer dqn.Stop()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_, _, _ = dqn.Next(p, 0)
			}
		}()
	}
	go func() {
		for j := 0; j < 100; j++ {
			dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: float64(j)}})
		}
	}()
	wg.Wait()
}

func (s *DQNSchedSuite) TestRewardHighPriorityWeightedMore() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()

	dqn.latencySums[0].Store(uint64(1 * 1e9))
	dqn.latencyCounts[0].Store(1)
	dqn.latencySums[9].Store(uint64(0.01 * 1e9))
	dqn.latencyCounts[9].Store(1)

	rewardHighPrioSlow := dqn.computePerPriorityReward()

	for i := 0; i < broker.MaxPriority; i++ {
		dqn.latencySums[i].Store(0)
		dqn.latencyCounts[i].Store(0)
	}

	dqn.latencySums[0].Store(uint64(0.01 * 1e9))
	dqn.latencyCounts[0].Store(1)
	dqn.latencySums[9].Store(uint64(1 * 1e9))
	dqn.latencyCounts[9].Store(1)

	rewardHighPrioFast := dqn.computePerPriorityReward()

	require.Greater(s.T(), rewardHighPrioFast, rewardHighPrioSlow,
		"fast high-priority + slow low-priority should yield better reward")
}

func (s *DQNSchedSuite) TestRewardPriorityOrderBonus() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()

	dqn.latencySums[0].Store(uint64(0.1 * 1e9))
	dqn.latencyCounts[0].Store(1)
	dqn.latencySums[9].Store(uint64(1.0 * 1e9))
	dqn.latencyCounts[9].Store(1)

	rewardCorrectOrder := dqn.computePerPriorityReward()

	for i := 0; i < broker.MaxPriority; i++ {
		dqn.latencySums[i].Store(0)
		dqn.latencyCounts[i].Store(0)
	}

	dqn.latencySums[0].Store(uint64(1.0 * 1e9))
	dqn.latencyCounts[0].Store(1)
	dqn.latencySums[9].Store(uint64(0.1 * 1e9))
	dqn.latencyCounts[9].Store(1)

	rewardInvertedOrder := dqn.computePerPriorityReward()

	require.Greater(s.T(), rewardCorrectOrder, rewardInvertedOrder,
		"serving high-prio faster should yield bonus")
}

func (s *DQNSchedSuite) TestDQNSchedulerNoEpsilon() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone, 1)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 100; i++ {
		_, _ = p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
	}

	dqn := NewDQNScheduler()
	defer dqn.Stop()

	for i := 0; i < 50; i++ {
		msg, _, err := dqn.Next(p, 0)
		if err != nil {
			continue
		}
		require.NotNil(s.T(), msg, "DQN scheduler should always use cached threshold (argmax)")
	}
}

func (s *DQNSchedSuite) TestNoExpertSeeds() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()

	require.Equal(s.T(), 0, dqn.replayBuffer.Len(),
		"replay buffer should start empty (no expert seeding)")
}
