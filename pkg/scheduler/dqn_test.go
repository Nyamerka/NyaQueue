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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 5; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler(WithDQNSchedEpsilon(0.0))
	defer dqn.Stop()
	msg, _, err := dqn.Next(p, 0)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), msg)
}

func (s *DQNSchedSuite) TestFallbackFIFO() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	_, err = p.Append(broker.NewMessage(5, []byte("k"), []byte("v")))
	require.NoError(s.T(), err)

	dqn := NewDQNScheduler()
	defer dqn.Stop()
	dqn.SetFallbackFIFO(true)

	msg, nextOff, err := dqn.Next(p, 1)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), msg)
	require.Equal(s.T(), uint64(2), nextOff)
}

func (s *DQNSchedSuite) TestFunctionalOptions() {
	tests := []struct {
		name string
		opts []DQNSchedOption
	}{
		{"default", nil},
		{"custom_epsilon", []DQNSchedOption{WithDQNSchedEpsilon(0.1)}},
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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 3; i++ {
		_, _ = p.Append(broker.NewMessage(5, []byte("k"), []byte("v")))
	}

	// Call Next to let policyLoop discover the PriorityIndex.
	_, _, _ = dqn.Next(p, 0)

	// Wait for policyLoop to produce snapshots from the cached PI.
	require.Eventually(s.T(), func() bool {
		return dqn.prevSnap.Load() != nil && dqn.currSnap.Load() != nil
	}, 2*time.Second, 10*time.Millisecond)

	dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: 10.0}})

	dqn.Stop()
	require.Greater(s.T(), dqn.replayBuffer.Len(), 0)
}

func (s *DQNSchedSuite) TestOnMetricsWithoutPriorState() {
	dqn := NewDQNScheduler()
	dqn.OnMetrics(broker.Metrics{BusinessMetrics: broker.BusinessMetrics{AvgLatency: 5.0}})
	dqn.Stop()
	require.Equal(s.T(), 0, dqn.replayBuffer.Len(), "no push without prior state")
}

func (s *DQNSchedSuite) TestEnqueueNoop() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()
	require.NotPanics(s.T(), func() {
		dqn.Enqueue(&broker.Message{}, 0)
	})
}

func (s *DQNSchedSuite) TestSetFallbackFIFOToggle() {
	dqn := NewDQNScheduler()
	defer dqn.Stop()
	require.False(s.T(), dqn.fallbackFIFO)
	dqn.SetFallbackFIFO(true)
	require.True(s.T(), dqn.fallbackFIFO)
	dqn.SetFallbackFIFO(false)
	require.False(s.T(), dqn.fallbackFIFO)
}

func (s *DQNSchedSuite) TestReplayBufSizeOption() {
	dqn := NewDQNScheduler(WithDQNSchedReplayBufSize(500))
	defer dqn.Stop()
	require.NotNil(s.T(), dqn.replayBuffer)
}

func (s *DQNSchedSuite) TestNoPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	dqn := NewDQNScheduler()
	defer dqn.Stop()
	_, _, err = dqn.Next(p, 0)
	require.Error(s.T(), err)
}

func (s *DQNSchedSuite) TestAsyncTrainingDoesNotBlockInference() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 200; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler(WithDQNSchedEpsilon(0.5))
	defer dqn.Stop()

	// Call Next to feed PI to policyLoop.
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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 10; i++ {
		_, _ = p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
	}

	// Call Next to let policyLoop discover PI.
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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 500; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i%10), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler(WithDQNSchedEpsilon(0.5))
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
