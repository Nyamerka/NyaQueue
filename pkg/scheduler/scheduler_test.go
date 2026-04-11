package scheduler

import (
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// --- FIFO ---

type FIFOSuite struct {
	suite.Suite
}

func TestFIFOSuite(t *testing.T) { suite.Run(t, new(FIFOSuite)) }

func (s *FIFOSuite) TestSequentialRead() {
	tests := []struct {
		name     string
		messages int
	}{
		{"single", 1},
		{"ten", 10},
		{"hundred", 100},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			dir := s.T().TempDir()
			p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
			require.NoError(s.T(), err)
			defer p.Close()

			for i := 0; i < tc.messages; i++ {
				_, err := p.Append(broker.NewMessage(0, []byte("k"), []byte("v")))
				require.NoError(s.T(), err)
			}

			fifo := NewFIFO()
			offset := uint64(1) // WAL starts at 1
			for i := 0; i < tc.messages; i++ {
				msg, nextOff, err := fifo.Next(p, offset)
				require.NoError(s.T(), err)
				require.NotNil(s.T(), msg)
				require.Equal(s.T(), offset+1, nextOff)
				offset = nextOff
			}
		})
	}
}

func (s *FIFOSuite) TestNoMessages() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
	require.NoError(s.T(), err)
	defer p.Close()

	fifo := NewFIFO()
	_, _, err = fifo.Next(p, 1)
	require.Error(s.T(), err)
}

// --- StrictPriority ---

type PrioritySuite struct {
	suite.Suite
}

func TestPrioritySuite(t *testing.T) { suite.Run(t, new(PrioritySuite)) }

func (s *PrioritySuite) TestHighestFirst() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	priorities := []uint8{1, 9, 5, 3, 7}
	for _, pri := range priorities {
		_, err := p.Append(broker.NewMessage(pri, []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	sched := NewStrictPriority()
	expectedOrder := []uint8{9, 7, 5, 3, 1}
	for _, expected := range expectedOrder {
		msg, _, err := sched.Next(p, 0)
		require.NoError(s.T(), err)
		require.Equal(s.T(), expected, msg.Header.Priority)
	}
}

func (s *PrioritySuite) TestEmptyQueue() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	sched := NewStrictPriority()
	_, _, err = sched.Next(p, 0)
	require.Error(s.T(), err)
}

func (s *PrioritySuite) TestNoPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
	require.NoError(s.T(), err)
	defer p.Close()

	sched := NewStrictPriority()
	_, _, err = sched.Next(p, 0)
	require.Error(s.T(), err)
}

// --- DQN Scheduler ---

type DQNSchedSuite struct {
	suite.Suite
}

func TestDQNSchedSuite(t *testing.T) { suite.Run(t, new(DQNSchedSuite)) }

func (s *DQNSchedSuite) TestSelectsFromPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 5; i++ {
		_, err := p.Append(broker.NewMessage(uint8(i), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	dqn := NewDQNScheduler(WithDQNSchedEpsilon(0.0))
	msg, _, err := dqn.Next(p, 0)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), msg)
}

func (s *DQNSchedSuite) TestFallbackFIFO() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	_, err = p.Append(broker.NewMessage(5, []byte("k"), []byte("v")))
	require.NoError(s.T(), err)

	dqn := NewDQNScheduler()
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
			require.NotNil(s.T(), dqn)
			require.NotNil(s.T(), dqn.w1)
			require.NotNil(s.T(), dqn.w2)
		})
	}
}

func (s *DQNSchedSuite) TestOnMetrics() {
	dqn := NewDQNScheduler()

	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	_, _ = p.Append(broker.NewMessage(5, []byte("k"), []byte("v")))
	_, _, _ = dqn.Next(p, 0)

	dqn.OnMetrics(broker.Metrics{AvgLatency: 10.0})
	require.Greater(s.T(), dqn.replayBuffer.Len(), 0)
}

func (s *DQNSchedSuite) TestOnMetricsWithoutPriorState() {
	dqn := NewDQNScheduler()
	dqn.OnMetrics(broker.Metrics{AvgLatency: 5.0})
	require.Equal(s.T(), 0, dqn.replayBuffer.Len(), "no push without prior state")
}

func (s *DQNSchedSuite) TestForwardOutputSize() {
	dqn := NewDQNScheduler()
	state := make([]float64, dqn.stateSize)
	q := dqn.forward(state)
	require.Len(s.T(), q, dqn.numActions)
}

func (s *DQNSchedSuite) TestEnqueueNoop() {
	dqn := NewDQNScheduler()
	require.NotPanics(s.T(), func() {
		dqn.Enqueue(&broker.Message{}, 0)
	})
}

func (s *DQNSchedSuite) TestSetFallbackFIFOToggle() {
	dqn := NewDQNScheduler()
	require.False(s.T(), dqn.fallbackFIFO)
	dqn.SetFallbackFIFO(true)
	require.True(s.T(), dqn.fallbackFIFO)
	dqn.SetFallbackFIFO(false)
	require.False(s.T(), dqn.fallbackFIFO)
}

func (s *DQNSchedSuite) TestThrottleOnLoadOption() {
	dqn := NewDQNScheduler(WithDQNSchedThrottleOnLoad(0.7))
	require.Equal(s.T(), 0.7, dqn.throttleOnLoad)
}

func (s *DQNSchedSuite) TestReplayBufSizeOption() {
	dqn := NewDQNScheduler(WithDQNSchedReplayBufSize(500))
	require.NotNil(s.T(), dqn.replayBuffer)
}

func (s *DQNSchedSuite) TestNoPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
	require.NoError(s.T(), err)
	defer p.Close()

	dqn := NewDQNScheduler()
	_, _, err = dqn.Next(p, 0)
	require.Error(s.T(), err)
}

// --- FIFO extras ---

func (s *FIFOSuite) TestEnqueueNoop() {
	fifo := NewFIFO()
	require.NotPanics(s.T(), func() {
		fifo.Enqueue(&broker.Message{}, 0)
	})
}

// --- StrictPriority extras ---

func (s *PrioritySuite) TestEnqueueNoop() {
	sp := NewStrictPriority()
	require.NotPanics(s.T(), func() {
		sp.Enqueue(&broker.Message{}, 0)
	})
}

func (s *PrioritySuite) TestSamePriorityFIFO() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority)
	require.NoError(s.T(), err)
	defer p.Close()

	for i := 0; i < 3; i++ {
		_, err := p.Append(broker.NewMessage(5, []byte("k"), []byte{byte(i)}))
		require.NoError(s.T(), err)
	}

	sched := NewStrictPriority()
	var values []byte
	for i := 0; i < 3; i++ {
		msg, _, err := sched.Next(p, 0)
		require.NoError(s.T(), err)
		values = append(values, msg.Value[0])
	}
	require.Equal(s.T(), []byte{0, 1, 2}, values, "same priority: FIFO order")
}
