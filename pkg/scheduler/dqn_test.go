package scheduler

import (
	"testing"

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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
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
	q, hidden := dqn.forward(state)
	require.Len(s.T(), q, dqn.numActions)
	require.Len(s.T(), hidden, dqn.hiddenSize)
}

func (s *DQNSchedSuite) TestForwardDeterministic() {
	dqn := NewDQNScheduler()
	state := make([]float64, dqn.stateSize)
	for i := range state {
		state[i] = float64(i) * 0.1
	}
	q1, h1 := dqn.forward(state)
	q2, h2 := dqn.forward(state)
	require.InDeltaSlice(s.T(), q1, q2, 1e-12)
	require.InDeltaSlice(s.T(), h1, h2, 1e-12)
}

func (s *DQNSchedSuite) TestUpdateWeightsChangesWeights() {
	dqn := NewDQNScheduler(WithDQNSchedLearningRate(0.1))
	state := make([]float64, dqn.stateSize)
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

func (s *DQNSchedSuite) TestForwardShortState() {
	dqn := NewDQNScheduler()
	state := []float64{0.1, 0.2}
	q, hidden := dqn.forward(state)
	require.Len(s.T(), q, dqn.numActions)
	require.Len(s.T(), hidden, dqn.hiddenSize)
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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	dqn := NewDQNScheduler()
	_, _, err = dqn.Next(p, 0)
	require.Error(s.T(), err)
}
