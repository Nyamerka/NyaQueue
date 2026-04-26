package scheduler

import (
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type PrioritySuite struct {
	suite.Suite
}

func TestPrioritySuite(t *testing.T) { suite.Run(t, new(PrioritySuite)) }

func (s *PrioritySuite) TestHighestFirst() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
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
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	sched := NewStrictPriority()
	_, _, err = sched.Next(p, 0)
	require.Error(s.T(), err)
}

func (s *PrioritySuite) TestNoPriorityIndex() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO, broker.SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	sched := NewStrictPriority()
	_, _, err = sched.Next(p, 0)
	require.Error(s.T(), err)
}

func (s *PrioritySuite) TestEnqueueNoop() {
	sp := NewStrictPriority()
	require.NotPanics(s.T(), func() {
		sp.Enqueue(&broker.Message{}, 0)
	})
}

func (s *PrioritySuite) TestSamePriorityFIFO() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeStrictPriority, broker.SyncNone)
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
