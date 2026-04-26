package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type BrokerSuite struct {
	suite.Suite
}

func TestBrokerSuite(t *testing.T) { suite.Run(t, new(BrokerSuite)) }

func (s *BrokerSuite) newBroker() (*Broker, func()) {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	return b, func() { b.Stop() }
}

func (s *BrokerSuite) TestCreateAndListTopics() {
	b, cleanup := s.newBroker()
	defer cleanup()

	require.NoError(s.T(), b.CreateTopic("t1", DefaultTopicConfig()))
	require.NoError(s.T(), b.CreateTopic("t2", DefaultTopicConfig()))

	topics := b.ListTopics()
	require.Len(s.T(), topics, 2)
}

func (s *BrokerSuite) TestCreateDuplicateTopic() {
	b, cleanup := s.newBroker()
	defer cleanup()

	require.NoError(s.T(), b.CreateTopic("t1", DefaultTopicConfig()))
	require.Error(s.T(), b.CreateTopic("t1", DefaultTopicConfig()))
}

func (s *BrokerSuite) TestDeleteTopic() {
	b, cleanup := s.newBroker()
	defer cleanup()

	require.NoError(s.T(), b.CreateTopic("t1", DefaultTopicConfig()))
	require.NoError(s.T(), b.DeleteTopic("t1"))
	require.Error(s.T(), b.DeleteTopic("t1"))
	require.Empty(s.T(), b.ListTopics())
}

func (s *BrokerSuite) TestPublishToUnknownTopic() {
	b, cleanup := s.newBroker()
	defer cleanup()

	_, _, err := b.Publish("nonexistent", NewMessage(0, nil, nil))
	require.Error(s.T(), err)
}

func (s *BrokerSuite) TestPublishAndConsume() {
	b, cleanup := s.newBroker()
	defer cleanup()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("t", cfg))
	b.SetScheduler("t", fifoScheduler{})

	msg := NewMessage(0, []byte("k"), []byte("hello"))
	part, off, err := b.Publish("t", msg)
	require.NoError(s.T(), err)
	require.Equal(s.T(), 0, part)
	require.Equal(s.T(), uint64(1), off)

	got, nextOff, err := b.Consume("t", "grp", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), []byte("hello"), got.Value)
	require.Equal(s.T(), uint64(2), nextOff)
}

func (s *BrokerSuite) TestCommit() {
	b, cleanup := s.newBroker()
	defer cleanup()

	require.NoError(s.T(), b.Commit("grp", "t", 0, 42))
}

func (s *BrokerSuite) TestPublishBatch() {
	b, cleanup := s.newBroker()
	defer cleanup()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 2
	require.NoError(s.T(), b.CreateTopic("t", cfg))
	b.SetScheduler("t", fifoScheduler{})

	msgs := make([]*Message, 20)
	for i := range msgs {
		msgs[i] = NewMessage(0, []byte("k"), []byte("v"))
	}

	results := b.PublishBatch("t", msgs)
	require.Len(s.T(), results, 20)

	for _, r := range results {
		require.NoError(s.T(), r.Err)
		require.Greater(s.T(), r.Offset, uint64(0))
	}
}

func (s *BrokerSuite) TestPublishBatchToUnknownTopic() {
	b, cleanup := s.newBroker()
	defer cleanup()

	msgs := []*Message{NewMessage(0, nil, nil)}
	results := b.PublishBatch("nonexistent", msgs)
	require.Len(s.T(), results, 1)
	require.Error(s.T(), results[0].Err)
}

func (s *BrokerSuite) TestConsumeNoScheduler() {
	b, cleanup := s.newBroker()
	defer cleanup()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("t", cfg))

	_, _, err := b.Consume("t", "grp", 0)
	require.Error(s.T(), err)
}

// fifoScheduler is a minimal in-package scheduler for testing.
type fifoScheduler struct{}

func (fifoScheduler) Next(p *Partition, offset uint64) (*Message, uint64, error) {
	msg, err := p.Read(offset)
	if err != nil {
		return nil, offset, err
	}
	return msg, offset + 1, nil
}

func (fifoScheduler) Enqueue(*Message, int64) {}
