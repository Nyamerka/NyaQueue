package broker

import (
	"os"
	"path/filepath"
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

func (s *BrokerSuite) TestDeleteTopicCleansWAL() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	defer b.Stop()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 2
	require.NoError(s.T(), b.CreateTopic("wal-clean", cfg))
	b.SetScheduler("wal-clean", fifoScheduler{})

	msg := NewMessage(0, []byte("k"), []byte("v"))
	_, _, err = b.Publish("wal-clean", msg)
	require.NoError(s.T(), err)

	walDir := filepath.Join(dir, "wal-clean")
	_, err = os.Stat(walDir)
	require.NoError(s.T(), err, "WAL directory should exist before delete")

	require.NoError(s.T(), b.DeleteTopic("wal-clean"))

	_, err = os.Stat(walDir)
	require.True(s.T(), os.IsNotExist(err), "WAL directory should be removed after delete")
}

func (s *BrokerSuite) TestDeleteTopicCleansOffsets() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	defer b.Stop()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("off-clean", cfg))
	b.SetScheduler("off-clean", fifoScheduler{})

	msg := NewMessage(0, []byte("k"), []byte("v"))
	_, _, err = b.Publish("off-clean", msg)
	require.NoError(s.T(), err)

	require.NoError(s.T(), b.Commit("grp", "off-clean", 0, 42))
	got, err := store.Load("grp", "off-clean", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(42), got)

	require.NoError(s.T(), b.DeleteTopic("off-clean"))

	_, err = store.Load("grp", "off-clean", 0)
	require.Error(s.T(), err, "offset should be gone after topic delete")
}

func (s *BrokerSuite) TestDeleteAndRecreateTopic() {
	b, cleanup := s.newBroker()
	defer cleanup()

	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("recreate", cfg))
	b.SetScheduler("recreate", fifoScheduler{})

	msg := NewMessage(0, []byte("k"), []byte("old"))
	_, _, err := b.Publish("recreate", msg)
	require.NoError(s.T(), err)

	require.NoError(s.T(), b.DeleteTopic("recreate"))
	require.NoError(s.T(), b.CreateTopic("recreate", cfg))
	b.SetScheduler("recreate", fifoScheduler{})

	msg2 := NewMessage(0, []byte("k"), []byte("new"))
	_, _, err = b.Publish("recreate", msg2)
	require.NoError(s.T(), err)

	got, _, err := b.Consume("recreate", "grp", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), []byte("new"), got.Value, "should see only the new message, not stale data")
}

func (s *BrokerSuite) TestPublishRejectsOversizedMessage() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	cfg := DefaultConfig()
	cfg.MaxMessageBytes = 100
	b := New(cfg, dir, noopBalancer{}, store)
	defer b.Stop()

	tcfg := DefaultTopicConfig()
	tcfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("t", tcfg))
	b.SetScheduler("t", fifoScheduler{})

	small := NewMessage(0, []byte("k"), make([]byte, 50))
	_, _, err = b.Publish("t", small)
	require.NoError(s.T(), err)

	big := NewMessage(0, []byte("k"), make([]byte, 200))
	_, _, err = b.Publish("t", big)
	require.ErrorIs(s.T(), err, ErrMessageTooLarge)
}

func (s *BrokerSuite) TestPublishBatchRejectsOversizedMessage() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	cfg := DefaultConfig()
	cfg.MaxMessageBytes = 100
	b := New(cfg, dir, noopBalancer{}, store)
	defer b.Stop()

	tcfg := DefaultTopicConfig()
	tcfg.NumPartitions = 1
	require.NoError(s.T(), b.CreateTopic("t", tcfg))
	b.SetScheduler("t", fifoScheduler{})

	msgs := []*Message{
		NewMessage(0, []byte("k"), make([]byte, 50)),
		NewMessage(0, []byte("k"), make([]byte, 200)),
		NewMessage(0, []byte("k"), make([]byte, 30)),
	}

	results := b.PublishBatch("t", msgs)
	require.Len(s.T(), results, 3)
	require.NoError(s.T(), results[0].Err)
	require.ErrorIs(s.T(), results[1].Err, ErrMessageTooLarge)
	require.NoError(s.T(), results[2].Err)
}

func (s *BrokerSuite) TestCompressionRoundTrip() {
	for _, codec := range []int{CompressionSnappy, CompressionGzip, CompressionLZ4} {
		s.Run(compressionName(codec), func() {
			dir := s.T().TempDir()
			store, err := NewOffsetStore(dir)
			require.NoError(s.T(), err)

			cfg := DefaultConfig()
			cfg.CompressionType = codec
			b := New(cfg, dir, noopBalancer{}, store)
			defer b.Stop()

			tcfg := DefaultTopicConfig()
			tcfg.NumPartitions = 1
			require.NoError(s.T(), b.CreateTopic("t", tcfg))
			b.SetScheduler("t", fifoScheduler{})

			original := []byte("hello world this is a compression test payload")
			msg := NewMessage(0, []byte("k"), original)
			_, _, err = b.Publish("t", msg)
			require.NoError(s.T(), err)

			got, _, err := b.Consume("t", "grp", 0)
			require.NoError(s.T(), err)
			require.Equal(s.T(), original, got.Value)
		})
	}
}

func compressionName(codec int) string {
	switch codec {
	case CompressionSnappy:
		return "snappy"
	case CompressionGzip:
		return "gzip"
	case CompressionLZ4:
		return "lz4"
	default:
		return "none"
	}
}

func (s *BrokerSuite) TestIOPoolConcurrentBatchWrites() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	cfg := DefaultConfig()
	cfg.NumIOGoroutines = 2
	b := New(cfg, dir, noopBalancer{}, store)
	defer b.Stop()

	tcfg := DefaultTopicConfig()
	tcfg.NumPartitions = 4
	require.NoError(s.T(), b.CreateTopic("t", tcfg))
	b.SetScheduler("t", fifoScheduler{})

	msgs := make([]*Message, 100)
	for i := range msgs {
		msgs[i] = NewMessage(0, []byte("k"), []byte("v"))
	}

	results := b.PublishBatch("t", msgs)
	require.Len(s.T(), results, 100)

	for _, r := range results {
		require.NoError(s.T(), r.Err)
		require.Greater(s.T(), r.Offset, uint64(0))
	}
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
