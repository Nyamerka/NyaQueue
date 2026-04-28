package transport

import (
	"context"
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type HTTPTransportSuite struct {
	suite.Suite
	server *HTTPServer
	client *HTTPClient
	broker *broker.Broker
}

func TestHTTPTransportSuite(t *testing.T) { suite.Run(t, new(HTTPTransportSuite)) }

func (s *HTTPTransportSuite) SetupTest() {
	dir := s.T().TempDir()

	offsetStore, err := broker.NewOffsetStore(dir)
	require.NoError(s.T(), err)

	bal := balancer.NewRoundRobin()
	s.broker = broker.New(broker.DefaultConfig(), dir, bal, offsetStore)
	s.broker.Start()

	s.server = NewHTTPServer(s.broker)
	err = s.server.Start(":0")
	require.NoError(s.T(), err)

	s.client = NewHTTPClient(s.server.Addr())
}

func (s *HTTPTransportSuite) TearDownTest() {
	if s.client != nil {
		s.client.Close()
	}
	if s.server != nil {
		s.server.Stop()
	}
	if s.broker != nil {
		s.broker.Stop()
	}
}

func (s *HTTPTransportSuite) createTopicWithScheduler(topic string, partitions int32, mode string) {
	ctx := context.Background()
	err := s.client.CreateTopic(ctx, topic, partitions, mode)
	require.NoError(s.T(), err)
	s.broker.SetScheduler(topic, scheduler.NewFIFO())
}

func (s *HTTPTransportSuite) TestCreateAndListTopics() {
	ctx := context.Background()

	err := s.client.CreateTopic(ctx, "test-topic", 4, "fifo")
	require.NoError(s.T(), err)

	topics, err := s.client.ListTopics(ctx)
	require.NoError(s.T(), err)

	found := false
	for _, t := range topics {
		if t.Topic == "test-topic" {
			require.Equal(s.T(), int32(4), t.NumPartitions)
			found = true
		}
	}
	require.True(s.T(), found)
}

func (s *HTTPTransportSuite) TestProduceAndConsume() {
	ctx := context.Background()
	s.createTopicWithScheduler("pc-test", 1, "fifo")

	partition, offset, err := s.client.Produce(ctx, "pc-test", []byte("key1"), []byte("value1"), 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), 0, partition)
	require.Greater(s.T(), offset, int64(0))

	msgs, err := s.client.Consume(ctx, "pc-test", "g1", 0, 65536)
	require.NoError(s.T(), err)
	require.Len(s.T(), msgs, 1)
	require.Equal(s.T(), []byte("key1"), msgs[0].Key)
	require.Equal(s.T(), []byte("value1"), msgs[0].Value)
}

func (s *HTTPTransportSuite) TestProduceBatch() {
	ctx := context.Background()
	s.createTopicWithScheduler("batch-test", 2, "fifo")

	records := make([]HTTPProduceRecord, 10)
	for i := range records {
		records[i] = HTTPProduceRecord{Key: []byte("k"), Value: []byte("v"), Priority: 0}
	}

	results, err := s.client.ProduceBatch(ctx, "batch-test", records)
	require.NoError(s.T(), err)
	require.Len(s.T(), results, 10)
}

func (s *HTTPTransportSuite) TestBatchConsume() {
	ctx := context.Background()
	s.createTopicWithScheduler("bc-test", 1, "fifo")

	for i := 0; i < 5; i++ {
		_, _, err := s.client.Produce(ctx, "bc-test", []byte("k"), []byte("v"), 0)
		require.NoError(s.T(), err)
	}

	msgs, err := s.client.Consume(ctx, "bc-test", "g1", 0, 1<<20)
	require.NoError(s.T(), err)
	require.GreaterOrEqual(s.T(), len(msgs), 1, "should return at least 1 message")
}

func (s *HTTPTransportSuite) TestCommit() {
	ctx := context.Background()
	s.createTopicWithScheduler("commit-test", 1, "fifo")

	_, _, err := s.client.Produce(ctx, "commit-test", []byte("k"), []byte("v"), 0)
	require.NoError(s.T(), err)

	_, err = s.client.Consume(ctx, "commit-test", "g1", 0, 65536)
	require.NoError(s.T(), err)

	err = s.client.Commit(ctx, "commit-test", "g1", 0, 2)
	require.NoError(s.T(), err)
}

func (s *HTTPTransportSuite) TestDeleteTopic() {
	ctx := context.Background()

	err := s.client.CreateTopic(ctx, "del-test", 1, "fifo")
	require.NoError(s.T(), err)

	err = s.client.DeleteTopic(ctx, "del-test")
	require.NoError(s.T(), err)
}

func (s *HTTPTransportSuite) TestConsumeEmpty() {
	ctx := context.Background()
	s.createTopicWithScheduler("empty-test", 1, "fifo")

	msgs, err := s.client.Consume(ctx, "empty-test", "g1", 0, 65536)
	require.NoError(s.T(), err)
	require.Empty(s.T(), msgs)
}

func (s *HTTPTransportSuite) TestDeleteNonExistentTopic() {
	ctx := context.Background()
	err := s.client.DeleteTopic(ctx, "nonexistent")
	require.Error(s.T(), err)
}

func (s *HTTPTransportSuite) TestCreateDuplicateTopic() {
	ctx := context.Background()
	err := s.client.CreateTopic(ctx, "dup-test", 1, "fifo")
	require.NoError(s.T(), err)

	err = s.client.CreateTopic(ctx, "dup-test", 1, "fifo")
	require.Error(s.T(), err)
}

func (s *HTTPTransportSuite) TestHealthz() {
	resp, err := s.client.client.Get(s.client.base + "/healthz")
	require.NoError(s.T(), err)
	defer resp.Body.Close()
	require.Equal(s.T(), 200, resp.StatusCode)
}
