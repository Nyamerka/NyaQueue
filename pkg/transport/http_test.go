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

type HTTPConfigWiringSuite struct {
	suite.Suite
}

func TestHTTPConfigWiringSuite(t *testing.T) { suite.Run(t, new(HTTPConfigWiringSuite)) }

func (s *HTTPConfigWiringSuite) newWithConfig(cfg broker.Config) (*broker.Broker, *HTTPServer, *HTTPClient, func()) {
	dir := s.T().TempDir()
	store, err := broker.NewOffsetStore(dir)
	require.NoError(s.T(), err)

	bal := balancer.NewRoundRobin()
	b := broker.New(cfg, dir, bal, store)
	b.Start()

	srv := NewHTTPServer(b)
	err = srv.Start(":0")
	require.NoError(s.T(), err)

	c := NewHTTPClient(srv.Addr())
	return b, srv, c, func() {
		c.Close()
		srv.Stop()
		b.Stop()
	}
}

func (s *HTTPConfigWiringSuite) TestMaxMessageBytesHTTP() {
	cfg := broker.DefaultConfig()
	cfg.MaxMessageBytes = 128
	b, _, c, cleanup := s.newWithConfig(cfg)
	defer cleanup()

	ctx := context.Background()
	require.NoError(s.T(), c.CreateTopic(ctx, "t", 1, "fifo"))
	b.SetScheduler("t", scheduler.NewFIFO())

	_, _, err := c.Produce(ctx, "t", []byte("k"), make([]byte, 50), 0)
	require.NoError(s.T(), err)

	_, _, err = c.Produce(ctx, "t", []byte("k"), make([]byte, 200), 0)
	require.Error(s.T(), err)
}

func (s *HTTPConfigWiringSuite) TestCompressionHTTP() {
	cfg := broker.DefaultConfig()
	cfg.CompressionType = broker.CompressionLZ4
	b, _, c, cleanup := s.newWithConfig(cfg)
	defer cleanup()

	ctx := context.Background()
	require.NoError(s.T(), c.CreateTopic(ctx, "t", 1, "fifo"))
	b.SetScheduler("t", scheduler.NewFIFO())

	payload := []byte("lz4 compression test data for http transport")
	_, _, err := c.Produce(ctx, "t", []byte("k"), payload, 0)
	require.NoError(s.T(), err)

	msgs, err := c.Consume(ctx, "t", "g1", 0, 1<<20)
	require.NoError(s.T(), err)
	require.Len(s.T(), msgs, 1)
	require.Equal(s.T(), payload, msgs[0].Value)
}

func (s *HTTPConfigWiringSuite) TestHTTPServerUsesTimeouts() {
	cfg := broker.DefaultConfig()
	cfg.ReadTimeoutMs = 5000
	cfg.WriteTimeoutMs = 10000
	_, srv, _, cleanup := s.newWithConfig(cfg)
	defer cleanup()

	require.NotNil(s.T(), srv.server)
}

func (s *HTTPConfigWiringSuite) TestHTTPMaxQueuedRequests() {
	cfg := broker.DefaultConfig()
	cfg.MaxQueuedRequests = 2
	_, srv, _, cleanup := s.newWithConfig(cfg)
	defer cleanup()

	require.NotNil(s.T(), srv.reqSem)
	require.Equal(s.T(), 2, cap(srv.reqSem))
}
