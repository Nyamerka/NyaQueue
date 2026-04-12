package transport

import (
	"context"
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type TransportSuite struct {
	suite.Suite
	server *Server
	client *Client
	broker *broker.Broker
	dir    string
}

func TestTransportSuite(t *testing.T) { suite.Run(t, new(TransportSuite)) }

func (s *TransportSuite) SetupTest() {
	s.dir = s.T().TempDir()

	offsetStore, err := broker.NewOffsetStore(s.dir)
	require.NoError(s.T(), err)

	bal := balancer.NewRoundRobin()
	s.broker = broker.New(broker.DefaultConfig(), s.dir, bal, offsetStore)
	s.broker.Start()

	s.server = NewServer(s.broker)
	err = s.server.Start(":0")
	require.NoError(s.T(), err)

	s.client, err = NewClient(s.server.Addr())
	require.NoError(s.T(), err)
}

func (s *TransportSuite) TearDownTest() {
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

func (s *TransportSuite) createTopicWithScheduler(topic string, partitions int32, mode pb.ScheduleMode) {
	ctx := context.Background()
	err := s.client.CreateTopic(ctx, topic, partitions, mode)
	require.NoError(s.T(), err)
	s.broker.SetScheduler(topic, scheduler.NewFIFO())
}

func (s *TransportSuite) TestServerAddr() {
	require.NotEmpty(s.T(), s.server.Addr())
}

func (s *TransportSuite) TestCreateAndListTopics() {
	ctx := context.Background()

	err := s.client.CreateTopic(ctx, "test-topic", 4, pb.ScheduleMode_FIFO)
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
	require.True(s.T(), found, "topic should be listed")
}

func (s *TransportSuite) TestProduceAndConsume() {
	ctx := context.Background()

	s.createTopicWithScheduler("pc-test", 1, pb.ScheduleMode_FIFO)

	partition, offset, err := s.client.Produce(ctx, "pc-test", []byte("key1"), []byte("value1"), 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int32(0), partition)
	require.Greater(s.T(), offset, int64(0))

	msgs, err := s.client.Consume(ctx, "pc-test", "g1", 0, 65536)
	require.NoError(s.T(), err)
	require.Len(s.T(), msgs, 1)
	require.Equal(s.T(), []byte("key1"), msgs[0].Key)
	require.Equal(s.T(), []byte("value1"), msgs[0].Value)
}

func (s *TransportSuite) TestCommit() {
	ctx := context.Background()

	s.createTopicWithScheduler("commit-test", 1, pb.ScheduleMode_FIFO)

	_, _, err := s.client.Produce(ctx, "commit-test", []byte("k"), []byte("v"), 0)
	require.NoError(s.T(), err)

	_, err = s.client.Consume(ctx, "commit-test", "g1", 0, 65536)
	require.NoError(s.T(), err)

	err = s.client.Commit(ctx, "commit-test", "g1", 0, 2)
	require.NoError(s.T(), err)
}

func (s *TransportSuite) TestDeleteTopic() {
	ctx := context.Background()

	err := s.client.CreateTopic(ctx, "del-test", 1, pb.ScheduleMode_FIFO)
	require.NoError(s.T(), err)

	err = s.client.DeleteTopic(ctx, "del-test")
	require.NoError(s.T(), err)
}

func (s *TransportSuite) TestGetMetrics() {
	ctx := context.Background()

	resp, err := s.client.GetMetrics(ctx)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), resp)
}

func (s *TransportSuite) TestProduceToNonExistentTopic() {
	ctx := context.Background()
	_, _, err := s.client.Produce(ctx, "nonexistent", []byte("k"), []byte("v"), 0)
	require.Error(s.T(), err)
}

func (s *TransportSuite) TestCreateTopicWithPriority() {
	ctx := context.Background()

	err := s.client.CreateTopic(ctx, "pri-test", 2, pb.ScheduleMode_STRICT_PRIORITY)
	require.NoError(s.T(), err)

	topics, err := s.client.ListTopics(ctx)
	require.NoError(s.T(), err)

	for _, t := range topics {
		if t.Topic == "pri-test" {
			require.Equal(s.T(), pb.ScheduleMode_STRICT_PRIORITY, t.Mode)
		}
	}
}

func (s *TransportSuite) TestMultipleProduceConsume() {
	ctx := context.Background()

	s.createTopicWithScheduler("multi-test", 1, pb.ScheduleMode_FIFO)

	for i := 0; i < 10; i++ {
		_, _, err := s.client.Produce(ctx, "multi-test", []byte("k"), []byte("v"), 0)
		require.NoError(s.T(), err)
	}

	msgs, err := s.client.Consume(ctx, "multi-test", "g1", 0, 65536)
	require.NoError(s.T(), err)
	require.NotEmpty(s.T(), msgs)
}

func (s *TransportSuite) TestServerAddrBeforeStart() {
	srv := NewServer(s.broker)
	require.Empty(s.T(), srv.Addr())
}
