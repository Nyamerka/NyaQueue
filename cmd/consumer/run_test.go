package main

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

type ConsumerSuite struct {
	suite.Suite
	server *transport.Server
	client *transport.Client
	broker *broker.Broker
	dir    string
}

func TestConsumerSuite(t *testing.T) { suite.Run(t, new(ConsumerSuite)) }

func (s *ConsumerSuite) SetupTest() {
	s.dir = s.T().TempDir()

	offsetStore, err := broker.NewOffsetStore(s.dir)
	require.NoError(s.T(), err)

	s.broker = broker.New(broker.DefaultConfig(), s.dir, balancer.NewRoundRobin(), offsetStore)
	s.broker.Start()

	s.server = transport.NewServer(s.broker)
	require.NoError(s.T(), s.server.Start(":0"))

	s.client, err = transport.NewClient(s.server.Addr())
	require.NoError(s.T(), err)
}

func (s *ConsumerSuite) TearDownTest() {
	s.client.Close()
	s.server.Stop()
	s.broker.Stop()
}

func (s *ConsumerSuite) publishWithTimestamp(topic string, n int) {
	ctx := context.Background()
	require.NoError(s.T(), s.client.CreateTopic(ctx, topic, 2, pb.ScheduleMode_FIFO))
	s.broker.SetScheduler(topic, scheduler.NewFIFO())

	for i := 0; i < n; i++ {
		value := make([]byte, 32)
		binary.BigEndian.PutUint64(value[:8], uint64(time.Now().UnixNano()))
		_, _, err := s.client.Produce(ctx, topic, []byte("k"), value, 0)
		require.NoError(s.T(), err)
	}
}

func (s *ConsumerSuite) TestConsumesAndRecordsLatency() {
	topic := "consumer-test"
	const n = 20
	s.publishWithTimestamp(topic, n)

	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	cfg := DefaultConfig()
	cfg.Addr = s.server.Addr()
	cfg.Topic = topic
	cfg.Group = "g1"
	cfg.Partitions = []int{0, 1}
	cfg.Workers = 2

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	require.NoError(s.T(), Run(ctx, cfg, s.client, m))

	consumedCount := testutil.CollectAndCount(m.consumed)
	require.Greater(s.T(), consumedCount, 0)

	latencyCount := testutil.CollectAndCount(m.e2eLatency)
	require.Greater(s.T(), latencyCount, 0)
}

func (s *ConsumerSuite) TestMalformedPayloadCountsAsParseError() {
	topic := "bad-payload"
	ctx := context.Background()
	require.NoError(s.T(), s.client.CreateTopic(ctx, topic, 1, pb.ScheduleMode_FIFO))
	s.broker.SetScheduler(topic, scheduler.NewFIFO())

	_, _, err := s.client.Produce(ctx, topic, []byte("k"), []byte("xx"), 0)
	require.NoError(s.T(), err)

	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	cfg := DefaultConfig()
	cfg.Addr = s.server.Addr()
	cfg.Topic = topic
	cfg.Group = "g1"
	cfg.Partitions = []int{0}
	cfg.Workers = 1

	runCtx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	require.NoError(s.T(), Run(runCtx, cfg, s.client, m))

	parseErrors := testutil.ToFloat64(m.parseErrors.WithLabelValues(topic))
	require.GreaterOrEqual(s.T(), parseErrors, 1.0)
}

func (s *ConsumerSuite) TestResolvePartitionsAutoDetect() {
	topic := "autodetect"
	ctx := context.Background()
	require.NoError(s.T(), s.client.CreateTopic(ctx, topic, 3, pb.ScheduleMode_FIFO))
	s.broker.SetScheduler(topic, scheduler.NewFIFO())

	cfg := DefaultConfig()
	cfg.Topic = topic

	partitions, err := resolvePartitions(ctx, s.client, cfg)
	require.NoError(s.T(), err)
	require.Equal(s.T(), []int{0, 1, 2}, partitions)
}

func (s *ConsumerSuite) TestResolvePartitionsCancelsWhenTopicAbsent() {
	cfg := DefaultConfig()
	cfg.Topic = "never-created"

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := resolvePartitions(ctx, s.client, cfg)
	require.Error(s.T(), err)
}
