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

type ProducerSuite struct {
	suite.Suite
	server *transport.Server
	client *transport.Client
	broker *broker.Broker
	dir    string
}

func TestProducerSuite(t *testing.T) { suite.Run(t, new(ProducerSuite)) }

func (s *ProducerSuite) SetupTest() {
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

func (s *ProducerSuite) TearDownTest() {
	s.client.Close()
	s.server.Stop()
	s.broker.Stop()
}

func (s *ProducerSuite) TestRunPublishesMessages() {
	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	cfg := DefaultConfig()
	cfg.Addr = s.server.Addr()
	cfg.Topic = "producer-test"
	cfg.Partitions = 2
	cfg.Producers = 2
	cfg.Duration = 300 * time.Millisecond

	s.broker.SetScheduler(cfg.Topic, scheduler.NewFIFO())

	require.NoError(s.T(), Run(context.Background(), cfg, s.client, m))

	total := testutil.CollectAndCount(m.published)
	require.Greater(s.T(), total, 0)

	published := testutil.ToFloat64(m.published.WithLabelValues(cfg.Topic, "0"))
	require.Greater(s.T(), published, 0.0)
}

func (s *ProducerSuite) TestTopicAlreadyExistsIsIgnored() {
	ctx := context.Background()
	require.NoError(s.T(), s.client.CreateTopic(ctx, "exists", 2, pb.ScheduleMode_FIFO))
	s.broker.SetScheduler("exists", scheduler.NewFIFO())

	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	cfg := DefaultConfig()
	cfg.Addr = s.server.Addr()
	cfg.Topic = "exists"
	cfg.Partitions = 2
	cfg.Producers = 1
	cfg.Duration = 150 * time.Millisecond

	require.NoError(s.T(), Run(ctx, cfg, s.client, m))
}

func (s *ProducerSuite) TestEncodeValueTimestampPrefix() {
	before := time.Now().UnixNano()
	buf := encodeValue(64)
	after := time.Now().UnixNano()

	require.Len(s.T(), buf, 64)
	ts := int64(binary.BigEndian.Uint64(buf[:timestampPrefixBytes]))
	require.GreaterOrEqual(s.T(), ts, before)
	require.LessOrEqual(s.T(), ts, after)
}

func (s *ProducerSuite) TestEncodeValuePadsSmallSizes() {
	buf := encodeValue(3)
	require.Len(s.T(), buf, timestampPrefixBytes)
}
