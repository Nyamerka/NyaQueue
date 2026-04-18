package main

import (
	"context"
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

type ExporterSuite struct {
	suite.Suite
	server *transport.Server
	client *transport.Client
	broker *broker.Broker
	dir    string
}

func TestExporterSuite(t *testing.T) { suite.Run(t, new(ExporterSuite)) }

func (s *ExporterSuite) SetupTest() {
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

func (s *ExporterSuite) TearDownTest() {
	s.client.Close()
	s.server.Stop()
	s.broker.Stop()
}

func (s *ExporterSuite) TestScrapeUpdatesGauges() {
	ctx := context.Background()
	require.NoError(s.T(), s.client.CreateTopic(ctx, "scrape", 4, pb.ScheduleMode_FIFO))
	s.broker.SetScheduler("scrape", scheduler.NewFIFO())

	for i := 0; i < 50; i++ {
		_, _, err := s.client.Produce(ctx, "scrape", []byte("k"), make([]byte, 16), 0)
		require.NoError(s.T(), err)
	}

	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	require.Eventually(s.T(), func() bool {
		if err := scrape(ctx, s.client, m); err != nil {
			return false
		}
		return testutil.CollectAndCount(m.partitionLoad) >= 1 &&
			testutil.CollectAndCount(m.queueDepth) >= 1
	}, 2*time.Second, 50*time.Millisecond)

	require.Equal(s.T(), 1.0, testutil.ToFloat64(m.lastScrapeSuccess))
}

func (s *ExporterSuite) TestRunUntilContextDone() {
	cfg := DefaultConfig()
	cfg.BrokerAddr = s.server.Addr()
	cfg.ScrapeInterval = 100 * time.Millisecond

	reg := prometheus.NewRegistry()
	m := registerMetrics(reg)

	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()
	require.NoError(s.T(), Run(ctx, cfg, s.client, m))

	require.Equal(s.T(), 1.0, testutil.ToFloat64(m.lastScrapeSuccess))
}
