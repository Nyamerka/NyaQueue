package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type MetricsSuite struct {
	suite.Suite
}

func TestMetricsSuite(t *testing.T) { suite.Run(t, new(MetricsSuite)) }

func (s *MetricsSuite) TestRecordAndCollect() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store.Close()

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	require.NoError(s.T(), b.CreateTopic("t", DefaultTopicConfig()))

	mc := b.metrics
	mc.RecordProduce("t", 0)
	mc.RecordProduce("t", 1)
	mc.RecordConsume("t", 0)

	snap := mc.Collect()
	require.Greater(s.T(), snap.Throughput, 0.0)
	require.NotZero(s.T(), snap.Timestamp)
}

func (s *MetricsSuite) TestSnapshotEmpty() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store.Close()

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	snap := b.metrics.Snapshot()
	require.Equal(s.T(), 0.0, snap.Throughput)
}

type noopBalancer struct{}

func (noopBalancer) SelectPartition(string, []byte, int) int { return 0 }
func (noopBalancer) OnMetrics(Metrics)                       {}
