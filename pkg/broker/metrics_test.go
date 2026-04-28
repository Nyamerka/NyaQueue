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

func (s *MetricsSuite) TestDeltaBasedPartitionLoads() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store.Close()

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 2
	require.NoError(s.T(), b.CreateTopic("t", cfg))

	mc := b.metrics

	mc.RecordProduceBatch("t", 0, 100)
	mc.RecordProduceBatch("t", 1, 100)
	mc.RecordConsume("t", 0)
	mc.RecordConsume("t", 1)

	snap1 := mc.Collect()
	require.Len(s.T(), snap1.PartitionLoads, 2)

	mc.RecordProduceBatch("t", 0, 50)
	mc.RecordProduceBatch("t", 1, 10)
	mc.RecordConsume("t", 0)
	mc.RecordConsume("t", 1)

	snap2 := mc.Collect()
	require.Len(s.T(), snap2.PartitionLoads, 2)
	require.Greater(s.T(), snap2.PartitionLoads[0], 0.0,
		"partition 0 produced 50, consumed 1 → load should be >0")
	require.Greater(s.T(), snap2.PartitionLoads[1], 0.0,
		"partition 1 produced 10, consumed 1 → load should be >0")
}

func (s *MetricsSuite) TestDeltaLoadsReflectRecentActivity() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store.Close()

	b := New(DefaultConfig(), dir, noopBalancer{}, store)
	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 2
	require.NoError(s.T(), b.CreateTopic("t", cfg))

	mc := b.metrics

	for i := 0; i < 1000; i++ {
		mc.RecordProduce("t", 0)
		mc.RecordProduce("t", 1)
		mc.RecordConsume("t", 0)
		mc.RecordConsume("t", 1)
	}
	_ = mc.Collect()

	for i := 0; i < 100; i++ {
		mc.RecordProduce("t", 0)
		mc.RecordConsume("t", 0)
		mc.RecordConsume("t", 1)
	}
	snap := mc.Collect()

	require.Less(s.T(), snap.PartitionLoads[0], 0.5,
		"partition 0: produced and consumed equally in this interval")
	require.InDelta(s.T(), 0.0, snap.PartitionLoads[1], 0.001,
		"partition 1: no new produces, load should be ~0")
}

type noopBalancer struct{}

func (noopBalancer) SelectPartition(string, []byte, int) int { return 0 }
func (noopBalancer) OnMetrics(Metrics)                       {}
