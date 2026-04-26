package broker

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OffsetStoreSuite struct {
	suite.Suite
}

func TestOffsetStoreSuite(t *testing.T) { suite.Run(t, new(OffsetStoreSuite)) }

func (s *OffsetStoreSuite) TestCommitAndLoad() {
	tests := []struct {
		name      string
		group     string
		topic     string
		partition int
		offset    int64
	}{
		{"basic", "grp1", "topic-a", 0, 42},
		{"large_offset", "grp2", "topic-b", 3, 1_000_000},
		{"zero_offset", "grp3", "topic-c", 0, 0},
	}

	store, err := NewOffsetStore(s.T().TempDir())
	require.NoError(s.T(), err)
	defer store.Close()

	for _, tc := range tests {
		s.Run(tc.name, func() {
			err := store.Commit(tc.group, tc.topic, tc.partition, tc.offset)
			require.NoError(s.T(), err)

			got, err := store.Load(tc.group, tc.topic, tc.partition)
			require.NoError(s.T(), err)
			require.Equal(s.T(), tc.offset, got)
		})
	}
}

func (s *OffsetStoreSuite) TestLoadNotFound() {
	store, err := NewOffsetStore(s.T().TempDir())
	require.NoError(s.T(), err)
	defer store.Close()

	_, err = store.Load("nonexistent", "topic", 0)
	require.Error(s.T(), err)
}

func (s *OffsetStoreSuite) TestCommitFloor() {
	store, err := NewOffsetStore(s.T().TempDir())
	require.NoError(s.T(), err)
	defer store.Close()

	require.NoError(s.T(), store.Commit("g1", "t", 0, 100))
	require.NoError(s.T(), store.Commit("g2", "t", 0, 50))
	require.NoError(s.T(), store.Commit("g3", "t", 0, 200))

	floor, err := store.CommitFloor("t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(50), floor)
}

func (s *OffsetStoreSuite) TestCommitFloorEmpty() {
	store, err := NewOffsetStore(s.T().TempDir())
	require.NoError(s.T(), err)
	defer store.Close()

	floor, err := store.CommitFloor("nonexistent", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(0), floor)
}

func (s *OffsetStoreSuite) TestDumpMode() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir, 50*time.Millisecond)
	require.NoError(s.T(), err)

	require.NoError(s.T(), store.Commit("g1", "t", 0, 100))
	require.NoError(s.T(), store.Commit("g2", "t", 0, 200))

	got, err := store.Load("g1", "t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(100), got)

	floor, err := store.CommitFloor("t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(100), floor)

	store.Close()

	store2, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store2.Close()

	got, err = store2.Load("g1", "t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(100), got)

	got, err = store2.Load("g2", "t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(200), got)
}

func (s *OffsetStoreSuite) TestDeleteTopic() {
	store, err := NewOffsetStore(s.T().TempDir())
	require.NoError(s.T(), err)
	defer store.Close()

	require.NoError(s.T(), store.Commit("g1", "t1", 0, 10))
	require.NoError(s.T(), store.Commit("g1", "t1", 1, 20))
	require.NoError(s.T(), store.Commit("g2", "t1", 0, 30))
	require.NoError(s.T(), store.Commit("g1", "t2", 0, 99))

	store.DeleteTopic("t1")

	_, err = store.Load("g1", "t1", 0)
	require.Error(s.T(), err, "t1 offsets should be gone")
	_, err = store.Load("g1", "t1", 1)
	require.Error(s.T(), err)
	_, err = store.Load("g2", "t1", 0)
	require.Error(s.T(), err)

	got, err := store.Load("g1", "t2", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(99), got, "t2 offsets should be untouched")
}

func (s *OffsetStoreSuite) TestDeleteTopicPersistence() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	require.NoError(s.T(), store.Commit("g1", "t1", 0, 42))
	store.DeleteTopic("t1")
	store.Close()

	store2, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store2.Close()

	_, err = store2.Load("g1", "t1", 0)
	require.Error(s.T(), err, "deleted topic offsets should not survive restart")
}

func (s *OffsetStoreSuite) TestCacheWarmup() {
	dir := s.T().TempDir()

	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	require.NoError(s.T(), store.Commit("g1", "t", 0, 42))
	store.Close()

	store2, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)
	defer store2.Close()

	got, err := store2.Load("g1", "t", 0)
	require.NoError(s.T(), err)
	require.Equal(s.T(), int64(42), got)
}
