package broker

import (
	"testing"

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
