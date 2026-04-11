package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type TopicSuite struct {
	suite.Suite
}

func TestTopicSuite(t *testing.T) { suite.Run(t, new(TopicSuite)) }

func (s *TopicSuite) TestNewTopic() {
	tests := []struct {
		name       string
		partitions int
		mode       ScheduleMode
	}{
		{"fifo_4", 4, ModeFIFO},
		{"priority_2", 2, ModeStrictPriority},
		{"default_partitions", 0, ModeFIFO}, // should default to 4
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			cfg := DefaultTopicConfig()
			cfg.NumPartitions = tc.partitions
			cfg.ScheduleMode = tc.mode

			t, err := NewTopic("test", s.T().TempDir(), cfg)
			require.NoError(s.T(), err)
			defer t.Close()

			expected := tc.partitions
			if expected <= 0 {
				expected = 4
			}
			require.Equal(s.T(), expected, t.NumPartitions())
			require.Equal(s.T(), "test", t.Name())
		})
	}
}

func (s *TopicSuite) TestPartitionOutOfRange() {
	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 2
	t, err := NewTopic("test", s.T().TempDir(), cfg)
	require.NoError(s.T(), err)
	defer t.Close()

	_, err = t.Partition(-1)
	require.Error(s.T(), err)

	_, err = t.Partition(5)
	require.Error(s.T(), err)

	p, err := t.Partition(1)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), p)
}

func (s *TopicSuite) TestPartitions() {
	cfg := DefaultTopicConfig()
	cfg.NumPartitions = 3
	t, err := NewTopic("test", s.T().TempDir(), cfg)
	require.NoError(s.T(), err)
	defer t.Close()

	parts := t.Partitions()
	require.Len(s.T(), parts, 3)
}
