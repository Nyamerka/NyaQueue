package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ConfigSuite struct {
	suite.Suite
}

func TestConfigSuite(t *testing.T) { suite.Run(t, new(ConfigSuite)) }

func (s *ConfigSuite) TestDefaultConfig() {
	cfg := DefaultConfig()
	require.Equal(s.T(), 20*1024*1024, cfg.SegmentMaxBytes)
	require.Equal(s.T(), 100, cfg.SegmentMaxCount)
	require.Equal(s.T(), 1024, cfg.MaxConnections)
	require.Equal(s.T(), 100, cfg.BatchSize)
	require.Equal(s.T(), 4, cfg.NumIOGoroutines)
}

func (s *ConfigSuite) TestDefaultTopicConfig() {
	cfg := DefaultTopicConfig()
	require.Equal(s.T(), ModeFIFO, cfg.ScheduleMode)
	require.Equal(s.T(), 4, cfg.NumPartitions)
	require.Equal(s.T(), 1, cfg.PriorityLevels)
}

func (s *ConfigSuite) TestTunableParamRanges() {
	ranges := TunableParamRanges()
	require.Equal(s.T(), NumTunableParams, len(ranges))

	for _, r := range ranges {
		require.NotEmpty(s.T(), r.Name, "each param must have a name")
		require.Less(s.T(), r.Min, r.Max, "min < max for %s", r.Name)
	}
}
