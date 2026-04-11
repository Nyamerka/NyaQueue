package benchmarks

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ScenariosSuite struct {
	suite.Suite
}

func TestScenariosSuite(t *testing.T) {
	suite.Run(t, new(ScenariosSuite))
}

func (s *ScenariosSuite) TestAllScenariosNotEmpty() {
	all := AllScenarios()
	require.GreaterOrEqual(s.T(), len(all), 5)
}

func (s *ScenariosSuite) TestScenarioNames() {
	expected := []string{"uniform", "skewed", "bursty", "growing", "mixed_priority"}
	all := AllScenarios()
	names := make([]string, len(all))
	for i, sc := range all {
		names[i] = sc.Name
	}
	require.Equal(s.T(), expected, names)
}

func (s *ScenariosSuite) TestUniformPriorities() {
	sc := Uniform()
	total := 0.0
	for _, p := range sc.Priorities {
		require.InDelta(s.T(), 0.1, p, 0.001)
		total += p
	}
	require.InDelta(s.T(), 1.0, total, 0.01)
}

func (s *ScenariosSuite) TestMixedPriorityDistribution() {
	sc := MixedPriority()
	total := 0.0
	for _, p := range sc.Priorities {
		total += p
	}
	require.InDelta(s.T(), 1.0, total, 0.01)
	require.Greater(s.T(), sc.Priorities[0], sc.Priorities[9])
}

func (s *ScenariosSuite) TestGenerateMessageSize() {
	tests := []struct {
		name string
		size int
	}{
		{"zero", 0},
		{"small", 64},
		{"typical", 256},
		{"large", 4096},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			msg := GenerateMessage(tc.size)
			require.Len(s.T(), msg, tc.size)
		})
	}
}

func (s *ScenariosSuite) TestSamplePriorityRange() {
	sc := MixedPriority()
	for i := 0; i < 100; i++ {
		p := sc.SamplePriority()
		require.LessOrEqual(s.T(), p, uint8(9))
	}
}

func (s *ScenariosSuite) TestSkewedHasNonZeroSkew() {
	sc := Skewed()
	require.Greater(s.T(), sc.SkewRatio, 0.0)
}

func (s *ScenariosSuite) TestBurstyHasBurstConfig() {
	sc := Bursty()
	require.Greater(s.T(), sc.BurstEvery.Nanoseconds(), int64(0))
	require.Greater(s.T(), sc.BurstFactor, 1)
}
