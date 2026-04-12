package preprocessing

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type SamplerSuite struct {
	suite.Suite
}

func TestSamplerSuite(t *testing.T) { suite.Run(t, new(SamplerSuite)) }

func (s *SamplerSuite) TestGenerateConfigs() {
	tests := []struct {
		name   string
		params int
		n      int
	}{
		{"small", 3, 10},
		{"medium", 10, 100},
		{"single_param", 1, 50},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			ranges := make([]ParamRange, tc.params)
			for i := range ranges {
				ranges[i] = ParamRange{Name: "p", Min: 0, Max: 100}
			}

			configs := GenerateConfigs(ranges, tc.n)
			require.Len(s.T(), configs, tc.n)

			for _, cfg := range configs {
				require.Len(s.T(), cfg, tc.params)
				for j, v := range cfg {
					require.GreaterOrEqual(s.T(), v, ranges[j].Min)
					require.LessOrEqual(s.T(), v, ranges[j].Max)
				}
			}
		})
	}
}

func (s *SamplerSuite) TestStratification() {
	ranges := []ParamRange{{Name: "x", Min: 0, Max: 1}}
	configs := GenerateConfigs(ranges, 100)

	buckets := make([]int, 10)
	for _, cfg := range configs {
		bucket := int(cfg[0] * 10)
		if bucket >= 10 {
			bucket = 9
		}
		buckets[bucket]++
	}

	for _, count := range buckets {
		require.Greater(s.T(), count, 0, "LHS should cover all strata")
	}
}

func (s *SamplerSuite) TestLatinHypercubeColumn() {
	col := latinHypercubeColumn(100)
	require.Len(s.T(), col, 100)
	for _, v := range col {
		require.GreaterOrEqual(s.T(), v, 0.0)
		require.Less(s.T(), v, 1.0)
	}
}

func (s *SamplerSuite) TestGenerateConfigsSorted() {
	ranges := []ParamRange{
		{Name: "a", Min: 0, Max: 100},
		{Name: "b", Min: 0, Max: 50},
	}
	configs := GenerateConfigsSorted(ranges, 20)
	require.Len(s.T(), configs, 20)
	for i := 1; i < len(configs); i++ {
		require.LessOrEqual(s.T(), configs[i-1][0], configs[i][0],
			"sorted by first parameter")
	}
}
