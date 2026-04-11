package preprocessing

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// --- Sampler ---

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

	// Each value should be in [0, 1] and roughly uniformly distributed
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

// --- PCA ---

type PCASuite struct {
	suite.Suite
}

func TestPCASuite(t *testing.T) { suite.Run(t, new(PCASuite)) }

func (s *PCASuite) TestSelectRepresentative() {
	tests := []struct {
		name    string
		n       int
		k       int
		wantLen int
	}{
		{"reduce", 100, 10, 10},
		{"k_equals_n", 10, 10, 10},
		{"k_greater_than_n", 5, 10, 5},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			configs := make([][]float64, tc.n)
			for i := range configs {
				configs[i] = []float64{float64(i), float64(i * 2), float64(i * 3)}
			}

			result := SelectRepresentative(configs, tc.k)
			require.Len(s.T(), result, tc.wantLen)
		})
	}
}

func (s *PCASuite) TestSelectRepresentativePreservesData() {
	configs := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
	}

	result := SelectRepresentative(configs, 2)
	require.Len(s.T(), result, 2)

	for _, cfg := range result {
		require.Len(s.T(), cfg, 3)
	}
}

// --- Lasso ---

type LassoSuite struct {
	suite.Suite
}

func TestLassoSuite(t *testing.T) { suite.Run(t, new(LassoSuite)) }

func (s *LassoSuite) TestSelectParameters() {
	tests := []struct {
		name  string
		alpha float64
	}{
		{"low_alpha", 0.001},
		{"medium_alpha", 0.1},
		{"high_alpha", 1.0},
	}

	configs := make([][]float64, 100)
	throughputs := make([]float64, 100)
	for i := range configs {
		x := float64(i)
		configs[i] = []float64{x, x * 0.5, float64(i % 7)} // param0 strongly correlated
		throughputs[i] = x*2.0 + 10
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			indices, weights := SelectParameters(configs, throughputs, tc.alpha)

			if tc.alpha < 0.5 {
				require.Greater(s.T(), len(indices), 0, "at least one param selected")
				require.Equal(s.T(), len(indices), len(weights))
				for _, w := range weights {
					require.Greater(s.T(), w, 0.0)
					require.LessOrEqual(s.T(), w, 1.0)
				}
			}
		})
	}
}

func (s *LassoSuite) TestEmptyInput() {
	indices, weights := SelectParameters(nil, nil, 0.1)
	require.Nil(s.T(), indices)
	require.Nil(s.T(), weights)
}

func (s *LassoSuite) TestHighAlphaZerosAll() {
	configs := make([][]float64, 10)
	throughputs := make([]float64, 10)
	for i := range configs {
		configs[i] = []float64{float64(i)}
		throughputs[i] = 1.0
	}

	indices, _ := SelectParameters(configs, throughputs, 100.0)
	require.Empty(s.T(), indices, "very high alpha should zero out all coefficients")
}

func (s *LassoSuite) TestSoftThreshold() {
	tests := []struct {
		name           string
		x, lambda, exp float64
	}{
		{"positive_above", 1.0, 0.5, 0.5},
		{"positive_below", 0.3, 0.5, 0.0},
		{"negative_above", -1.0, 0.5, -0.5},
		{"negative_below", -0.3, 0.5, 0.0},
		{"zero", 0.0, 0.5, 0.0},
		{"exact_lambda", 0.5, 0.5, 0.0},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			require.InDelta(s.T(), tc.exp, softThreshold(tc.x, tc.lambda), 1e-9)
		})
	}
}

func (s *LassoSuite) TestSingleFeatureCorrelation() {
	n := 50
	configs := make([][]float64, n)
	throughputs := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		configs[i] = []float64{x, 1.0} // only first param matters
		throughputs[i] = x * 3.0
	}
	indices, weights := SelectParameters(configs, throughputs, 0.001)
	require.Contains(s.T(), indices, 0)
	require.Greater(s.T(), len(weights), 0)
}

// --- Sampler extra ---

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

// --- PCA extra ---

func (s *PCASuite) TestEmptyDimension() {
	configs := [][]float64{{}, {}, {}}
	result := SelectRepresentative(configs, 2)
	require.Len(s.T(), result, 2)
}

func (s *PCASuite) TestSingleDimension() {
	configs := make([][]float64, 20)
	for i := range configs {
		configs[i] = []float64{float64(i)}
	}
	result := SelectRepresentative(configs, 5)
	require.Len(s.T(), result, 5)
}
