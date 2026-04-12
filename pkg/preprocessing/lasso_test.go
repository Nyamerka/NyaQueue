package preprocessing

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

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
		configs[i] = []float64{x, x * 0.5, float64(i % 7)}
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
		configs[i] = []float64{x, 1.0}
		throughputs[i] = x * 3.0
	}
	indices, weights := SelectParameters(configs, throughputs, 0.001)
	require.Contains(s.T(), indices, 0)
	require.Greater(s.T(), len(weights), 0)
}
