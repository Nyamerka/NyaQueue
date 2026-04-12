package preprocessing

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

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
