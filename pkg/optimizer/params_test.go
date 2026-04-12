package optimizer

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ParamsSuite struct {
	suite.Suite
}

func TestParamsSuite(t *testing.T) { suite.Run(t, new(ParamsSuite)) }

func (s *ParamsSuite) TestNormalize() {
	tests := []struct {
		name          string
		val, min, max float64
		want          float64
	}{
		{"midpoint", 50, 0, 100, 0.5},
		{"min", 0, 0, 100, 0.0},
		{"max", 100, 0, 100, 1.0},
		{"below_min", -10, 0, 100, 0.0},
		{"above_max", 200, 0, 100, 1.0},
		{"equal_range", 5, 5, 5, 0.0},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			got := Normalize(tc.val, tc.min, tc.max)
			require.InDelta(s.T(), tc.want, got, 1e-9)
		})
	}
}

func (s *ParamsSuite) TestDenormalize() {
	tests := []struct {
		name           string
		norm, min, max float64
		want           float64
	}{
		{"midpoint", 0.5, 0, 100, 50},
		{"zero", 0.0, 10, 20, 10},
		{"one", 1.0, 10, 20, 20},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			got := Denormalize(tc.norm, tc.min, tc.max)
			require.InDelta(s.T(), tc.want, got, 1e-9)
		})
	}
}

func (s *ParamsSuite) TestClipAction() {
	tests := []struct {
		name                     string
		current, delta, min, max float64
		want                     float64
	}{
		{"within_range", 50, 10, 0, 100, 60},
		{"clip_high", 90, 20, 0, 100, 100},
		{"clip_low", 10, -20, 0, 100, 0},
		{"negative_delta", 50, -30, 0, 100, 20},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			got := ClipAction(tc.current, tc.delta, tc.min, tc.max)
			require.InDelta(s.T(), tc.want, got, 1e-9)
		})
	}
}

func (s *ParamsSuite) TestActiveParams() {
	all := []TunableParam{
		{"a", 0, 1, 1.0},
		{"b", 0, 1, 0.0},
		{"c", 0, 1, 0.5},
	}

	active := ActiveParams(all)
	require.Len(s.T(), active, 2)
	require.Equal(s.T(), "a", active[0].Name)
	require.Equal(s.T(), "c", active[1].Name)
}

func (s *ParamsSuite) TestDefaultTunableParams() {
	params := DefaultTunableParams()
	require.Equal(s.T(), 22, len(params))

	for _, p := range params {
		require.Less(s.T(), p.Min, p.Max)
		require.GreaterOrEqual(s.T(), p.Weight, 0.0)
	}
}

func (s *ParamsSuite) TestNormalizeDenormalizeRoundTrip() {
	tests := []struct {
		name          string
		val, min, max float64
	}{
		{"typical", 50, 0, 100},
		{"float", 3.14, 0, 10},
		{"negative_range", -5, -10, 10},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			n := Normalize(tc.val, tc.min, tc.max)
			d := Denormalize(n, tc.min, tc.max)
			require.InDelta(s.T(), tc.val, d, 1e-9)
		})
	}
}
