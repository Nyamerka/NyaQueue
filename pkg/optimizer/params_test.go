package optimizer

import (
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
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

func (s *ParamsSuite) TestCalibrateWeightsUpdatesFromLasso() {
	params := []TunableParam{
		{"A", 0, 100, 1.0},
		{"B", 0, 100, 1.0},
		{"C", 0, 100, 1.0},
	}

	n := 100
	configs := make([][]float64, n)
	throughputs := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		configs[i] = []float64{x, 1.0, float64(i % 3)}
		throughputs[i] = x*3.0 + 10
	}

	calibrated := CalibrateWeights(params, configs, throughputs, 0.001)
	require.Len(s.T(), calibrated, 3)

	hasNonZero := false
	for _, p := range calibrated {
		if p.Weight > 0 {
			hasNonZero = true
		}
	}
	require.True(s.T(), hasNonZero, "at least one parameter should have nonzero weight")

	require.Greater(s.T(), calibrated[0].Weight, calibrated[1].Weight,
		"param A (correlated with throughput) should have higher weight than B (constant)")
}

func (s *ParamsSuite) TestCalibrateWeightsHighAlphaZeros() {
	params := DefaultTunableParams()

	n := 50
	configs := make([][]float64, n)
	throughputs := make([]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, len(params))
		for j := range row {
			row[j] = float64(i + j)
		}
		configs[i] = row
		throughputs[i] = 100.0
	}

	calibrated := CalibrateWeights(params, configs, throughputs, 100.0)
	active := ActiveParams(calibrated)
	require.Empty(s.T(), active, "very high alpha should zero all params")
}

func (s *ParamsSuite) TestCalibrateWeightsEmptyConfigs() {
	params := DefaultTunableParams()
	calibrated := CalibrateWeights(params, nil, nil, 0.1)
	require.Equal(s.T(), params, calibrated, "empty configs should return original params")
}

func (s *ParamsSuite) TestNewOptimizerWithPilotData() {
	params := []TunableParam{
		{"A", 0, 100, 1.0},
		{"B", 0, 100, 1.0},
	}

	configs := make([][]float64, 50)
	throughputs := make([]float64, 50)
	for i := 0; i < 50; i++ {
		configs[i] = []float64{float64(i), 1.0}
		throughputs[i] = float64(i) * 2
	}

	pilot := PilotData{
		Configs:     configs,
		Throughputs: throughputs,
		Alpha:       0.5,
	}

	opt := NewOptimizer(nil, params, DefaultOptimizerConfig(), pilot)
	require.NotNil(s.T(), opt)
	require.Greater(s.T(), len(opt.params), 0)
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

func (s *ParamsSuite) TestOptimizerAppliesConfig() {
	dir := s.T().TempDir()
	cfg := broker.DefaultConfig()
	b := broker.New(cfg, dir, noopBal{}, nil)
	b.Start()
	defer b.Stop()

	params := []TunableParam{
		{"BatchSize", 1, 1000, 1.0},
	}

	opt := NewOptimizer(b, params, OptimizerConfig{Interval: 100 * time.Millisecond, BatchSize: 32, WarmupTicks: 2, Hysteresis: 0.001})
	opt.Start()
	defer opt.Stop()

	time.Sleep(3 * time.Second)

	newCfg := b.Config()
	require.NotEqual(s.T(), cfg.BatchSize, newCfg.BatchSize,
		"optimizer should have changed BatchSize via ApplyConfig")
}

type noopBal struct{}

func (noopBal) SelectPartition(string, []byte, int) int { return 0 }
func (noopBal) OnMetrics(broker.Metrics)                {}
