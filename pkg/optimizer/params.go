package optimizer

import (
	"github.com/Nyamerka/NyaQueue/pkg/preprocessing"
)

type TunableParam struct {
	Name   string
	Min    float64
	Max    float64
	Weight float64 // Lasso-derived importance (0 = excluded from DDPG action space)
}

func DefaultTunableParams() []TunableParam {
	return []TunableParam{
		{"SegmentMaxBytes", 1 << 20, 64 << 20, 1.0},
		{"SegmentMaxCount", 10, 500, 1.0},
		{"FlushIntervalMs", 100, 10000, 1.0},
		{"BatchSize", 1, 1000, 1.0},
		{"BatchMemoryBytes", 1024, 1 << 20, 0.9},
		{"LingerMs", 0, 100, 1.0},
		{"NumIOGoroutines", 1, 32, 1.0},
		{"NumNetworkGoroutines", 1, 32, 1.0},
		{"CompressionType", 0, 3, 0.3},
		{"MaxMessageBytes", 1024, 10 << 20, 0.7},
	}
}

// CalibrateWeights runs Lasso regression on pilot run data to determine which
// parameters actually affect throughput. Params with zero Lasso weight are
// excluded from the DDPG action space, reducing dimensionality.
//
// configs: matrix of pilot configurations (each row = one run, len(row) == len(params))
// throughputs: observed throughput for each pilot config
// alpha: L1 regularisation strength (higher => more parameters zeroed out)
func CalibrateWeights(params []TunableParam, configs [][]float64, throughputs []float64, alpha float64) []TunableParam {
	if len(configs) == 0 || len(configs[0]) != len(params) {
		return params
	}

	indices, weights := preprocessing.SelectParameters(configs, throughputs, alpha)

	calibrated := make([]TunableParam, len(params))
	copy(calibrated, params)

	for i := range calibrated {
		calibrated[i].Weight = 0
	}
	for i, idx := range indices {
		if idx < len(calibrated) {
			calibrated[idx].Weight = weights[i]
		}
	}

	return calibrated
}

func ActiveParams(all []TunableParam) []TunableParam {
	var active []TunableParam
	for _, p := range all {
		if p.Weight > 0 {
			active = append(active, p)
		}
	}
	return active
}

func Normalize(val, min, max float64) float64 {
	if max <= min {
		return 0
	}
	n := (val - min) / (max - min)
	if n < 0 {
		return 0
	}
	if n > 1 {
		return 1
	}
	return n
}

func Denormalize(norm, min, max float64) float64 {
	return min + norm*(max-min)
}

func ClipAction(current, delta, min, max float64) float64 {
	result := current + delta
	if result < min {
		return min
	}
	if result > max {
		return max
	}
	return result
}
