package optimizer

import (
	"github.com/Nyamerka/NyaQueue/pkg/preprocessing"
)

type TunableParam struct {
	Name   string
	Min    float64
	Max    float64
	Weight float64 // Lasso-derived importance (0 = excluded from TD3 action space)
}

func DefaultTunableParams() []TunableParam {
	return []TunableParam{
		{"NumIOGoroutines", 1, 32, 1.0},
		{"CompressionType", 0, 3, 0.3},
		{"MaxMessageBytes", 1024, 10 << 20, 0.7},
		{"FetchMaxWaitMs", 10, 5000, 0.6},
		{"FetchMinBytes", 1, 1 << 20, 0.8},
		{"MaxFetchBytes", 1024, 10 << 20, 0.8},
	}
}

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
