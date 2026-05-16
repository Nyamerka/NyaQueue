package preprocessing

import (
	"cmp"
	"math/rand/v2"
	"slices"
)

type ParamRange struct {
	Name string
	Min  float64
	Max  float64
}

func GenerateConfigs(paramRanges []ParamRange, n int) [][]float64 {
	d := len(paramRanges)
	configs := make([][]float64, n)

	for i := range configs {
		configs[i] = make([]float64, d)
	}

	for j := 0; j < d; j++ {
		perm := latinHypercubeColumn(n)
		for i := 0; i < n; i++ {
			lo := paramRanges[j].Min
			hi := paramRanges[j].Max
			configs[i][j] = lo + perm[i]*(hi-lo)
		}
	}

	return configs
}

func latinHypercubeColumn(n int) []float64 {
	samples := make([]float64, n)
	for i := 0; i < n; i++ {
		samples[i] = (float64(i) + rand.Float64()) / float64(n)
	}
	rand.Shuffle(n, func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
	return samples
}

func GenerateConfigsSorted(paramRanges []ParamRange, n int) [][]float64 {
	configs := GenerateConfigs(paramRanges, n)
	slices.SortFunc(configs, func(a, b []float64) int {
		return cmp.Compare(a[0], b[0])
	})
	return configs
}
