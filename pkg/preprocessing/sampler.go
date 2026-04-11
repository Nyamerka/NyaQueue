package preprocessing

import (
	"math/rand"
	"sort"
)

// ParamRange defines the bounds for a single tunable parameter.
type ParamRange struct {
	Name string
	Min  float64
	Max  float64
}

// GenerateConfigs produces n configurations via Latin Hypercube Sampling,
// ensuring uniform coverage of the parameter space.
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
			// Map the i-th sample in [0,1] to the parameter range
			configs[i][j] = lo + perm[i]*(hi-lo)
		}
	}

	return configs
}

// latinHypercubeColumn generates n stratified samples in [0,1] for one dimension.
func latinHypercubeColumn(n int) []float64 {
	samples := make([]float64, n)
	for i := 0; i < n; i++ {
		// Stratified: sample within [i/n, (i+1)/n]
		samples[i] = (float64(i) + rand.Float64()) / float64(n)
	}
	// Shuffle to break correlation between dimensions
	rand.Shuffle(n, func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
	return samples
}

// GenerateConfigsSorted returns configs sorted by the first parameter for reproducibility.
func GenerateConfigsSorted(paramRanges []ParamRange, n int) [][]float64 {
	configs := GenerateConfigs(paramRanges, n)
	sort.Slice(configs, func(i, j int) bool {
		return configs[i][0] < configs[j][0]
	})
	return configs
}
