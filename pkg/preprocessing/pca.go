package preprocessing

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// SelectRepresentative reduces N configs to K via PCA + grid-based selection.
func SelectRepresentative(configs [][]float64, k int) [][]float64 {
	n := len(configs)
	if n <= k {
		return configs
	}
	if len(configs[0]) == 0 {
		return configs[:k]
	}

	d := len(configs[0])

	data := mat.NewDense(n, d, nil)
	means := make([]float64, d)
	for j := 0; j < d; j++ {
		for i := 0; i < n; i++ {
			means[j] += configs[i][j]
		}
		means[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			data.Set(i, j, configs[i][j]-means[j])
		}
	}

	var svd mat.SVD
	if !svd.Factorize(data, mat.SVDThin) {
		return configs[:k]
	}

	numPC := 2
	if d < numPC {
		numPC = d
	}

	var vt mat.Dense
	svd.VTo(&vt)

	pcBasis := mat.NewDense(numPC, d, nil)
	for i := 0; i < numPC; i++ {
		for j := 0; j < d; j++ {
			pcBasis.Set(i, j, vt.At(i, j))
		}
	}

	projected := mat.NewDense(n, numPC, nil)
	projected.Mul(data, pcBasis.T())

	selected := gridSelect(projected, n, numPC, k)

	result := make([][]float64, len(selected))
	for i, idx := range selected {
		result[i] = configs[idx]
	}
	return result
}

func gridSelect(projected *mat.Dense, n, numPC, k int) []int {
	type scored struct {
		idx  int
		dist float64
	}

	mins := make([]float64, numPC)
	maxs := make([]float64, numPC)
	for j := 0; j < numPC; j++ {
		mins[j] = math.Inf(1)
		maxs[j] = math.Inf(-1)
		for i := 0; i < n; i++ {
			v := projected.At(i, j)
			if v < mins[j] {
				mins[j] = v
			}
			if v > maxs[j] {
				maxs[j] = v
			}
		}
	}

	gridSize := int(math.Ceil(math.Sqrt(float64(k))))
	targets := make([][]float64, 0, gridSize*gridSize)
	for i := 0; i < gridSize; i++ {
		for j := 0; j < gridSize; j++ {
			t := make([]float64, numPC)
			if numPC > 0 {
				t[0] = mins[0] + (maxs[0]-mins[0])*float64(i)/float64(gridSize-1+1)
			}
			if numPC > 1 {
				t[1] = mins[1] + (maxs[1]-mins[1])*float64(j)/float64(gridSize-1+1)
			}
			targets = append(targets, t)
		}
	}

	used := make(map[int]bool)
	selected := make([]int, 0, k)

	for _, target := range targets {
		if len(selected) >= k {
			break
		}

		bestIdx := -1
		bestDist := math.Inf(1)

		for i := 0; i < n; i++ {
			if used[i] {
				continue
			}
			dist := 0.0
			for j := 0; j < numPC; j++ {
				d := projected.At(i, j) - target[j]
				dist += d * d
			}
			if dist < bestDist {
				bestDist = dist
				bestIdx = i
			}
		}

		if bestIdx >= 0 {
			selected = append(selected, bestIdx)
			used[bestIdx] = true
		}
	}

	sort.Ints(selected)
	return selected
}
