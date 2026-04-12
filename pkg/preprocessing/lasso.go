package preprocessing

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// SelectParameters runs L1-regularized coordinate descent (Lasso) to identify
// parameters most correlated with throughput.
func SelectParameters(configs [][]float64, throughputs []float64, alpha float64) (indices []int, weights []float64) {
	n := len(configs)
	if n == 0 {
		return nil, nil
	}
	d := len(configs[0])
	fn := float64(n)

	flat := make([]float64, n*d)
	for i, row := range configs {
		copy(flat[i*d:], row)
	}
	X := mat.NewDense(n, d, flat)

	colBuf := make([]float64, n)
	for j := 0; j < d; j++ {
		mat.Col(colBuf, j, X)
		m := stat.Mean(colBuf, nil)
		sd := stat.StdDev(colBuf, nil)
		if sd < 1e-10 {
			sd = 1
		}
		for i := 0; i < n; i++ {
			X.Set(i, j, (X.At(i, j)-m)/sd)
		}
	}

	yMean := stat.Mean(throughputs, nil)
	Y := make([]float64, n)
	floats.SubTo(Y, throughputs, make([]float64, n))
	for i := range Y {
		Y[i] = throughputs[i] - yMean
	}

	cols := make([][]float64, d)
	for j := 0; j < d; j++ {
		cols[j] = make([]float64, n)
		mat.Col(cols[j], j, X)
	}

	beta := make([]float64, d)
	residual := make([]float64, n)
	copy(residual, Y)

	const maxIter = 1000
	const tol = 1e-6

	for iter := 0; iter < maxIter; iter++ {
		maxChange := 0.0

		for j := 0; j < d; j++ {
			col := cols[j]
			floats.AddScaled(residual, beta[j], col)

			rho := floats.Dot(col, residual) / fn

			oldBeta := beta[j]
			beta[j] = softThreshold(rho, alpha)

			floats.AddScaled(residual, -beta[j], col)

			if change := math.Abs(beta[j] - oldBeta); change > maxChange {
				maxChange = change
			}
		}

		if maxChange < tol {
			break
		}
	}

	for j := 0; j < d; j++ {
		if math.Abs(beta[j]) > 1e-8 {
			indices = append(indices, j)
			weights = append(weights, math.Abs(beta[j]))
		}
	}

	if len(weights) > 0 {
		maxW := floats.Max(weights)
		if maxW > 0 {
			floats.Scale(1.0/maxW, weights)
		}
	}

	return indices, weights
}

func softThreshold(x, lambda float64) float64 {
	if x > lambda {
		return x - lambda
	}
	if x < -lambda {
		return x + lambda
	}
	return 0
}
