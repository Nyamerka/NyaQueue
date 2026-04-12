package preprocessing

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// SelectParameters runs L1-regularized coordinate descent (Lasso) to identify
// parameters most correlated with throughput.
func SelectParameters(configs [][]float64, throughputs []float64, alpha float64) (indices []int, weights []float64) {
	n := len(configs)
	if n == 0 {
		return nil, nil
	}
	d := len(configs[0])

	means := make([]float64, d)
	stds := make([]float64, d)
	for j := 0; j < d; j++ {
		for i := 0; i < n; i++ {
			means[j] += configs[i][j]
		}
		means[j] /= float64(n)

		for i := 0; i < n; i++ {
			diff := configs[i][j] - means[j]
			stds[j] += diff * diff
		}
		stds[j] = math.Sqrt(stds[j] / float64(n))
		if stds[j] < 1e-10 {
			stds[j] = 1
		}
	}

	X := make([][]float64, n)
	for i := range X {
		X[i] = make([]float64, d)
		floats.SubTo(X[i], configs[i], means)
		floats.Div(X[i], stds)
	}

	yMean := 0.0
	for _, y := range throughputs {
		yMean += y
	}
	yMean /= float64(n)

	Y := make([]float64, n)
	for i := range Y {
		Y[i] = throughputs[i] - yMean
	}

	beta := make([]float64, d)
	residual := make([]float64, n)
	copy(residual, Y)

	maxIter := 1000
	tol := 1e-6

	col := make([]float64, n)
	for iter := 0; iter < maxIter; iter++ {
		maxChange := 0.0

		for j := 0; j < d; j++ {
			for i := 0; i < n; i++ {
				col[i] = X[i][j]
			}

			floats.AddScaled(residual, beta[j], col)

			rho := floats.Dot(col, residual) / float64(n)

			oldBeta := beta[j]
			beta[j] = softThreshold(rho, alpha)

			floats.AddScaled(residual, -beta[j], col)

			change := math.Abs(beta[j] - oldBeta)
			if change > maxChange {
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
