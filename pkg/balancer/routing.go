package balancer

import (
	"math"
	"math/rand/v2"
)

// currentEntropyTarget returns the normalized entropy target via cosine
// annealing from EntropyStart → EntropyEnd over dqnBoltzmannAnnealingExp.
func currentEntropyTarget(experiencesSeen int64) float64 {
	if experiencesSeen >= dqnBoltzmannAnnealingExp {
		return dqnBoltzmannEntropyEnd
	}
	progress := float64(experiencesSeen) / float64(dqnBoltzmannAnnealingExp)
	return dqnBoltzmannEntropyEnd + 0.5*(dqnBoltzmannEntropyStart-dqnBoltzmannEntropyEnd)*(1+math.Cos(progress*math.Pi))
}

func normalizedEntropy(q []float64, tau float64) float64 {
	K := len(q)
	if K <= 1 {
		return 1.0
	}
	logK := math.Log(float64(K))
	maxQ := q[0]
	for _, v := range q[1:] {
		if v > maxQ {
			maxQ = v
		}
	}
	var sum float64
	for _, v := range q {
		sum += math.Exp((v - maxQ) / tau)
	}
	var H float64
	for _, v := range q {
		p := math.Exp((v-maxQ)/tau) / sum
		if p > 1e-15 {
			H -= p * math.Log(p)
		}
	}
	return H / logK
}

// solveForEntropy finds τ such that H(softmax(q/τ))/log(K) ≈ normEntTarget
// via binary search (16 iterations).
func solveForEntropy(q []float64, normEntTarget float64) float64 {
	K := len(q)
	if K <= 1 {
		return 1.0
	}
	allSame := true
	for i := 1; i < K; i++ {
		if math.Abs(q[i]-q[0]) > 1e-10 {
			allSame = false
			break
		}
	}
	if allSame {
		return 1.0
	}
	lo, hi := 1e-4, 100.0
	for range 16 {
		mid := (lo + hi) / 2
		if normalizedEntropy(q, mid) < normEntTarget {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

func softmaxWithEntropy(q []float64, normEntTarget float64) []float64 {
	tau := solveForEntropy(q, normEntTarget)
	return softmaxFromQ(q, tau)
}

// softmaxFromQ computes Boltzmann probabilities:
// p[i] = exp(q[i]/τ) / Σ exp(q[j]/τ).
func softmaxFromQ(q []float64, tau float64) []float64 {
	if tau <= 0 {
		tau = 1e-8
	}
	maxQ := q[0]
	for _, v := range q[1:] {
		if v > maxQ {
			maxQ = v
		}
	}
	probs := make([]float64, len(q))
	var sum float64
	for i, v := range q {
		e := math.Exp((v - maxQ) / tau)
		probs[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	} else {
		uniform := 1.0 / float64(len(probs))
		for i := range probs {
			probs[i] = uniform
		}
	}
	return probs
}

func sampleCDF(probs []float64) int {
	r := rand.Float64()
	var cumulative float64
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return len(probs) - 1
}
