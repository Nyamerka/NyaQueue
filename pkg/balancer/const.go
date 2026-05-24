package balancer

import (
	"math"
	"math/rand/v2"
	"time"
)

// DQN hyperparameters (user-facing defaults).
const (
	DefaultDQNHiddenSize     = 128
	DefaultDQNGamma          = 0.9
	DefaultDQNLearningRate   = 0.001
	DefaultDQNReplayBufSize  = 50_000
	DefaultDQNBatchSize      = 32
	DefaultDQNMinReplay      = 64
	DefaultDQNTrainEvery     = 1
	DefaultDQNUpdatesPerStep = 4
	DefaultDQNExpChannelSize = 16384
	DefaultDQNWeightDecay    = 1e-4
)

// DQN reward shaping constants.
//
// Two orthogonal signals via smooth pressure-based interpolation:
//
//	balance  = -CV(loads), where CV = stddev/mean (coefficient of variation)
//	utility  = log1p(throughput/scale) / log1p(maxThroughput/scale)  ∈ [0,1]
//	pressure = clamp(meanLoad / overloadThreshold, 0, 1)
//	w_bal    = baseWeight + (maxWeight - baseWeight) * pressure
//	reward   = w_bal * balance + (1 - w_bal) * utility
const (
	dqnThroughputScale   = 10_000.0
	dqnMaxThroughput     = 300_000.0
	dqnOverloadThreshold = 0.7
	dqnBaseBalanceWeight = 0.4
	dqnMaxBalanceWeight  = 0.7
)

var dqnThroughputLogNorm = math.Log1p(dqnMaxThroughput / dqnThroughputScale)

// DQN state normalization scales.
const (
	dqnMsgRateScale      = 100_000.0
	dqnMsgSizeScale      = 1024.0
	dqnInflightScale     = 10_000.0
	dqnDepthFeatureScale = 1_000.0
)

// DQN queue-depth penalty in reward.
const (
	dqnDepthSoftCap = 100_000.0
	dqnDepthWeight  = 0.3
)

// DQN max-load penalty: explicitly punishes the hottest partition when it
// exceeds capacity (load > 1.0), preventing the DQN from funneling traffic
// into a single partition under overload.
const dqnMaxLoadWeight = 1.0

// Boltzmann (softmax) routing parameters.
const (
	dqnBoltzmannTauStart      = 0.5
	dqnBoltzmannTauEnd        = 0.05
	dqnBoltzmannAnnealingExp  = 20_000
	dqnBoltzmannInflightAlpha = 2.0
)

// currentTemperature implements cosine annealing for the Boltzmann temperature:
// τ decays from dqnBoltzmannTauStart → dqnBoltzmannTauEnd over the first
// dqnBoltzmannAnnealingExp experiences using cosine schedule.
func currentTemperature(experiencesSeen int64) float64 {
	if experiencesSeen >= dqnBoltzmannAnnealingExp {
		return dqnBoltzmannTauEnd
	}
	progress := float64(experiencesSeen) / float64(dqnBoltzmannAnnealingExp)
	return dqnBoltzmannTauEnd + 0.5*(dqnBoltzmannTauStart-dqnBoltzmannTauEnd)*(1+math.Cos(progress*math.Pi))
}

// Policy tick interval and adaptive slow-tick for low-rate scenarios.
const (
	dqnPolicyTickInterval     = 100 * time.Millisecond
	dqnSlowPolicyTickInterval = 500 * time.Millisecond
	dqnLowRateThreshold       = 5_000.0
)

// softmaxFromQ computes Boltzmann probabilities from Q-values:
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

// sampleCDF samples an index from a probability distribution using inverse-CDF.
// O(K) per call which is fast for K ≤ 32.
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

// PSA constants.
const (
	PSARebalanceLoadFactor = 2.0
)
