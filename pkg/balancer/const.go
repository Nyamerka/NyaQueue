package balancer

import (
	"math"
	"time"
)

const (
	DefaultDQNHiddenSize     = 128
	DefaultDQNGamma          = 0.9
	DefaultDQNLearningRate   = 0.001
	DefaultDQNReplayBufSize  = 50_000
	DefaultDQNBatchSize      = 64
	DefaultDQNMinReplay      = 64
	DefaultDQNTrainEvery     = 4
	DefaultDQNUpdatesPerStep = 1
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

const (
	dqnMsgRateScale      = 100_000.0
	dqnMsgSizeScale      = 1024.0
	dqnInflightScale     = 10_000.0
	dqnDepthFeatureScale = 1_000.0
)

// Entropy-calibrated Boltzmann routing parameters.
//
// Instead of annealing a fixed τ (which breaks when Q-value scale drifts),
// we anneal a *normalized entropy target* H(softmax(q/τ))/log(K).
// τ is solved per-inference via binary search to hit that target.
const (
	dqnBoltzmannEntropyStart = 0.85
	dqnBoltzmannEntropyEnd   = 0.50
	dqnBoltzmannAnnealingExp = 5_000

	dqnInflightCouplingBeta = 3.0
	dqnLoadCouplingBeta     = 2.5
)

const (
	dqnPolicyTickInterval     = 100 * time.Millisecond
	dqnSlowPolicyTickInterval = 500 * time.Millisecond
	dqnFastPolicyTickInterval = 50 * time.Millisecond
	dqnLowRateThreshold       = 5_000.0
	dqnOverloadRateThreshold  = 50_000.0
)

const dqnNumQuantiles = 21
const dqnThompsonK = 1

const PSARebalanceLoadFactor = 2.0
