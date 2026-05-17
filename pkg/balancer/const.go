package balancer

import (
	"math"
	"time"
)

// DQN hyperparameters (user-facing defaults).
const (
	DefaultDQNHiddenSize     = 128
	DefaultDQNEpsilon        = 0.05
	DefaultDQNGamma          = 0.9
	DefaultDQNLearningRate   = 0.001
	DefaultDQNReplayBufSize  = 50_000
	DefaultDQNBatchSize      = 32
	DefaultDQNMinReplay      = 64
	DefaultDQNTrainEvery     = 4
	DefaultDQNFallbackRatio  = 0.8
	DefaultDQNLoadThreshold  = 0.75
	DefaultDQNExpChannelSize = 4096
)

// DQN reward shaping constants.
//
// The reward combines two signals via smooth pressure-based interpolation:
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
	dqnBaseBalanceWeight = 0.5
	dqnMaxBalanceWeight  = 0.95
)

var dqnThroughputLogNorm = math.Log1p(dqnMaxThroughput / dqnThroughputScale)

// DQN state normalization scales.
const (
	dqnMsgRateScale       = 100_000.0
	dqnMsgSizeScale       = 1024.0 // normalize to KB
	dqnInflightScale      = 10_000.0
	dqnPolicyTickInterval = 100 * time.Millisecond
)

// Adaptive epsilon: below this msg rate, epsilon stays at base value.
const dqnNormalRateThreshold = 10_000.0

// DQN queue-depth penalty in reward.
const (
	dqnDepthSoftCap = 100_000.0
	dqnDepthWeight  = 0.3
)

// DQN fallback hysteresis: consecutive ticks required to enter/exit RR fallback.
const (
	fallbackEnterTicks = 3
	fallbackExitTicks  = 10
)

// PSA constants.
const (
	PSARebalanceLoadFactor = 2.0
)
