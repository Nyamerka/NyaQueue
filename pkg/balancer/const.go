package balancer

import (
	"math"
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

// Policy tick interval and adaptive slow-tick for low-rate scenarios.
const (
	dqnPolicyTickInterval     = 100 * time.Millisecond
	dqnSlowPolicyTickInterval = 500 * time.Millisecond
	dqnLowRateThreshold       = 5_000.0
)

// PSA constants.
const (
	PSARebalanceLoadFactor = 2.0
)
