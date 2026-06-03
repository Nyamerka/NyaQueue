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

const (
	dqnPolicyTickInterval     = 100 * time.Millisecond
	dqnSlowPolicyTickInterval = 500 * time.Millisecond
	dqnFastPolicyTickInterval = 50 * time.Millisecond
	dqnLowRateThreshold       = 5_000.0
	dqnOverloadRateThreshold  = 50_000.0
)

const dqnNumQuantiles = 21

const PSARebalanceLoadFactor = 2.0
