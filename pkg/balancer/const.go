package balancer

import "math"

const (
	DefaultDQNHiddenSize     = 64
	DefaultDQNEpsilon        = 0.05
	DefaultDQNGamma          = 0.99
	DefaultDQNLearningRate   = 0.001
	DefaultDQNReplayBufSize  = 50_000
	DefaultDQNBatchSize      = 32
	DefaultDQNMinReplay      = 64
	DefaultDQNTrainEvery     = 4    // train every N experiences (SB3 default)
	DefaultDQNFallbackRatio  = 0.8  // fallback to RR when throughput < ratio * baseline
	DefaultDQNLoadThreshold  = 0.75 // proactive fallback when mean partition load exceeds this
	DefaultDQNExpChannelSize = 4096
)

const (
	dqnMaxThroughput   = 300_000.0
	fallbackEnterTicks = 3
	fallbackExitTicks  = 10
)

var dqnThroughputLogNorm = math.Log1p(dqnMaxThroughput / 10_000.0)

const (
	PSARebalanceLoadFactor = 2.0
)
