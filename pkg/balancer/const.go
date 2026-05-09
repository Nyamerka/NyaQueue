package balancer

const (
	DefaultDQNHiddenSize    = 64
	DefaultDQNEpsilon       = 0.05
	DefaultDQNGamma         = 0.99
	DefaultDQNLearningRate  = 0.001
	DefaultDQNReplayBufSize = 100_000
	DefaultDQNBatchSize     = 32
	DefaultDQNMinReplay     = 64
	DefaultDQNFallbackRatio = 0.8  // fallback to RR when throughput < ratio * baseline
	DefaultDQNLoadThreshold = 0.75 // proactive fallback when mean partition load exceeds this
	DefaultDQNWeightInit    = 0.1
	DefaultWRRMinLoad       = 0.01
)

const (
	PSARebalanceLoadFactor = 2.0
)
