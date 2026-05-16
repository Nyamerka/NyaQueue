package optimizer

const (
	DefaultOptimizerWindowCap   = 10
	DefaultOptimizerWarmupTicks = 2 * DefaultOptimizerWindowCap
	DefaultQueueSoftThreshold   = 1000.0
	DefaultQueuePenaltyWeight   = 1.5
	DefaultLatencyPenaltyWeight = 0.3
	DefaultQueuePenaltyClamp    = -2.0
	DefaultRewardClamp          = 2.0
	DefaultSafetyConsumeRatio   = 0.7
	DefaultLogEveryTicks        = 20

	DefaultMinApplyInterval    = 2_000_000_000 // 2s in nanoseconds (time.Duration)
	DefaultErrorPenaltyWeight  = 2.0
	DefaultErrorPenaltyThresh  = 0.01
	DefaultEmergencyRollbackDR = 0.5
	DefaultEmergencyRollbackN  = 2
)

// State normalization scales for DDPG optimizer.
const (
	optimizerThroughputScale = 100_000.0
	optimizerLatencyScale    = 1000.0 // normalize to seconds
)
