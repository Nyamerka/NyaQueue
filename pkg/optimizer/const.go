package optimizer

const (
	DefaultOptimizerWindowCap   = 10
	DefaultOptimizerWarmupTicks = 400
	DefaultQueueSoftThreshold   = 1000.0
	DefaultQueuePenaltyWeight   = 1.5
	DefaultLatencyPenaltyWeight = 0.3
	DefaultQueuePenaltyClamp    = -2.0
	DefaultRewardClamp          = 2.0
	DefaultSafetyConsumeRatio   = 0.7
	DefaultLogEveryTicks        = 20

	DefaultMaxActionDelta = 0.05 // max parameter change per apply step

	DefaultMinApplyInterval          = 2_000_000_000 // 2s in nanoseconds (time.Duration)
	DefaultErrorPenaltyWeight        = 2.0
	DefaultErrorPenaltyThresh        = 0.01
	DefaultEmergencyRollbackDR       = 0.5
	DefaultEmergencyRollbackN        = 5
	DefaultEmergencyRollbackCooldown = 120
	DefaultEmergencyMinMsgRate       = 100.0
)

// State normalization scales for TD3 optimizer.
const (
	optimizerThroughputScale = 100_000.0
	optimizerLatencyScale    = 1000.0 // normalize to seconds
	optimizerDepthScale      = 10_000.0
)
