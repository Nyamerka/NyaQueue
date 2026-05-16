package optimizer

const (
	DefaultOptimizerWindowCap   = 10
	DefaultOptimizerWarmupTicks = 2 * DefaultOptimizerWindowCap
	DefaultQueueSoftThreshold   = 1000.0
	DefaultQueuePenaltyWeight   = 1.5
	DefaultLatencyPenaltyWeight = 0.3
	DefaultQueuePenaltyClamp    = -2.0
)
