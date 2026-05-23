package scheduler

import "github.com/Nyamerka/NyaQueue/pkg/broker"

const (
	DefaultDQNSchedStateSize  = broker.MaxPriority*2 + 3 // level dist + wait times + depth + velocity + ratio
	DefaultDQNSchedHiddenSize = 128
	DefaultDQNSchedActions    = broker.MaxPriority

	DefaultDQNSchedGamma         = 0.9
	DefaultDQNSchedLearningRate  = 0.001
	DefaultDQNSchedReplayBufSize = 50_000
	DefaultDQNSchedBatchSize     = 32
	DefaultDQNSchedMinReplay     = 64
	DefaultDQNSchedThreshold     = 5

	dqnSchedVelocityScale      = 10_000.0
	dqnSchedGrowthPenaltyW     = 0.3
	dqnSchedPriorityOrderBonus = 0.1
)
