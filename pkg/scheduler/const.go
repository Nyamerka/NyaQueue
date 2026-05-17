package scheduler

import "github.com/Nyamerka/NyaQueue/pkg/broker"

const (
	DefaultDQNSchedStateSize  = broker.MaxPriority*2 + 3 // level dist + wait times + depth + velocity + ratio
	DefaultDQNSchedHiddenSize = 64
	DefaultDQNSchedActions    = broker.MaxPriority

	DefaultDQNSchedEpsilon       = 0.05
	DefaultDQNSchedGamma         = 0.99
	DefaultDQNSchedLearningRate  = 0.001
	DefaultDQNSchedReplayBufSize = 50_000
	DefaultDQNSchedBatchSize     = 32
	DefaultDQNSchedMinReplay     = 64
	DefaultDQNSchedThreshold     = 5

	dqnSchedVelocityScale      = 10_000.0
	dqnSchedGrowthPenaltyW     = 0.3
	dqnSchedPriorityOrderBonus = 0.1
	dqnSchedOverloadDepth      = 1_000
	dqnSchedExpertSeeds        = 200
	dqnSchedCrisisBufSize      = 2_000
	dqnSchedCrisisRatio        = 0.3 // 30% of batch from crisis buffer
)
