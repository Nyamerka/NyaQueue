package scheduler

import "github.com/Nyamerka/NyaQueue/pkg/broker"

const (
	DefaultDQNSchedStateSize  = 22 // 10 level dist + 10 wait times + depth + lag
	DefaultDQNSchedHiddenSize = 64
	DefaultDQNSchedActions    = broker.MaxPriority

	DefaultDQNSchedEpsilon       = 0.05
	DefaultDQNSchedGamma         = 0.99
	DefaultDQNSchedLearningRate  = 0.001
	DefaultDQNSchedReplayBufSize = 50_000
	DefaultDQNSchedBatchSize     = 32
	DefaultDQNSchedMinReplay     = 64
	DefaultDQNSchedThreshold     = 5
	DefaultDQNSchedWeightInit    = 0.1
)
