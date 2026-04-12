package scheduler

import "github.com/Nyamerka/NyaQueue/pkg/broker"

const (
	DefaultDQNSchedStateSize  = broker.MaxPriority*2 + 2 // level dist + wait times + depth + lag
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
