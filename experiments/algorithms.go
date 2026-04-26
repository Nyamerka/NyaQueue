package experiments

import (
	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
)

// AlgorithmConfig describes a balancer+scheduler combination to benchmark.
//
// Mode picks the partition storage layout the scheduler depends on. FIFO-based
// schedulers read sequentially from the WAL; priority and DQN schedulers pop
// from the per-partition PriorityIndex, which only exists when the partition
// is created with a non-FIFO mode.
type AlgorithmConfig struct {
	Name          string
	NewBalancer   func() broker.Balancer
	NewScheduler  func() broker.Scheduler
	Mode          broker.ScheduleMode
	WithOptimizer bool
}

func AllAlgorithms() []AlgorithmConfig {
	return []AlgorithmConfig{
		{
			Name:         "RR+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewRoundRobin() },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
			Mode:         broker.ModeFIFO,
		},
		{
			Name:         "WRR+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewWeightedRoundRobin() },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
			Mode:         broker.ModeFIFO,
		},
		{
			Name:         "PSA+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewPSA(4) },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
			Mode:         broker.ModeFIFO,
		},
		{
			Name:         "DQN+Priority",
			NewBalancer:  func() broker.Balancer { return balancer.NewDQNBalancer(4) },
			NewScheduler: func() broker.Scheduler { return scheduler.NewStrictPriority() },
			Mode:         broker.ModeStrictPriority,
		},
		{
			Name:          "DQN+DDPG",
			NewBalancer:   func() broker.Balancer { return balancer.NewDQNBalancer(4) },
			NewScheduler:  func() broker.Scheduler { return scheduler.NewDQNScheduler() },
			Mode:          broker.ModeDQNAdaptive,
			WithOptimizer: true,
		},
	}
}

// KafkaBaseline returns a stub config used for labeling Kafka results.
func KafkaBaseline() AlgorithmConfig {
	return AlgorithmConfig{
		Name: "Kafka",
	}
}
