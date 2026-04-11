package experiments

import (
	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
)

// AlgorithmConfig describes a balancer+scheduler combination to benchmark.
type AlgorithmConfig struct {
	Name          string
	NewBalancer   func() broker.Balancer
	NewScheduler  func() broker.Scheduler
	WithOptimizer bool
}

func AllAlgorithms() []AlgorithmConfig {
	return []AlgorithmConfig{
		{
			Name:         "RR+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewRoundRobin() },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
		},
		{
			Name:         "WRR+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewWeightedRoundRobin() },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
		},
		{
			Name:         "PSA+FIFO",
			NewBalancer:  func() broker.Balancer { return balancer.NewPSA(4) },
			NewScheduler: func() broker.Scheduler { return scheduler.NewFIFO() },
		},
		{
			Name:         "DQN+Priority",
			NewBalancer:  func() broker.Balancer { return balancer.NewDQNBalancer(4) },
			NewScheduler: func() broker.Scheduler { return scheduler.NewStrictPriority() },
		},
		{
			Name:          "DQN+DDPG",
			NewBalancer:   func() broker.Balancer { return balancer.NewDQNBalancer(4) },
			NewScheduler:  func() broker.Scheduler { return scheduler.NewDQNScheduler() },
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
