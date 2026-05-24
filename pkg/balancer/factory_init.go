package balancer

import "github.com/Nyamerka/NyaQueue/pkg/broker"

func init() {
	broker.RegisterBalancerFactory(broker.KindRoundRobin, func(k int) broker.Balancer {
		return NewRoundRobin()
	})
	broker.RegisterBalancerFactory(broker.KindPowerOfTwoChoices, func(k int) broker.Balancer {
		return NewPowerOfTwoChoices()
	})
	broker.RegisterBalancerFactory(broker.KindPSA, func(k int) broker.Balancer {
		return NewPSA(k)
	})
	broker.RegisterBalancerFactory(broker.KindDQN, func(k int) broker.Balancer {
		return NewDQNBalancer(k)
	})
}
