package balancer

import "github.com/Nyamerka/NyaQueue/pkg/broker"

// Balancer selects a target partition for an incoming message.
type Balancer interface {
	// SelectPartition returns the partition index for a given topic/key pair.
	SelectPartition(topic string, key []byte, numPartitions int) int

	// OnMetrics is called periodically with fresh broker metrics.
	OnMetrics(m broker.Metrics)
}
