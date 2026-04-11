package scheduler

import "github.com/Nyamerka/NyaQueue/pkg/broker"

// Scheduler determines the order in which messages are delivered to consumers.
type Scheduler interface {
	// Next returns the next message for consumption from the given partition.
	// In FIFO mode, reads WAL sequentially.
	// In Priority/DQN modes, uses the PriorityIndex.
	Next(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error)

	// Enqueue is called when a new message is written (Priority/DQN: add to index).
	// FIFO schedulers can ignore this call.
	Enqueue(msg *broker.Message, walOffset int64)
}
