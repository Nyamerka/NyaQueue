package scheduler

import (
	"fmt"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

// FIFO reads messages sequentially from the WAL by consumer offset.
type FIFO struct{}

func NewFIFO() *FIFO { return &FIFO{} }

func (f *FIFO) Next(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	hwm := partition.HighWaterMark()
	if consumerOffset > hwm {
		return nil, consumerOffset, fmt.Errorf("no new messages")
	}

	msg, err := partition.Read(consumerOffset)
	if err != nil {
		return nil, consumerOffset, err
	}

	return msg, consumerOffset + 1, nil
}

func (f *FIFO) Enqueue(_ *broker.Message, _ int64) {}
