package scheduler

import (
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/samber/oops"
)

// StrictPriority delivers highest-priority messages first via PriorityIndex.
// Within the same priority level, order is FIFO.
type StrictPriority struct{}

func NewStrictPriority() *StrictPriority { return &StrictPriority{} }

func (s *StrictPriority) Next(partition *broker.Partition, _ uint64) (*broker.Message, uint64, error) {
	pi := partition.PriorityIndex()
	if pi == nil {
		return nil, 0, oops.Errorf("partition %d has no PriorityIndex", partition.ID())
	}

	entry, ok := pi.PopHighest()
	if !ok {
		return nil, 0, oops.Errorf("no pending messages")
	}

	msg, err := partition.Read(uint64(entry.WalOffset))
	if err != nil {
		return nil, 0, err
	}

	return msg, uint64(entry.WalOffset), nil
}

func (s *StrictPriority) Enqueue(_ *broker.Message, _ int64) {}
