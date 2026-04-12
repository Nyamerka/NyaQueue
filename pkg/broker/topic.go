package broker

import (
	"sync"

	"github.com/samber/oops"
)

type Topic struct {
	mu         sync.RWMutex
	name       string
	partitions []*Partition
	config     TopicConfig
}

func NewTopic(name, dataDir string, cfg TopicConfig) (*Topic, error) {
	if cfg.NumPartitions <= 0 {
		cfg.NumPartitions = 4
	}

	t := &Topic{
		name:       name,
		partitions: make([]*Partition, cfg.NumPartitions),
		config:     cfg,
	}

	for i := 0; i < cfg.NumPartitions; i++ {
		p, err := NewPartition(i, name, dataDir, cfg.ScheduleMode)
		if err != nil {
			t.Close()
			return nil, oops.Wrapf(err, "create partition %d for topic %q", i, name)
		}
		t.partitions[i] = p
	}

	return t, nil
}

func (t *Topic) Name() string { return t.name }

func (t *Topic) NumPartitions() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.partitions)
}

func (t *Topic) Config() TopicConfig {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.config
}

func (t *Topic) Partition(id int) (*Partition, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if id < 0 || id >= len(t.partitions) {
		return nil, oops.Errorf("partition %d out of range [0, %d)", id, len(t.partitions))
	}
	return t.partitions[id], nil
}

func (t *Topic) Partitions() []*Partition {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]*Partition, len(t.partitions))
	copy(out, t.partitions)
	return out
}

func (t *Topic) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	var firstErr error
	for _, p := range t.partitions {
		if p != nil {
			if err := p.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}
