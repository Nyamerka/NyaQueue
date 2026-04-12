package broker

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/samber/oops"
	"github.com/tidwall/wal"
)

type Partition struct {
	mu            sync.RWMutex
	id            int
	topicName     string
	log           *wal.Log
	nextOffset    uint64
	priorityIndex *PriorityIndex
	scheduleMode  ScheduleMode
}

func NewPartition(id int, topicName, dataDir string, mode ScheduleMode) (*Partition, error) {
	dir := filepath.Join(dataDir, topicName, fmt.Sprintf("partition-%d", id))
	log, err := wal.Open(dir, nil)
	if err != nil {
		return nil, oops.Wrapf(err, "open WAL partition %d", id)
	}

	lastIndex, err := log.LastIndex()
	if err != nil {
		log.Close()
		return nil, oops.Wrapf(err, "read last index partition %d", id)
	}

	p := &Partition{
		id:           id,
		topicName:    topicName,
		log:          log,
		nextOffset:   lastIndex + 1,
		scheduleMode: mode,
	}

	if mode != ModeFIFO {
		p.priorityIndex = NewPriorityIndex()
	}

	return p, nil
}

func (p *Partition) ID() int { return p.id }

// Append writes a message to the WAL and returns its offset.
// If the partition uses priority scheduling, the offset is added to the PriorityIndex.
func (p *Partition) Append(msg *Message) (uint64, error) {
	data := msg.Marshal()

	p.mu.Lock()
	offset := p.nextOffset
	err := p.log.Write(offset, data)
	if err != nil {
		p.mu.Unlock()
		return 0, oops.Wrapf(err, "WAL write")
	}
	p.nextOffset++
	pi := p.priorityIndex
	p.mu.Unlock()

	if pi != nil {
		pi.Add(int(msg.Header.Priority), int64(offset), time.Now())
	}

	return offset, nil
}

func (p *Partition) Read(offset uint64) (*Message, error) {
	p.mu.RLock()
	data, err := p.log.Read(offset)
	p.mu.RUnlock()

	if err != nil {
		return nil, oops.Wrapf(err, "WAL read offset %d", offset)
	}
	return UnmarshalMessage(data)
}

func (p *Partition) HighWaterMark() uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.nextOffset == 0 {
		return 0
	}
	return p.nextOffset - 1
}

func (p *Partition) PriorityIndex() *PriorityIndex {
	return p.priorityIndex
}

// QueueDepth returns the number of undelivered messages (used by metrics).
func (p *Partition) QueueDepth() int {
	if p.priorityIndex != nil {
		return p.priorityIndex.Len()
	}
	return 0
}

// Rebuild walks the WAL from startOffset to HighWaterMark and re-populates the PriorityIndex.
// Used at startup for partitions with priority scheduling.
func (p *Partition) Rebuild(startOffset uint64, isCommitted func(offset uint64) bool) error {
	if p.priorityIndex == nil {
		return nil
	}

	p.mu.RLock()
	hwm := p.nextOffset
	p.mu.RUnlock()

	for off := startOffset; off < hwm; off++ {
		p.mu.RLock()
		data, err := p.log.Read(off)
		p.mu.RUnlock()
		if err != nil {
			return oops.Wrapf(err, "rebuild read offset %d", off)
		}

		if isCommitted(off) {
			continue
		}

		hdr, err := UnmarshalHeader(data)
		if err != nil {
			return oops.Wrapf(err, "rebuild unmarshal header offset %d", off)
		}

		p.priorityIndex.Add(int(hdr.Priority), int64(off), time.Unix(0, hdr.Timestamp))
	}

	return nil
}

func (p *Partition) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.log.Close()
}
