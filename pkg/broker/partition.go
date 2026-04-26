package broker

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/samber/oops"
	"github.com/tidwall/wal"
)

func walOptions(sp SyncPolicy) *wal.Options {
	opts := *wal.DefaultOptions
	switch sp {
	case SyncNone, SyncInterval:
		opts.NoSync = true
	}
	return &opts
}

type Partition struct {
	mu            sync.RWMutex
	id            int
	topicName     string
	log           *wal.Log
	nextOffset    uint64
	priorityIndex *PriorityIndex
	scheduleMode  ScheduleMode
}

func NewPartition(id int, topicName, dataDir string, mode ScheduleMode, syncPolicy SyncPolicy) (*Partition, error) {
	dir := filepath.Join(dataDir, topicName, fmt.Sprintf("partition-%d", id))
	log, err := wal.Open(dir, walOptions(syncPolicy))
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

// Append writes a single message to the WAL and returns its offset.
func (p *Partition) Append(msg *Message) (uint64, error) {
	offsets, err := p.AppendBatch([]*Message{msg})
	if err != nil {
		return 0, err
	}
	return offsets[0], nil
}

// AppendBatch writes a slice of messages to the WAL in a single WriteBatch
// call, removing per-message fsync overhead. Returns the offsets assigned to
// each message.
func (p *Partition) AppendBatch(msgs []*Message) ([]uint64, error) {
	if len(msgs) == 0 {
		return nil, nil
	}

	var batch wal.Batch
	offsets := make([]uint64, len(msgs))
	buffers := make([][]byte, len(msgs))

	p.mu.Lock()
	for i, msg := range msgs {
		offsets[i] = p.nextOffset
		buffers[i] = msg.MarshalPooled()
		batch.Write(p.nextOffset, buffers[i])
		p.nextOffset++
	}

	err := p.log.WriteBatch(&batch)
	pi := p.priorityIndex
	p.mu.Unlock()

	for _, buf := range buffers {
		ReleaseMarshalBuf(buf)
	}

	if err != nil {
		return nil, oops.Wrapf(err, "WAL write batch")
	}

	if pi != nil {
		now := time.Now()
		for i, msg := range msgs {
			pi.Add(int(msg.Header.Priority), int64(offsets[i]), now)
		}
	}

	return offsets, nil
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
