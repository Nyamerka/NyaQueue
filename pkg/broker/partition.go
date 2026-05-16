package broker

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samber/oops"
	"github.com/tidwall/wal"
)

const (
	walBatchMagic    byte = 0xBA
	walRedirectMagic byte = 0xFF
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
	lastFlushNs   atomic.Int64
	batchCodec    Codec
}

func (p *Partition) SetBatchCodec(c Codec) {
	p.batchCodec = c
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
		return 0, oops.Wrapf(err, "append to partition %d", p.id)
	}
	return offsets[0], nil
}

// AppendBatch writes a slice of messages to the WAL in a single WriteBatch
// call, removing per-message fsync overhead. Returns the offsets assigned to
// each message.
// LastFlushLatency returns the duration of the most recent WAL write batch.
func (p *Partition) LastFlushLatency() time.Duration {
	return time.Duration(p.lastFlushNs.Load())
}

func (p *Partition) AppendBatch(msgs []*Message) ([]uint64, error) {
	if len(msgs) == 0 {
		return nil, nil
	}

	var batch wal.Batch
	offsets := make([]uint64, len(msgs))
	buffers := make([][]byte, len(msgs))

	appendTime := time.Now().UnixNano()

	p.mu.Lock()
	for i, msg := range msgs {
		msg.Header.AppendTime = appendTime
		offsets[i] = p.nextOffset
		buffers[i] = msg.MarshalPooled()
		batch.Write(p.nextOffset, buffers[i])
		p.nextOffset++
	}

	start := time.Now()
	err := p.log.WriteBatch(&batch)
	p.lastFlushNs.Store(int64(time.Since(start)))
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

// AppendBatchCompressed serializes all messages, compresses as a single blob,
// and stores in WAL. The first offset gets the full compressed batch; subsequent
// offsets get small redirect records pointing back to the batch leader.
func (p *Partition) AppendBatchCompressed(msgs []*Message, codec Codec) ([]uint64, error) {
	if len(msgs) == 0 {
		return nil, nil
	}

	n := len(msgs)
	appendTime := time.Now().UnixNano()

	var totalPayload int
	for _, msg := range msgs {
		msg.Header.AppendTime = appendTime
		totalPayload += msg.encodedSize() + 4
	}

	payload := make([]byte, totalPayload)
	off := 0
	for _, msg := range msgs {
		encoded := msg.Marshal()
		binary.BigEndian.PutUint32(payload[off:], uint32(len(encoded)))
		off += 4
		copy(payload[off:], encoded)
		off += len(encoded)
	}

	compressed, err := codec.Encode(payload)
	if err != nil {
		return nil, oops.Wrapf(err, "batch compress")
	}

	header := make([]byte, 5+len(compressed))
	header[0] = walBatchMagic
	binary.BigEndian.PutUint32(header[1:5], uint32(n))
	copy(header[5:], compressed)

	var batch wal.Batch
	offsets := make([]uint64, n)

	p.mu.Lock()
	baseOffset := p.nextOffset
	offsets[0] = baseOffset
	batch.Write(baseOffset, header)
	p.nextOffset++

	for i := 1; i < n; i++ {
		offsets[i] = p.nextOffset
		redirect := make([]byte, 13)
		redirect[0] = walRedirectMagic
		binary.BigEndian.PutUint64(redirect[1:9], baseOffset)
		binary.BigEndian.PutUint32(redirect[9:13], uint32(i))
		batch.Write(p.nextOffset, redirect)
		p.nextOffset++
	}

	start := time.Now()
	err = p.log.WriteBatch(&batch)
	p.lastFlushNs.Store(int64(time.Since(start)))
	pi := p.priorityIndex
	p.mu.Unlock()

	if err != nil {
		return nil, oops.Wrapf(err, "WAL write compressed batch")
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

	if len(data) == 0 {
		return nil, oops.Errorf("empty WAL entry at offset %d", offset)
	}

	switch data[0] {
	case walRedirectMagic:
		if len(data) < 13 {
			return nil, oops.Errorf("corrupt redirect at offset %d", offset)
		}
		batchBase := binary.BigEndian.Uint64(data[1:9])
		idx := int(binary.BigEndian.Uint32(data[9:13]))
		return p.readFromBatch(batchBase, idx)

	case walBatchMagic:
		return p.readFromBatchData(data, 0)

	default:
		return UnmarshalMessage(data)
	}
}

func (p *Partition) readFromBatch(batchOffset uint64, index int) (*Message, error) {
	p.mu.RLock()
	data, err := p.log.Read(batchOffset)
	p.mu.RUnlock()
	if err != nil {
		return nil, oops.Wrapf(err, "WAL read batch at offset %d", batchOffset)
	}
	return p.readFromBatchData(data, index)
}

// readFromBatchData extracts a message from a batch WAL record.
// The data format: [0xBA][count:4][compressed payload...]
// The compressed payload, when decompressed, is: [len0:4][msg0...][len1:4][msg1...]...
// NOTE: The decompression codec is not stored in the record; the broker must
// call SetBatchCodec before using batch-compressed partitions.
func (p *Partition) readFromBatchData(data []byte, index int) (*Message, error) {
	if len(data) < 5 || data[0] != walBatchMagic {
		return nil, oops.Errorf("not a batch record")
	}
	count := int(binary.BigEndian.Uint32(data[1:5]))
	if index >= count {
		return nil, oops.Errorf("batch index %d out of range [0, %d)", index, count)
	}

	compressed := data[5:]

	var payload []byte
	if p.batchCodec != nil {
		var err error
		payload, err = p.batchCodec.Decode(compressed)
		if err != nil {
			return nil, oops.Wrapf(err, "decompress batch record")
		}
	} else {
		payload = compressed
	}

	off := 0
	for i := 0; i <= index; i++ {
		if off+4 > len(payload) {
			return nil, oops.Errorf("corrupt batch payload at message %d", i)
		}
		msgLen := int(binary.BigEndian.Uint32(payload[off:]))
		off += 4
		if i == index {
			if off+msgLen > len(payload) {
				return nil, oops.Errorf("corrupt batch payload: message %d truncated", i)
			}
			msg, err := UnmarshalMessage(payload[off : off+msgLen])
			if err != nil {
				return nil, oops.Wrapf(err, "unmarshal message %d in batch", i)
			}
			msg.BatchDecoded = true
			return msg, nil
		}
		off += msgLen
	}
	return nil, oops.Errorf("batch index %d not found", index)
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

// TruncateFront removes WAL entries before the given offset.
func (p *Partition) TruncateFront(offset uint64) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if offset < 1 {
		return nil
	}
	return p.log.TruncateFront(offset)
}

// TruncateBefore removes WAL segments whose entries are all older than cutoff.
// Scans from firstIndex forward until finding an entry newer than cutoff,
// then truncates everything before that offset.
func (p *Partition) TruncateBefore(cutoff time.Time) (deletedSegments int, deletedBytes int64, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	firstIdx, err := p.log.FirstIndex()
	if err != nil || firstIdx == 0 {
		return 0, 0, nil
	}

	cutoffNano := cutoff.UnixNano()
	truncateAt := uint64(0)

	for off := firstIdx; off < p.nextOffset; off++ {
		data, readErr := p.log.Read(off)
		if readErr != nil {
			break
		}
		hdr, hdrErr := UnmarshalHeader(data)
		if hdrErr != nil {
			break
		}
		if hdr.Timestamp >= cutoffNano {
			break
		}
		truncateAt = off + 1
		deletedSegments++
		deletedBytes += int64(len(data))
	}

	if truncateAt > 0 {
		if tErr := p.log.TruncateFront(truncateAt); tErr != nil {
			return 0, 0, tErr
		}
	}
	return deletedSegments, deletedBytes, nil
}

// TruncateOldestUntilSize removes oldest WAL entries until the total remaining
// is at most maxBytes. Returns the number of entries and bytes removed.
func (p *Partition) TruncateOldestUntilSize(maxBytes int64) (deletedEntries int, deletedBytes int64, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	firstIdx, err := p.log.FirstIndex()
	if err != nil || firstIdx == 0 {
		return 0, 0, nil
	}

	var totalSize int64
	for off := firstIdx; off < p.nextOffset; off++ {
		data, readErr := p.log.Read(off)
		if readErr != nil {
			break
		}
		totalSize += int64(len(data))
	}

	if totalSize <= maxBytes {
		return 0, 0, nil
	}

	truncateAt := uint64(0)
	for off := firstIdx; off < p.nextOffset && totalSize > maxBytes; off++ {
		data, readErr := p.log.Read(off)
		if readErr != nil {
			break
		}
		totalSize -= int64(len(data))
		truncateAt = off + 1
		deletedEntries++
		deletedBytes += int64(len(data))
	}

	if truncateAt > 0 {
		if tErr := p.log.TruncateFront(truncateAt); tErr != nil {
			return 0, 0, tErr
		}
	}
	return deletedEntries, deletedBytes, nil
}

func (p *Partition) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.log.Close()
}
