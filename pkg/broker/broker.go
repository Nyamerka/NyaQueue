package broker

import (
	"context"
	"errors"
	"log"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
	"github.com/samber/oops"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

var (
	ErrTopicAlreadyExists = errors.New("topic already exists")
	ErrTopicNotFound      = errors.New("topic not found")
	ErrNoMessages         = errors.New("no messages available")
)

type Balancer interface {
	SelectPartition(topic string, key []byte, numPartitions int) int
	OnMetrics(m Metrics)
}

// BaseThroughputSetter is optionally implemented by balancers that support
// reactive watchdog fallback based on throughput degradation.
type BaseThroughputSetter interface {
	SetBaseThroughput(t float64)
}

type Scheduler interface {
	Next(partition *Partition, consumerOffset uint64) (*Message, uint64, error)
	Enqueue(msg *Message, walOffset int64)
}

// configSnapshot is an immutable snapshot published via a single atomic.Pointer.
// All fields are consistent with each other: Publish/PublishBatch read exactly
// one snapshot and operate on it without racing with ApplyConfig.
type configSnapshot struct {
	cfg   Config
	codec Codec
	gate  *ioGate
}

type Broker struct {
	mu   sync.RWMutex // protects balancer, backpressure, schedulerFactory
	snap atomic.Pointer[configSnapshot]

	dataDir    string
	topics     *xsync.MapOf[string, *Topic]
	schedulers *xsync.MapOf[string, Scheduler]
	balancer   Balancer

	schedulerFactory func(TopicConfig) Scheduler

	metrics      *MetricsCollector
	predictor    *LoadPredictor
	backpressure *BackpressureController
	offsetStore  *OffsetStore

	eg     *errgroup.Group
	cancel context.CancelFunc
}

type ioGate struct {
	sem *semaphore.Weighted
	cap int64
}

func New(cfg Config, dataDir string, bal Balancer, offsetStore *OffsetStore) *Broker {
	ioPoolSize := int64(cfg.NumIOGoroutines)
	if ioPoolSize <= 0 {
		ioPoolSize = 4
	}

	b := &Broker{
		dataDir:     dataDir,
		topics:      xsync.NewMapOf[string, *Topic](),
		balancer:    bal,
		schedulers:  xsync.NewMapOf[string, Scheduler](),
		offsetStore: offsetStore,
	}

	codec := NewCodec(cfg.CompressionType)
	b.snap.Store(&configSnapshot{
		cfg:   cfg,
		codec: codec,
		gate:  &ioGate{sem: semaphore.NewWeighted(ioPoolSize), cap: ioPoolSize},
	})

	b.predictor = NewLoadPredictor(100, 8, 100*time.Millisecond)
	b.metrics = NewMetricsCollector(b)
	b.backpressure = NewBackpressureController(0.85)

	return b
}

func (b *Broker) SetBalancer(bal Balancer) {
	b.mu.Lock()
	old := b.balancer
	b.balancer = bal
	b.mu.Unlock()

	if old != nil {
		if s, ok := old.(Stopper); ok {
			go s.Stop()
		}
	}
}

func (b *Broker) SetScheduler(topic string, sched Scheduler) {
	b.stopScheduler(topic)
	b.schedulers.Store(topic, sched)
}

func (b *Broker) SetSchedulerFactory(fn func(TopicConfig) Scheduler) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.schedulerFactory = fn
}

func (b *Broker) SetBackpressure(bp *BackpressureController) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.backpressure = bp
}

func (b *Broker) CreateTopic(name string, cfg TopicConfig) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.topics.Load(name); exists {
		return oops.Wrapf(ErrTopicAlreadyExists, "create topic %q", name)
	}

	snap := b.snap.Load()
	t, err := NewTopic(name, b.dataDir, cfg, snap.cfg.SyncPolicy)
	if err != nil {
		return oops.Wrapf(err, "create topic %q", name)
	}
	if snap.cfg.CompressionType != CompressionNone {
		for _, p := range t.Partitions() {
			p.SetBatchCodec(snap.codec)
		}
	}
	b.topics.Store(name, t)

	if b.schedulerFactory != nil {
		b.schedulers.Store(name, b.schedulerFactory(cfg))
	}

	return nil
}

func (b *Broker) stopScheduler(name string) {
	sched, ok := b.schedulers.LoadAndDelete(name)
	if !ok {
		return
	}
	if s, ok := sched.(Stopper); ok {
		s.Stop()
	}
}

func (b *Broker) DeleteTopic(name string) error {
	t, exists := b.topics.LoadAndDelete(name)
	if !exists {
		return oops.Wrapf(ErrTopicNotFound, "delete topic %q", name)
	}

	partIDs := make([]int, 0, len(t.Partitions()))
	for _, p := range t.Partitions() {
		partIDs = append(partIDs, p.ID())
	}

	b.stopScheduler(name)

	if err := t.Close(); err != nil {
		return oops.Wrapf(err, "close topic %q", name)
	}

	// Clean up partition-level state to prevent memory leaks in long-lived brokers.
	for _, pid := range partIDs {
		b.predictor.RemovePartition(pid)
		b.metrics.RemovePartition(pid)
	}

	if err := os.RemoveAll(filepath.Join(b.dataDir, name)); err != nil {
		return oops.Wrapf(err, "remove topic data dir %q", name)
	}

	if b.offsetStore != nil {
		b.offsetStore.DeleteTopic(name)
	}

	return nil
}

func (b *Broker) GetTopic(name string) (*Topic, error) {
	t, exists := b.topics.Load(name)
	if !exists {
		return nil, oops.Errorf("topic %q not found", name)
	}
	return t, nil
}

func (b *Broker) ListTopics() []*Topic {
	var topics []*Topic
	b.topics.Range(func(_ string, t *Topic) bool {
		topics = append(topics, t)
		return true
	})
	return topics
}

func (b *Broker) compressMessage(msg *Message, codec Codec) error {
	compressed, err := codec.Encode(msg.Value)
	if err != nil {
		return oops.Wrapf(err, "compress message value")
	}
	msg.Value = compressed
	return nil
}

func (b *Broker) decompressMessage(msg *Message) error {
	if msg.BatchDecoded {
		return nil
	}
	snap := b.snap.Load()
	decompressed, err := snap.codec.Decode(msg.Value)
	if err != nil {
		return oops.Wrapf(err, "decompress message value")
	}
	msg.Value = decompressed
	return nil
}

func (b *Broker) Publish(ctx context.Context, topicName string, msg *Message) (partition int, offset uint64, err error) {
	snap := b.snap.Load()
	if snap.cfg.MaxMessageBytes > 0 && len(msg.Key)+len(msg.Value) > snap.cfg.MaxMessageBytes {
		return 0, 0, oops.Wrapf(ErrMessageTooLarge, "size %d > limit %d",
			len(msg.Key)+len(msg.Value), snap.cfg.MaxMessageBytes)
	}

	if err := b.compressMessage(msg, snap.codec); err != nil {
		return 0, 0, oops.Wrapf(err, "compress message for topic %q", topicName)
	}

	t, exists := b.topics.Load(topicName)
	if !exists {
		return 0, 0, oops.Errorf("topic %q not found", topicName)
	}

	b.mu.RLock()
	bal := b.balancer
	bp := b.backpressure
	b.mu.RUnlock()

	numParts := t.NumPartitions()
	partIdx := bal.SelectPartition(topicName, msg.Key, numParts)

	if bp != nil {
		state := bp.Check(partIdx)
		if state == BPClosed {
			return 0, 0, ErrThrottled
		}
	}

	p, err := t.Partition(partIdx)
	if err != nil {
		return 0, 0, oops.Wrapf(err, "get partition %d for topic %q", partIdx, topicName)
	}

	msg.Header.ProduceTime = time.Now().UnixNano()
	offset, err = p.Append(msg)
	if err != nil {
		return 0, 0, oops.Wrapf(err, "append to partition %d", partIdx)
	}

	b.metrics.RecordProduce(topicName, partIdx)

	return partIdx, offset, nil
}

type PublishResult struct {
	Partition int
	Offset    uint64
	Err       error
}

func (b *Broker) PublishBatch(ctx context.Context, topicName string, msgs []*Message) []PublishResult {
	results := make([]PublishResult, len(msgs))

	t, exists := b.topics.Load(topicName)
	if !exists {
		for i := range results {
			results[i].Err = oops.Errorf("topic %q not found", topicName)
		}
		return results
	}

	b.mu.RLock()
	bal := b.balancer
	bp := b.backpressure
	b.mu.RUnlock()

	numParts := t.NumPartitions()

	type indexed struct {
		msg *Message
		idx int
	}
	groups := make(map[int][]indexed, numParts)

	snap := b.snap.Load()
	cfg := &snap.cfg
	codec := snap.codec
	useBatchCompression := cfg.CompressionType != CompressionNone

	now := time.Now().UnixNano()
	for i, msg := range msgs {
		if cfg.MaxMessageBytes > 0 && len(msg.Key)+len(msg.Value) > cfg.MaxMessageBytes {
			results[i].Err = oops.Wrapf(ErrMessageTooLarge, "size %d > limit %d",
				len(msg.Key)+len(msg.Value), cfg.MaxMessageBytes)
			continue
		}

		msg.Header.ProduceTime = now
		partIdx := bal.SelectPartition(topicName, msg.Key, numParts)
		results[i].Partition = partIdx

		if bp != nil && bp.Check(partIdx) == BPClosed {
			results[i].Err = ErrThrottled
			continue
		}
		groups[partIdx] = append(groups[partIdx], indexed{msg, i})
	}

	appendPartitionBatch := func(partIdx int, batch []indexed) {
		p, err := t.Partition(partIdx)
		if err != nil {
			for _, item := range batch {
				results[item.idx].Err = err
			}
			return
		}

		batchMsgs := make([]*Message, len(batch))
		for j, item := range batch {
			batchMsgs[j] = item.msg
		}

		var offsets []uint64
		if useBatchCompression && len(batchMsgs) > 1 {
			// Batch compression: the entire batch is compressed as a single block.
			// Individual message Values remain uncompressed in the WAL payload;
			// ReadFromBatchData decompresses the whole block on read.
			offsets, err = p.AppendBatchCompressed(batchMsgs, codec)
		} else {
			if cfg.CompressionType != CompressionNone {
				for _, msg := range batchMsgs {
					compressed, cErr := codec.Encode(msg.Value)
					if cErr != nil {
						err = oops.Wrapf(cErr, "compress message value")
						break
					}
					msg.Value = compressed
				}
			}
			if err == nil {
				offsets, err = p.AppendBatch(batchMsgs)
			}
		}

		for j, item := range batch {
			if err != nil {
				results[item.idx].Err = err
			} else {
				results[item.idx].Offset = offsets[j]
			}
		}
		b.metrics.RecordProduceBatch(topicName, partIdx, len(batch))
	}

	if len(groups) == 1 {
		for partIdx, batch := range groups {
			appendPartitionBatch(partIdx, batch)
		}
		return results
	}

	var eg errgroup.Group
	for partIdx, batch := range groups {
		eg.Go(func() error {
			gate := snap.gate
			if err := gate.sem.Acquire(ctx, 1); err != nil {
				for _, item := range batch {
					results[item.idx].Err = err
				}
				return nil
			}
			defer gate.sem.Release(1)

			appendPartitionBatch(partIdx, batch)
			return nil
		})
	}
	_ = eg.Wait()

	return results
}

func (b *Broker) Consume(ctx context.Context, topicName, group string, partIdx int) (*Message, uint64, error) {
	var consumerOffset uint64
	if b.offsetStore != nil {
		off, err := b.offsetStore.Load(group, topicName, partIdx)
		if err != nil {
			consumerOffset = 1
		} else {
			consumerOffset = uint64(off)
		}
	}
	return b.ConsumeFrom(ctx, topicName, group, partIdx, consumerOffset)
}

// ConsumeFrom reads a message starting from the given offset. Used by the
// transport layer's batch-consume loop which tracks offsets locally and commits
// once at the end of the batch, rather than after every message.
func (b *Broker) ConsumeFrom(ctx context.Context, topicName, group string, partIdx int, offset uint64) (*Message, uint64, error) {
	t, exists := b.topics.Load(topicName)
	sched, _ := b.schedulers.Load(topicName)

	if !exists {
		return nil, 0, oops.Errorf("topic %q not found", topicName)
	}

	p, err := t.Partition(partIdx)
	if err != nil {
		return nil, 0, oops.Wrapf(err, "get partition %d for topic %q", partIdx, topicName)
	}

	if sched == nil {
		return nil, 0, oops.Errorf("no scheduler configured for topic %q", topicName)
	}

	msg, nextOffset, err := sched.Next(p, offset)
	if err != nil {
		return nil, 0, oops.Wrapf(err, "scheduler.Next topic=%q part=%d", topicName, partIdx)
	}

	if err := b.decompressMessage(msg); err != nil {
		return nil, 0, oops.Wrapf(err, "decompress message topic=%q part=%d", topicName, partIdx)
	}

	b.metrics.RecordConsume(topicName, partIdx, group)

	return msg, nextOffset, nil
}

func (b *Broker) ConsumeBatch(ctx context.Context, topicName, group string, partIdx int, maxMessages int) ([]*Message, uint64, error) {
	var msgs []*Message
	var lastOffset uint64

	// Load offset once; advance locally in the loop.
	var currentOffset uint64
	if b.offsetStore != nil {
		off, err := b.offsetStore.Load(group, topicName, partIdx)
		if err != nil {
			currentOffset = 1
		} else {
			currentOffset = uint64(off)
		}
	}

	for i := 0; i < maxMessages; i++ {
		msg, nextOff, err := b.ConsumeFrom(ctx, topicName, group, partIdx, currentOffset)
		if err != nil {
			if errors.Is(err, ErrNoMessages) {
				break
			}
			if len(msgs) > 0 {
				break
			}
			return nil, 0, err
		}
		msgs = append(msgs, msg)
		lastOffset = nextOff
		currentOffset = nextOff
	}

	if len(msgs) == 0 {
		return nil, 0, ErrNoMessages
	}

	if err := b.Commit(group, topicName, partIdx, int64(lastOffset)); err != nil {
		return msgs, lastOffset, oops.Wrapf(err, "commit offset topic=%q part=%d", topicName, partIdx)
	}
	return msgs, lastOffset, nil
}

// LoadOffset returns the committed offset for the given consumer group.
func (b *Broker) LoadOffset(group, topicName string, partIdx int) (uint64, error) {
	if b.offsetStore == nil {
		return 1, nil
	}
	off, err := b.offsetStore.Load(group, topicName, partIdx)
	if err != nil {
		return 1, oops.Wrapf(err, "load offset group=%q topic=%q part=%d", group, topicName, partIdx)
	}
	return uint64(off), nil
}

func (b *Broker) Commit(group, topicName string, partIdx int, offset int64) error {
	if b.offsetStore == nil {
		return nil
	}
	return b.offsetStore.Commit(group, topicName, partIdx, offset)
}

func (b *Broker) Metrics() Metrics {
	return b.metrics.Snapshot()
}

func (b *Broker) MetricsCollector() *MetricsCollector {
	return b.metrics
}

func (b *Broker) Config() Config {
	return b.snap.Load().cfg
}

func (b *Broker) ApplyConfig(cfg Config) error {
	if err := cfg.Validate(); err != nil {
		return oops.Wrapf(err, "validate config")
	}

	old := b.snap.Load()

	// Idempotent: if config is structurally equal, skip all re-initialization.
	if old.cfg == cfg {
		return nil
	}

	newSnap := &configSnapshot{
		cfg:   cfg,
		codec: old.codec,
		gate:  old.gate,
	}

	if cfg.CompressionType != old.cfg.CompressionType {
		newSnap.codec = NewCodec(cfg.CompressionType)
	}

	if cfg.NumIOGoroutines != old.cfg.NumIOGoroutines {
		newSize := int64(cfg.NumIOGoroutines)
		if newSize <= 0 {
			newSize = 4
		}
		newSnap.gate = &ioGate{sem: semaphore.NewWeighted(newSize), cap: newSize}
	}

	b.snap.Store(newSnap)
	return nil
}

func (b *Broker) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	b.cancel = cancel
	b.eg, _ = errgroup.WithContext(ctx)

	b.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[broker] metricsLoop panic: %v", r)
			}
		}()
		b.metricsLoop(ctx)
		return nil
	})
	b.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[broker] retentionLoop panic: %v", r)
			}
		}()
		b.retentionLoop(ctx)
		return nil
	})
	b.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[broker] sessionTimeoutLoop panic: %v", r)
			}
		}()
		b.sessionTimeoutLoop(ctx)
		return nil
	})
}

func (b *Broker) metricsLoop(ctx context.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	const warmupTicks = 50
	const historyCapacity = 100

	for {
		select {
		case <-ticker.C:
			snap := b.metrics.Collect()

			if b.predictor != nil && len(snap.PartitionLoads) > 0 {
				b.predictor.Update(snap.PartitionLoads)
				snap.PredictedLoads = b.predictor.PredictAll(8)
				b.metrics.lastSnap.Store(&snap)
			}

			b.mu.RLock()
			bal := b.balancer
			bp := b.backpressure
			b.mu.RUnlock()

			if bal != nil {
				bal.OnMetrics(snap)
			}
			if bp != nil {
				bp.UpdatePredictions(snap.PredictedLoads)

				var avgPred float64
				if len(snap.PredictedLoads) > 0 {
					for _, pl := range snap.PredictedLoads {
						avgPred += pl
					}
					avgPred /= float64(len(snap.PredictedLoads))
				}

			var maxFlushNs int64
			var sumFlushNs int64
			var partCount int64
			b.topics.Range(func(_ string, t *Topic) bool {
				for _, p := range t.Partitions() {
					flushNs := int64(p.LastFlushLatency())
					if flushNs > maxFlushNs {
						maxFlushNs = flushNs
					}
					sumFlushNs += flushNs
					partCount++
				}
				return true
			})
				if partCount > 0 {
					b.metrics.avgFlushLatencyNs.Store(sumFlushNs / partCount)
				}
				maxFlushMs := float64(maxFlushNs) / float64(time.Millisecond)
				bp.UpdateSystem(SystemSignal{AvgPredictedLoad: avgPred, WALFlushLatencyMs: maxFlushMs})
			}

			b.metrics.historyMu.Lock()
			b.metrics.throughputHistory = append(b.metrics.throughputHistory, snap.Throughput)
			if len(b.metrics.throughputHistory) > historyCapacity {
				b.metrics.throughputHistory = b.metrics.throughputHistory[1:]
			}
			th := b.metrics.throughputHistory
			b.metrics.historyMu.Unlock()

			if len(th) >= warmupTicks {
				if setter, ok := bal.(BaseThroughputSetter); ok {
					baseWindow := th[:len(th)*4/5]
					sum := 0.0
					for _, v := range baseWindow {
						sum += v
					}
					baseTP := sum / float64(len(baseWindow))
					setter.SetBaseThroughput(baseTP)
				}
			}
		case <-ctx.Done():
			return
		}
	}
}

func (b *Broker) retentionLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cfg := &b.snap.Load().cfg

			var topics []*Topic
			b.topics.Range(func(_ string, t *Topic) bool {
				topics = append(topics, t)
				return true
			})

			for _, t := range topics {
				if t.IsClosed() {
					continue
				}
				for _, p := range t.Partitions() {
					if cfg.RetentionMaxAge > 0 {
						cutoff := time.Now().Add(-cfg.RetentionMaxAge)
						segs, bytes, _ := p.TruncateBefore(cutoff)
						if segs > 0 {
							b.metrics.RecordRetention(segs, bytes)
						}
					}

					if cfg.RetentionMaxBytes > 0 {
						entries, bytes, _ := p.TruncateOldestUntilSize(cfg.RetentionMaxBytes)
						if entries > 0 {
							b.metrics.RecordRetention(entries, bytes)
						}
					}

					// Commit-floor fallback: only when no time/size policy is set.
					if cfg.RetentionMaxAge == 0 && cfg.RetentionMaxBytes == 0 && b.offsetStore != nil {
						floor, err := b.offsetStore.CommitFloor(t.Name(), p.ID())
						if err == nil && floor > 1 {
							_ = p.TruncateFront(uint64(floor))
						}
					}
				}
			}
		case <-ctx.Done():
			return
		}
	}
}

func (b *Broker) sessionTimeoutLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cfg := &b.snap.Load().cfg
			if cfg.ConsumerSessionTimeoutMs <= 0 || b.offsetStore == nil {
				continue
			}
			b.offsetStore.ExpireSessions(time.Duration(cfg.ConsumerSessionTimeoutMs) * time.Millisecond)
		case <-ctx.Done():
			return
		}
	}
}

type Stopper interface {
	Stop()
}

func (b *Broker) Stop() {
	if b.cancel != nil {
		b.cancel()
	}
	if b.eg != nil {
		_ = b.eg.Wait()
	}

	b.mu.Lock()
	if s, ok := b.balancer.(Stopper); ok {
		s.Stop()
	}
	b.mu.Unlock()

	b.schedulers.Range(func(name string, _ Scheduler) bool {
		b.stopScheduler(name)
		return true
	})

	b.topics.Range(func(_ string, t *Topic) bool {
		if err := t.Close(); err != nil {
			log.Printf("error closing topic %s: %v", t.Name(), err)
		}
		return true
	})

	if b.offsetStore != nil {
		b.offsetStore.Close()
	}
}

var (
	ErrThrottled       = errors.New("backpressure: partition overloaded, try again later")
	ErrMessageTooLarge = errors.New("message exceeds MaxMessageBytes")
)
