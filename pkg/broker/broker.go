package broker

import (
	"errors"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/samber/oops"
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

type Scheduler interface {
	Next(partition *Partition, consumerOffset uint64) (*Message, uint64, error)
	Enqueue(msg *Message, walOffset int64)
}

type Broker struct {
	mu         sync.RWMutex
	config     Config
	dataDir    string
	topics     map[string]*Topic
	balancer   Balancer
	schedulers map[string]Scheduler

	schedulerFactory func(TopicConfig) Scheduler

	metrics      *MetricsCollector
	backpressure *BackpressureController
	offsetStore  *OffsetStore

	ioPool chan struct{}

	stopCh chan struct{}
}

func New(cfg Config, dataDir string, bal Balancer, offsetStore *OffsetStore) *Broker {
	ioPoolSize := cfg.NumIOGoroutines
	if ioPoolSize <= 0 {
		ioPoolSize = 4
	}

	b := &Broker{
		config:      cfg,
		dataDir:     dataDir,
		topics:      make(map[string]*Topic),
		balancer:    bal,
		schedulers:  make(map[string]Scheduler),
		offsetStore: offsetStore,
		ioPool:      make(chan struct{}, ioPoolSize),
		stopCh:      make(chan struct{}),
	}

	b.metrics = NewMetricsCollector(b)
	b.backpressure = NewBackpressureController(nil, 0.85, 3)

	return b
}

func (b *Broker) SetBalancer(bal Balancer) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.balancer = bal
}

func (b *Broker) SetScheduler(topic string, sched Scheduler) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.schedulers[topic] = sched
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

	if _, exists := b.topics[name]; exists {
		return oops.Wrapf(ErrTopicAlreadyExists, "create topic %q", name)
	}

	t, err := NewTopic(name, b.dataDir, cfg, b.config.SyncPolicy)
	if err != nil {
		return err
	}
	b.topics[name] = t

	if b.schedulerFactory != nil {
		b.schedulers[name] = b.schedulerFactory(cfg)
	}

	return nil
}

func (b *Broker) DeleteTopic(name string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	t, exists := b.topics[name]
	if !exists {
		return oops.Wrapf(ErrTopicNotFound, "delete topic %q", name)
	}
	delete(b.topics, name)
	delete(b.schedulers, name)

	if err := t.Close(); err != nil {
		return err
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
	b.mu.RLock()
	defer b.mu.RUnlock()

	t, exists := b.topics[name]
	if !exists {
		return nil, oops.Errorf("topic %q not found", name)
	}
	return t, nil
}

func (b *Broker) ListTopics() []*Topic {
	b.mu.RLock()
	defer b.mu.RUnlock()

	topics := make([]*Topic, 0, len(b.topics))
	for _, t := range b.topics {
		topics = append(topics, t)
	}
	return topics
}

func (b *Broker) compressMessage(msg *Message) error {
	if b.config.CompressionType == CompressionNone {
		return nil
	}
	compressed, err := Compress(msg.Value, b.config.CompressionType)
	if err != nil {
		return oops.Wrapf(err, "compress message value")
	}
	msg.Value = compressed
	return nil
}

func (b *Broker) decompressMessage(msg *Message) error {
	if b.config.CompressionType == CompressionNone {
		return nil
	}
	decompressed, err := Decompress(msg.Value, b.config.CompressionType)
	if err != nil {
		return oops.Wrapf(err, "decompress message value")
	}
	msg.Value = decompressed
	return nil
}

func (b *Broker) Publish(topicName string, msg *Message) (partition int, offset uint64, err error) {
	if err := b.checkMessageSize(msg); err != nil {
		return 0, 0, err
	}

	if err := b.compressMessage(msg); err != nil {
		return 0, 0, err
	}

	b.mu.RLock()
	t, exists := b.topics[topicName]
	bal := b.balancer
	bp := b.backpressure
	b.mu.RUnlock()

	if !exists {
		return 0, 0, oops.Errorf("topic %q not found", topicName)
	}

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
		return 0, 0, err
	}

	offset, err = p.Append(msg)
	if err != nil {
		return 0, 0, err
	}

	b.metrics.RecordProduce(topicName, partIdx)

	return partIdx, offset, nil
}

type PublishResult struct {
	Partition int
	Offset    uint64
	Err       error
}

func (b *Broker) PublishBatch(topicName string, msgs []*Message) []PublishResult {
	results := make([]PublishResult, len(msgs))

	b.mu.RLock()
	t, exists := b.topics[topicName]
	bal := b.balancer
	bp := b.backpressure
	b.mu.RUnlock()

	if !exists {
		for i := range results {
			results[i].Err = oops.Errorf("topic %q not found", topicName)
		}
		return results
	}

	numParts := t.NumPartitions()

	type indexed struct {
		msg *Message
		idx int
	}
	groups := make(map[int][]indexed, numParts)

	for i, msg := range msgs {
		if err := b.checkMessageSize(msg); err != nil {
			results[i].Err = err
			continue
		}
		if err := b.compressMessage(msg); err != nil {
			results[i].Err = err
			continue
		}

		partIdx := bal.SelectPartition(topicName, msg.Key, numParts)
		results[i].Partition = partIdx

		if bp != nil && bp.Check(partIdx) == BPClosed {
			results[i].Err = ErrThrottled
			continue
		}
		groups[partIdx] = append(groups[partIdx], indexed{msg, i})
	}

	var wg sync.WaitGroup
	for partIdx, batch := range groups {
		wg.Add(1)
		go func(partIdx int, batch []indexed) {
			defer wg.Done()

			b.ioPool <- struct{}{}
			defer func() { <-b.ioPool }()

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

			offsets, err := p.AppendBatch(batchMsgs)
			for j, item := range batch {
				if err != nil {
					results[item.idx].Err = err
				} else {
					results[item.idx].Offset = offsets[j]
				}
			}

			b.metrics.RecordProduceBatch(topicName, partIdx, len(batch))
		}(partIdx, batch)
	}
	wg.Wait()

	return results
}

func (b *Broker) Consume(topicName, group string, partIdx int) (*Message, uint64, error) {
	b.mu.RLock()
	t, exists := b.topics[topicName]
	sched := b.schedulers[topicName]
	b.mu.RUnlock()

	if !exists {
		return nil, 0, oops.Errorf("topic %q not found", topicName)
	}

	p, err := t.Partition(partIdx)
	if err != nil {
		return nil, 0, err
	}

	var consumerOffset uint64
	if b.offsetStore != nil {
		off, err := b.offsetStore.Load(group, topicName, partIdx)
		if err != nil {
			consumerOffset = 1
		} else {
			consumerOffset = uint64(off)
		}
	}

	if sched == nil {
		return nil, 0, oops.Errorf("no scheduler configured for topic %q", topicName)
	}

	msg, nextOffset, err := sched.Next(p, consumerOffset)
	if err != nil {
		return nil, 0, err
	}

	if err := b.decompressMessage(msg); err != nil {
		return nil, 0, err
	}

	b.metrics.RecordConsume(topicName, partIdx)

	return msg, nextOffset, nil
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

func (b *Broker) Config() Config {
	return b.config
}

func (b *Broker) Start() {
	go b.metricsLoop()
}

func (b *Broker) metricsLoop() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			snap := b.metrics.Collect()

			b.mu.RLock()
			bal := b.balancer
			b.mu.RUnlock()

			bal.OnMetrics(snap)
		case <-b.stopCh:
			return
		}
	}
}

type Stopper interface {
	Stop()
}

func (b *Broker) Stop() {
	close(b.stopCh)

	b.mu.Lock()
	defer b.mu.Unlock()

	if s, ok := b.balancer.(Stopper); ok {
		s.Stop()
	}

	for _, t := range b.topics {
		if err := t.Close(); err != nil {
			log.Printf("error closing topic %s: %v", t.Name(), err)
		}
	}

	if b.offsetStore != nil {
		b.offsetStore.Close()
	}
}

var (
	ErrThrottled       = errors.New("backpressure: partition overloaded, try again later")
	ErrMessageTooLarge = errors.New("message exceeds MaxMessageBytes")
)

func (b *Broker) checkMessageSize(msg *Message) error {
	if b.config.MaxMessageBytes > 0 && len(msg.Key)+len(msg.Value) > b.config.MaxMessageBytes {
		return oops.Wrapf(ErrMessageTooLarge, "size %d > limit %d",
			len(msg.Key)+len(msg.Value), b.config.MaxMessageBytes)
	}
	return nil
}
