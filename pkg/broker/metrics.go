package broker

import (
	"sync"
	"sync/atomic"
	"time"
)

type Metrics struct {
	Throughput     float64
	AvgLatency     float64
	PartitionLoads []float64
	SuccessRate    float64
	QueueDepth     []int
	Timestamp      time.Time
}

type MetricsCollector struct {
	broker *Broker

	mu           sync.Mutex
	produceCount int64
	consumeCount int64
	lastSnap     Metrics
	lastCollect  time.Time

	partProduces []atomic.Int64
	partConsumes []atomic.Int64
}

func NewMetricsCollector(b *Broker) *MetricsCollector {
	return &MetricsCollector{
		broker:      b,
		lastCollect: time.Now(),
	}
}

func (mc *MetricsCollector) RecordProduce(_ string, partition int) {
	atomic.AddInt64(&mc.produceCount, 1)
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.ensurePartCountersLocked(partition + 1)
	mc.partProduces[partition].Add(1)
}

func (mc *MetricsCollector) RecordProduceBatch(_ string, partition int, n int) {
	atomic.AddInt64(&mc.produceCount, int64(n))
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.ensurePartCountersLocked(partition + 1)
	mc.partProduces[partition].Add(int64(n))
}

func (mc *MetricsCollector) RecordConsume(_ string, partition int) {
	atomic.AddInt64(&mc.consumeCount, 1)
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.ensurePartCountersLocked(partition + 1)
	mc.partConsumes[partition].Add(1)
}

func (mc *MetricsCollector) ensurePartCountersLocked(minLen int) {
	for len(mc.partProduces) < minLen {
		mc.partProduces = append(mc.partProduces, atomic.Int64{})
		mc.partConsumes = append(mc.partConsumes, atomic.Int64{})
	}
}

func (mc *MetricsCollector) Collect() Metrics {
	mc.mu.Lock()
	now := time.Now()
	elapsed := now.Sub(mc.lastCollect).Seconds()
	if elapsed < 0.001 {
		elapsed = 0.001
	}

	produced := atomic.LoadInt64(&mc.produceCount)
	consumed := atomic.LoadInt64(&mc.consumeCount)
	mc.lastCollect = now
	mc.mu.Unlock()

	throughput := float64(produced+consumed) / elapsed

	atomic.StoreInt64(&mc.produceCount, 0)
	atomic.StoreInt64(&mc.consumeCount, 0)

	mc.broker.mu.RLock()
	topics := make([]*Topic, 0, len(mc.broker.topics))
	for _, t := range mc.broker.topics {
		topics = append(topics, t)
	}
	mc.broker.mu.RUnlock()

	var loads []float64
	var depths []int
	for _, t := range topics {
		for _, p := range t.Partitions() {
			pID := p.ID()

			mc.mu.Lock()
			var produces, consumes int64
			if pID < len(mc.partProduces) {
				produces = mc.partProduces[pID].Load()
				consumes = mc.partConsumes[pID].Load()
			}
			mc.mu.Unlock()

			pending := produces - consumes
			if pending < 0 {
				pending = 0
			}
			depths = append(depths, int(pending))

			load := 0.0
			if produces > 0 {
				load = float64(pending) / float64(produces)
			}
			loads = append(loads, load)
		}
	}

	successRate := 1.0
	if produced > 0 && consumed == 0 {
		successRate = 0.0
	}

	snap := Metrics{
		Throughput:     throughput,
		PartitionLoads: loads,
		QueueDepth:     depths,
		SuccessRate:    successRate,
		Timestamp:      now,
	}

	mc.mu.Lock()
	mc.lastSnap = snap
	mc.mu.Unlock()

	return snap
}

func (mc *MetricsCollector) Snapshot() Metrics {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	return mc.lastSnap
}
