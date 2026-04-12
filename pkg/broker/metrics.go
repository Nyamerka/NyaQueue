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
	mc.ensurePartCounters(partition + 1)
	mc.partProduces[partition].Add(1)
}

func (mc *MetricsCollector) RecordConsume(_ string, partition int) {
	atomic.AddInt64(&mc.consumeCount, 1)
	mc.ensurePartCounters(partition + 1)
	mc.partConsumes[partition].Add(1)
}

func (mc *MetricsCollector) ensurePartCounters(minLen int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

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
			hwm := float64(p.HighWaterMark())
			load := 0.0
			if hwm > 0 {
				depth := p.QueueDepth()
				load = float64(depth) / hwm
				if load > 1 {
					load = 1
				}
				depths = append(depths, depth)
			} else {
				depths = append(depths, 0)
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
