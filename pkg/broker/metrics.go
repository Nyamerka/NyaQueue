package broker

import (
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"gonum.org/v1/gonum/stat"
)

// BusinessMetrics holds raw counters-derived metrics.
type BusinessMetrics struct {
	Throughput  float64
	AvgLatency  float64
	SuccessRate float64
	MsgRate     float64
	AvgMsgSize  float64
}

// DerivedMetrics holds pre-computed analytical metrics.
type DerivedMetrics struct {
	PartitionLoads []float64
	PredictedLoads []float64
	QueueDepth     []int
	LoadStdDev     float64
}

// Metrics is the unified snapshot exposed to balancers, transport, and experiments.
type Metrics struct {
	BusinessMetrics
	DerivedMetrics
	Timestamp time.Time
}

type MetricsCollector struct {
	broker *Broker

	mu           sync.Mutex
	produceCount int64
	consumeCount int64
	produceBytes int64
	lastSnap     Metrics
	lastCollect  time.Time

	partProduces []atomic.Int64
	partConsumes []atomic.Int64

	prevPartProduces []int64
	prevPartConsumes []int64

	historyMu         sync.Mutex
	throughputHistory []float64
	loadStdDevHistory []float64

	retentionDeletedBytes    atomic.Int64
	retentionDeletedSegments atomic.Int64

	promRegistered bool
	promProduce    *prometheus.CounterVec
	promConsume    *prometheus.CounterVec
	promLoad       *prometheus.GaugeVec
	promDepth      *prometheus.GaugeVec
	promThroughput prometheus.Gauge
	promLoadStdDev prometheus.Gauge
	promRetBytes   prometheus.Counter
	promRetSegs    prometheus.Counter
	promBPClosed   prometheus.Counter
	promBPWarn     prometheus.Counter
}

func NewMetricsCollector(b *Broker) *MetricsCollector {
	return &MetricsCollector{
		broker:      b,
		lastCollect: time.Now(),
	}
}

func (mc *MetricsCollector) RecordProduce(topic string, partition int) {
	atomic.AddInt64(&mc.produceCount, 1)
	mc.mu.Lock()
	mc.ensurePartCountersLocked(partition + 1)
	mc.partProduces[partition].Add(1)
	mc.mu.Unlock()
	mc.RecordProduceLabeled(topic, partition)
}

func (mc *MetricsCollector) RecordProduceWithSize(_ string, partition int, msgBytes int) {
	atomic.AddInt64(&mc.produceCount, 1)
	atomic.AddInt64(&mc.produceBytes, int64(msgBytes))
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

func (mc *MetricsCollector) RecordConsume(topic string, partition int, group string) {
	atomic.AddInt64(&mc.consumeCount, 1)
	mc.mu.Lock()
	mc.ensurePartCountersLocked(partition + 1)
	mc.partConsumes[partition].Add(1)
	mc.mu.Unlock()
	mc.RecordConsumeLabeled(topic, partition, group)
}

func (mc *MetricsCollector) ensurePartCountersLocked(minLen int) {
	for len(mc.partProduces) < minLen {
		mc.partProduces = append(mc.partProduces, atomic.Int64{})
		mc.partConsumes = append(mc.partConsumes, atomic.Int64{})
	}
}

func (mc *MetricsCollector) savePrevCountersLocked() {
	n := len(mc.partProduces)
	if len(mc.prevPartProduces) < n {
		mc.prevPartProduces = make([]int64, n)
		mc.prevPartConsumes = make([]int64, n)
	}
	for i := 0; i < n; i++ {
		mc.prevPartProduces[i] = mc.partProduces[i].Load()
		mc.prevPartConsumes[i] = mc.partConsumes[i].Load()
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

			var prevProd, prevCons int64
			if pID < len(mc.prevPartProduces) {
				prevProd = mc.prevPartProduces[pID]
				prevCons = mc.prevPartConsumes[pID]
			}
			mc.mu.Unlock()

			pending := produces - consumes
			if pending < 0 {
				pending = 0
			}
			depths = append(depths, int(pending))

			deltaProd := produces - prevProd
			deltaCons := consumes - prevCons
			load := 0.0
			if deltaProd > 0 {
				deltaPending := deltaProd - deltaCons
				if deltaPending < 0 {
					deltaPending = 0
				}
				load = float64(deltaPending) / float64(deltaProd)
			}
			loads = append(loads, load)
		}
	}

	mc.mu.Lock()
	mc.savePrevCountersLocked()
	mc.mu.Unlock()

	successRate := 1.0
	if produced > 0 && consumed == 0 {
		successRate = 0.0
	}

	producedBytes := atomic.LoadInt64(&mc.produceBytes)
	atomic.StoreInt64(&mc.produceBytes, 0)

	msgRate := float64(produced) / elapsed
	avgMsgSize := 0.0
	if produced > 0 {
		avgMsgSize = float64(producedBytes) / float64(produced)
	}

	var loadSD float64
	if len(loads) >= 2 {
		sd := stat.StdDev(loads, nil)
		mc.historyMu.Lock()
		mc.loadStdDevHistory = append(mc.loadStdDevHistory, sd)
		if len(mc.loadStdDevHistory) > 100 {
			mc.loadStdDevHistory = mc.loadStdDevHistory[1:]
		}
		loadSD = stat.Mean(mc.loadStdDevHistory, nil)
		mc.historyMu.Unlock()
	}

	snap := Metrics{
		BusinessMetrics: BusinessMetrics{
			Throughput:  throughput,
			SuccessRate: successRate,
			MsgRate:     msgRate,
			AvgMsgSize:  avgMsgSize,
		},
		DerivedMetrics: DerivedMetrics{
			PartitionLoads: loads,
			QueueDepth:     depths,
			LoadStdDev:     loadSD,
		},
		Timestamp: now,
	}

	mc.mu.Lock()
	mc.lastSnap = snap
	mc.mu.Unlock()

	mc.updatePrometheus(snap)

	return snap
}

func (mc *MetricsCollector) RecordRetention(segments int, bytes int64) {
	mc.retentionDeletedSegments.Add(int64(segments))
	mc.retentionDeletedBytes.Add(bytes)
}

func (mc *MetricsCollector) Snapshot() Metrics {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	return mc.lastSnap
}

func (mc *MetricsCollector) RetentionDeletedBytes() int64 {
	return mc.retentionDeletedBytes.Load()
}

func (mc *MetricsCollector) RetentionDeletedSegments() int64 {
	return mc.retentionDeletedSegments.Load()
}

// RegisterPrometheus registers labeled Prometheus metrics with the given registry.
// Safe to call once; subsequent calls are no-ops.
func (mc *MetricsCollector) RegisterPrometheus(reg prometheus.Registerer) {
	if mc.promRegistered {
		return
	}
	mc.promRegistered = true
	f := promauto.With(reg)

	mc.promProduce = f.NewCounterVec(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "produced_total",
		Help:      "Total produced messages.",
	}, []string{"topic", "partition"})

	mc.promConsume = f.NewCounterVec(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "consumed_total",
		Help:      "Total consumed messages.",
	}, []string{"topic", "partition", "group"})

	mc.promLoad = f.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "partition_load",
		Help:      "Normalized partition load [0,1].",
	}, []string{"topic", "partition"})

	mc.promDepth = f.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "queue_depth",
		Help:      "Pending messages per partition.",
	}, []string{"topic", "partition"})

	mc.promThroughput = f.NewGauge(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "throughput_messages_per_sec",
		Help:      "Messages per second across all partitions.",
	})

	mc.promLoadStdDev = f.NewGauge(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "partition_load_stddev",
		Help:      "Standard deviation of partition load.",
	})

	mc.promRetBytes = f.NewCounter(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "retention_deleted_bytes_total",
		Help:      "Total bytes deleted by retention.",
	})

	mc.promRetSegs = f.NewCounter(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "retention_deleted_segments_total",
		Help:      "Total segments deleted by retention.",
	})

	mc.promBPClosed = f.NewCounter(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "backpressure_closed_total",
		Help:      "Backpressure closed activations.",
	})

	mc.promBPWarn = f.NewCounter(prometheus.CounterOpts{
		Namespace: "nyaqueue",
		Name:      "backpressure_warn_total",
		Help:      "Backpressure warn activations.",
	})
}

// updatePrometheus pushes the latest snapshot to Prometheus gauges.
func (mc *MetricsCollector) updatePrometheus(snap Metrics) {
	if !mc.promRegistered {
		return
	}

	mc.promThroughput.Set(snap.Throughput)
	mc.promLoadStdDev.Set(snap.LoadStdDev)

	mc.broker.mu.RLock()
	topics := make([]*Topic, 0, len(mc.broker.topics))
	for _, t := range mc.broker.topics {
		topics = append(topics, t)
	}
	mc.broker.mu.RUnlock()

	partIdx := 0
	for _, t := range topics {
		for _, p := range t.Partitions() {
			pStr := strconv.Itoa(p.ID())
			if partIdx < len(snap.PartitionLoads) {
				mc.promLoad.WithLabelValues(t.Name(), pStr).Set(snap.PartitionLoads[partIdx])
			}
			if partIdx < len(snap.QueueDepth) {
				mc.promDepth.WithLabelValues(t.Name(), pStr).Set(float64(snap.QueueDepth[partIdx]))
			}
			partIdx++
		}
	}
}

// RecordProduceLabeled records a produce for Prometheus with topic/partition labels.
func (mc *MetricsCollector) RecordProduceLabeled(topic string, partition int) {
	if mc.promRegistered && mc.promProduce != nil {
		mc.promProduce.WithLabelValues(topic, strconv.Itoa(partition)).Inc()
	}
}

// RecordConsumeLabeled records a consume for Prometheus with topic/partition/group labels.
func (mc *MetricsCollector) RecordConsumeLabeled(topic string, partition int, group string) {
	if mc.promRegistered && mc.promConsume != nil {
		mc.promConsume.WithLabelValues(topic, strconv.Itoa(partition), group).Inc()
	}
}
