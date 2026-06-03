package broker

import (
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/puzpuzpuz/xsync/v3"
	"gonum.org/v1/gonum/stat"
)

type BusinessMetrics struct {
	Throughput    float64
	ConsumeRate   float64
	AvgLatency    float64
	DeliveryRatio float64
	MsgRate       float64
	AvgMsgSize    float64
}

type DerivedMetrics struct {
	PartitionLoads []float64
	PredictedLoads []float64
	QueueDepth     []int
	LoadStdDev     float64
}

type ObservabilityMetrics struct {
	BackpressureClosedCount int64 `json:"bp_closed_count"`
	BackpressureWarnCount   int64 `json:"bp_warn_count"`
	BackpressureOpenCount   int64 `json:"bp_open_count"`
	BalancerDroppedExp      int64 `json:"balancer_dropped_exp,omitempty"`
	BalancerEvictions       int64 `json:"balancer_evictions,omitempty"`
}

type Metrics struct {
	BusinessMetrics
	DerivedMetrics
	ObservabilityMetrics
	Timestamp time.Time
}

type partCounters struct {
	produces []atomic.Int64
	consumes []atomic.Int64
}

type MetricsCollector struct {
	broker *Broker

	produceCount atomic.Int64
	consumeCount atomic.Int64
	produceBytes atomic.Int64
	lastSnap     atomic.Pointer[Metrics]

	avgFlushLatencyNs atomic.Int64

	counters atomic.Pointer[partCounters]

	mu          xsync.RBMutex
	lastCollect time.Time

	prevPartProduces []int64
	prevPartConsumes []int64

	historyMu         xsync.RBMutex
	throughputHistory []float64
	loadStdDevHistory []float64

	retentionDeletedBytes    atomic.Int64
	retentionDeletedSegments atomic.Int64

	promRegistered   bool
	promProduce      *prometheus.CounterVec
	promConsume      *prometheus.CounterVec
	promLoad         *prometheus.GaugeVec
	promDepth        *prometheus.GaugeVec
	promThroughput   prometheus.Gauge
	promLoadStdDev   prometheus.Gauge
	promRetBytes     prometheus.Counter
	promRetSegs      prometheus.Counter
	promBPClosed     prometheus.Counter
	promBPWarn       prometheus.Counter
	promBPOpen       prometheus.Gauge
	promBalDropped   prometheus.Gauge
	promBalEvictions prometheus.Gauge
}

func NewMetricsCollector(b *Broker) *MetricsCollector {
	mc := &MetricsCollector{
		broker:      b,
		lastCollect: time.Now(),
	}
	mc.lastSnap.Store(&Metrics{})
	initial := &partCounters{
		produces: make([]atomic.Int64, 16),
		consumes: make([]atomic.Int64, 16),
	}
	mc.counters.Store(initial)
	return mc
}

func (mc *MetricsCollector) getCounters(minPartitions int) *partCounters {
	for {
		pc := mc.counters.Load()
		if len(pc.produces) > minPartitions {
			return pc
		}
		newSize := max(len(pc.produces)*2, minPartitions+1)
		grown := &partCounters{
			produces: make([]atomic.Int64, newSize),
			consumes: make([]atomic.Int64, newSize),
		}
		for i := range pc.produces {
			grown.produces[i].Store(pc.produces[i].Load())
			grown.consumes[i].Store(pc.consumes[i].Load())
		}
		if mc.counters.CompareAndSwap(pc, grown) {
			return grown
		}
	}
}

func (mc *MetricsCollector) RecordProduce(_ string, partition int) {
	mc.produceCount.Add(1)
	pc := mc.getCounters(partition)
	pc.produces[partition].Add(1)
}

func (mc *MetricsCollector) RecordProduceWithSize(_ string, partition int, msgBytes int) {
	mc.produceCount.Add(1)
	mc.produceBytes.Add(int64(msgBytes))
	pc := mc.getCounters(partition)
	pc.produces[partition].Add(1)
}

func (mc *MetricsCollector) RecordProduceBatch(_ string, partition int, n int) {
	mc.produceCount.Add(int64(n))
	pc := mc.getCounters(partition)
	pc.produces[partition].Add(int64(n))
}

func (mc *MetricsCollector) RecordConsume(_ string, partition int, _ string) {
	mc.consumeCount.Add(1)
	pc := mc.getCounters(partition)
	pc.consumes[partition].Add(1)
}

func (mc *MetricsCollector) savePrevCounters() {
	pc := mc.counters.Load()
	n := len(pc.produces)
	mc.mu.Lock()
	if len(mc.prevPartProduces) < n {
		mc.prevPartProduces = make([]int64, n)
		mc.prevPartConsumes = make([]int64, n)
	}
	for i := 0; i < n; i++ {
		mc.prevPartProduces[i] = pc.produces[i].Load()
		mc.prevPartConsumes[i] = pc.consumes[i].Load()
	}
	mc.mu.Unlock()
}

func (mc *MetricsCollector) Collect() Metrics {
	mc.mu.Lock()
	now := time.Now()
	elapsed := now.Sub(mc.lastCollect).Seconds()
	if elapsed < 0.001 {
		elapsed = 0.001
	}
	mc.lastCollect = now

	prevProdSnap := append([]int64(nil), mc.prevPartProduces...)
	prevConsSnap := append([]int64(nil), mc.prevPartConsumes...)
	mc.mu.Unlock()

	produced := mc.produceCount.Swap(0)
	consumed := mc.consumeCount.Swap(0)

	throughput := float64(produced+consumed) / elapsed

	var topics []*Topic
	mc.broker.topics.Range(func(_ string, t *Topic) bool {
		topics = append(topics, t)
		return true
	})

	var loads []float64
	var depths []int
	pc := mc.counters.Load()
	for _, t := range topics {
		for _, p := range t.Partitions() {
			pID := p.ID()

			var produces, consumes int64
			if pID < len(pc.produces) {
				produces = pc.produces[pID].Load()
				consumes = pc.consumes[pID].Load()
			}

			var prevProd, prevCons int64
			if pID < len(prevProdSnap) {
				prevProd = prevProdSnap[pID]
				prevCons = prevConsSnap[pID]
			}

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

	mc.savePrevCounters()

	deliveryRatio := 1.0
	if produced > 0 {
		deliveryRatio = float64(consumed) / float64(produced)
		if deliveryRatio > 1.0 {
			deliveryRatio = 1.0
		}
	}

	producedBytes := mc.produceBytes.Swap(0)

	msgRate := float64(produced) / elapsed
	consumeRate := float64(consumed) / elapsed
	avgMsgSize := 0.0
	if produced > 0 {
		avgMsgSize = float64(producedBytes) / float64(produced)
	}

	avgLatencyMs := float64(mc.avgFlushLatencyNs.Load()) / float64(time.Millisecond)

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

	var obs ObservabilityMetrics
	rt := mc.broker.runtime.Load()
	bp := rt.backpressure
	bal := rt.balancer

	if bp != nil {
		obs.BackpressureClosedCount = bp.ClosedCount()
		obs.BackpressureWarnCount = bp.WarnCount()
		obs.BackpressureOpenCount = bp.OpenCount()
	}
	if de, ok := bal.(interface{ DroppedExperience() int64 }); ok {
		obs.BalancerDroppedExp = de.DroppedExperience()
	}
	if ev, ok := bal.(interface{ EvictionCount() int64 }); ok {
		obs.BalancerEvictions = ev.EvictionCount()
	}

	snap := Metrics{
		BusinessMetrics: BusinessMetrics{
			Throughput:    throughput,
			ConsumeRate:   consumeRate,
			AvgLatency:    avgLatencyMs,
			DeliveryRatio: deliveryRatio,
			MsgRate:       msgRate,
			AvgMsgSize:    avgMsgSize,
		},
		DerivedMetrics: DerivedMetrics{
			PartitionLoads: loads,
			QueueDepth:     depths,
			LoadStdDev:     loadSD,
		},
		ObservabilityMetrics: obs,
		Timestamp:            now,
	}

	mc.lastSnap.Store(&snap)

	mc.updatePrometheus(snap)

	return snap
}

func (mc *MetricsCollector) RecordRetention(segments int, bytes int64) {
	mc.retentionDeletedSegments.Add(int64(segments))
	mc.retentionDeletedBytes.Add(bytes)
}

func (mc *MetricsCollector) Snapshot() Metrics {
	if p := mc.lastSnap.Load(); p != nil {
		return *p
	}
	return Metrics{}
}

func (mc *MetricsCollector) RetentionDeletedBytes() int64 {
	return mc.retentionDeletedBytes.Load()
}

func (mc *MetricsCollector) RetentionDeletedSegments() int64 {
	return mc.retentionDeletedSegments.Load()
}

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

	mc.promBPOpen = f.NewGauge(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "backpressure_open_total",
		Help:      "Backpressure open activations.",
	})

	mc.promBalDropped = f.NewGauge(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "balancer_dropped_experience_total",
		Help:      "DQN balancer dropped experience tuples.",
	})

	mc.promBalEvictions = f.NewGauge(prometheus.GaugeOpts{
		Namespace: "nyaqueue",
		Name:      "balancer_evictions_total",
		Help:      "PSA balancer LRU evictions.",
	})
}

func (mc *MetricsCollector) updatePrometheus(snap Metrics) {
	if !mc.promRegistered {
		return
	}

	mc.promThroughput.Set(snap.Throughput)
	mc.promLoadStdDev.Set(snap.LoadStdDev)

	var promTopics []*Topic
	mc.broker.topics.Range(func(_ string, t *Topic) bool {
		promTopics = append(promTopics, t)
		return true
	})

	pc := mc.counters.Load()
	partIdx := 0
	for _, t := range promTopics {
		for _, p := range t.Partitions() {
			pID := p.ID()
			pStr := strconv.Itoa(pID)
			if partIdx < len(snap.PartitionLoads) {
				mc.promLoad.WithLabelValues(t.Name(), pStr).Set(snap.PartitionLoads[partIdx])
			}
			if partIdx < len(snap.QueueDepth) {
				mc.promDepth.WithLabelValues(t.Name(), pStr).Set(float64(snap.QueueDepth[partIdx]))
			}
			if mc.promProduce != nil && pID < len(pc.produces) {
				mc.promProduce.WithLabelValues(t.Name(), pStr).Add(float64(pc.produces[pID].Load()))
			}
			if mc.promConsume != nil && pID < len(pc.consumes) {
				mc.promConsume.WithLabelValues(t.Name(), pStr, "").Add(float64(pc.consumes[pID].Load()))
			}
			partIdx++
		}
	}

	mc.promBPOpen.Set(float64(snap.BackpressureOpenCount))
	mc.promBalDropped.Set(float64(snap.BalancerDroppedExp))
	mc.promBalEvictions.Set(float64(snap.BalancerEvictions))
}

func (mc *MetricsCollector) RemovePartition(partID int) {
	pc := mc.counters.Load()
	if partID < len(pc.produces) {
		pc.produces[partID].Store(0)
		pc.consumes[partID].Store(0)
	}
	mc.mu.Lock()
	if partID < len(mc.prevPartProduces) {
		mc.prevPartProduces[partID] = 0
		mc.prevPartConsumes[partID] = 0
	}
	mc.mu.Unlock()
}

func (mc *MetricsCollector) RecordProduceLabeled(topic string, partition int) {
	if mc.promRegistered && mc.promProduce != nil {
		mc.promProduce.WithLabelValues(topic, strconv.Itoa(partition)).Inc()
	}
}

func (mc *MetricsCollector) RecordConsumeLabeled(topic string, partition int, group string) {
	if mc.promRegistered && mc.promConsume != nil {
		mc.promConsume.WithLabelValues(topic, strconv.Itoa(partition), group).Inc()
	}
}
