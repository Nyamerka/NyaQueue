package experiments

import (
	"math/rand/v2"
	"slices"
	"time"

	hdrhistogram "github.com/HdrHistogram/hdrhistogram-go"
	"github.com/puzpuzpuz/xsync/v3"
	"gonum.org/v1/gonum/stat"
)

// maxPrioritySamples caps how many latency samples are kept per priority level.
const maxPrioritySamples = 10_000

// prioritySampler holds a reservoir of latency samples for one priority level.
type prioritySampler struct {
	mu      xsync.RBMutex
	samples []time.Duration
	total   int64 // total observations, including those not in reservoir
}

func (s *prioritySampler) record(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.total++
	n := s.total
	if n <= maxPrioritySamples {
		s.samples = append(s.samples, d)
		return
	}
	// Reservoir sampling: replace a random existing slot with probability
	// maxPrioritySamples/n so every observation has equal chance of inclusion.
	j := rand.Int64N(n)
	if j < maxPrioritySamples {
		s.samples[j] = d
	}
}

func (s *prioritySampler) snapshot() PriorityStats {
	t := s.mu.RLock()
	defer s.mu.RUnlock(t)
	ps := PriorityStats{Count: s.total}
	if len(s.samples) == 0 {
		return ps
	}
	sorted := make([]float64, len(s.samples))
	for i, l := range s.samples {
		sorted[i] = float64(l)
	}
	slices.Sort(sorted)
	ps.P50 = time.Duration(percentile(sorted, 0.50))
	ps.P99 = time.Duration(percentile(sorted, 0.99))
	return ps
}

type PriorityStats struct {
	P50   time.Duration `json:"p50_ns"`
	P99   time.Duration `json:"p99_ns"`
	Count int64         `json:"count"`
}

const defaultNumShards = 16

// latencyShard holds a per-partition HDR histogram, avoiding cross-goroutine contention.
type latencyShard struct {
	mu      xsync.RBMutex
	latency *hdrhistogram.Histogram
}

// MetricsCollector gathers per-message latencies, per-priority latencies,
// partition-load samples and error counters during an experiment run.
// Latency histograms are sharded by consumer partition to minimize contention.
type MetricsCollector struct {
	shards []latencyShard

	mu          xsync.RBMutex
	loadStddevs []float64
	byPriority  [10]prioritySampler

	multiStageMu    xsync.RBMutex
	enqueueToFlush  *hdrhistogram.Histogram
	flushToAppend   *hdrhistogram.Histogram
	appendToConsume *hdrhistogram.Histogram

	produced         *xsync.Counter
	consumed         *xsync.Counter
	publishErrors    *xsync.Counter
	publishThrottled *xsync.Counter
	consumeErrors    *xsync.Counter
	startTime        time.Time
	endTime          time.Time
}

func NewMetricsCollector() *MetricsCollector {
	shards := make([]latencyShard, defaultNumShards)
	for i := range shards {
		shards[i].latency = hdrhistogram.New(1, 30_000_000, 4)
	}
	return &MetricsCollector{
		shards:           shards,
		enqueueToFlush:   hdrhistogram.New(1, 30_000_000, 4),
		flushToAppend:    hdrhistogram.New(1, 30_000_000, 4),
		appendToConsume:  hdrhistogram.New(1, 30_000_000, 4),
		produced:         xsync.NewCounter(),
		consumed:         xsync.NewCounter(),
		publishErrors:    xsync.NewCounter(),
		publishThrottled: xsync.NewCounter(),
		consumeErrors:    xsync.NewCounter(),
	}
}

func (c *MetricsCollector) Start() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.startTime = time.Now()
}

func (c *MetricsCollector) Stop() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.endTime = time.Now()
}

// Reset clears latency histograms and per-priority samples so that
// reconnection downtime latency does not pollute P99 measurements.
// Throughput counters are intentionally preserved.
func (c *MetricsCollector) Reset() {
	for i := range c.shards {
		c.shards[i].mu.Lock()
		c.shards[i].latency.Reset()
		c.shards[i].mu.Unlock()
	}

	for i := range c.byPriority {
		c.byPriority[i].mu.Lock()
		c.byPriority[i].samples = c.byPriority[i].samples[:0]
		c.byPriority[i].total = 0
		c.byPriority[i].mu.Unlock()
	}

	c.multiStageMu.Lock()
	c.enqueueToFlush.Reset()
	c.flushToAppend.Reset()
	c.appendToConsume.Reset()
	c.multiStageMu.Unlock()

	c.mu.Lock()
	c.loadStddevs = c.loadStddevs[:0]
	c.mu.Unlock()
}

func (c *MetricsCollector) RecordProduce() {
	c.produced.Add(1)
}

// RecordConsume records aggregate latency (no priority tracking).
// Use RecordConsumeWithPriority when the priority level is known.
func (c *MetricsCollector) RecordConsume(latency time.Duration) {
	c.RecordConsumeWithPriority(0, latency)
}

// RecordConsumeWithPriority records both the aggregate latency and per-priority
// latency for the given message. priority must be in [0, 9].
func (c *MetricsCollector) RecordConsumeWithPriority(priority uint8, latency time.Duration) {
	c.consumed.Add(1)
	us := latency.Microseconds()
	if us < 1 {
		us = 1
	}

	shard := &c.shards[int(priority)%len(c.shards)]
	shard.mu.Lock()
	shard.latency.RecordValue(us)
	shard.mu.Unlock()

	if int(priority) < len(c.byPriority) {
		c.byPriority[priority].record(latency)
	}
}

// RecordConsumeMultiStage records the breakdown of latency phases.
// enqueueTime: client timestamp, produceTime: broker receive, appendTime: WAL write.
func (c *MetricsCollector) RecordConsumeMultiStage(enqueueTime, produceTime, appendTime int64) {
	now := time.Now().UnixNano()
	if produceTime > 0 && appendTime > 0 {
		c.multiStageMu.Lock()
		if produceTime > enqueueTime {
			us := (produceTime - enqueueTime) / 1000
			if us < 1 {
				us = 1
			}
			c.enqueueToFlush.RecordValue(us)
		}
		if appendTime > produceTime {
			us := (appendTime - produceTime) / 1000
			if us < 1 {
				us = 1
			}
			c.flushToAppend.RecordValue(us)
		}
		if now > appendTime {
			us := (now - appendTime) / 1000
			if us < 1 {
				us = 1
			}
			c.appendToConsume.RecordValue(us)
		}
		c.multiStageMu.Unlock()
	}
}

func (c *MetricsCollector) RecordPublishError() {
	c.publishErrors.Add(1)
}

func (c *MetricsCollector) RecordPublishThrottled() {
	c.publishThrottled.Add(1)
}

func (c *MetricsCollector) RecordConsumeError() {
	c.consumeErrors.Add(1)
}

const loadStdDevGracePeriod = 5 * time.Second

func (c *MetricsCollector) RecordLoadSample(loads []float64) {
	if len(loads) < 2 {
		return
	}
	rt := c.mu.RLock()
	start := c.startTime
	c.mu.RUnlock(rt)
	if time.Since(start) < loadStdDevGracePeriod {
		return
	}
	stddev := stat.StdDev(loads, nil)
	c.mu.Lock()
	c.loadStddevs = append(c.loadStddevs, stddev)
	c.mu.Unlock()
}

func (c *MetricsCollector) RecordLoadStdDev(sd float64) {
	rt := c.mu.RLock()
	start := c.startTime
	c.mu.RUnlock(rt)
	if time.Since(start) < loadStdDevGracePeriod {
		return
	}
	c.mu.Lock()
	c.loadStddevs = append(c.loadStddevs, sd)
	c.mu.Unlock()
}

// ExperimentResult holds the aggregated metrics for a single experiment run.
type ExperimentResult struct {
	Scenario          string        `json:"scenario"`
	Algorithm         string        `json:"algorithm"`
	System            string        `json:"system"`
	Mode              string        `json:"mode"`
	Throughput        float64       `json:"throughput_msg_per_sec"`
	LatencyP50        time.Duration `json:"latency_p50_ns"`
	LatencyP95        time.Duration `json:"latency_p95_ns"`
	LatencyP99        time.Duration `json:"latency_p99_ns"`
	LoadStdDev        float64       `json:"load_stddev"`
	Produced          int64         `json:"produced"`
	Consumed          int64         `json:"consumed"`
	PublishErrors     int64         `json:"publish_errors"`
	MessagesThrottled int64         `json:"messages_throttled"`
	ConsumeErrors     int64         `json:"consume_errors"`
	Duration          time.Duration `json:"duration_ns"`

	LatencyEnqueueToFlushP50  time.Duration `json:"latency_enqueue_to_flush_p50_ns"`
	LatencyFlushToAppendP50   time.Duration `json:"latency_flush_to_append_p50_ns"`
	LatencyAppendToConsumeP50 time.Duration `json:"latency_append_to_consume_p50_ns"`
	LatencyAppendToConsumeP95 time.Duration `json:"latency_append_to_consume_p95_ns"`

	// LatencyByPriority breaks down latency per priority level (0=highest,
	// 9=lowest). Meaningful only when the scenario uses mixed priorities and
	// the system under test supports priority scheduling.
	LatencyByPriority [10]PriorityStats `json:"latency_by_priority"`
}

func (c *MetricsCollector) Snapshot(scenario, algorithm, system, mode string) ExperimentResult {
	rt := c.mu.RLock()

	duration := c.endTime.Sub(c.startTime)
	if duration <= 0 {
		duration = time.Since(c.startTime)
	}

	consumed := c.consumed.Value()
	result := ExperimentResult{
		Scenario:          scenario,
		Algorithm:         algorithm,
		System:            system,
		Mode:              mode,
		Produced:          c.produced.Value(),
		Consumed:          consumed,
		PublishErrors:     c.publishErrors.Value(),
		MessagesThrottled: c.publishThrottled.Value(),
		ConsumeErrors:     c.consumeErrors.Value(),
		Duration:          duration,
	}

	if duration.Seconds() > 0 {
		result.Throughput = float64(consumed) / duration.Seconds()
	}

	if len(c.loadStddevs) > 0 {
		result.LoadStdDev = stat.Mean(c.loadStddevs, nil)
	}

	c.mu.RUnlock(rt)

	merged := hdrhistogram.New(1, 30_000_000, 4)
	for i := range c.shards {
		st := c.shards[i].mu.RLock()
		merged.Merge(c.shards[i].latency)
		c.shards[i].mu.RUnlock(st)
	}

	if merged.TotalCount() > 0 {
		result.LatencyP50 = time.Duration(merged.ValueAtQuantile(50)) * time.Microsecond
		result.LatencyP95 = time.Duration(merged.ValueAtQuantile(95)) * time.Microsecond
		result.LatencyP99 = time.Duration(merged.ValueAtQuantile(99)) * time.Microsecond
	}

	mt := c.multiStageMu.RLock()
	if c.enqueueToFlush.TotalCount() > 0 {
		result.LatencyEnqueueToFlushP50 = time.Duration(c.enqueueToFlush.ValueAtQuantile(50)) * time.Microsecond
	}
	if c.flushToAppend.TotalCount() > 0 {
		result.LatencyFlushToAppendP50 = time.Duration(c.flushToAppend.ValueAtQuantile(50)) * time.Microsecond
	}
	if c.appendToConsume.TotalCount() > 0 {
		result.LatencyAppendToConsumeP50 = time.Duration(c.appendToConsume.ValueAtQuantile(50)) * time.Microsecond
		result.LatencyAppendToConsumeP95 = time.Duration(c.appendToConsume.ValueAtQuantile(95)) * time.Microsecond
	}
	c.multiStageMu.RUnlock(mt)

	for i := range result.LatencyByPriority {
		result.LatencyByPriority[i] = c.byPriority[i].snapshot()
	}

	return result
}

func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 || p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[n-1]
	}

	pos := p * float64(n-1)
	i := int(pos)
	frac := pos - float64(i)
	return sorted[i]*(1-frac) + sorted[i+1]*frac
}
