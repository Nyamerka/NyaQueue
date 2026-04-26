package experiments

import (
	"math/rand/v2"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/stat"
)

// maxPrioritySamples caps how many latency samples are kept per priority level.
const maxPrioritySamples = 10_000

// prioritySampler holds a reservoir of latency samples for one priority level.
type prioritySampler struct {
	mu      sync.Mutex
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
	s.mu.Lock()
	defer s.mu.Unlock()
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

// MetricsCollector gathers per-message latencies, per-priority latencies,
// partition-load samples and error counters during an experiment run.
type MetricsCollector struct {
	mu          sync.Mutex
	latencies   []time.Duration
	loadStddevs []float64
	byPriority  [10]prioritySampler

	produced      atomic.Int64
	consumed      atomic.Int64
	publishErrors atomic.Int64
	consumeErrors atomic.Int64
	startTime     time.Time
	endTime       time.Time
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{}
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
	c.mu.Lock()
	c.latencies = append(c.latencies, latency)
	c.mu.Unlock()

	if int(priority) < len(c.byPriority) {
		c.byPriority[priority].record(latency)
	}
}

func (c *MetricsCollector) RecordPublishError() {
	c.publishErrors.Add(1)
}

func (c *MetricsCollector) RecordConsumeError() {
	c.consumeErrors.Add(1)
}

// RecordLoadSample appends one stddev value computed across partitions at a
// single point in time. Snapshot averages these to get the run-level imbalance.
func (c *MetricsCollector) RecordLoadSample(loads []float64) {
	if len(loads) < 2 {
		return
	}
	stddev := stat.StdDev(loads, nil)
	c.mu.Lock()
	c.loadStddevs = append(c.loadStddevs, stddev)
	c.mu.Unlock()
}

// ExperimentResult holds the aggregated metrics for a single experiment run.
type ExperimentResult struct {
	Scenario      string        `json:"scenario"`
	Algorithm     string        `json:"algorithm"`
	System        string        `json:"system"`
	Mode          string        `json:"mode"`
	Throughput    float64       `json:"throughput_msg_per_sec"`
	LatencyP50    time.Duration `json:"latency_p50_ns"`
	LatencyP95    time.Duration `json:"latency_p95_ns"`
	LatencyP99    time.Duration `json:"latency_p99_ns"`
	LoadStdDev    float64       `json:"load_stddev"`
	Produced      int64         `json:"produced"`
	Consumed      int64         `json:"consumed"`
	PublishErrors int64         `json:"publish_errors"`
	ConsumeErrors int64         `json:"consume_errors"`
	Duration      time.Duration `json:"duration_ns"`

	// LatencyByPriority breaks down latency per priority level (0=highest,
	// 9=lowest). Meaningful only when the scenario uses mixed priorities and
	// the system under test supports priority scheduling.
	LatencyByPriority [10]PriorityStats `json:"latency_by_priority"`
}

func (c *MetricsCollector) Snapshot(scenario, algorithm, system, mode string) ExperimentResult {
	c.mu.Lock()

	duration := c.endTime.Sub(c.startTime)
	if duration <= 0 {
		duration = time.Since(c.startTime)
	}

	consumed := c.consumed.Load()
	result := ExperimentResult{
		Scenario:      scenario,
		Algorithm:     algorithm,
		System:        system,
		Mode:          mode,
		Produced:      c.produced.Load(),
		Consumed:      consumed,
		PublishErrors: c.publishErrors.Load(),
		ConsumeErrors: c.consumeErrors.Load(),
		Duration:      duration,
	}

	if duration.Seconds() > 0 {
		result.Throughput = float64(consumed) / duration.Seconds()
	}

	if len(c.latencies) > 0 {
		sorted := make([]float64, len(c.latencies))
		for i, l := range c.latencies {
			sorted[i] = float64(l)
		}
		slices.Sort(sorted)
		result.LatencyP50 = time.Duration(percentile(sorted, 0.50))
		result.LatencyP95 = time.Duration(percentile(sorted, 0.95))
		result.LatencyP99 = time.Duration(percentile(sorted, 0.99))
	}

	if len(c.loadStddevs) > 0 {
		result.LoadStdDev = stat.Mean(c.loadStddevs, nil)
	}

	c.mu.Unlock()

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
