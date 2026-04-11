package experiments

import (
	"math"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/stat"
)

// MetricsCollector gathers per-message latencies and partition-level loads,
// then computes aggregate statistics for an experiment run.
type MetricsCollector struct {
	mu        sync.Mutex
	latencies []time.Duration
	partLoads []float64
	produced  int64
	consumed  int64
	startTime time.Time
	endTime   time.Time
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
	c.mu.Lock()
	c.produced++
	c.mu.Unlock()
}

func (c *MetricsCollector) RecordConsume(latency time.Duration) {
	c.mu.Lock()
	c.consumed++
	c.latencies = append(c.latencies, latency)
	c.mu.Unlock()
}

func (c *MetricsCollector) RecordPartitionLoads(loads []float64) {
	c.mu.Lock()
	c.partLoads = append(c.partLoads[:0], loads...)
	c.mu.Unlock()
}

// ExperimentResult holds the aggregated metrics for a single experiment run.
type ExperimentResult struct {
	Scenario   string        `json:"scenario"`
	Algorithm  string        `json:"algorithm"`
	System     string        `json:"system"` // "nyaqueue" or "kafka"
	Mode       string        `json:"mode"`   // "inprocess", "grpc", or "kafka"
	Throughput float64       `json:"throughput_msg_per_sec"`
	LatencyP50 time.Duration `json:"latency_p50_ns"`
	LatencyP95 time.Duration `json:"latency_p95_ns"`
	LatencyP99 time.Duration `json:"latency_p99_ns"`
	LoadStdDev float64       `json:"load_stddev"`
	Produced   int64         `json:"produced"`
	Consumed   int64         `json:"consumed"`
	Duration   time.Duration `json:"duration_ns"`
}

// Snapshot computes aggregate results from the collected data.
func (c *MetricsCollector) Snapshot(scenario, algorithm, system, mode string) ExperimentResult {
	c.mu.Lock()
	defer c.mu.Unlock()

	duration := c.endTime.Sub(c.startTime)
	if duration <= 0 {
		duration = time.Since(c.startTime)
	}

	result := ExperimentResult{
		Scenario:  scenario,
		Algorithm: algorithm,
		System:    system,
		Mode:      mode,
		Produced:  c.produced,
		Consumed:  c.consumed,
		Duration:  duration,
	}

	if duration.Seconds() > 0 {
		result.Throughput = float64(c.consumed) / duration.Seconds()
	}

	if len(c.latencies) > 0 {
		sorted := make([]float64, len(c.latencies))
		for i, l := range c.latencies {
			sorted[i] = float64(l)
		}
		sort.Float64s(sorted)

		result.LatencyP50 = time.Duration(percentile(sorted, 0.50))
		result.LatencyP95 = time.Duration(percentile(sorted, 0.95))
		result.LatencyP99 = time.Duration(percentile(sorted, 0.99))
	}

	if len(c.partLoads) > 1 {
		result.LoadStdDev = stat.StdDev(c.partLoads, nil)
	}

	return result
}

func (c *MetricsCollector) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.latencies = c.latencies[:0]
	c.partLoads = c.partLoads[:0]
	c.produced = 0
	c.consumed = 0
	c.startTime = time.Time{}
	c.endTime = time.Time{}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := p * float64(len(sorted)-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	if lower == upper || upper >= len(sorted) {
		return sorted[lower]
	}
	frac := idx - float64(lower)
	return sorted[lower]*(1-frac) + sorted[upper]*frac
}
