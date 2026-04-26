package experiments

import (
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/stat"
)

// MetricsCollector gathers per-message latencies, partition-load samples and
// error counters during an experiment run.
type MetricsCollector struct {
	mu              sync.Mutex
	latencies       []time.Duration
	loadStddevs     []float64
	produced        atomic.Int64
	consumed        atomic.Int64
	publishErrors   atomic.Int64
	consumeErrors   atomic.Int64
	startTime       time.Time
	endTime         time.Time
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

func (c *MetricsCollector) RecordConsume(latency time.Duration) {
	c.consumed.Add(1)
	c.mu.Lock()
	c.latencies = append(c.latencies, latency)
	c.mu.Unlock()
}

func (c *MetricsCollector) RecordPublishError() {
	c.publishErrors.Add(1)
}

func (c *MetricsCollector) RecordConsumeError() {
	c.consumeErrors.Add(1)
}

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
}

func (c *MetricsCollector) Snapshot(scenario, algorithm, system, mode string) ExperimentResult {
	c.mu.Lock()
	defer c.mu.Unlock()

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

	return result
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}

	pos := p * float64(len(sorted)-1)
	i := int(pos)
	frac := pos - float64(i)
	return sorted[i]*(1-frac) + sorted[i+1]*frac
}