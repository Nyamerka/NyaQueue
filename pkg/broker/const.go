package broker

import "time"

const DefaultBPThreshold = 0.85

const (
	bpWarnRatio          = 0.9   // warn when load > threshold * bpWarnRatio
	bpFlushLatencyWarnMs = 100.0 // WAL flush latency (ms) that triggers BPWarn
)

const (
	metricsTickInterval = 100 * time.Millisecond
	metricsWarmupTicks  = 50
	metricsHistoryCap   = 100
)

const (
	defaultPredictorWindowCap = 100
	defaultPredictorHorizon   = 8
	defaultPredictorInterval  = 100 * time.Millisecond
)
