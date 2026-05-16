package broker

import (
	"sync/atomic"
)

type BackpressureState int

const (
	BPOpen BackpressureState = iota
	BPWarn
	BPClosed
)

// SystemSignal represents system-wide pressure indicators.
type SystemSignal struct {
	AvgPredictedLoad  float64
	WALFlushLatencyMs float64
}

// BackpressureController throttles producers based on predicted partition loads.
// Single source of truth: only uses externally-provided predicted loads
// (no direct predictor reference to avoid dual sources).
// Also supports system-wide backpressure when avg predicted load exceeds threshold.
type BackpressureController struct {
	threshold float64
	predicted atomic.Pointer[[]float64]

	systemSignal atomic.Pointer[SystemSignal]

	bpClosed atomic.Int64
	bpWarn   atomic.Int64
	bpOpen   atomic.Int64
}

func NewBackpressureController(threshold float64) *BackpressureController {
	if threshold <= 0 {
		threshold = DefaultBPThreshold
	}
	return &BackpressureController{
		threshold: threshold,
	}
}

func (bp *BackpressureController) UpdatePredictions(predicted []float64) {
	cp := make([]float64, len(predicted))
	copy(cp, predicted)
	bp.predicted.Store(&cp)
}

func (bp *BackpressureController) UpdateSystem(sig SystemSignal) {
	bp.systemSignal.Store(&sig)
}

func (bp *BackpressureController) Check(partitionID int) BackpressureState {
	state := bp.evaluate(partitionID)
	switch state {
	case BPClosed:
		bp.bpClosed.Add(1)
	case BPWarn:
		bp.bpWarn.Add(1)
	case BPOpen:
		bp.bpOpen.Add(1)
	}
	return state
}

func (bp *BackpressureController) evaluate(partitionID int) BackpressureState {
	// System-wide check first: catastrophic backpressure.
	if sig := bp.systemSignal.Load(); sig != nil {
		if sig.AvgPredictedLoad > bp.threshold {
			return BPClosed
		}
		if sig.WALFlushLatencyMs > bpFlushLatencyWarnMs {
			return BPWarn
		}
	}

	loads := bp.predicted.Load()
	if loads == nil || partitionID >= len(*loads) {
		return BPOpen
	}

	load := (*loads)[partitionID]
	switch {
	case load > bp.threshold:
		return BPClosed
	case load > bp.threshold*bpWarnRatio:
		return BPWarn
	default:
		return BPOpen
	}
}

func (bp *BackpressureController) ClosedCount() int64 { return bp.bpClosed.Load() }
func (bp *BackpressureController) WarnCount() int64   { return bp.bpWarn.Load() }
func (bp *BackpressureController) OpenCount() int64   { return bp.bpOpen.Load() }
