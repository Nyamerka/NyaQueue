package broker

import (
	"sync"
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
	threshold      float64
	predictedLoads []float64
	predictedMu    sync.RWMutex

	systemSignal atomic.Pointer[SystemSignal]

	bpClosed atomic.Int64
	bpWarn   atomic.Int64
	bpOpen   atomic.Int64
}

func NewBackpressureController(threshold float64) *BackpressureController {
	if threshold <= 0 {
		threshold = 0.85
	}
	return &BackpressureController{
		threshold: threshold,
	}
}

func (bp *BackpressureController) UpdatePredictions(predicted []float64) {
	bp.predictedMu.Lock()
	if cap(bp.predictedLoads) < len(predicted) {
		bp.predictedLoads = make([]float64, len(predicted))
	}
	bp.predictedLoads = bp.predictedLoads[:len(predicted)]
	copy(bp.predictedLoads, predicted)
	bp.predictedMu.Unlock()
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
		if sig.WALFlushLatencyMs > 100 {
			return BPWarn
		}
	}

	bp.predictedMu.RLock()
	defer bp.predictedMu.RUnlock()

	if partitionID >= len(bp.predictedLoads) {
		return BPOpen
	}

	load := bp.predictedLoads[partitionID]
	switch {
	case load > bp.threshold:
		return BPClosed
	case load > bp.threshold*0.9:
		return BPWarn
	default:
		return BPOpen
	}
}

func (bp *BackpressureController) ClosedCount() int64 { return bp.bpClosed.Load() }
func (bp *BackpressureController) WarnCount() int64   { return bp.bpWarn.Load() }
func (bp *BackpressureController) OpenCount() int64   { return bp.bpOpen.Load() }
