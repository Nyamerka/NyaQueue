package broker

import "sync"

type BackpressureState int

const (
	BPOpen BackpressureState = iota
	BPWarn
	BPClosed
)

// BackpressureController throttles producers based on LoadPredictor predictions.
// Falls back to open-by-default when no predictor is connected.
type BackpressureController struct {
	predictor      *LoadPredictor
	threshold      float64
	horizon        int
	predictedLoads []float64
	predictedMu    sync.RWMutex
}

func NewBackpressureController(predictor *LoadPredictor, threshold float64, horizon int) *BackpressureController {
	if threshold <= 0 {
		threshold = 0.85
	}
	return &BackpressureController{
		predictor: predictor,
		threshold: threshold,
		horizon:   horizon,
	}
}

func (bp *BackpressureController) UpdatePredictions(predicted []float64) {
	bp.predictedMu.Lock()
	bp.predictedLoads = predicted
	bp.predictedMu.Unlock()
}

func (bp *BackpressureController) Check(partitionID int) BackpressureState {
	bp.predictedMu.RLock()
	predicted := bp.predictedLoads
	bp.predictedMu.RUnlock()

	if partitionID < len(predicted) {
		load := predicted[partitionID]
		if load > bp.threshold {
			return BPClosed
		}
		if load > bp.threshold*0.9 {
			return BPWarn
		}
	}

	if bp.predictor == nil {
		return BPOpen
	}

	preds := bp.predictor.Predictions()
	if preds == nil {
		return BPOpen
	}

	for _, p := range preds {
		if p.PartitionID != partitionID {
			continue
		}

		idx := bp.horizon
		if idx >= len(p.Predicted) {
			idx = len(p.Predicted) - 1
		}
		if idx < 0 {
			return BPOpen
		}

		pv := p.Predicted[idx]
		if pv > bp.threshold {
			return BPClosed
		}
		if pv > bp.threshold*0.9 {
			return BPWarn
		}
		return BPOpen
	}

	return BPOpen
}
