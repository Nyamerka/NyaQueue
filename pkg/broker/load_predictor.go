package broker

import (
	"sync"
	"sync/atomic"
	"time"
)

type PartitionPrediction struct {
	PartitionID int
	Current     float64
	Predicted   []float64 // predictions K steps ahead
}

// LoadPredictor runs a background loop that publishes predictions via atomic.Value.
// Initially uses a simple moving-average heuristic; will be replaced with LSTM (GoMLX).
type LoadPredictor struct {
	predictions atomic.Value // stores []PartitionPrediction
	history     map[int]*RingBuffer
	window      int
	horizon     int
	interval    time.Duration
	stopCh      chan struct{}
}

func NewLoadPredictor(window, horizon int, interval time.Duration) *LoadPredictor {
	lp := &LoadPredictor{
		history:  make(map[int]*RingBuffer),
		window:   window,
		horizon:  horizon,
		interval: interval,
		stopCh:   make(chan struct{}),
	}
	lp.predictions.Store([]PartitionPrediction{})
	return lp
}

// Predictions returns the latest predictions (lock-free read).
func (lp *LoadPredictor) Predictions() []PartitionPrediction {
	v := lp.predictions.Load()
	if v == nil {
		return nil
	}
	return v.([]PartitionPrediction)
}

// Update feeds current partition loads into the predictor's history buffers.
func (lp *LoadPredictor) Update(loads []float64) {
	for i, load := range loads {
		buf, ok := lp.history[i]
		if !ok {
			buf = NewRingBuffer(lp.window)
			lp.history[i] = buf
		}
		buf.Push(load)
	}
	lp.predict()
}

func (lp *LoadPredictor) predict() {
	preds := make([]PartitionPrediction, 0, len(lp.history))
	for id, buf := range lp.history {
		vals := buf.Values()
		current := 0.0
		if len(vals) > 0 {
			current = vals[len(vals)-1]
		}

		// Simple moving-average extrapolation as baseline;
		// will be replaced by LSTM inference via GoMLX.
		avg := movingAverage(vals)
		predicted := make([]float64, lp.horizon)
		for k := range predicted {
			predicted[k] = avg
		}

		preds = append(preds, PartitionPrediction{
			PartitionID: id,
			Current:     current,
			Predicted:   predicted,
		})
	}
	lp.predictions.Store(preds)
}

func (lp *LoadPredictor) Start() {
	go func() {
		ticker := time.NewTicker(lp.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				lp.predict()
			case <-lp.stopCh:
				return
			}
		}
	}()
}

func (lp *LoadPredictor) Stop() {
	close(lp.stopCh)
}

func movingAverage(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// RingBuffer is a fixed-size circular buffer of float64 values.
type RingBuffer struct {
	mu   sync.Mutex
	data []float64
	pos  int
	full bool
}

func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{data: make([]float64, size)}
}

func (rb *RingBuffer) Push(v float64) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.data[rb.pos] = v
	rb.pos++
	if rb.pos >= len(rb.data) {
		rb.pos = 0
		rb.full = true
	}
}

func (rb *RingBuffer) Values() []float64 {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if !rb.full {
		out := make([]float64, rb.pos)
		copy(out, rb.data[:rb.pos])
		return out
	}

	out := make([]float64, len(rb.data))
	copy(out, rb.data[rb.pos:])
	copy(out[len(rb.data)-rb.pos:], rb.data[:rb.pos])
	return out
}
