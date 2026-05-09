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

// LoadPredictor publishes partition load predictions via atomic.Value.
// Uses moving-average as baseline; LSTM inference can replace predict().
type LoadPredictor struct {
	predictions atomic.Value // stores []PartitionPrediction
	mu          sync.Mutex
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

func (lp *LoadPredictor) Predictions() []PartitionPrediction {
	v := lp.predictions.Load()
	if v == nil {
		return nil
	}
	return v.([]PartitionPrediction)
}

func (lp *LoadPredictor) Update(loads []float64) {
	lp.mu.Lock()
	for i, load := range loads {
		buf, ok := lp.history[i]
		if !ok {
			buf = NewRingBuffer(lp.window)
			lp.history[i] = buf
		}
		buf.Push(load)
	}
	lp.mu.Unlock()
	lp.predict()
}

func (lp *LoadPredictor) predict() {
	lp.mu.Lock()
	snapshot := make(map[int]*RingBuffer, len(lp.history))
	for id, buf := range lp.history {
		snapshot[id] = buf
	}
	lp.mu.Unlock()

	preds := make([]PartitionPrediction, 0, len(snapshot))
	for id, buf := range snapshot {
		vals := buf.Values()
		current := 0.0
		if len(vals) > 0 {
			current = vals[len(vals)-1]
		}

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
