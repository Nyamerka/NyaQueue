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
// Uses AR(p) autoregression with Yule-Walker coefficient estimation.
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

		predicted := arPredict(vals, lp.horizon)

		preds = append(preds, PartitionPrediction{
			PartitionID: id,
			Current:     current,
			Predicted:   predicted,
		})
	}
	lp.predictions.Store(preds)
}

func (lp *LoadPredictor) PredictAll(horizon int) []float64 {
	preds := lp.Predictions()
	if len(preds) == 0 {
		return nil
	}
	result := make([]float64, len(preds))
	for i, p := range preds {
		if len(p.Predicted) > 0 {
			result[i] = p.Predicted[0]
		} else {
			result[i] = p.Current
		}
	}
	return result
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

// arPredict forecasts `horizon` steps ahead using AR(p) autoregression.
// The AR order p is automatically chosen as min(len(vals)/2, 8).
// Coefficients are estimated via Yule-Walker equations solved with
// Levinson-Durbin recursion (O(p²) time, no matrix inversion).
// Falls back to simple mean when data is insufficient or constant.
func arPredict(vals []float64, horizon int) []float64 {
	predicted := make([]float64, horizon)
	n := len(vals)
	if n == 0 {
		return predicted
	}

	mean := 0.0
	for _, v := range vals {
		mean += v
	}
	mean /= float64(n)

	p := n / 2
	if p > 8 {
		p = 8
	}
	if p < 1 || n < 2*p {
		for k := range predicted {
			_ = k
			predicted[k] = mean
		}
		return predicted
	}

	coeffs := yuleWalker(vals, mean, p)
	if coeffs == nil {
		for k := range predicted {
			predicted[k] = mean
		}
		return predicted
	}

	centered := make([]float64, n)
	for i, v := range vals {
		centered[i] = v - mean
	}

	buf := make([]float64, n+horizon)
	copy(buf, centered)

	for k := 0; k < horizon; k++ {
		idx := n + k
		pred := 0.0
		for j := 0; j < p; j++ {
			pred += coeffs[j] * buf[idx-1-j]
		}
		buf[idx] = pred
		predicted[k] = pred + mean
		if predicted[k] < 0 {
			predicted[k] = 0
		}
		if predicted[k] > 1 {
			predicted[k] = 1
		}
	}

	return predicted
}

// yuleWalker estimates AR(p) coefficients via Levinson-Durbin recursion.
// Returns nil if the series has zero variance.
func yuleWalker(vals []float64, mean float64, p int) []float64 {
	n := len(vals)

	r := make([]float64, p+1)
	for lag := 0; lag <= p; lag++ {
		sum := 0.0
		for i := lag; i < n; i++ {
			sum += (vals[i] - mean) * (vals[i-lag] - mean)
		}
		r[lag] = sum / float64(n)
	}

	if r[0] < 1e-15 {
		return nil
	}

	a := make([]float64, p)
	aOld := make([]float64, p)
	a[0] = r[1] / r[0]
	var e float64 = r[0] * (1 - a[0]*a[0])

	for m := 1; m < p; m++ {
		if e < 1e-15 {
			return a[:m]
		}

		sum := r[m+1]
		for j := 0; j < m; j++ {
			sum -= a[j] * r[m-j]
		}
		k := sum / e

		copy(aOld, a)
		a[m] = k
		for j := 0; j < m; j++ {
			a[j] = aOld[j] - k*aOld[m-1-j]
		}
		e *= (1 - k*k)
	}

	return a
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
