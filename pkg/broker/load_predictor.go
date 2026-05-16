package broker

import (
	"context"
	"log"
	"math"
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
	"golang.org/x/sync/errgroup"
)

type PartitionPrediction struct {
	PartitionID int
	Current     float64
	Predicted   []float64
}

// LoadPredictor publishes partition load predictions via atomic.Value.
// Uses AR(p) autoregression with Yule-Walker coefficient estimation.
// Pre-allocated buffers eliminate per-predict allocations on the hot path.
type LoadPredictor struct {
	predictions atomic.Value // stores []PartitionPrediction
	history     *xsync.MapOf[int, *RingBuffer]
	window      int
	horizon     int
	interval    time.Duration
	eg          *errgroup.Group
	cancel      context.CancelFunc

	predictMu xsync.RBMutex // serializes predict() which reuses scratch buffers
	predBuf   []float64
	centerBuf []float64
	arBuf     []float64
	coeffBuf  []float64
	valsBuf   []float64
	rBuf      []float64
	aOldBuf   []float64
}

func NewLoadPredictor(window, horizon int, interval time.Duration) *LoadPredictor {
	lp := &LoadPredictor{
		history:   xsync.NewMapOf[int, *RingBuffer](),
		window:    window,
		horizon:   horizon,
		interval:  interval,
		predBuf:   make([]float64, horizon),
		centerBuf: make([]float64, window),
		arBuf:     make([]float64, window+horizon),
		coeffBuf:  make([]float64, 16),
		valsBuf:   make([]float64, window),
		rBuf:      make([]float64, 16+1),
		aOldBuf:   make([]float64, 16),
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
	for i, load := range loads {
		buf, _ := lp.history.LoadOrCompute(i, func() *RingBuffer {
			return NewRingBuffer(lp.window)
		})
		buf.Push(load)
	}
	lp.predict()
}

func (lp *LoadPredictor) predict() {
	lp.predictMu.Lock()
	defer lp.predictMu.Unlock()

	var preds []PartitionPrediction
	lp.history.Range(func(id int, buf *RingBuffer) bool {
		n := buf.ValuesInto(lp.valsBuf)
		vals := lp.valsBuf[:n]

		current := 0.0
		if n > 0 {
			current = vals[n-1]
		}

		predicted := lp.arPredictInto(vals, lp.horizon)

		predCopy := make([]float64, len(predicted))
		copy(predCopy, predicted)

		preds = append(preds, PartitionPrediction{
			PartitionID: id,
			Current:     current,
			Predicted:   predCopy,
		})
		return true
	})

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
	ctx, cancel := context.WithCancel(context.Background())
	lp.cancel = cancel
	lp.eg, _ = errgroup.WithContext(ctx)
	lp.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[load-predictor] loop panic: %v", r)
			}
		}()
		ticker := time.NewTicker(lp.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				lp.predict()
			case <-ctx.Done():
				return nil
			}
		}
	})
}

func (lp *LoadPredictor) Stop() {
	if lp.cancel != nil {
		lp.cancel()
	}
	if lp.eg != nil {
		_ = lp.eg.Wait()
	}
}

// RemovePartition deletes stored history for the given partition ID,
// preventing unbounded memory growth when topics/partitions are deleted.
func (lp *LoadPredictor) RemovePartition(partID int) {
	lp.history.Delete(partID)
}

// arPredictInto forecasts `horizon` steps ahead using AR(p) autoregression.
// Uses pre-allocated buffers to avoid per-call allocations.
func (lp *LoadPredictor) arPredictInto(vals []float64, horizon int) []float64 {
	if len(lp.predBuf) < horizon {
		lp.predBuf = make([]float64, horizon)
	}
	predicted := lp.predBuf[:horizon]
	for i := range predicted {
		predicted[i] = 0
	}

	n := len(vals)
	if n == 0 {
		return predicted
	}

	mean := 0.0
	for _, v := range vals {
		mean += v
	}
	mean /= float64(n)

	maxOrder := lp.window / 4
	if maxOrder < 1 {
		maxOrder = 1
	}
	p := int(math.Sqrt(float64(n)))
	if p > maxOrder {
		p = maxOrder
	}
	if p < 1 || n < 2*p {
		for k := range predicted {
			predicted[k] = mean
		}
		return predicted
	}

	coeffs := lp.yuleWalker(vals, mean, p)
	if coeffs == nil {
		for k := range predicted {
			predicted[k] = mean
		}
		return predicted
	}

	if len(lp.centerBuf) < n {
		lp.centerBuf = make([]float64, n)
	}
	centered := lp.centerBuf[:n]
	for i, v := range vals {
		centered[i] = v - mean
	}

	needed := n + horizon
	if len(lp.arBuf) < needed {
		lp.arBuf = make([]float64, needed)
	}
	buf := lp.arBuf[:needed]
	copy(buf, centered)

	order := len(coeffs)
	for k := 0; k < horizon; k++ {
		idx := n + k
		pred := 0.0
		for j := 0; j < order; j++ {
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
// Uses pre-allocated struct buffers to eliminate per-call allocations.
// Returns nil if zero variance.
func (lp *LoadPredictor) yuleWalker(vals []float64, mean float64, p int) []float64 {
	n := len(vals)
	coeffBuf, rBuf, aOldBuf := lp.coeffBuf, lp.rBuf, lp.aOldBuf

	if len(rBuf) < p+1 {
		rBuf = make([]float64, p+1)
	}
	r := rBuf[:p+1]
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

	if len(coeffBuf) < p {
		coeffBuf = make([]float64, p)
	}
	a := coeffBuf[:p]
	if len(aOldBuf) < p {
		aOldBuf = make([]float64, p)
	}
	aOld := aOldBuf[:p]

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

// RingBuffer is a circular buffer for time-series data, safe for concurrent use.
type RingBuffer struct {
	mu       xsync.RBMutex
	data     []float64
	writeIdx int64
}

func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{data: make([]float64, size)}
}

func (rb *RingBuffer) Push(v float64) {
	rb.mu.Lock()
	rb.data[rb.writeIdx%int64(len(rb.data))] = v
	rb.writeIdx++
	rb.mu.Unlock()
}

// Values allocates and returns a copy of the buffer contents in chronological order.
func (rb *RingBuffer) Values() []float64 {
	t := rb.mu.RLock()
	defer rb.mu.RUnlock(t)

	w := rb.writeIdx
	size := int64(len(rb.data))
	if w == 0 {
		return nil
	}

	n := w
	if n > size {
		n = size
	}
	out := make([]float64, n)
	rb.copyInto(out, w, size, int(n))
	return out
}

// ValuesInto copies buffer contents into dst without allocating. Returns the
// number of values written. dst must be at least as large as the ring size.
func (rb *RingBuffer) ValuesInto(dst []float64) int {
	t := rb.mu.RLock()
	defer rb.mu.RUnlock(t)

	w := rb.writeIdx
	size := int64(len(rb.data))
	if w == 0 {
		return 0
	}

	n := int(w)
	if int64(n) > size {
		n = int(size)
	}
	if n > len(dst) {
		n = len(dst)
	}
	rb.copyInto(dst[:n], w, size, n)
	return n
}

func (rb *RingBuffer) copyInto(dst []float64, w, size int64, n int) {
	if w <= size {
		copy(dst, rb.data[:n])
	} else {
		start := w % size
		tail := int(size - start)
		if tail > n {
			tail = n
		}
		copy(dst, rb.data[start:start+int64(tail)])
		if tail < n {
			copy(dst[tail:], rb.data[:n-tail])
		}
	}
}
