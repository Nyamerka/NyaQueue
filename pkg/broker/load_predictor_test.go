package broker

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type LoadPredictorSuite struct {
	suite.Suite
}

func TestLoadPredictorSuite(t *testing.T) { suite.Run(t, new(LoadPredictorSuite)) }

func (s *LoadPredictorSuite) TestUpdateAndPredict() {
	lp := NewLoadPredictor(10, 3, 100*time.Millisecond)

	lp.Update([]float64{0.5, 0.3})
	preds := lp.Predictions()
	predByID := make(map[int]PartitionPrediction, len(preds))
	for _, p := range preds {
		predByID[p.PartitionID] = p
	}

	require.Len(s.T(), preds, 2)
	require.Contains(s.T(), predByID, 0)
	require.Contains(s.T(), predByID, 1)
	require.Equal(s.T(), 0.5, predByID[0].Current)
	require.Equal(s.T(), 0.3, predByID[1].Current)
	require.Len(s.T(), predByID[0].Predicted, 3)
}

func (s *LoadPredictorSuite) TestRingBuffer() {
	tests := []struct {
		name   string
		size   int
		pushes []float64
		want   []float64
	}{
		{"under_capacity", 5, []float64{1, 2, 3}, []float64{1, 2, 3}},
		{"exact_capacity", 3, []float64{1, 2, 3}, []float64{1, 2, 3}},
		{"overflow", 3, []float64{1, 2, 3, 4, 5}, []float64{3, 4, 5}},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			rb := NewRingBuffer(tc.size)
			for _, v := range tc.pushes {
				rb.Push(v)
			}
			require.Equal(s.T(), tc.want, rb.Values())
		})
	}
}

func (s *LoadPredictorSuite) TestEmptyPredictions() {
	lp := NewLoadPredictor(10, 3, time.Second)
	preds := lp.Predictions()
	require.Empty(s.T(), preds)
}

// testArPredict is a helper that creates a temporary LoadPredictor to call arPredictInto.
func testArPredict(vals []float64, horizon int) []float64 {
	lp := NewLoadPredictor(len(vals)+horizon, horizon, time.Second)
	return lp.arPredictInto(vals, horizon)
}

func (s *LoadPredictorSuite) TestARPredictLinearTrend() {
	vals := make([]float64, 30)
	for i := range vals {
		vals[i] = float64(i) * 0.01
	}

	predicted := testArPredict(vals, 5)
	require.Len(s.T(), predicted, 5)

	lastVal := vals[len(vals)-1]
	for _, p := range predicted {
		require.Greater(s.T(), p, lastVal*0.5,
			"AR should extrapolate upward trend, not collapse to mean")
	}
}

func (s *LoadPredictorSuite) TestARPredictConstant() {
	vals := make([]float64, 20)
	for i := range vals {
		vals[i] = 0.5
	}

	predicted := testArPredict(vals, 3)
	require.Len(s.T(), predicted, 3)
	for _, p := range predicted {
		require.InDelta(s.T(), 0.5, p, 0.01, "constant series should predict constant")
	}
}

func (s *LoadPredictorSuite) TestARPredictEmptyInput() {
	predicted := testArPredict(nil, 5)
	require.Len(s.T(), predicted, 5)
	for _, p := range predicted {
		require.Equal(s.T(), 0.0, p)
	}
}

func (s *LoadPredictorSuite) TestARPredictShortSeries() {
	predicted := testArPredict([]float64{0.5}, 3)
	require.Len(s.T(), predicted, 3)
	for _, p := range predicted {
		require.InDelta(s.T(), 0.5, p, 0.01)
	}
}

func (s *LoadPredictorSuite) TestARPredictClampsBounds() {
	vals := make([]float64, 30)
	for i := range vals {
		vals[i] = 0.99 + float64(i)*0.001
	}

	predicted := testArPredict(vals, 5)
	for _, p := range predicted {
		require.GreaterOrEqual(s.T(), p, 0.0)
		require.LessOrEqual(s.T(), p, 1.0)
	}
}

func (s *LoadPredictorSuite) TestYuleWalkerCoefficients() {
	vals := make([]float64, 50)
	for i := range vals {
		vals[i] = float64(i) * 0.1
	}
	mean := 0.0
	for _, v := range vals {
		mean += v
	}
	mean /= float64(len(vals))

	rBuf := make([]float64, 5)
	aOldBuf := make([]float64, 4)
	coeffBuf := make([]float64, 4)
	coeffs := yuleWalker(vals, mean, 4, coeffBuf, rBuf, aOldBuf)
	require.NotNil(s.T(), coeffs)
	require.Len(s.T(), coeffs, 4)
}

func (s *LoadPredictorSuite) TestYuleWalkerZeroVariance() {
	vals := []float64{1, 1, 1, 1, 1}
	rBuf := make([]float64, 3)
	aOldBuf := make([]float64, 2)
	coeffBuf := make([]float64, 2)
	coeffs := yuleWalker(vals, 1.0, 2, coeffBuf, rBuf, aOldBuf)
	require.Nil(s.T(), coeffs, "zero-variance series should return nil coefficients")
}

func (s *LoadPredictorSuite) TestPredictAll() {
	lp := NewLoadPredictor(20, 5, 100*time.Millisecond)

	for i := 0; i < 15; i++ {
		lp.Update([]float64{0.3 + float64(i)*0.01, 0.5 + float64(i)*0.01})
	}

	predicted := lp.PredictAll(8)
	require.Len(s.T(), predicted, 2)
	require.Greater(s.T(), predicted[0], 0.3, "predicted load should be above initial for partition 0")
	require.Greater(s.T(), predicted[1], 0.4, "predicted load should be reasonable for partition 1")
}

func (s *LoadPredictorSuite) TestBrokerMetricsIncludesPredictions() {
	dir := s.T().TempDir()
	store, err := NewOffsetStore(dir)
	require.NoError(s.T(), err)

	cfg := DefaultConfig()
	b := New(cfg, dir, noopBalancer{}, store)
	b.SetBackpressure(NewBackpressureController(0.99))
	b.Start()
	defer b.Stop()

	tcfg := DefaultTopicConfig()
	tcfg.NumPartitions = 2
	require.NoError(s.T(), b.CreateTopic("t", tcfg))
	b.SetScheduler("t", fifoScheduler{})

	for i := 0; i < 50; i++ {
		msg := NewMessage(0, []byte("k"), []byte("v"))
		b.Publish("t", msg)
		time.Sleep(5 * time.Millisecond)
	}

	time.Sleep(500 * time.Millisecond)
	m := b.Metrics()
	require.NotNil(s.T(), m.PartitionLoads)
}

func (s *LoadPredictorSuite) TestConcurrentUpdateAndPredict() {
	lp := NewLoadPredictor(10, 3, 1*time.Millisecond)
	lp.Start()
	defer lp.Stop()

	const goroutines = 8
	const iterations = 200

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				lp.Update([]float64{0.1, 0.2, 0.3, 0.4})
			}
		}()
	}

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				_ = lp.Predictions()
			}
		}()
	}

	wg.Wait()
}
