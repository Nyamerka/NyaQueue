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
