package broker

import (
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

	require.Len(s.T(), preds, 2)
	require.Equal(s.T(), 0.5, preds[0].Current)
	require.Len(s.T(), preds[0].Predicted, 3) // horizon=3
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
