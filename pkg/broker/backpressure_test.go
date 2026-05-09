package broker

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type BackpressureSuite struct {
	suite.Suite
}

func TestBackpressureSuite(t *testing.T) { suite.Run(t, new(BackpressureSuite)) }

func (s *BackpressureSuite) TestDefaultOpen() {
	bp := NewBackpressureController(0.85)
	require.Equal(s.T(), BPOpen, bp.Check(0))
}

func (s *BackpressureSuite) TestThresholdStates() {
	tests := []struct {
		name      string
		predicted float64
		threshold float64
		want      BackpressureState
	}{
		{"open", 0.5, 0.85, BPOpen},
		{"warn", 0.80, 0.85, BPWarn},
		{"closed", 0.90, 0.85, BPClosed},
		{"exact_threshold", 0.85, 0.85, BPWarn},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			bp := NewBackpressureController(tc.threshold)
			bp.UpdatePredictions([]float64{tc.predicted})
			got := bp.Check(0)
			require.Equal(s.T(), tc.want, got)
		})
	}
}

func (s *BackpressureSuite) TestActivationCounters() {
	bp := NewBackpressureController(0.85)
	bp.UpdatePredictions([]float64{0.5, 0.90, 0.80})

	bp.Check(0)
	bp.Check(1)
	bp.Check(2)

	require.Equal(s.T(), int64(1), bp.OpenCount())
	require.Equal(s.T(), int64(1), bp.ClosedCount())
	require.Equal(s.T(), int64(1), bp.WarnCount())
}

func (s *BackpressureSuite) TestSystemWideSignal() {
	bp := NewBackpressureController(0.85)
	bp.UpdatePredictions([]float64{0.1})
	bp.UpdateSystem(SystemSignal{AvgPredictedLoad: 0.95})

	require.Equal(s.T(), BPClosed, bp.Check(0))
}

func (s *BackpressureSuite) TestConcurrentCheckAndUpdate() {
	bp := NewBackpressureController(0.85)
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 500; j++ {
				bp.Check(0)
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 200; j++ {
			bp.UpdatePredictions([]float64{float64(j%100) / 100.0})
			bp.UpdateSystem(SystemSignal{AvgPredictedLoad: float64(j%100) / 100.0})
		}
	}()

	wg.Wait()
}
