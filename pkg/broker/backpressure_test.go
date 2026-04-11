package broker

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type BackpressureSuite struct {
	suite.Suite
}

func TestBackpressureSuite(t *testing.T) { suite.Run(t, new(BackpressureSuite)) }

func (s *BackpressureSuite) TestNilPredictor() {
	bp := NewBackpressureController(nil, 0.85, 3)
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
			lp := NewLoadPredictor(10, 5, 100*time.Millisecond)
			lp.Update([]float64{tc.predicted})

			bp := NewBackpressureController(lp, tc.threshold, 0)
			got := bp.Check(0)
			require.Equal(s.T(), tc.want, got)
		})
	}
}
