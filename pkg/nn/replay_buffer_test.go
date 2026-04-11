package nn

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ReplayBufferSuite struct {
	suite.Suite
}

func TestReplayBufferSuite(t *testing.T) { suite.Run(t, new(ReplayBufferSuite)) }

func (s *ReplayBufferSuite) TestPushAndLen() {
	tests := []struct {
		name     string
		capacity int
		pushes   int
		wantLen  int
	}{
		{"empty", 10, 0, 0},
		{"under_cap", 10, 5, 5},
		{"at_cap", 10, 10, 10},
		{"over_cap", 10, 15, 10},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			rb := NewReplayBuffer(tc.capacity)
			for i := 0; i < tc.pushes; i++ {
				rb.Push(Transition{
					State:     []float64{float64(i)},
					Action:    []float64{0},
					Reward:    float64(i),
					NextState: []float64{float64(i + 1)},
				})
			}
			require.Equal(s.T(), tc.wantLen, rb.Len())
		})
	}
}

func (s *ReplayBufferSuite) TestSampleSize() {
	rb := NewReplayBuffer(100)
	for i := 0; i < 50; i++ {
		rb.Push(Transition{
			State:  []float64{float64(i)},
			Action: []float64{0},
			Reward: 1.0,
		})
	}

	tests := []struct {
		name      string
		batchSize int
		wantSize  int
	}{
		{"small_batch", 10, 10},
		{"exact", 50, 50},
		{"larger_than_buffer", 100, 50},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			batch := rb.Sample(tc.batchSize)
			require.Len(s.T(), batch, tc.wantSize)
		})
	}
}

func (s *ReplayBufferSuite) TestSampleContentValid() {
	rb := NewReplayBuffer(100)
	for i := 0; i < 10; i++ {
		rb.Push(Transition{
			State:     []float64{float64(i)},
			Action:    []float64{float64(i % 3)},
			Reward:    float64(i) * 0.1,
			NextState: []float64{float64(i + 1)},
			Done:      i == 9,
		})
	}

	batch := rb.Sample(5)
	for _, t := range batch {
		require.NotEmpty(s.T(), t.State)
		require.NotEmpty(s.T(), t.Action)
	}
}

func (s *ReplayBufferSuite) TestRingOverwrite() {
	rb := NewReplayBuffer(3)
	for i := 0; i < 5; i++ {
		rb.Push(Transition{
			State:  []float64{float64(i)},
			Action: []float64{0},
			Reward: float64(i),
		})
	}

	require.Equal(s.T(), 3, rb.Len())

	batch := rb.Sample(3)
	for _, t := range batch {
		require.GreaterOrEqual(s.T(), t.Reward, 2.0, "oldest entries should be overwritten")
	}
}
