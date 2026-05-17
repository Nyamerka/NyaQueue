package nn

import (
	"sync"
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

func (s *ReplayBufferSuite) TestSampleInto() {
	rb := NewReplayBuffer(100)
	for i := 0; i < 50; i++ {
		rb.Push(Transition{
			State:  []float64{float64(i)},
			Action: []float64{0},
			Reward: 1.0,
		})
	}

	dst := make([]Transition, 10)
	n := rb.SampleInto(dst)
	require.Equal(s.T(), 10, n)
	for _, t := range dst[:n] {
		require.NotEmpty(s.T(), t.State)
	}
}

func (s *ReplayBufferSuite) TestDeepCopyOnPush() {
	rb := NewReplayBuffer(10)
	state := []float64{1.0, 2.0, 3.0}
	rb.Push(Transition{State: state, Action: []float64{0}, Reward: 1.0, NextState: []float64{4.0}})

	state[0] = 999.0
	batch := rb.Sample(1)
	require.Equal(s.T(), 1.0, batch[0].State[0], "push must deep-copy to prevent aliasing")
}

func (s *ReplayBufferSuite) TestConcurrentPushAndSample() {
	rb := NewReplayBuffer(1000)
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 500; j++ {
				rb.Push(Transition{
					State:     []float64{float64(id), float64(j)},
					Action:    []float64{0},
					Reward:    float64(j),
					NextState: []float64{float64(id), float64(j + 1)},
				})
			}
		}(i)
	}

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				rb.Sample(32)
			}
		}()
	}

	wg.Wait()
	require.Equal(s.T(), 1000, rb.Len())
}

type PrioritizedReplayBufferSuite struct {
	suite.Suite
}

func TestPrioritizedReplayBufferSuite(t *testing.T) {
	suite.Run(t, new(PrioritizedReplayBufferSuite))
}

func (s *PrioritizedReplayBufferSuite) TestPushAndLen() {
	b := NewPrioritizedReplayBuffer(10, 0.6)
	require.Equal(s.T(), 0, b.Len())

	for i := 0; i < 5; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: float64(i)})
	}
	require.Equal(s.T(), 5, b.Len())

	for i := 5; i < 15; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: float64(i)})
	}
	require.Equal(s.T(), 10, b.Len())
}

func (s *PrioritizedReplayBufferSuite) TestSampleSize() {
	b := NewPrioritizedReplayBuffer(100, 0.6)
	for i := 0; i < 50; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: 1.0})
	}

	batch, indices, weights := b.Sample(10, 1.0)
	require.Len(s.T(), batch, 10)
	require.Len(s.T(), indices, 10)
	require.Len(s.T(), weights, 10)

	batch, indices, weights = b.Sample(100, 1.0)
	require.Len(s.T(), batch, 50)
	require.Len(s.T(), indices, 50)
	require.Len(s.T(), weights, 50)
}

func (s *PrioritizedReplayBufferSuite) TestHighTDErrorSampledMore() {
	b := NewPrioritizedReplayBuffer(100, 0.6)
	for i := 0; i < 100; i++ {
		b.Push(Transition{
			State:  []float64{float64(i)},
			Action: []float64{0},
			Reward: float64(i),
		})
	}

	b.UpdatePriority(0, 100.0)
	for i := 1; i < 100; i++ {
		b.UpdatePriority(i, 0.001)
	}

	counts := make([]int, 100)
	const trials = 10_000
	for trial := 0; trial < trials; trial++ {
		batch, indices, _ := b.Sample(1, 1.0)
		require.Len(s.T(), batch, 1)
		counts[indices[0]]++
	}

	require.Greater(s.T(), counts[0], trials/10,
		"index 0 with high TD error should be sampled much more frequently")
}

func (s *PrioritizedReplayBufferSuite) TestPERSampleReturnsISWeights() {
	b := NewPrioritizedReplayBuffer(100, 0.6)
	for i := 0; i < 50; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: 1.0})
	}

	_, _, weights := b.Sample(10, 0.6)
	require.Len(s.T(), weights, 10)

	var maxW float64
	for _, w := range weights {
		require.Greater(s.T(), w, 0.0, "IS-weight must be > 0")
		require.LessOrEqual(s.T(), w, 1.0, "IS-weight must be <= 1.0")
		if w > maxW {
			maxW = w
		}
	}
	require.InDelta(s.T(), 1.0, maxW, 1e-9, "max IS-weight must be 1.0")
}

func (s *PrioritizedReplayBufferSuite) TestPERBetaOneGivesUniformWeights() {
	b := NewPrioritizedReplayBuffer(100, 0.6)
	for i := 0; i < 100; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: 1.0})
	}

	_, _, weights := b.Sample(50, 1.0)
	for _, w := range weights {
		require.InDelta(s.T(), 1.0, w, 0.01,
			"with uniform priorities and beta=1.0, all weights should be ~1.0")
	}
}

func (s *PrioritizedReplayBufferSuite) TestBetaScheduleAnnealsProperly() {
	bs := NewBetaSchedule(0.4, 1.0, 100)

	first := bs.Next()
	require.InDelta(s.T(), 0.4, first, 0.01, "first value should be ~start")

	for i := 0; i < 49; i++ {
		bs.Next()
	}
	mid := bs.Next()
	require.InDelta(s.T(), 0.7, mid, 0.02, "halfway should be ~0.7")

	for i := 0; i < 49; i++ {
		bs.Next()
	}
	last := bs.Next()
	require.InDelta(s.T(), 1.0, last, 0.02, "at end should be ~1.0")

	beyond := bs.Next()
	require.InDelta(s.T(), 1.0, beyond, 0.01, "beyond steps should stay at end")
}

func (s *PrioritizedReplayBufferSuite) TestUpdatePriority() {
	b := NewPrioritizedReplayBuffer(10, 0.6)
	for i := 0; i < 5; i++ {
		b.Push(Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: float64(i)})
	}

	b.UpdatePriority(2, 50.0)
	b.UpdatePriority(-1, 1.0)
	b.UpdatePriority(100, 1.0)
}
