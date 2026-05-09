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

func (s *ReplayBufferSuite) TestDeterministicWithSeed() {
	rb1 := NewReplayBufferWithSeed(100, 12345)
	rb2 := NewReplayBufferWithSeed(100, 12345)
	for i := 0; i < 50; i++ {
		t := Transition{State: []float64{float64(i)}, Action: []float64{0}, Reward: float64(i)}
		rb1.Push(t)
		rb2.Push(t)
	}

	b1 := rb1.Sample(10)
	b2 := rb2.Sample(10)
	require.Equal(s.T(), len(b1), len(b2))
	for i := range b1 {
		require.Equal(s.T(), b1[i].Reward, b2[i].Reward)
	}
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
