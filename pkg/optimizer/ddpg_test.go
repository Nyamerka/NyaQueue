package optimizer

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type DDPGSuite struct {
	suite.Suite
}

func TestDDPGSuite(t *testing.T) { suite.Run(t, new(DDPGSuite)) }

func (s *DDPGSuite) TestAct() {
	tests := []struct {
		name       string
		stateSize  int
		actionSize int
	}{
		{"small", 5, 3},
		{"medium", 20, 10},
		{"single_action", 10, 1},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			d := NewDDPG(tc.stateSize, tc.actionSize, 0.001)
			state := make([]float64, tc.stateSize)
			for i := range state {
				state[i] = float64(i) * 0.1
			}

			action := d.Act(state)
			require.Len(s.T(), action, tc.actionSize)
			for _, a := range action {
				require.GreaterOrEqual(s.T(), a, -1.0)
				require.LessOrEqual(s.T(), a, 1.0)
			}
		})
	}
}

func (s *DDPGSuite) TestStoreAndTrain() {
	d := NewDDPG(5, 3, 0.001)

	for i := 0; i < 200; i++ {
		state := []float64{float64(i), 0.1, 0.2, 0.3, 0.4}
		action := []float64{0.1, -0.2, 0.3}
		nextState := []float64{float64(i + 1), 0.2, 0.3, 0.4, 0.5}
		d.Store(state, action, 1.0, nextState, false)
	}

	require.NotPanics(s.T(), func() { d.Train(64) })
}

func (s *DDPGSuite) TestTrainInsufficientData() {
	d := NewDDPG(5, 3, 0.001)
	d.Store([]float64{1, 2, 3, 4, 5}, []float64{0.1, 0.2, 0.3}, 1.0, []float64{2, 3, 4, 5, 6}, false)

	require.NotPanics(s.T(), func() { d.Train(64) })
}

func (s *DDPGSuite) TestResetNoise() {
	d := NewDDPG(5, 3, 0.001)
	require.NotPanics(s.T(), func() { d.ResetNoise() })
}

func (s *DDPGSuite) TestActorForwardOutputInRange() {
	d := NewDDPG(5, 3, 0.001)
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

	d.mu.Lock()
	exec := context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, st *Node) *Node {
			return d.actorGraph(ctx, st)
		})
	result := exec.MustExec1(state)
	d.mu.Unlock()

	out := tensorToFloat64(result)
	require.Len(s.T(), out, 3)
	for _, v := range out {
		require.GreaterOrEqual(s.T(), v, -1.0)
		require.LessOrEqual(s.T(), v, 1.0)
	}
}

func (s *DDPGSuite) TestCriticForwardReturnsScalar() {
	d := NewDDPG(5, 3, 0.001)
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	action := []float64{0.1, 0.2, 0.3}

	d.mu.Lock()
	q := d.criticForward(state, action)
	d.mu.Unlock()

	require.IsType(s.T(), float64(0), q)
}

func (s *DDPGSuite) TestTrainUpdatesActorWeights() {
	d := NewDDPG(5, 3, 0.01)

	// Get initial action for a fixed state.
	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	actionBefore := d.Act(state)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}
	d.Train(64)

	actionAfter := d.Act(state)

	changed := false
	for i := range actionBefore {
		if actionAfter[i] != actionBefore[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "actor should produce different actions after training")
}

func (s *DDPGSuite) TestSoftUpdateChangesTarget() {
	d := NewDDPG(5, 3, 0.001)

	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}

	// Target actor should initially produce same output as main actor.
	d.mu.Lock()
	mainAction := d.targetActorForward(state)
	d.mu.Unlock()
	require.Len(s.T(), mainAction, 3)

	// After training + soft update, target should change.
	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}
	d.Train(64) // includes soft update

	d.mu.Lock()
	targetAction := d.targetActorForward(state)
	d.mu.Unlock()

	changed := false
	for i := range mainAction {
		if targetAction[i] != mainAction[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "target network should update via soft update")
}
