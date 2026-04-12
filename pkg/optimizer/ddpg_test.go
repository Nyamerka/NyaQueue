package optimizer

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
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

func (s *DDPGSuite) TestNewDDPGLayer() {
	l := newDDPGLayer(4, 3)
	r, c := l.W.Dims()
	require.Equal(s.T(), 4, r)
	require.Equal(s.T(), 3, c)
	require.Len(s.T(), l.B, 4)
}

func (s *DDPGSuite) TestLayerClone() {
	l := newDDPGLayer(4, 3)
	lc := l.clone()

	r, c := lc.W.Dims()
	require.Equal(s.T(), 4, r)
	require.Equal(s.T(), 3, c)

	l.W.Set(0, 0, 999)
	require.NotEqual(s.T(), 999.0, lc.W.At(0, 0), "clone must be independent")
}

func (s *DDPGSuite) TestLayerForward() {
	l := ddpgLayer{
		W: mat.NewDense(3, 3, []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		}),
		B: []float64{0, 0, 0},
	}
	got := l.forward([]float64{1, 2, 3})
	require.InDeltaSlice(s.T(), []float64{1, 2, 3}, got, 1e-9)

	lBias := ddpgLayer{
		W: mat.NewDense(1, 2, []float64{1, 1}),
		B: []float64{5},
	}
	got2 := lBias.forward([]float64{1, 1})
	require.InDeltaSlice(s.T(), []float64{7}, got2, 1e-9)
}

func (s *DDPGSuite) TestLayerForwardReLU() {
	l := ddpgLayer{
		W: mat.NewDense(3, 2, []float64{1, 0, 0, 1, -1, 0}),
		B: []float64{0, 0, 0},
	}
	got := l.forwardReLU([]float64{1, -1})
	require.Len(s.T(), got, 3)
	require.InDelta(s.T(), 1.0, got[0], 1e-9)
	require.InDelta(s.T(), 0.0, got[1], 1e-9)
	require.InDelta(s.T(), 0.0, got[2], 1e-9)
}

func (s *DDPGSuite) TestSoftUpdateLayer() {
	src := ddpgLayer{
		W: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
		B: []float64{10, 20},
	}
	dst := ddpgLayer{
		W: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		B: []float64{0, 0},
	}
	softUpdateLayer(src, dst, 0.5)
	require.InDelta(s.T(), 0.5, dst.W.At(0, 0), 1e-9)
	require.InDelta(s.T(), 1.0, dst.W.At(0, 1), 1e-9)
	require.InDelta(s.T(), 5.0, dst.B[0], 1e-9)
}

func (s *DDPGSuite) TestDDPGUpdateLayer() {
	l := ddpgLayer{
		W: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		B: []float64{0, 0},
	}
	ddpgUpdateLayer([]float64{1.0, 0.5}, []float64{1.0, -1.0}, l, 0.1)
	require.InDelta(s.T(), 0.1, l.W.At(0, 0), 1e-9)
	require.InDelta(s.T(), 0.05, l.W.At(0, 1), 1e-9)
	require.InDelta(s.T(), -0.1, l.W.At(1, 0), 1e-9)
	require.InDelta(s.T(), 0.1, l.B[0], 1e-9)
	require.InDelta(s.T(), -0.1, l.B[1], 1e-9)
}

func (s *DDPGSuite) TestActorForwardOutputInRange() {
	d := NewDDPG(5, 3, 0.001)
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	out := d.actorFwd(state, d.actor1, d.actor2, d.actor3)
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
	q := d.criticFwd(state, action, d.critic1, d.critic2, d.critic3)
	require.IsType(s.T(), float64(0), q)
}

func (s *DDPGSuite) TestTrainUpdatesAllActorLayers() {
	d := NewDDPG(5, 3, 0.01)
	w1Before := cloneMatData(d.actor1.W)
	w2Before := cloneMatData(d.actor2.W)
	w3Before := cloneMatData(d.actor3.W)

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

	w1Changed := !mat.Equal(w1Before, d.actor1.W)
	w2Changed := !mat.Equal(w2Before, d.actor2.W)
	w3Changed := !mat.Equal(w3Before, d.actor3.W)

	require.True(s.T(), w3Changed, "actor W3 should change")
	require.True(s.T(), w2Changed, "actor W2 should change (full backprop)")
	require.True(s.T(), w1Changed, "actor W1 should change (full backprop)")
}

func (s *DDPGSuite) TestMatTransposeVecMul() {
	w := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	v := []float64{1, 1}
	got := matTransposeVecMul(w, v)
	require.Len(s.T(), got, 3)
	require.InDelta(s.T(), 5.0, got[0], 1e-9)
	require.InDelta(s.T(), 7.0, got[1], 1e-9)
	require.InDelta(s.T(), 9.0, got[2], 1e-9)
}
