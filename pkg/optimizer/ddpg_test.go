package optimizer

import (
	"testing"

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

func (s *DDPGSuite) TestInitLayer() {
	w, b := initLayer(4, 3)
	require.Len(s.T(), w, 4)
	require.Len(s.T(), b, 4)
	for _, row := range w {
		require.Len(s.T(), row, 3)
	}
}

func (s *DDPGSuite) TestCloneLayer() {
	w, b := initLayer(4, 3)
	wc, bc := cloneLayer(w, b)
	require.Equal(s.T(), w, wc)
	require.Equal(s.T(), b, bc)

	w[0][0] = 999
	require.NotEqual(s.T(), w[0][0], wc[0][0], "clone must be independent")
}

func (s *DDPGSuite) TestLinearForward() {
	tests := []struct {
		name  string
		input []float64
		w     [][]float64
		b     []float64
		want  []float64
	}{
		{
			"identity",
			[]float64{1, 2, 3},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[]float64{0, 0, 0},
			[]float64{1, 2, 3},
		},
		{
			"with_bias",
			[]float64{1, 1},
			[][]float64{{1, 1}},
			[]float64{5},
			[]float64{7},
		},
	}
	for _, tc := range tests {
		s.Run(tc.name, func() {
			got := linearForward(tc.input, tc.w, tc.b)
			require.InDeltaSlice(s.T(), tc.want, got, 1e-9)
		})
	}
}

func (s *DDPGSuite) TestLinearReLU() {
	input := []float64{1, -1}
	w := [][]float64{{1, 0}, {0, 1}, {-1, 0}}
	b := []float64{0, 0, 0}

	got := linearReLU(input, w, b)
	require.Equal(s.T(), 3, len(got))
	require.InDelta(s.T(), 1.0, got[0], 1e-9)
	require.InDelta(s.T(), 0.0, got[1], 1e-9)
	require.InDelta(s.T(), 0.0, got[2], 1e-9)
}

func (s *DDPGSuite) TestSoftUpdate() {
	src := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	dst := [][]float64{{0.0, 0.0}, {0.0, 0.0}}
	softUpdate(src, dst, 0.5)
	require.InDelta(s.T(), 0.5, dst[0][0], 1e-9)
	require.InDelta(s.T(), 1.0, dst[0][1], 1e-9)
}

func (s *DDPGSuite) TestSoftUpdateBias() {
	src := []float64{10.0, 20.0}
	dst := []float64{0.0, 0.0}
	softUpdateBias(src, dst, 0.1)
	require.InDelta(s.T(), 1.0, dst[0], 1e-9)
	require.InDelta(s.T(), 2.0, dst[1], 1e-9)
}

func (s *DDPGSuite) TestUpdateLinear() {
	input := []float64{1.0, 0.5}
	grad := []float64{1.0, -1.0}
	w := [][]float64{{0.0, 0.0}, {0.0, 0.0}}
	b := []float64{0.0, 0.0}
	updateLinear(input, grad, w, b, 0.1)
	require.InDelta(s.T(), 0.1, w[0][0], 1e-9)
	require.InDelta(s.T(), 0.05, w[0][1], 1e-9)
	require.InDelta(s.T(), -0.1, w[1][0], 1e-9)
	require.InDelta(s.T(), 0.1, b[0], 1e-9)
	require.InDelta(s.T(), -0.1, b[1], 1e-9)
}

func (s *DDPGSuite) TestActorForwardOutputInRange() {
	d := NewDDPG(5, 3, 0.001)
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	out := d.actorForward(state, d.actorW1, d.actorB1, d.actorW2, d.actorB2, d.actorW3, d.actorB3)
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
	q := d.criticForward(state, action, d.criticW1, d.criticB1, d.criticW2, d.criticB2, d.criticW3, d.criticB3)
	require.IsType(s.T(), float64(0), q)
}
