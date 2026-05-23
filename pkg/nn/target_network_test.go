package nn

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

type TargetNetworkSuite struct {
	suite.Suite
	backend backends.Backend
}

func TestTargetNetworkSuite(t *testing.T) { suite.Run(t, new(TargetNetworkSuite)) }

func (s *TargetNetworkSuite) SetupSuite() {
	s.backend = backends.MustNew()
}

func buildTestNetwork(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	h := layers.Dense(ctx.In("hidden"), x, true, 16)
	h = activations.Relu(h)
	q := layers.Dense(ctx.In("output"), h, true, 4)
	return Reshape(q, 4)
}

func (s *TargetNetworkSuite) TestHardSyncCopiesWeights() {
	mainCtx := context.New()
	fwd := context.MustNewExec(s.backend, mainCtx, buildTestNetwork)
	fwd.MustExec(make([]float64, 8))

	tn := NewTargetNetwork(mainCtx, WithSyncEvery(1))

	mainCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			val := v.MustValue()
			val.MustMutableFlatData(func(flat any) {
				switch f := flat.(type) {
				case []float32:
					for i := range f {
						f[i] = 100.0
					}
				case []float64:
					for i := range f {
						f[i] = 100.0
					}
				}
			})
		}
	})

	tn.HardSync()

	tn.targetCtx.EnumerateVariables(func(v *context.Variable) {
		if !v.Trainable {
			return
		}
		val := v.MustValue()
		val.MustConstFlatData(func(flat any) {
			switch f := flat.(type) {
			case []float32:
				for _, w := range f {
					require.InDelta(s.T(), 100.0, float64(w), 0.01,
						"after hard sync, target weight should equal main weight")
				}
			case []float64:
				for _, w := range f {
					require.InDelta(s.T(), 100.0, w, 0.01,
						"after hard sync, target weight should equal main weight")
				}
			}
		})
	})
}

func (s *TargetNetworkSuite) TestTargetCtxIndependentFromMain() {
	mainCtx := context.New()
	fwd := context.MustNewExec(s.backend, mainCtx, buildTestNetwork)
	fwd.MustExec(make([]float64, 8))

	tn := NewTargetNetwork(mainCtx)

	var targetBefore []float64
	tn.targetCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			val := v.MustValue()
			val.MustConstFlatData(func(flat any) {
				switch f := flat.(type) {
				case []float32:
					for _, x := range f {
						targetBefore = append(targetBefore, float64(x))
					}
				case []float64:
					cp := make([]float64, len(f))
					copy(cp, f)
					targetBefore = append(targetBefore, cp...)
				}
			})
		}
	})

	mainCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			val := v.MustValue()
			val.MustMutableFlatData(func(flat any) {
				switch f := flat.(type) {
				case []float32:
					for i := range f {
						f[i] = 999.0
					}
				case []float64:
					for i := range f {
						f[i] = 999.0
					}
				}
			})
		}
	})

	var targetAfter []float64
	tn.targetCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			val := v.MustValue()
			val.MustConstFlatData(func(flat any) {
				switch f := flat.(type) {
				case []float32:
					for _, x := range f {
						targetAfter = append(targetAfter, float64(x))
					}
				case []float64:
					cp := make([]float64, len(f))
					copy(cp, f)
					targetAfter = append(targetAfter, cp...)
				}
			})
		}
	})

	require.Equal(s.T(), targetBefore, targetAfter,
		"target weights must not change when main weights change (until explicit sync)")
}
