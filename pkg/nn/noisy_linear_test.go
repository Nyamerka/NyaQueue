package nn

import (
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

type NoisyLinearSuite struct {
	suite.Suite
	backend backends.Backend
}

func TestNoisyLinearSuite(t *testing.T) { suite.Run(t, new(NoisyLinearSuite)) }

func (s *NoisyLinearSuite) SetupSuite() {
	s.backend = backends.MustNew()
}

func (s *NoisyLinearSuite) TestForwardVaries() {
	ctx := context.New()
	exec := context.MustNewExec(s.backend, ctx, func(ctx *context.Context, state *Node) *Node {
		x := InsertAxes(state, 0)
		out := NoisyLinear(ctx.In("test"), x, 4, DefaultNoisySigmaInit)
		return Reshape(out, 4)
	})

	input := make([]float64, 8)
	for i := range input {
		input[i] = 0.5
	}

	result1 := exec.MustExec1(input)
	v1 := tensorToSlice(result1)

	result2 := exec.MustExec1(input)
	v2 := tensorToSlice(result2)

	differ := false
	for i := range v1 {
		if v1[i] != v2[i] {
			differ = true
			break
		}
	}
	require.True(s.T(), differ,
		"two forward passes with same input must produce different outputs due to noise resampling")
}

func (s *NoisyLinearSuite) TestOutputShape() {
	ctx := context.New()
	outDim := 6
	exec := context.MustNewExec(s.backend, ctx, func(ctx *context.Context, state *Node) *Node {
		x := InsertAxes(state, 0)
		out := NoisyLinear(ctx.In("shape_test"), x, outDim, DefaultNoisySigmaInit)
		return Reshape(out, outDim)
	})

	input := make([]float64, 4)
	result := exec.MustExec1(input)
	v := tensorToSlice(result)
	require.Len(s.T(), v, outDim, "output should have outFeatures elements")
}

func (s *NoisyLinearSuite) TestSigmaVariablesExist() {
	ctx := context.New()
	exec := context.MustNewExec(s.backend, ctx, func(ctx *context.Context, state *Node) *Node {
		x := InsertAxes(state, 0)
		return NoisyLinear(ctx.In("sigma_check"), x, 4, DefaultNoisySigmaInit)
	})
	exec.MustExec(make([]float64, 8))

	var foundWSigma, foundBSigma bool
	ctx.EnumerateVariables(func(v *context.Variable) {
		name := v.ScopeAndName()
		if contains(name, "w_sigma") {
			foundWSigma = true
		}
		if contains(name, "b_sigma") {
			foundBSigma = true
		}
	})
	require.True(s.T(), foundWSigma, "w_sigma variable must be registered")
	require.True(s.T(), foundBSigma, "b_sigma variable must be registered")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func (s *NoisyLinearSuite) TestNoisyLinearEvalDeterministic() {
	ctx := context.New()

	initExec := context.MustNewExec(s.backend, ctx,
		func(ctx *context.Context, state *Node) *Node {
			x := InsertAxes(state, 0)
			return NoisyLinear(ctx.In("eval_det"), x, 4, DefaultNoisySigmaInit)
		})
	initExec.MustExec(make([]float64, 8))
	initExec.Finalize()

	evalExec := context.MustNewExec(s.backend, ctx.Reuse(),
		func(ctx *context.Context, state *Node) *Node {
			x := InsertAxes(state, 0)
			out := NoisyLinearEval(ctx.In("eval_det"), x, 4, DefaultNoisySigmaInit)
			return Reshape(out, 4)
		})

	input := make([]float64, 8)
	for i := range input {
		input[i] = 0.5
	}

	r1 := tensorToSlice(evalExec.MustExec1(input))
	r2 := tensorToSlice(evalExec.MustExec1(input))
	for i := range r1 {
		require.Equal(s.T(), r1[i], r2[i],
			"NoisyLinearEval must be deterministic (same input → same output)")
	}
}

func (s *NoisyLinearSuite) TestNoisyLinearExternalExploresOnInference() {
	ctx := context.New()

	initExec := context.MustNewExec(s.backend, ctx,
		func(ctx *context.Context, state *Node) *Node {
			x := InsertAxes(state, 0)
			return NoisyLinear(ctx.In("ext_explore"), x, 4, DefaultNoisySigmaInit)
		})
	initExec.MustExec(make([]float64, 8))
	initExec.Finalize()

	extExec := context.MustNewExec(s.backend, ctx.Reuse(),
		func(ctx *context.Context, state, epsIn, epsOut *Node) *Node {
			x := InsertAxes(state, 0)
			out := NoisyLinearExternal(ctx.In("ext_explore"), x, epsIn, epsOut, 4, DefaultNoisySigmaInit)
			return Reshape(out, 4)
		})

	input := make([]float64, 8)
	for i := range input {
		input[i] = 0.5
	}

	epsIn1 := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	epsOut1 := []float64{0.1, 0.2, 0.3, 0.4}
	epsIn2 := []float64{-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1}
	epsOut2 := []float64{-0.4, -0.3, -0.2, -0.1}

	r1 := tensorToSlice(extExec.MustExec1(input, epsIn1, epsOut1))
	r2 := tensorToSlice(extExec.MustExec1(input, epsIn2, epsOut2))

	differ := false
	for i := range r1 {
		if r1[i] != r2[i] {
			differ = true
			break
		}
	}
	require.True(s.T(), differ,
		"NoisyLinearExternal with different eps should produce different outputs (exploration)")

	argmax1, argmax2 := 0, 0
	for i := 1; i < len(r1); i++ {
		if r1[i] > r1[argmax1] {
			argmax1 = i
		}
		if r2[i] > r2[argmax2] {
			argmax2 = i
		}
	}
	_ = argmax1
	_ = argmax2
}

func (s *NoisyLinearSuite) TestNoisyLinearExternalConcurrent() {
	ctx := context.New()

	initExec := context.MustNewExec(s.backend, ctx,
		func(ctx *context.Context, state *Node) *Node {
			x := InsertAxes(state, 0)
			return NoisyLinear(ctx.In("ext_conc"), x, 4, DefaultNoisySigmaInit)
		})
	initExec.MustExec(make([]float64, 8))
	initExec.Finalize()

	input := make([]float64, 8)
	for i := range input {
		input[i] = 0.5
	}

	const numGoroutines = 32
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	require.NotPanics(s.T(), func() {
		for g := 0; g < numGoroutines; g++ {
			go func(id int) {
				defer wg.Done()
				exec := context.MustNewExec(s.backend, ctx.Reuse(),
					func(ctx *context.Context, state, epsIn, epsOut *Node) *Node {
						x := InsertAxes(state, 0)
						out := NoisyLinearExternal(ctx.In("ext_conc"), x, epsIn, epsOut, 4, DefaultNoisySigmaInit)
						return Reshape(out, 4)
					})
				for iter := 0; iter < 10; iter++ {
					eps := make([]float64, 8)
					epsOut := make([]float64, 4)
					for j := range eps {
						eps[j] = float64(id*10+iter) * 0.01
					}
					for j := range epsOut {
						epsOut[j] = float64(id*10+iter) * 0.02
					}
					exec.MustExec(input, eps, epsOut)
				}
			}(g)
		}
		wg.Wait()
	}, "32 concurrent goroutines with NoisyLinearExternal must not panic on #rngState")
}

func (s *NoisyLinearSuite) TestNoisyLinearExternalSigmaShrinks() {
	ctx := context.New()
	ctx.SetParams(map[string]any{
		"optimizer":     "adam",
		"learning_rate": 0.01,
	})

	inFeatures := 8
	outFeatures := 4

	trainExec := context.MustNewExec(s.backend, ctx,
		func(mlCtx *context.Context, state, epsIn, epsOut, target *Node) *Node {
			g := state.Graph()
			x := InsertAxes(state, 0)
			out := NoisyLinearExternal(mlCtx.In("sigma_shrink"), x, epsIn, epsOut, outFeatures, DefaultNoisySigmaInit)
			out = Reshape(out, outFeatures)
			diff := Sub(out, target)
			loss := ReduceAllMean(Mul(diff, diff))

			opt := optimizers.FromContext(mlCtx)
			opt.UpdateGraph(mlCtx, g, loss)
			return loss
		})

	state := make([]float64, inFeatures)
	for i := range state {
		state[i] = 0.5
	}
	target := make([]float64, outFeatures)

	// Initialize with non-zero noise so sigma variables participate in the loss.
	epsIn := make([]float64, inFeatures)
	epsOut := make([]float64, outFeatures)
	for i := range epsIn {
		epsIn[i] = 1.0
	}
	for i := range epsOut {
		epsOut[i] = 1.0
	}
	trainExec.MustExec(state, epsIn, epsOut, target)

	collectSigmaNorm := func() float64 {
		var norm float64
		ctx.EnumerateVariables(func(v *context.Variable) {
			if !contains(v.ScopeAndName(), "sigma") {
				return
			}
			val := v.MustValue()
			switch d := val.Value().(type) {
			case []float64:
				for _, x := range d {
					norm += x * x
				}
			case [][]float64:
				for _, row := range d {
					for _, x := range row {
						norm += x * x
					}
				}
			}
		})
		return norm
	}

	sigmaBefore := collectSigmaNorm()
	require.Greater(s.T(), sigmaBefore, 0.0, "sigma norm should be positive after initialization")

	// Train with varying non-zero noise against zero target.
	// With a stable target, sigma should shrink because noise is detrimental.
	for step := 0; step < 2000; step++ {
		sign := float64(1 - 2*(step%2))
		for i := range epsIn {
			epsIn[i] = sign * float64(i+1) * 0.3
		}
		for i := range epsOut {
			epsOut[i] = sign * float64(i+1) * 0.3
		}
		trainExec.MustExec(state, epsIn, epsOut, target)
	}

	sigmaAfter := collectSigmaNorm()
	require.Less(s.T(), sigmaAfter, sigmaBefore,
		"sigma parameters should shrink after training on a stable target (before=%.6f, after=%.6f)",
		sigmaBefore, sigmaAfter)
}

func tensorToSlice(t interface{ Value() any }) []float64 {
	val := t.Value()
	switch v := val.(type) {
	case []float64:
		out := make([]float64, len(v))
		copy(out, v)
		return out
	case []float32:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	case float64:
		return []float64{v}
	case float32:
		return []float64{float64(v)}
	default:
		return nil
	}
}
