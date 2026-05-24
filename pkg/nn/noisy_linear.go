package nn

import (
	"math"
	"math/rand/v2"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

const DefaultNoisySigmaInit = 0.5

// NoisyLinear implements a Factorised Gaussian Noisy Linear layer
// (Fortunato et al., ICLR 2018).
//
// Instead of external epsilon-greedy or softmax exploration, the noise is baked
// into the network weights themselves. Each forward pass samples fresh noise
// via RandomNormal, and the learnable sigma parameters control the noise
// amplitude per-weight. As training progresses and the network becomes more
// confident, sigma shrinks toward zero, making the policy naturally greedy.
//
// The factorised variant reduces the noise parameter count from
// (p*q + q) to (p + q) by decomposing the noise matrix as:
//
//	f(eps_in) ⊗ f(eps_out)    where f(x) = sign(x) * sqrt(|x|)
func NoisyLinear(ctx *context.Context, x *Node, outFeatures int, sigmaInit float64) *Node {
	g := x.Graph()
	xShape := x.Shape()
	inFeatures := xShape.Dimensions[xShape.Rank()-1]

	ctx = ctx.In("noisy")

	bound := 1.0 / math.Sqrt(float64(inFeatures))
	sigmaVal := sigmaInit / math.Sqrt(float64(inFeatures))

	wMuData := uniformSlice(inFeatures*outFeatures, -bound, bound)
	wMuT := tensors.FromFlatDataAndDimensions(wMuData, inFeatures, outFeatures)
	wMu := ctx.VariableWithValue("w_mu", wMuT).ValueGraph(g)

	wSigmaData := constSlice(inFeatures*outFeatures, sigmaVal)
	wSigmaT := tensors.FromFlatDataAndDimensions(wSigmaData, inFeatures, outFeatures)
	wSigma := ctx.VariableWithValue("w_sigma", wSigmaT).ValueGraph(g)

	bMuData := uniformSlice(outFeatures, -bound, bound)
	bMuT := tensors.FromFlatDataAndDimensions(bMuData, outFeatures)
	bMu := ctx.VariableWithValue("b_mu", bMuT).ValueGraph(g)

	bSigmaData := constSlice(outFeatures, sigmaVal)
	bSigmaT := tensors.FromFlatDataAndDimensions(bSigmaData, outFeatures)
	bSigma := ctx.VariableWithValue("b_sigma", bSigmaT).ValueGraph(g)

	epsInShape := shapes.Make(xShape.DType, inFeatures)
	epsOutShape := shapes.Make(xShape.DType, outFeatures)

	epsIn := ctx.RandomNormal(g, epsInShape)
	epsOut := ctx.RandomNormal(g, epsOutShape)

	fIn := factorisedNoise(epsIn)
	fOut := factorisedNoise(epsOut)

	// Outer product: [inFeatures] x [outFeatures] -> [inFeatures, outFeatures]
	noiseW := Einsum("i,j->ij", fIn, fOut)
	noiseB := fOut

	W := Add(wMu, Mul(wSigma, noiseW))
	b := Add(bMu, Mul(bSigma, noiseB))

	return nn.Dense(x, W, b)
}

// NoisyLinearExternal is identical to NoisyLinear except that epsIn and epsOut
// are provided as graph *Node inputs sampled by the caller (e.g. via
// math/rand/v2.NormFloat64 on the CPU) rather than ctx.RandomNormal.
//
// This avoids the gomlx #rngState concurrency issue (concurrent Exec objects
// that share a context call Reset() on the RNG variable during setSideParams,
// racing with each other) while preserving exploration noise on every
// forward pass — which is the core mechanism of NoisyNets (Fortunato 2017 §4).
func NoisyLinearExternal(ctx *context.Context, x, epsIn, epsOut *Node, outFeatures int, sigmaInit float64) *Node {
	g := x.Graph()
	xShape := x.Shape()
	inFeatures := xShape.Dimensions[xShape.Rank()-1]

	ctx = ctx.In("noisy")

	bound := 1.0 / math.Sqrt(float64(inFeatures))
	sigmaVal := sigmaInit / math.Sqrt(float64(inFeatures))

	wMuData := uniformSlice(inFeatures*outFeatures, -bound, bound)
	wMuT := tensors.FromFlatDataAndDimensions(wMuData, inFeatures, outFeatures)
	wMu := ctx.VariableWithValue("w_mu", wMuT).ValueGraph(g)

	wSigmaData := constSlice(inFeatures*outFeatures, sigmaVal)
	wSigmaT := tensors.FromFlatDataAndDimensions(wSigmaData, inFeatures, outFeatures)
	wSigma := ctx.VariableWithValue("w_sigma", wSigmaT).ValueGraph(g)

	bMuData := uniformSlice(outFeatures, -bound, bound)
	bMuT := tensors.FromFlatDataAndDimensions(bMuData, outFeatures)
	bMu := ctx.VariableWithValue("b_mu", bMuT).ValueGraph(g)

	bSigmaData := constSlice(outFeatures, sigmaVal)
	bSigmaT := tensors.FromFlatDataAndDimensions(bSigmaData, outFeatures)
	bSigma := ctx.VariableWithValue("b_sigma", bSigmaT).ValueGraph(g)

	fIn := factorisedNoise(epsIn)
	fOut := factorisedNoise(epsOut)

	noiseW := Einsum("i,j->ij", fIn, fOut)
	noiseB := fOut

	W := Add(wMu, Mul(wSigma, noiseW))
	b := Add(bMu, Mul(bSigma, noiseB))

	return nn.Dense(x, W, b)
}

// NoisyLinearEval performs inference using only the mu (mean) weights,
// without sampling noise. Useful for final evaluation metrics or ONNX export
// after training is complete. NOT suitable as the main inference path during
// training — see NoisyLinearExternal for exploration-preserving inference.
func NoisyLinearEval(ctx *context.Context, x *Node, outFeatures int, sigmaInit float64) *Node {
	g := x.Graph()
	xShape := x.Shape()
	inFeatures := xShape.Dimensions[xShape.Rank()-1]

	ctx = ctx.In("noisy")

	bound := 1.0 / math.Sqrt(float64(inFeatures))

	wMuData := uniformSlice(inFeatures*outFeatures, -bound, bound)
	wMuT := tensors.FromFlatDataAndDimensions(wMuData, inFeatures, outFeatures)
	wMu := ctx.VariableWithValue("w_mu", wMuT).ValueGraph(g)

	bMuData := uniformSlice(outFeatures, -bound, bound)
	bMuT := tensors.FromFlatDataAndDimensions(bMuData, outFeatures)
	bMu := ctx.VariableWithValue("b_mu", bMuT).ValueGraph(g)

	return nn.Dense(x, wMu, bMu)
}

// factorisedNoise applies f(x) = sign(x) * sqrt(|x|).
func factorisedNoise(eps *Node) *Node {
	return Mul(Sign(eps), Sqrt(Abs(eps)))
}

func uniformSlice(n int, lo, hi float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = lo + rand.Float64()*(hi-lo)
	}
	return s
}

func constSlice(n int, val float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = val
	}
	return s
}
