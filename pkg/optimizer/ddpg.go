package optimizer

import (
	"math"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

const (
	ddpgHiddenSize = 128
	ddpgTau        = 0.005
)

// DDPG implements Deep Deterministic Policy Gradient using GoMLX for automatic
// differentiation. Actor maps state → action (tanh-scaled), Critic maps
// (state, action) → Q-value. Training uses exact gradients via GoMLX autograd
// instead of numerical finite differences.
type DDPG struct {
	mu sync.Mutex

	stateSize  int
	actionSize int
	lr         float64
	gamma      float64

	backend   backends.Backend
	mainCtx   *context.Context
	targetCtx *context.Context

	replayBuffer *nn.ReplayBuffer
	noise        *nn.OUNoise
}

func NewDDPG(stateSize, actionSize int, lr float64) *DDPG {
	d := &DDPG{
		stateSize:    stateSize,
		actionSize:   actionSize,
		lr:           lr,
		gamma:        0.99,
		replayBuffer: nn.NewReplayBuffer(1000000),
		noise:        nn.NewOUNoise(actionSize, 0, 0.15, 0.2),
	}

	d.backend = backends.MustNew()
	d.mainCtx = context.New()

	d.initNetworks()

	d.targetCtx = d.cloneContext(d.mainCtx)

	return d
}

func (d *DDPG) initNetworks() {
	dummyState := make([]float64, d.stateSize)
	dummyAction := make([]float64, d.actionSize)

	// Init actor variables.
	actorExec := context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, state *Node) *Node {
			return d.actorGraph(ctx, state)
		})
	actorExec.MustExec(dummyState)

	// Init critic variables.
	criticExec := context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, state, action *Node) *Node {
			return d.criticGraph(ctx, state, action)
		})
	criticExec.MustExec(dummyState, dummyAction)
}

// actorGraph: Dense(128) → ReLU → Dense(128) → ReLU → Dense(actionSize) → Tanh
// Variables are scoped under "actor/".
func (d *DDPG) actorGraph(ctx *context.Context, state *Node) *Node {
	actorCtx := ctx.In("actor")
	x := InsertAxes(state, 0)
	h := layers.Dense(actorCtx.In("l1"), x, true, ddpgHiddenSize)
	h = activations.Relu(h)
	h = layers.Dense(actorCtx.In("l2"), h, true, ddpgHiddenSize)
	h = activations.Relu(h)
	out := layers.Dense(actorCtx.In("l3"), h, true, d.actionSize)
	out = Tanh(out)
	return Reshape(out, d.actionSize)
}

// criticGraph: concat(state, action) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(1)
// Variables are scoped under "critic/".
func (d *DDPG) criticGraph(ctx *context.Context, state, action *Node) *Node {
	criticCtx := ctx.In("critic")
	s := InsertAxes(state, 0)
	a := InsertAxes(action, 0)
	input := Concatenate([]*Node{s, a}, -1)
	h := layers.Dense(criticCtx.In("l1"), input, true, ddpgHiddenSize)
	h = activations.Relu(h)
	h = layers.Dense(criticCtx.In("l2"), h, true, ddpgHiddenSize)
	h = activations.Relu(h)
	q := layers.Dense(criticCtx.In("l3"), h, true, 1)
	return Reshape(q) // scalar
}

func (d *DDPG) Act(state []float64) []float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	exec := context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, st *Node) *Node {
			return d.actorGraph(ctx, st)
		})
	result := exec.MustExec1(state)
	action := tensorToFloat64(result)

	noise := d.noise.Sample()
	for i := range action {
		action[i] = math.Max(-1, math.Min(1, action[i]+noise[i]))
	}
	return action
}

func (d *DDPG) Store(state, action []float64, reward float64, nextState []float64, done bool) {
	ac := make([]float64, len(action))
	copy(ac, action)
	d.replayBuffer.Push(nn.Transition{
		State: state, Action: ac, Reward: reward,
		NextState: nextState, Done: done,
	})
}

func (d *DDPG) Train(batchSize int) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.replayBuffer.Len() < batchSize {
		return
	}

	batch := d.replayBuffer.Sample(batchSize)

	for _, t := range batch {
		action := t.Action
		if len(action) < d.actionSize {
			padded := make([]float64, d.actionSize)
			copy(padded, action)
			action = padded
		}

		nextAction := d.targetActorForward(t.NextState)
		nextQ := d.targetCriticForward(t.NextState, nextAction)
		targetQ := t.Reward
		if !t.Done {
			targetQ += d.gamma * nextQ
		}

		currentQ := d.criticForward(t.State, action)
		tdError := targetQ - currentQ
		d.updateCritic(t.State, action, tdError)
		d.updateActor(t.State)
	}

	d.softUpdateContext(d.mainCtx, d.targetCtx, ddpgTau)
}

func (d *DDPG) ResetNoise() {
	d.noise.Reset()
}

func (d *DDPG) targetActorForward(state []float64) []float64 {
	exec := context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, st *Node) *Node {
			return d.actorGraph(ctx, st)
		})
	result := exec.MustExec1(state)
	return tensorToFloat64(result)
}

func (d *DDPG) targetCriticForward(state, action []float64) float64 {
	exec := context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, st, act *Node) *Node {
			return d.criticGraph(ctx, st, act)
		})
	result := exec.MustExec1(state, action)
	return tensorToScalar(result)
}

func (d *DDPG) criticForward(state, action []float64) float64 {
	exec := context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, st, act *Node) *Node {
			return d.criticGraph(ctx, st, act)
		})
	result := exec.MustExec1(state, action)
	return tensorToScalar(result)
}

// updateCritic applies gradient descent on the critic to minimize TD error.
func (d *DDPG) updateCritic(state, action []float64, tdError float64) {
	exec := context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, st, act, tdErr *Node) *Node {
			g := st.Graph()
			q := d.criticGraph(ctx, st, act)
			loss := Neg(Mul(tdErr, q))

			// Only compute gradients for critic variables.
			criticCtx := ctx.In("critic")
			grads := criticCtx.BuildTrainableVariablesGradientsGraph(loss)
			lr := Const(g, d.lr)
			idx := 0
			criticCtx.EnumerateVariables(func(v *context.Variable) {
				if v.Trainable && v.InUseByGraph(g) {
					w := v.ValueGraph(g)
					v.SetValueGraph(Sub(w, Mul(lr, grads[idx])))
					idx++
				}
			})
			return q
		})
	exec.MustExec(state, action, tdError)
}

// updateActor uses exact GoMLX autograd to compute dQ/dθ_actor where
// Q = critic(s, actor(s)). This replaces numerical finite differences.
func (d *DDPG) updateActor(state []float64) {
	exec := context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, st *Node) *Node {
			g := st.Graph()

			// Forward through actor.
			action := d.actorGraph(ctx, st)

			// Forward through critic (using same mainCtx).
			q := d.criticGraph(ctx, st, action)

			// Maximize Q → minimize -Q, but only update actor weights.
			loss := Neg(q)
			actorCtx := ctx.In("actor")
			grads := actorCtx.BuildTrainableVariablesGradientsGraph(loss)
			lr := Const(g, d.lr)
			idx := 0
			actorCtx.EnumerateVariables(func(v *context.Variable) {
				if v.Trainable && v.InUseByGraph(g) {
					w := v.ValueGraph(g)
					v.SetValueGraph(Sub(w, Mul(lr, grads[idx])))
					idx++
				}
			})
			return q
		})
	exec.MustExec(state)
}

func (d *DDPG) cloneContext(src *context.Context) *context.Context {
	dst := context.New()
	src.EnumerateVariables(func(v *context.Variable) {
		srcT := v.MustValue()
		cloned, _ := srcT.LocalClone()
		dst.InAbsPath(v.Scope()).VariableWithValue(v.Name(), cloned.Value())
	})
	return dst
}

func (d *DDPG) softUpdateContext(src, target *context.Context, tau float64) {
	src.EnumerateVariables(func(srcVar *context.Variable) {
		if !srcVar.Trainable {
			return
		}
		tgtVar := target.InAbsPath(srcVar.Scope()).GetVariable(srcVar.Name())
		if tgtVar == nil {
			return
		}
		srcT := srcVar.MustValue()
		tgtT := tgtVar.MustValue()

		srcT.MutableFlatData(func(srcFlat any) {
			tgtT.MutableFlatData(func(tgtFlat any) {
				switch srcData := srcFlat.(type) {
				case []float64:
					tgtData := tgtFlat.([]float64)
					for i := range tgtData {
						if i < len(srcData) {
							tgtData[i] = (1-tau)*tgtData[i] + tau*srcData[i]
						}
					}
				case []float32:
					tgtData := tgtFlat.([]float32)
					tau32 := float32(tau)
					for i := range tgtData {
						if i < len(srcData) {
							tgtData[i] = (1-tau32)*tgtData[i] + tau32*srcData[i]
						}
					}
				}
			})
		})
	})
}

func tensorToFloat64(t *tensors.Tensor) []float64 {
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

func tensorToScalar(t *tensors.Tensor) float64 {
	val := t.Value()
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case []float64:
		if len(v) > 0 {
			return v[0]
		}
	case []float32:
		if len(v) > 0 {
			return float64(v[0])
		}
	}
	return 0
}
