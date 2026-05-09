package optimizer

import (
	"log"
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

// DDPG hyperparameters — design parameters fixed at network creation time.
// These are NOT runtime-tunable via ApplyConfig because changing them would
// require rebuilding the neural network graph (hidden size changes layer
// dimensions; tau is baked into the compiled softUpdateExec graph).
//
// For the thesis (Chapter 4): lr, gamma, ddpgHiddenSize, ddpgTau, and batchSize
// are design choices selected during architecture design, not dynamic knobs.
// The 22 runtime-tunable broker parameters (SegmentMaxBytes, NumIOGoroutines, etc.)
// are adjusted by the DDPG optimizer itself.
const (
	ddpgHiddenSize = 128   // neurons per hidden layer (actor & critic)
	ddpgTau        = 0.005 // soft update blend coefficient (target ← main)
)

// DDPG implements Deep Deterministic Policy Gradient using GoMLX for automatic
// differentiation. Actor maps state → action (tanh-scaled), Critic maps
// (state, action) → Q-value. Training uses exact gradients via GoMLX autograd
// instead of numerical finite differences.
//
// All GoMLX Exec objects are pre-compiled once in initNetworks() and reused
// for every forward/backward call, eliminating per-call graph compilation overhead.
// Training operates on batched tensors [B, *] for proper SGD with gradient averaging.
//
// Soft update is performed via a pre-compiled GoMLX graph operation that blends
// main→target weights through SetValueGraph, natively handling any backend dtype.
type DDPG struct {
	mu sync.Mutex

	stateSize  int
	actionSize int
	lr         float64
	gamma      float64
	batchSize  int

	backend   backends.Backend
	mainCtx   *context.Context
	targetCtx *context.Context

	actorFwdExec       *context.Exec
	targetActorFwdExec *context.Exec
	targetCriticExec   *context.Exec
	criticTrainExec    *context.Exec
	actorTrainExec     *context.Exec

	// varPairs caches main→target variable pairs for softUpdate,
	// collected once during initialization to avoid repeated enumeration.
	varPairs []varPair

	replayBuffer *nn.ReplayBuffer
	noise        *nn.OUNoise

	statesBuf     []float64
	nextStatesBuf []float64
	actionsBuf    []float64
	rewardsBuf    []float64
	donesBuf      []float64
	targetQBuf    []float64
}

func NewDDPG(stateSize, actionSize int, lr float64) *DDPG {
	bs := 64
	d := &DDPG{
		stateSize:    stateSize,
		actionSize:   actionSize,
		lr:           lr,
		gamma:        0.99,
		batchSize:    bs,
		replayBuffer: nn.NewReplayBuffer(1000000),
		noise:        nn.NewOUNoise(actionSize, 0, 0.15, 0.2),

		statesBuf:     make([]float64, bs*stateSize),
		nextStatesBuf: make([]float64, bs*stateSize),
		actionsBuf:    make([]float64, bs*actionSize),
		rewardsBuf:    make([]float64, bs),
		donesBuf:      make([]float64, bs),
		targetQBuf:    make([]float64, bs),
	}

	d.backend = backends.MustNew()
	d.mainCtx = context.New()

	d.initNetworks()

	d.targetCtx = d.cloneContext(d.mainCtx)
	d.initTargetExecs()

	return d
}

func (d *DDPG) initNetworks() {
	dummyState := make([]float64, d.stateSize)

	// actorFwdExec: single-sample inference using batched graph with InsertAxes.
	// Output is flattened from [1, actionSize] → [actionSize] for Go consumption.
	// This is the first exec — it creates actor variables in mainCtx.
	d.actorFwdExec = context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, state *Node) *Node {
			out := d.actorGraphBatched(ctx, InsertAxes(state, 0))
			return Reshape(out, d.actionSize)
		})
	d.actorFwdExec.MustExec(dummyState)

	// criticTrainExec creates critic variables in mainCtx.
	// Reuse() because actor variables already exist.
	d.criticTrainExec = context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, states, actions, targetQs *Node) *Node {
			g := states.Graph()
			q := d.criticGraphBatched(ctx, states, actions)
			diff := Sub(q, targetQs)
			loss := ReduceAllMean(Mul(diff, diff))

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
			return loss
		})
	// Dummy exec to initialize critic variables.
	dummyBatchStates := make([]float64, d.stateSize)
	dummyBatchActions := make([]float64, d.actionSize)
	dummyTargetQ := []float64{0}
	d.criticTrainExec.MustExec(
		tensors.FromFlatDataAndDimensions(dummyBatchStates, 1, d.stateSize),
		tensors.FromFlatDataAndDimensions(dummyBatchActions, 1, d.actionSize),
		tensors.FromFlatDataAndDimensions(dummyTargetQ, 1),
	)

	d.actorTrainExec = context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			g := states.Graph()
			actions := d.actorGraphBatched(ctx, states)
			q := d.criticGraphBatched(ctx, states, actions)
			loss := Neg(ReduceAllMean(q))

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
			return loss
		})
}

// varPair holds a matched source (main) and target variable for soft updates.
type varPair struct {
	src *context.Variable
	tgt *context.Variable
}

func (d *DDPG) initTargetExecs() {
	d.targetActorFwdExec = context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			return d.actorGraphBatched(ctx, states)
		})

	d.targetCriticExec = context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, states, actions *Node) *Node {
			return d.criticGraphBatched(ctx, states, actions)
		})

	// Pre-collect variable pairs for softUpdate. Avoids repeated enumeration
	// and ensures stable ordering across calls.
	d.mainCtx.EnumerateVariables(func(srcVar *context.Variable) {
		if !srcVar.Trainable {
			return
		}
		tgtVar := d.targetCtx.InAbsPath(srcVar.Scope()).GetVariable(srcVar.Name())
		if tgtVar == nil {
			return
		}
		d.varPairs = append(d.varPairs, varPair{src: srcVar, tgt: tgtVar})
	})
}

// softUpdate blends main→target weights: tgt = (1-tau)*tgt + tau*src.
// Operates in pure Go to avoid GoMLX graph parameter name collisions between
// mainCtx and targetCtx (which share identical scope/name paths).
// Handles both float64 and float32 backend dtypes.
func (d *DDPG) softUpdate() {
	for _, p := range d.varPairs {
		srcVal := p.src.MustValue()
		tgtVal := p.tgt.MustValue()

		switch srcData := srcVal.Value().(type) {
		case []float64:
			tgtData, ok := tgtVal.Value().([]float64)
			if !ok {
				log.Printf("DDPG softUpdate: dtype mismatch for %s/%s: src=float64, tgt=%T",
					p.src.Scope(), p.src.Name(), tgtVal.Value())
				continue
			}
			for i := range tgtData {
				tgtData[i] = (1-ddpgTau)*tgtData[i] + ddpgTau*srcData[i]
			}
		case []float32:
			tgtData, ok := tgtVal.Value().([]float32)
			if !ok {
				log.Printf("DDPG softUpdate: dtype mismatch for %s/%s: src=float32, tgt=%T",
					p.src.Scope(), p.src.Name(), tgtVal.Value())
				continue
			}
			tau32 := float32(ddpgTau)
			for i := range tgtData {
				tgtData[i] = (1-tau32)*tgtData[i] + tau32*srcData[i]
			}
		default:
			log.Printf("DDPG softUpdate: unsupported dtype for %s/%s: %T",
				p.src.Scope(), p.src.Name(), srcVal.Value())
		}
	}
}

// actorGraphBatched: Dense(128) → ReLU → Dense(128) → ReLU → Dense(actionSize) → Tanh.
// Input shape [B, stateSize], output shape [B, actionSize].
func (d *DDPG) actorGraphBatched(ctx *context.Context, states *Node) *Node {
	actorCtx := ctx.In("actor")
	h := layers.Dense(actorCtx.In("l1"), states, true, ddpgHiddenSize)
	h = activations.Relu(h)
	h = layers.Dense(actorCtx.In("l2"), h, true, ddpgHiddenSize)
	h = activations.Relu(h)
	out := layers.Dense(actorCtx.In("l3"), h, true, d.actionSize)
	return Tanh(out)
}

// criticGraphBatched: concat(states, actions) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(1) → Squeeze.
// Input shapes [B, stateSize], [B, actionSize]; output shape [B].
func (d *DDPG) criticGraphBatched(ctx *context.Context, states, actions *Node) *Node {
	criticCtx := ctx.In("critic")
	input := Concatenate([]*Node{states, actions}, -1)
	h := layers.Dense(criticCtx.In("l1"), input, true, ddpgHiddenSize)
	h = activations.Relu(h)
	h = layers.Dense(criticCtx.In("l2"), h, true, ddpgHiddenSize)
	h = activations.Relu(h)
	q := layers.Dense(criticCtx.In("l3"), h, true, 1)
	return Squeeze(q, -1)
}

func (d *DDPG) Act(state []float64) []float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	result := d.actorFwdExec.MustExec1(state)
	action := tensorToFloat64(result)

	noise := d.noise.Sample()
	for i := range action {
		action[i] = math.Max(-1, math.Min(1, action[i]+noise[i]))
	}
	return action
}

func (d *DDPG) Store(state, action []float64, reward float64, nextState []float64, done bool) {
	d.replayBuffer.Push(nn.Transition{
		State:     append([]float64(nil), state...),
		Action:    append([]float64(nil), action...),
		Reward:    reward,
		NextState: append([]float64(nil), nextState...),
		Done:      done,
	})
}

// Train performs one batched SGD step on both critic and actor using
// the full mini-batch at once (proper gradient averaging).
// Pre-allocated buffers in the struct eliminate per-call allocations for tensor data.
func (d *DDPG) Train(batchSize int) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.replayBuffer.Len() < batchSize {
		return
	}

	batch := d.replayBuffer.Sample(batchSize)
	bs := len(batch)
	if bs < batchSize {
		return
	}

	d.ensureBuffers(bs)

	for i, t := range batch {
		copyPadded(d.statesBuf[i*d.stateSize:], t.State, d.stateSize)
		copyPadded(d.nextStatesBuf[i*d.stateSize:], t.NextState, d.stateSize)
		copyPadded(d.actionsBuf[i*d.actionSize:], t.Action, d.actionSize)
		d.rewardsBuf[i] = t.Reward
		if t.Done {
			d.donesBuf[i] = 1.0
		} else {
			d.donesBuf[i] = 0.0
		}
	}

	statesT := tensors.FromFlatDataAndDimensions(d.statesBuf[:bs*d.stateSize], bs, d.stateSize)
	nextStatesT := tensors.FromFlatDataAndDimensions(d.nextStatesBuf[:bs*d.stateSize], bs, d.stateSize)

	nextActionsT := d.targetActorFwdExec.MustExec1(nextStatesT)
	nextQT := d.targetCriticExec.MustExec1(nextStatesT, nextActionsT)

	nextQSlice := tensorToFloat64(nextQT)
	for i := 0; i < bs; i++ {
		d.targetQBuf[i] = d.rewardsBuf[i] + d.gamma*nextQSlice[i]*(1-d.donesBuf[i])
	}

	actionsT := tensors.FromFlatDataAndDimensions(d.actionsBuf[:bs*d.actionSize], bs, d.actionSize)
	targetQT := tensors.FromFlatDataAndDimensions(d.targetQBuf[:bs], bs)

	d.criticTrainExec.MustExec(statesT, actionsT, targetQT)
	d.actorTrainExec.MustExec(statesT)

	d.softUpdate()
}

// ensureBuffers grows pre-allocated buffers if batchSize changed.
func (d *DDPG) ensureBuffers(bs int) {
	if len(d.statesBuf) < bs*d.stateSize {
		d.statesBuf = make([]float64, bs*d.stateSize)
		d.nextStatesBuf = make([]float64, bs*d.stateSize)
	}
	if len(d.actionsBuf) < bs*d.actionSize {
		d.actionsBuf = make([]float64, bs*d.actionSize)
	}
	if len(d.rewardsBuf) < bs {
		d.rewardsBuf = make([]float64, bs)
		d.donesBuf = make([]float64, bs)
		d.targetQBuf = make([]float64, bs)
	}
}

func (d *DDPG) ResetNoise() {
	d.noise.Reset()
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

func copyPadded(dst, src []float64, n int) {
	if len(src) >= n {
		copy(dst[:n], src[:n])
	} else {
		copy(dst, src)
		for i := len(src); i < n; i++ {
			dst[i] = 0
		}
	}
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
