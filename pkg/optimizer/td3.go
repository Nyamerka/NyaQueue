package optimizer

import (
	"log"
	"math"
	"math/rand/v2"
	"strings"

	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"

	_ "github.com/gomlx/gomlx/backends/default"
)

const (
	td3HiddenSize        = 128
	td3Tau               = 0.005
	td3DefaultGamma      = 0.99
	td3DefaultBatchSize  = 64
	td3ReplayBufSize     = 50_000
	td3PolicyDelay       = 2
	td3TargetNoiseStddev = 0.2
	td3TargetNoiseClip   = 0.5
	td3ExploreNoiseInit  = 0.1
)

// TD3 implements Twin Delayed Deep Deterministic Policy Gradient (Fujimoto 2018).
//
// Key improvements over DDPG:
//   - Twin critics with min(Q1, Q2) to suppress overestimation.
//   - Clipped Gaussian noise on target actions for smoothing.
//   - Delayed actor updates (every td3PolicyDelay critic steps).
//   - Gaussian exploration noise instead of OU (no temporal correlation needed
//     for broker parameter tuning).
type TD3 struct {
	stateSize  int
	actionSize int
	hiddenSize int
	lr         float64
	gamma      float64
	batchSize  int

	backend   backends.Backend
	mainCtx   *context.Context
	targetCtx *context.Context

	actorFwdExec       *context.Exec
	targetActorFwdExec *context.Exec
	targetCritic1Exec  *context.Exec
	targetCritic2Exec  *context.Exec
	critic1TrainExec   *context.Exec
	critic2TrainExec   *context.Exec
	actorTrainExec     *context.Exec

	varPairs []varPair

	replayBuffer *nn.ReplayBuffer

	sigma      float64
	sigmaDecay float64
	sigmaFloor float64

	criticStep int

	statesBuf     []float64
	nextStatesBuf []float64
	actionsBuf    []float64
	rewardsBuf    []float64
	donesBuf      []float64
	targetQBuf    []float64
}

func NewTD3(stateSize, actionSize int, lr float64, batchSize int) *TD3 {
	if batchSize <= 0 {
		batchSize = td3DefaultBatchSize
	}
	hiddenSize := max(td3HiddenSize, (stateSize+actionSize)*3)
	d := &TD3{
		stateSize:    stateSize,
		actionSize:   actionSize,
		hiddenSize:   hiddenSize,
		lr:           lr,
		gamma:        td3DefaultGamma,
		batchSize:    batchSize,
		replayBuffer: nn.NewReplayBuffer(td3ReplayBufSize),
		sigma:        td3ExploreNoiseInit,
		sigmaDecay:   1.0,
		sigmaFloor:   0.0,

		statesBuf:     make([]float64, batchSize*stateSize),
		nextStatesBuf: make([]float64, batchSize*stateSize),
		actionsBuf:    make([]float64, batchSize*actionSize),
		rewardsBuf:    make([]float64, batchSize),
		donesBuf:      make([]float64, batchSize),
		targetQBuf:    make([]float64, batchSize),
	}

	d.backend = backends.MustNew()
	d.mainCtx = context.New()
	d.mainCtx.SetParams(map[string]any{
		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    d.lr,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamWeightDecay: 1e-4,
	})

	d.initNetworks()

	d.targetCtx = d.cloneContext(d.mainCtx)
	d.initTargetExecs()
	d.warmupStaticShapes()

	return d
}

func (d *TD3) initNetworks() {
	dummyState := make([]float64, d.stateSize)

	d.actorFwdExec = context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, state *Node) *Node {
			out := d.actorGraphBatched(ctx, InsertAxes(state, 0))
			return Reshape(out, d.actionSize)
		})
	d.actorFwdExec.MustExec(dummyState)

	dummyBatchStates := make([]float64, d.stateSize)
	dummyBatchActions := make([]float64, d.actionSize)
	dummyTargetQ := []float64{0}

	d.critic1TrainExec = context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, states, actions, targetQs *Node) *Node {
			g := states.Graph()
			q := d.criticGraphBatched(ctx.In("critic1"), states, actions)
			diff := Sub(q, targetQs)
			loss := ReduceAllMean(Mul(diff, diff))

			criticCtx := ctx.In("critic1")
			opt := optimizers.FromContext(criticCtx)
			opt.UpdateGraph(criticCtx, g, loss)
			return loss
		})
	d.critic1TrainExec.MustExec(
		tensors.FromFlatDataAndDimensions(dummyBatchStates, 1, d.stateSize),
		tensors.FromFlatDataAndDimensions(dummyBatchActions, 1, d.actionSize),
		tensors.FromFlatDataAndDimensions(dummyTargetQ, 1),
	)

	d.critic2TrainExec = context.MustNewExec(d.backend, d.mainCtx,
		func(ctx *context.Context, states, actions, targetQs *Node) *Node {
			g := states.Graph()
			q := d.criticGraphBatched(ctx.In("critic2"), states, actions)
			diff := Sub(q, targetQs)
			loss := ReduceAllMean(Mul(diff, diff))

			criticCtx := ctx.In("critic2")
			opt := optimizers.FromContext(criticCtx)
			opt.UpdateGraph(criticCtx, g, loss)
			return loss
		})
	d.critic2TrainExec.MustExec(
		tensors.FromFlatDataAndDimensions(dummyBatchStates, 1, d.stateSize),
		tensors.FromFlatDataAndDimensions(dummyBatchActions, 1, d.actionSize),
		tensors.FromFlatDataAndDimensions(dummyTargetQ, 1),
	)

	d.actorTrainExec = context.MustNewExec(d.backend, d.mainCtx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			g := states.Graph()
			actions := d.actorGraphBatched(ctx, states)
			q := d.criticGraphBatched(ctx.In("critic1"), states, actions)
			loss := Neg(ReduceAllMean(q))

			actorCtx := ctx.In("actor")
			opt := optimizers.FromContext(actorCtx)
			opt.UpdateGraph(actorCtx, g, loss)
			return loss
		})
}

func (d *TD3) initTargetExecs() {
	d.targetActorFwdExec = context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			return d.actorGraphBatched(ctx, states)
		})

	d.targetCritic1Exec = context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, states, actions *Node) *Node {
			return d.criticGraphBatched(ctx.In("critic1"), states, actions)
		})

	d.targetCritic2Exec = context.MustNewExec(d.backend, d.targetCtx.Reuse(),
		func(ctx *context.Context, states, actions *Node) *Node {
			return d.criticGraphBatched(ctx.In("critic2"), states, actions)
		})

	d.mainCtx.EnumerateVariables(func(srcVar *context.Variable) {
		if !srcVar.Trainable {
			return
		}
		tgtVar := d.targetCtx.InAbsPath(srcVar.Scope()).GetVariable(srcVar.Name())
		if tgtVar == nil {
			return
		}
		srcT := srcVar.MustValue()
		flatLen := 1
		for _, dim := range srcT.Shape().Dimensions {
			flatLen *= dim
		}
		d.varPairs = append(d.varPairs, varPair{
			src: srcVar,
			tgt: tgtVar,
			buf: make([]float64, flatLen),
		})
	})
}

func (d *TD3) warmupStaticShapes() {
	dummyState := make([]float64, d.stateSize)
	d.actorFwdExec.MustExec(dummyState)

	bs := d.batchSize
	dummyBatchStates := make([]float64, bs*d.stateSize)
	dummyBatchActions := make([]float64, bs*d.actionSize)

	statesT := tensors.FromFlatDataAndDimensions(dummyBatchStates, bs, d.stateSize)
	actionsT := tensors.FromFlatDataAndDimensions(dummyBatchActions, bs, d.actionSize)

	d.targetActorFwdExec.MustExec1(statesT)
	d.targetCritic1Exec.MustExec1(statesT, actionsT)
	d.targetCritic2Exec.MustExec1(statesT, actionsT)
}

func (d *TD3) actorGraphBatched(ctx *context.Context, states *Node) *Node {
	actorCtx := ctx.In("actor")
	h := layers.Dense(actorCtx.In("l1"), states, true, d.hiddenSize)
	h = layers.LayerNormalization(actorCtx.In("ln1"), h, -1).Done()
	h = activations.Relu(h)
	h = layers.Dense(actorCtx.In("l2"), h, true, d.hiddenSize)
	h = layers.LayerNormalization(actorCtx.In("ln2"), h, -1).Done()
	h = activations.Relu(h)
	out := layers.Dense(actorCtx.In("l3"), h, true, d.actionSize)
	return Tanh(out)
}

func (d *TD3) criticGraphBatched(ctx *context.Context, states, actions *Node) *Node {
	input := Concatenate([]*Node{states, actions}, -1)
	h := layers.Dense(ctx.In("l1"), input, true, d.hiddenSize)
	h = layers.LayerNormalization(ctx.In("ln1"), h, -1).Done()
	h = activations.Relu(h)
	h = layers.Dense(ctx.In("l2"), h, true, d.hiddenSize)
	h = layers.LayerNormalization(ctx.In("ln2"), h, -1).Done()
	h = activations.Relu(h)
	q := layers.Dense(ctx.In("l3"), h, true, 1)
	return Squeeze(q, -1)
}

func (d *TD3) Act(state []float64) []float64 {
	result := d.actorFwdExec.MustExec1(state)
	action := tensorToFloat64(result)

	for i := range action {
		noise := d.sigma * rand.NormFloat64()
		action[i] = math.Max(-1, math.Min(1, action[i]+noise))
	}
	return action
}

func (d *TD3) Store(state, action []float64, reward float64, nextState []float64, done bool) {
	d.replayBuffer.Push(nn.Transition{
		State:     append([]float64(nil), state...),
		Action:    append([]float64(nil), action...),
		Reward:    reward,
		NextState: append([]float64(nil), nextState...),
		Done:      done,
	})
}

func (d *TD3) Train(batchSize int) {
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

	// Target policy smoothing: a' = clip(actor_target(s') + clip(noise, -c, c), -1, 1)
	nextActionsT := d.targetActorFwdExec.MustExec1(nextStatesT)
	nextActions := tensorToFloat64(nextActionsT)
	for i := range nextActions {
		noise := td3TargetNoiseStddev * rand.NormFloat64()
		noise = math.Max(-td3TargetNoiseClip, math.Min(td3TargetNoiseClip, noise))
		nextActions[i] = math.Max(-1, math.Min(1, nextActions[i]+noise))
	}
	smoothedNextActionsT := tensors.FromFlatDataAndDimensions(nextActions, bs, d.actionSize)

	// Twin critics: use min(Q1_target, Q2_target)
	nextQ1T := d.targetCritic1Exec.MustExec1(nextStatesT, smoothedNextActionsT)
	nextQ2T := d.targetCritic2Exec.MustExec1(nextStatesT, smoothedNextActionsT)

	nextQ1 := tensorToFloat64(nextQ1T)
	nextQ2 := tensorToFloat64(nextQ2T)
	for i := 0; i < bs; i++ {
		minQ := math.Min(nextQ1[i], nextQ2[i])
		d.targetQBuf[i] = d.rewardsBuf[i] + d.gamma*minQ*(1-d.donesBuf[i])
	}

	actionsT := tensors.FromFlatDataAndDimensions(d.actionsBuf[:bs*d.actionSize], bs, d.actionSize)
	targetQT := tensors.FromFlatDataAndDimensions(d.targetQBuf[:bs], bs)

	// Train both critics
	d.critic1TrainExec.MustExec(statesT, actionsT, targetQT)
	d.critic2TrainExec.MustExec(statesT, actionsT, targetQT)

	d.criticStep++

	// Delayed policy update: actor + target soft-update every td3PolicyDelay critic steps
	if d.criticStep%td3PolicyDelay == 0 {
		d.freezeCritic()
		d.actorTrainExec.MustExec(statesT)
		d.unfreezeCritic()
		d.softUpdate()
	}
}

// softUpdate blends main→target weights: tgt = (1-tau)*tgt + tau*src.
func (d *TD3) softUpdate() {
	for _, p := range d.varPairs {
		srcVal := p.src.MustValue()
		tgtVal := p.tgt.MustValue()
		dims := tgtVal.Shape().Dimensions

		srcFlat := flattenAnyToFloat64(srcVal.Value())
		tgtFlat := flattenAnyToFloat64(tgtVal.Value())
		if srcFlat == nil || tgtFlat == nil || len(srcFlat) != len(tgtFlat) {
			log.Printf("TD3 softUpdate: skip %s/%s (nil or length mismatch)",
				p.src.Scope(), p.src.Name())
			continue
		}

		buf := p.buf[:len(tgtFlat)]
		for i := range buf {
			buf[i] = (1-td3Tau)*tgtFlat[i] + td3Tau*srcFlat[i]
		}
		if err := p.tgt.SetValue(tensors.FromFlatDataAndDimensions(buf, dims...)); err != nil {
			log.Printf("TD3 softUpdate SetValue failed: %v", err)
		}
	}
}

func (d *TD3) ensureBuffers(bs int) {
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

// freezeCritic marks all critic variables as non-trainable so that
// the actor's UpdateGraph does not shift critic weights.
func (d *TD3) freezeCritic() {
	d.mainCtx.EnumerateVariables(func(v *context.Variable) {
		scope := v.Scope()
		if strings.Contains(scope, "critic1") || strings.Contains(scope, "critic2") {
			v.Trainable = false
		}
	})
}

func (d *TD3) unfreezeCritic() {
	d.mainCtx.EnumerateVariables(func(v *context.Variable) {
		scope := v.Scope()
		if strings.Contains(scope, "critic1") || strings.Contains(scope, "critic2") {
			v.Trainable = true
		}
	})
}

func (d *TD3) ResetNoise() {
	d.sigma = td3ExploreNoiseInit
}

func (d *TD3) BoostNoise(factor float64) {
	d.ResetNoise()
	d.sigma *= factor
}

func (d *TD3) SetNoiseDecay(decay, floor float64) {
	d.sigmaDecay = decay
	d.sigmaFloor = floor
}

func (d *TD3) DecayNoise() {
	if d.sigmaDecay < 1.0 {
		d.sigma *= d.sigmaDecay
		if d.sigma < d.sigmaFloor {
			d.sigma = d.sigmaFloor
		}
	}
}

func (d *TD3) Sigma() float64 {
	return d.sigma
}

func flattenAnyToFloat64(v any) []float64 {
	switch d := v.(type) {
	case []float64:
		out := make([]float64, len(d))
		copy(out, d)
		return out
	case [][]float64:
		var out []float64
		for _, row := range d {
			out = append(out, row...)
		}
		return out
	case []float32:
		out := make([]float64, len(d))
		for i, x := range d {
			out[i] = float64(x)
		}
		return out
	case [][]float32:
		var out []float64
		for _, row := range d {
			for _, x := range row {
				out = append(out, float64(x))
			}
		}
		return out
	default:
		return nil
	}
}

type varPair struct {
	src *context.Variable
	tgt *context.Variable
	buf []float64
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
	case [][]float64:
		var out []float64
		for _, row := range v {
			out = append(out, row...)
		}
		return out
	case [][]float32:
		var out []float64
		for _, row := range v {
			for _, x := range row {
				out = append(out, float64(x))
			}
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

func (d *TD3) cloneContext(src *context.Context) *context.Context {
	dst := context.New()
	src.EnumerateVariables(func(v *context.Variable) {
		srcT := v.MustValue()
		cloned, _ := srcT.LocalClone()
		dst.InAbsPath(v.Scope()).VariableWithValue(v.Name(), cloned.Value())
	})
	return dst
}
