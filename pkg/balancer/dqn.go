package balancer

import (
	stdctx "context"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/stat"
)

var float64BufPool = sync.Pool{New: func() any {
	buf := make([]float64, 256)
	return &buf
}}

type paddedInt64 struct {
	n atomic.Int64
	_ [56]byte // pad to 64-byte cache line
}

const (
	dqnBiasWarmup = 200.0
	dqnBiasScale  = 2.0
	dqnBiasClip   = 1.0
)

type policySnap struct {
	state  []float64
	action int
}

type stateSnapshot struct {
	loads          []float64
	predictedLoads []float64
	queueDepth     []int
	msgRate        float64
	avgMsgSize     float64
	consumeRate    float64
}

type nStepBuffer struct {
	mu    sync.Mutex
	queue []nn.Transition
	n     int
	gamma float64
}

func newNStepBuffer(n int, gamma float64) *nStepBuffer {
	return &nStepBuffer{
		queue: make([]nn.Transition, 0, n),
		n:     n,
		gamma: gamma,
	}
}

func (b *nStepBuffer) Push(t nn.Transition) (nn.Transition, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.queue = append(b.queue, t)
	if len(b.queue) < b.n {
		return nn.Transition{}, false
	}
	first := b.queue[0]
	gPow := 1.0
	var R float64
	for _, tr := range b.queue {
		R += gPow * tr.Reward
		gPow *= b.gamma
	}
	out := nn.Transition{
		State:     first.State,
		Action:    first.Action,
		Reward:    R,
		NextState: b.queue[b.n-1].NextState,
		Done:      b.queue[b.n-1].Done,
	}
	b.queue = b.queue[1:]
	return out, true
}

func (b *nStepBuffer) Reset() {
	b.mu.Lock()
	b.queue = b.queue[:0]
	b.mu.Unlock()
}

type DQNBalancer struct {
	numPartitions  int
	hiddenSize     int
	gamma          float64
	lr             float64
	huberDelta     float64
	batchSize      int
	minReplay      int
	trainEvery     int
	updatesPerStep int

	backend               backends.Backend
	ctx                   *context.Context
	targetNet             *nn.TargetNetwork
	inferExec             atomic.Pointer[context.Exec]
	inferCtx              *context.Context
	inferVarMap           map[string]*context.Variable
	bestActionsExec       *context.Exec
	targetQForActionsExec *context.Exec
	batchQExec            *context.Exec
	trainExec             *context.Exec
	trainExecMu           sync.Mutex
	stateSize             int

	replayBuffer *nn.PrioritizedReplayBuffer
	betaSchedule *nn.BetaSchedule
	normalizer   *nn.RunningNormalizer

	expCh      chan nn.Transition
	snapshotCh chan chan []nn.Transition
	eg         *errgroup.Group
	cancel     stdctx.CancelFunc

	atomicState atomic.Pointer[stateSnapshot]

	droppedExperience atomic.Int64
	liveDepth         atomic.Pointer[[]paddedInt64]
	inflight          atomic.Pointer[[]atomic.Int64]

	cachedQ       atomic.Pointer[[]float64]
	cachedArgmax  atomic.Int32
	deterministic bool

	prevSnap atomic.Pointer[policySnap]
	currSnap atomic.Pointer[policySnap]

	experiencesSeen atomic.Int64
	trainingSteps   atomic.Int64

	nStep     *nStepBuffer
	xlaReady  chan struct{}
	xlaFailed atomic.Bool

	popArt struct {
		mean  float64
		m2    float64
		count int64
	}

	trainBuf struct {
		states     []float64
		nextStates []float64
		actions    []float64
		rewards    []float64
		dones      []float64
		targets    []float64
	}

	policyStateBuf []float64
	policyQBuf     [2][]float64
	policyBufIdx   int
	prevLoads      atomic.Pointer[[]float64]

	noiseValueIn  int
	noiseValueOut int
	noiseAdvIn    int
	noiseAdvOut   int

	noiseBufs struct {
		valueIn, valueOut, advIn, advOut []float64
	}

	qResultBuf []float64
}

type DQNOption func(*DQNBalancer)

func WithDQNGamma(g float64) DQNOption          { return func(d *DQNBalancer) { d.gamma = g } }
func WithDQNLearningRate(lr float64) DQNOption  { return func(d *DQNBalancer) { d.lr = lr } }
func WithDQNHiddenSize(n int) DQNOption         { return func(d *DQNBalancer) { d.hiddenSize = n } }
func WithDQNHuberDelta(delta float64) DQNOption { return func(d *DQNBalancer) { d.huberDelta = delta } }
func WithDQNReplayBufSize(n int) DQNOption {
	return func(d *DQNBalancer) { d.replayBuffer = nn.NewPrioritizedReplayBuffer(n, 0.6) }
}
func WithDQNBatchSize(n int) DQNOption { return func(d *DQNBalancer) { d.batchSize = n } }
func WithDQNMinReplay(n int) DQNOption { return func(d *DQNBalancer) { d.minReplay = n } }
func WithDQNTrainEvery(n int) DQNOption {
	return func(d *DQNBalancer) { d.trainEvery = n }
}
func WithDQNDeterministic(v bool) DQNOption {
	return func(d *DQNBalancer) { d.deterministic = v }
}

func NewDQNBalancer(numPartitions int, opts ...DQNOption) *DQNBalancer {
	d := &DQNBalancer{
		numPartitions:  numPartitions,
		hiddenSize:     DefaultDQNHiddenSize,
		gamma:          DefaultDQNGamma,
		lr:             DefaultDQNLearningRate,
		huberDelta:     1.0,
		batchSize:      DefaultDQNBatchSize,
		minReplay:      DefaultDQNMinReplay,
		trainEvery:     DefaultDQNTrainEvery,
		updatesPerStep: DefaultDQNUpdatesPerStep,
		replayBuffer:   nn.NewPrioritizedReplayBuffer(DefaultDQNReplayBufSize, 0.6),
		expCh:          make(chan nn.Transition, DefaultDQNExpChannelSize),
		snapshotCh:     make(chan chan []nn.Transition, 1),
	}
	d.atomicState.Store(&stateSnapshot{
		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	})

	for _, opt := range opts {
		opt(d)
	}

	d.stateSize = numPartitions*5 + 4
	d.hiddenSize = max(DefaultDQNHiddenSize, d.stateSize+64)
	d.policyStateBuf = make([]float64, d.stateSize)
	d.policyQBuf[0] = make([]float64, numPartitions)
	d.policyQBuf[1] = make([]float64, numPartitions)
	d.qResultBuf = make([]float64, numPartitions)
	d.betaSchedule = nn.NewBetaSchedule(0.4, 1.0, DefaultDQNReplayBufSize/d.batchSize)
	d.normalizer = nn.NewRunningNormalizer(d.stateSize)
	d.nStep = newNStepBuffer(5, d.gamma)
	d.xlaReady = make(chan struct{})

	ctx, cancel := stdctx.WithCancel(stdctx.Background())
	d.cancel = cancel
	d.eg, _ = errgroup.WithContext(ctx)

	d.eg.Go(func() error {
		defer close(d.xlaReady)
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-balancer] initGoMLX panic: %v", r)
				d.xlaFailed.Store(true)
			}
		}()
		d.initGoMLX()
		return nil
	})
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-balancer] trainLoop panic: %v", r)
			}
		}()
		select {
		case <-d.xlaReady:
			if d.xlaFailed.Load() {
				log.Printf("[dqn-balancer] XLA init failed, trainLoop staying on P2C fallback")
				return nil
			}
		case <-ctx.Done():
			return nil
		}
		d.trainLoop(ctx)
		return nil
	})
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-balancer] policyLoop panic: %v", r)
			}
		}()
		select {
		case <-d.xlaReady:
			if d.xlaFailed.Load() {
				log.Printf("[dqn-balancer] XLA init failed, policyLoop staying on P2C fallback")
				return nil
			}
		case <-ctx.Done():
			return nil
		}
		d.policyLoop(ctx)
		return nil
	})

	return d
}

func (d *DQNBalancer) initGoMLX() {
	d.backend = nn.SharedBackend()
	d.ctx = context.New()
	d.ctx.SetParams(map[string]any{
		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    d.lr,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamWeightDecay: DefaultDQNWeightDecay,
	})

	d.noiseValueIn = d.hiddenSize / 4
	d.noiseValueOut = dqnNumQuantiles
	d.noiseAdvIn = d.hiddenSize / 4
	d.noiseAdvOut = d.numPartitions * dqnNumQuantiles

	d.noiseBufs.valueIn = make([]float64, d.noiseValueIn)
	d.noiseBufs.valueOut = make([]float64, d.noiseValueOut)
	d.noiseBufs.advIn = make([]float64, d.noiseAdvIn)
	d.noiseBufs.advOut = make([]float64, d.noiseAdvOut)

	dummyNoise := make([]float64, max(d.noiseValueIn, d.noiseValueOut, d.noiseAdvIn, d.noiseAdvOut))
	initExec := context.MustNewExec(d.backend, d.ctx,
		func(ctx *context.Context, state, evi, evo, eai, eao *Node) *Node {
			return d.qNetworkGraphExternalNoise(ctx, state, evi, evo, eai, eao)
		})
	initExec.MustExec(
		make([]float64, d.stateSize),
		dummyNoise[:d.noiseValueIn],
		dummyNoise[:d.noiseValueOut],
		dummyNoise[:d.noiseAdvIn],
		dummyNoise[:d.noiseAdvOut],
	)
	initExec.Finalize()

	d.inferCtx = d.cloneContext(d.ctx)
	d.inferVarMap = make(map[string]*context.Variable)
	d.inferCtx.EnumerateVariables(func(v *context.Variable) {
		d.inferVarMap[v.ScopeAndName()] = v
	})
	fwdExec := context.MustNewExec(d.backend, d.inferCtx.Reuse(),
		func(ctx *context.Context, state, evi, evo, eai, eao *Node) *Node {
			return d.qNetworkGraphExternalNoise(ctx, state, evi, evo, eai, eao)
		})
	d.inferExec.Store(fwdExec)

	d.targetNet = nn.NewTargetNetwork(d.ctx)

	d.bestActionsExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			q := d.qNetworkBatch(ctx, states)
			return ArgMax(q, -1, dtypes.Int32)
		})

	d.targetQForActionsExec = context.MustNewExec(d.backend, d.targetNet.TargetCtx().Reuse(),
		func(ctx *context.Context, states, actionIdx *Node) *Node {
			qAll := d.qNetworkBatch(ctx, states)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numPartitions, qAll.DType())
			return ReduceSum(Mul(qAll, oneHot), -1)
		})

	d.batchQExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, actionIdx *Node) *Node {
			qAll := d.qNetworkBatch(ctx, states)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numPartitions, qAll.DType())
			return ReduceSum(Mul(qAll, oneHot), -1)
		})

	d.warmupStaticShapes()

	huber := losses.MakeHuberLoss(d.huberDelta)
	d.trainExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, targetQ, actionIdx, isWeights *Node) *Node {
			g := states.Graph()
			qAll := d.qNetworkBatch(ctx, states)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numPartitions, qAll.DType())
			qSelected := ReduceSum(Mul(qAll, oneHot), -1)
			loss := huber([]*Node{targetQ, isWeights}, []*Node{qSelected})
			opt := optimizers.FromContext(ctx)
			opt.UpdateGraph(ctx, g, loss)
			return loss
		})
}

func (d *DQNBalancer) warmupStaticShapes() {
	dummyState := make([]float64, d.stateSize)
	dummyNoise := make([]float64, max(d.noiseValueIn, d.noiseValueOut, d.noiseAdvIn, d.noiseAdvOut))
	exec := d.inferExec.Load()
	exec.MustExec1(dummyState,
		dummyNoise[:d.noiseValueIn], dummyNoise[:d.noiseValueOut],
		dummyNoise[:d.noiseAdvIn], dummyNoise[:d.noiseAdvOut])

	dummyBatch := make([]float64, d.batchSize*d.stateSize)
	batchT := tensors.FromFlatDataAndDimensions(dummyBatch, d.batchSize, d.stateSize)
	d.bestActionsExec.MustExec1(batchT)
}

func (d *DQNBalancer) syncInferWeights() {
	d.ctx.EnumerateVariables(func(mainVar *context.Variable) {
		if !mainVar.Trainable {
			return
		}
		inferVar, ok := d.inferVarMap[mainVar.ScopeAndName()]
		if !ok {
			return
		}
		mainVal := mainVar.MustValue()
		inferVal := inferVar.MustValue()
		if mainVal == nil || inferVal == nil {
			return
		}
		mainVal.ConstFlatData(func(mainFlat any) {
			inferVal.MutableFlatData(func(inferFlat any) {
				switch m := mainFlat.(type) {
				case []float32:
					copy(inferFlat.([]float32), m)
				case []float64:
					copy(inferFlat.([]float64), m)
				}
			})
		})
	})
}

func (d *DQNBalancer) cloneContext(src *context.Context) *context.Context {
	dst := context.New()
	src.EnumerateVariables(func(v *context.Variable) {
		srcT := v.MustValue()
		cloned, _ := srcT.LocalClone()
		dst.InAbsPath(v.Scope()).VariableWithValue(v.Name(), cloned.Value())
	})
	return dst
}

func (d *DQNBalancer) qNetworkBatch(ctx *context.Context, x *Node) *Node {
	h := layers.Dense(ctx.In("feature1"), x, true, d.hiddenSize)
	h = layers.LayerNormalization(ctx.In("ln1"), h, -1).Done()
	h = activations.Relu(h)
	h = layers.Dense(ctx.In("feature2"), h, true, d.hiddenSize/2)
	h = layers.LayerNormalization(ctx.In("ln2"), h, -1).Done()
	h = activations.Relu(h)

	vStream := layers.Dense(ctx.In("value_hidden"), h, true, d.hiddenSize/4)
	vStream = layers.LayerNormalization(ctx.In("ln_vh"), vStream, -1).Done()
	vStream = activations.Relu(vStream)
	v := nn.NoisyLinear(ctx.In("value_out"), vStream, dqnNumQuantiles, nn.DefaultNoisySigmaInit)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = layers.LayerNormalization(ctx.In("ln_ah"), aStream, -1).Done()
	aStream = activations.Relu(aStream)
	a := nn.NoisyLinear(ctx.In("adv_out"), aStream, d.numPartitions*dqnNumQuantiles, nn.DefaultNoisySigmaInit)

	batchSize := x.Shape().Dimensions[0]
	aReshaped := Reshape(a, batchSize, d.numPartitions, dqnNumQuantiles)
	vReshaped := Reshape(v, batchSize, 1, dqnNumQuantiles)

	aMean := ReduceMean(aReshaped, 1)
	aMean = ExpandDims(aMean, 1)
	quantiles := Add(vReshaped, Sub(aReshaped, aMean))
	return ReduceMean(quantiles, -1)
}

func (d *DQNBalancer) qNetworkBatchExternalNoise(ctx *context.Context, x *Node, epsValueIn, epsValueOut, epsAdvIn, epsAdvOut *Node) *Node {
	h := layers.Dense(ctx.In("feature1"), x, true, d.hiddenSize)
	h = layers.LayerNormalization(ctx.In("ln1"), h, -1).Done()
	h = activations.Relu(h)
	h = layers.Dense(ctx.In("feature2"), h, true, d.hiddenSize/2)
	h = layers.LayerNormalization(ctx.In("ln2"), h, -1).Done()
	h = activations.Relu(h)

	vStream := layers.Dense(ctx.In("value_hidden"), h, true, d.hiddenSize/4)
	vStream = layers.LayerNormalization(ctx.In("ln_vh"), vStream, -1).Done()
	vStream = activations.Relu(vStream)
	v := nn.NoisyLinearExternal(ctx.In("value_out"), vStream, epsValueIn, epsValueOut, dqnNumQuantiles, nn.DefaultNoisySigmaInit)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = layers.LayerNormalization(ctx.In("ln_ah"), aStream, -1).Done()
	aStream = activations.Relu(aStream)
	a := nn.NoisyLinearExternal(ctx.In("adv_out"), aStream, epsAdvIn, epsAdvOut, d.numPartitions*dqnNumQuantiles, nn.DefaultNoisySigmaInit)

	batchSize := x.Shape().Dimensions[0]
	aReshaped := Reshape(a, batchSize, d.numPartitions, dqnNumQuantiles)
	vReshaped := Reshape(v, batchSize, 1, dqnNumQuantiles)

	aMean := ReduceMean(aReshaped, 1)
	aMean = ExpandDims(aMean, 1)
	quantiles := Add(vReshaped, Sub(aReshaped, aMean))

	return ReduceMean(quantiles, -1)
}

func (d *DQNBalancer) qNetworkGraphExternalNoise(ctx *context.Context, state, epsValueIn, epsValueOut, epsAdvIn, epsAdvOut *Node) *Node {
	x := InsertAxes(state, 0)
	q := d.qNetworkBatchExternalNoise(ctx, x, epsValueIn, epsValueOut, epsAdvIn, epsAdvOut)
	return Reshape(q, d.numPartitions)
}

func (d *DQNBalancer) forward(state []float64) []float64 {
	buf := float64BufPool.Get().(*[]float64)
	norm := (*buf)[:len(state)]
	copy(norm, state)
	d.normalizer.NormalizeInPlace(norm)

	for i := range d.noiseBufs.valueIn {
		d.noiseBufs.valueIn[i] = rand.NormFloat64()
	}
	for i := range d.noiseBufs.valueOut {
		d.noiseBufs.valueOut[i] = rand.NormFloat64()
	}
	for i := range d.noiseBufs.advIn {
		d.noiseBufs.advIn[i] = rand.NormFloat64()
	}
	for i := range d.noiseBufs.advOut {
		d.noiseBufs.advOut[i] = rand.NormFloat64()
	}

	exec := d.inferExec.Load()
	result := exec.MustExec1(norm, d.noiseBufs.valueIn, d.noiseBufs.valueOut, d.noiseBufs.advIn, d.noiseBufs.advOut)

	float64BufPool.Put(buf)
	tensorToFloat64Into(result, d.qResultBuf)
	return d.qResultBuf
}

func (d *DQNBalancer) shouldRecordExperience() bool {
	return d.cachedQ.Load() != nil
}

func (d *DQNBalancer) ensureInflight(n int) *[]atomic.Int64 {
	for {
		ptr := d.inflight.Load()
		if ptr != nil && len(*ptr) >= n {
			return ptr
		}
		arr := make([]atomic.Int64, n)
		if d.inflight.CompareAndSwap(ptr, &arr) {
			return &arr
		}
	}
}

func (d *DQNBalancer) SelectPartition(topic string, key []byte, numPartitions int) int {
	depth := d.ensureLiveDepth(numPartitions)
	q := d.cachedQ.Load()

	conf := 0.0
	if steps := d.trainingSteps.Load(); steps > 0 {
		conf = math.Min(1.0, float64(steps)/dqnBiasWarmup)
	}

	i, j := d.pick2(numPartitions)

	si := float64((*depth)[i].n.Load())
	sj := float64((*depth)[j].n.Load())

	if q != nil && len(*q) >= numPartitions && conf > 0 {
		qi := clip64((*q)[i], -dqnBiasClip, dqnBiasClip)
		qj := clip64((*q)[j], -dqnBiasClip, dqnBiasClip)
		si -= dqnBiasScale * conf * qi
		sj -= dqnBiasScale * conf * qj
	}

	best := i
	if sj < si {
		best = j
	}

	(*depth)[best].n.Add(1)
	return best
}

func (d *DQNBalancer) ensureLiveDepth(n int) *[]paddedInt64 {
	dp := d.liveDepth.Load()
	if dp != nil && len(*dp) >= n {
		return dp
	}
	arr := make([]paddedInt64, n)
	d.liveDepth.Store(&arr)
	return &arr
}

func (d *DQNBalancer) pick2(n int) (int, int) {
	i := rand.IntN(n)
	j := rand.IntN(n - 1)
	if j >= i {
		j++
	}
	return i, j
}

func clip64(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	if len(m.QueueDepth) > 0 {
		dp := d.ensureLiveDepth(len(m.QueueDepth))
		for i, depth := range m.QueueDepth {
			(*dp)[i].n.Store(int64(depth))
		}
	}

	snap := &stateSnapshot{
		msgRate:     m.MsgRate,
		avgMsgSize:  m.AvgMsgSize,
		consumeRate: m.ConsumeRate,
	}
	if len(m.PartitionLoads) > 0 {
		snap.loads = append([]float64(nil), m.PartitionLoads...)
	}
	if len(m.PredictedLoads) > 0 {
		snap.predictedLoads = append([]float64(nil), m.PredictedLoads...)
	}
	if len(m.QueueDepth) > 0 {
		snap.queueDepth = append([]int(nil), m.QueueDepth...)
	}
	d.atomicState.Store(snap)

	if m.MsgRate < 1.0 && m.Throughput < 1.0 {
		return
	}

	reward := d.computeReward(m)

	if !d.shouldRecordExperience() {
		return
	}

	prev := d.prevSnap.Load()
	curr := d.currSnap.Load()
	if prev != nil && curr != nil {
		raw := nn.Transition{
			State:     append([]float64(nil), prev.state...),
			Action:    []float64{float64(prev.action)},
			Reward:    reward,
			NextState: append([]float64(nil), curr.state...),
			Done:      false,
		}
		if multi, ok := d.nStep.Push(raw); ok {
			d.experiencesSeen.Add(1)
			select {
			case d.expCh <- multi:
			case <-time.After(time.Millisecond):
				d.droppedExperience.Add(1)
			}
		}
	}
}

func (d *DQNBalancer) BalancerType() string                      { return "DQN" }
func (d *DQNBalancer) NumPartitions() int                        { return d.numPartitions }
func (d *DQNBalancer) ReplayBuffer() *nn.PrioritizedReplayBuffer { return d.replayBuffer }
func (d *DQNBalancer) XLAReady() <-chan struct{}                 { return d.xlaReady }
func (d *DQNBalancer) Ctx() *context.Context                     { return d.ctx }

func (d *DQNBalancer) DroppedExperience() int64 {
	return d.droppedExperience.Load()
}

func (d *DQNBalancer) TrainingSteps() int64 {
	return d.trainingSteps.Load()
}

func (d *DQNBalancer) ExperiencesSeen() int64 {
	return d.experiencesSeen.Load()
}

func (d *DQNBalancer) OnPublishComplete(partition int) {
	if ptr := d.inflight.Load(); ptr != nil && partition < len(*ptr) {
		(*ptr)[partition].Add(-1)
	}
}

func (d *DQNBalancer) WaitReady(ctx stdctx.Context) error {
	select {
	case <-d.xlaReady:
		if d.xlaFailed.Load() {
			return fmt.Errorf("XLA initialization failed")
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (d *DQNBalancer) XLAFailed() bool {
	return d.xlaFailed.Load()
}

func (d *DQNBalancer) Stop() {
	d.cancel()
	_ = d.eg.Wait()

	if d.trainExec != nil {
		d.trainExec.Finalize()
	}
	if d.bestActionsExec != nil {
		d.bestActionsExec.Finalize()
	}
	if d.targetQForActionsExec != nil {
		d.targetQForActionsExec.Finalize()
	}
	if d.batchQExec != nil {
		d.batchQExec.Finalize()
	}
	if exec := d.inferExec.Load(); exec != nil {
		exec.Finalize()
	}
}

func (d *DQNBalancer) trainLoop(ctx stdctx.Context) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	steps := 0
	var lastLoss float64
	const minTrainInterval = 20 * time.Millisecond
	lastTrainTime := time.Now()

	for {
		select {
		case t := <-d.expCh:
			d.replayBuffer.Push(t)
			steps++
			if steps%d.trainEvery == 0 {
				now := time.Now()
				if now.Sub(lastTrainTime) < minTrainInterval {
					continue
				}
				lastTrainTime = now
				for u := 0; u < d.updatesPerStep; u++ {
					lastLoss = d.trainStep()
					d.trainingSteps.Add(1)
				}
			}
			if steps%100 == 0 {
				log.Printf("[dqn-bal] step=%d loss=%.4f buf=%d/%d dropped=%d",
					steps, lastLoss, d.replayBuffer.Len(), d.replayBuffer.Cap(),
					d.droppedExperience.Load())
			}
		case respCh := <-d.snapshotCh:
			respCh <- d.replayBuffer.ToBuffer()
		case <-ctx.Done():
			for {
				select {
				case t := <-d.expCh:
					d.replayBuffer.Push(t)
				default:
					return
				}
			}
		}
	}
}

func (d *DQNBalancer) ExportReplayBuffer(ctx stdctx.Context) []nn.Transition {
	respCh := make(chan []nn.Transition, 1)
	select {
	case d.snapshotCh <- respCh:
	case <-ctx.Done():
		return nil
	}
	select {
	case buf := <-respCh:
		return buf
	case <-ctx.Done():
		return nil
	}
}

func (d *DQNBalancer) policyInterval() time.Duration {
	snap := d.atomicState.Load()
	if snap == nil {
		return dqnPolicyTickInterval
	}
	if snap.msgRate > 0 && snap.msgRate < dqnLowRateThreshold {
		return dqnSlowPolicyTickInterval
	}
	if snap.msgRate > dqnOverloadRateThreshold {
		return dqnFastPolicyTickInterval
	}
	return dqnPolicyTickInterval
}

func (d *DQNBalancer) policyLoop(ctx stdctx.Context) {
	ticker := time.NewTicker(dqnPolicyTickInterval)
	defer ticker.Stop()

	lastInterval := dqnPolicyTickInterval

	for {
		select {
		case <-ticker.C:
			d.buildStateInto(d.policyStateBuf)

			qValues := d.forward(d.policyStateBuf)

			buf := d.policyBufIdx ^ 1
			meanQ := d.policyQBuf[buf]
			copy(meanQ, qValues[:d.numPartitions])
			d.cachedQ.Store(&meanQ)
			d.cachedArgmax.Store(int32(argmax(meanQ)))

			stateCopy := make([]float64, len(d.policyStateBuf))
			copy(stateCopy, d.policyStateBuf)
			snap := &policySnap{
				state:  stateCopy,
				action: argmax(meanQ),
			}

			d.prevSnap.Store(d.currSnap.Load())
			d.currSnap.Store(snap)
			d.policyBufIdx = buf

			if newInterval := d.policyInterval(); newInterval != lastInterval {
				ticker.Reset(newInterval)
				lastInterval = newInterval
			}
		case <-ctx.Done():
			return
		}
	}
}

func (d *DQNBalancer) buildStateInto(dst []float64) {
	for i := range dst {
		dst[i] = 0
	}

	snap := d.atomicState.Load()
	if snap == nil {
		return
	}

	copy(dst, snap.loads)
	copy(dst[d.numPartitions:], snap.predictedLoads)

	inflightBase := d.numPartitions * 2
	if ptr := d.inflight.Load(); ptr != nil {
		for i := 0; i < d.numPartitions && i < len(*ptr); i++ {
			dst[inflightBase+i] = float64((*ptr)[i].Load()) / dqnInflightScale
		}
	}

	depthBase := d.numPartitions * 3
	for i := 0; i < d.numPartitions && i < len(snap.queueDepth); i++ {
		if depthBase+i < len(dst) {
			dst[depthBase+i] = float64(snap.queueDepth[i]) / dqnDepthFeatureScale
		}
	}

	deltaBase := d.numPartitions * 4
	if prev := d.prevLoads.Load(); prev != nil {
		for i := 0; i < d.numPartitions && i < len(*prev) && i < len(snap.loads); i++ {
			if deltaBase+i < len(dst) {
				dst[deltaBase+i] = snap.loads[i] - (*prev)[i]
			}
		}
	}
	loadsCopy := make([]float64, len(snap.loads))
	copy(loadsCopy, snap.loads)
	d.prevLoads.Store(&loadsCopy)

	idx := d.numPartitions * 5
	if idx < len(dst) {
		dst[idx] = snap.msgRate / dqnMsgRateScale
	}
	if idx+1 < len(dst) {
		dst[idx+1] = snap.avgMsgSize / dqnMsgSizeScale
	}
	if idx+2 < len(dst) {
		dst[idx+2] = snap.consumeRate / dqnMsgRateScale
	}
	if idx+3 < len(dst) && snap.consumeRate > 0 {
		dst[idx+3] = snap.msgRate / snap.consumeRate
	}

	d.normalizer.Observe(dst)
}

const (
	dqnTailWeight = 0.2
	dqnLatNorm    = 10_000.0 // microseconds normalizer for latency penalty
)

func (d *DQNBalancer) computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}

	meanLoad := stat.Mean(m.PartitionLoads, nil)

	var balance float64
	if meanLoad > 0 {
		cv := stat.StdDev(m.PartitionLoads, nil) / meanLoad
		balance = -math.Min(cv, 1.0)
	}

	var utility float64
	if m.Throughput > 0 {
		utility = math.Log1p(m.Throughput/dqnThroughputScale) / dqnThroughputLogNorm
	}

	pressure := math.Min(meanLoad/dqnOverloadThreshold, 1.0)
	balanceW := dqnBaseBalanceWeight + (dqnMaxBalanceWeight-dqnBaseBalanceWeight)*pressure

	tailPenalty := 0.0
	if m.AvgLatency > 0 {
		tailPenalty = math.Min(m.AvgLatency/dqnLatNorm, 1.0)
	}

	return balanceW*balance + (1-balanceW)*utility - dqnTailWeight*tailPenalty
}

func (d *DQNBalancer) ResetForNewScenario() {
	d.normalizer.Reset()
	d.nStep.Reset()
	d.trainExecMu.Lock()
	d.popArt.mean = 0
	d.popArt.m2 = 0
	d.popArt.count = 0
	d.trainExecMu.Unlock()
}

func (d *DQNBalancer) trainStep() float64 {
	if d.replayBuffer.Len() < d.minReplay {
		return 0
	}

	beta := d.betaSchedule.Next()
	batch, indices, isWeights := d.replayBuffer.Sample(d.batchSize, beta)
	bs := len(batch)
	if bs < d.batchSize {
		return 0
	}

	n := bs * d.stateSize
	if cap(d.trainBuf.states) < n {
		d.trainBuf.states = make([]float64, n)
		d.trainBuf.nextStates = make([]float64, n)
	}
	if cap(d.trainBuf.actions) < bs {
		d.trainBuf.actions = make([]float64, bs)
		d.trainBuf.rewards = make([]float64, bs)
		d.trainBuf.dones = make([]float64, bs)
		d.trainBuf.targets = make([]float64, bs)
	}
	statesBuf := d.trainBuf.states[:n]
	nextStatesBuf := d.trainBuf.nextStates[:n]
	actions := d.trainBuf.actions[:bs]
	rewards := d.trainBuf.rewards[:bs]
	dones := d.trainBuf.dones[:bs]

	for i, t := range batch {
		copy(statesBuf[i*d.stateSize:], padOrTrunc(t.State, d.stateSize))
		copy(nextStatesBuf[i*d.stateSize:], padOrTrunc(t.NextState, d.stateSize))
		if len(t.Action) > 0 {
			actions[i] = t.Action[0]
		} else {
			actions[i] = 0
		}
		rewards[i] = t.Reward
		if t.Done {
			dones[i] = 1.0
		} else {
			dones[i] = 0
		}
	}

	d.normalizer.NormalizeBatch(statesBuf, bs)
	d.normalizer.NormalizeBatch(nextStatesBuf, bs)

	statesT := tensors.FromFlatDataAndDimensions(statesBuf, bs, d.stateSize)
	nextStatesT := tensors.FromFlatDataAndDimensions(nextStatesBuf, bs, d.stateSize)

	d.trainExecMu.Lock()
	bestActionsT := d.bestActionsExec.MustExec1(nextStatesT)
	nextQT := d.targetQForActionsExec.MustExec1(nextStatesT, bestActionsT)
	actionsT := tensors.FromFlatDataAndDimensions(actions, bs)
	currentQT := d.batchQExec.MustExec1(statesT, actionsT)
	d.trainExecMu.Unlock()

	maxNextQ := tensorToFloat64Slice(nextQT)
	currentQ := tensorToFloat64Slice(currentQT)
	gammaPow := math.Pow(d.gamma, float64(d.nStep.n))
	targets := d.trainBuf.targets[:bs]
	for i := range batch {
		targets[i] = rewards[i] + gammaPow*maxNextQ[i]*(1-dones[i])
	}

	for _, t := range targets {
		d.popArt.count++
		delta := t - d.popArt.mean
		d.popArt.mean += delta / float64(d.popArt.count)
		d.popArt.m2 += delta * (t - d.popArt.mean)
	}
	paMean := d.popArt.mean
	var paSigma float64
	if d.popArt.count > 1 {
		paSigma = math.Sqrt(d.popArt.m2/float64(d.popArt.count-1)) + 1e-8
	} else {
		paSigma = 1.0
	}

	for i := range targets {
		targets[i] = (targets[i] - paMean) / paSigma
	}

	for i, idx := range indices {
		tdError := targets[i] - currentQ[i]
		d.replayBuffer.UpdatePriority(idx, tdError)
	}

	targetsT := tensors.FromFlatDataAndDimensions(targets, bs)
	isWeightsT := tensors.FromFlatDataAndDimensions(isWeights, bs)

	d.trainExecMu.Lock()
	lossT := d.trainExec.MustExec1(statesT, targetsT, actionsT, isWeightsT)
	d.targetNet.Step()
	d.trainExecMu.Unlock()

	if d.trainingSteps.Load()%25 == 0 {
		d.syncInferWeights()
	}

	lossVals := tensorToFloat64Slice(lossT)
	if len(lossVals) > 0 {
		return lossVals[0]
	}
	return 0
}

func padOrTrunc(s []float64, n int) []float64 {
	if len(s) == n {
		return s
	}
	out := make([]float64, n)
	copy(out, s)
	return out
}

func argmax(s []float64) int {
	if len(s) == 0 {
		return 0
	}
	best := 0
	for i := 1; i < len(s); i++ {
		if s[i] > s[best] {
			best = i
		}
	}
	return best
}

func tensorToFloat64Into(t *tensors.Tensor, dst []float64) {
	val := t.Value()
	switch v := val.(type) {
	case []float64:
		copy(dst, v)
	case []float32:
		for i := range min(len(v), len(dst)) {
			dst[i] = float64(v[i])
		}
	case float64:
		if len(dst) > 0 {
			dst[0] = v
		}
	case float32:
		if len(dst) > 0 {
			dst[0] = float64(v)
		}
	}
}

func tensorToFloat64Slice(t *tensors.Tensor) []float64 {
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
