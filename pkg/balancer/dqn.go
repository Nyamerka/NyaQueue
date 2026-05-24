package balancer

import (
	stdctx "context"
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

	_ "github.com/gomlx/gomlx/backends/simplego"
)

var float64BufPool = sync.Pool{New: func() any {
	buf := make([]float64, 256)
	return &buf
}}

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
	b.queue = b.queue[:0]
}

// DQNBalancer selects partitions via a DQN with softmax (Boltzmann) routing.
// State: [partition_loads, predicted_loads, inflight, msg_rate, avg_msg_size]; Action: partition index.
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
	fwdExec               *context.Exec
	bestActionsExec       *context.Exec
	targetQForActionsExec *context.Exec
	batchQExec            *context.Exec
	trainExec             *context.Exec
	stateSize             int

	replayBuffer *nn.PrioritizedReplayBuffer
	betaSchedule *nn.BetaSchedule
	normalizer   *nn.RunningNormalizer

	expCh  chan nn.Transition
	eg     *errgroup.Group
	cancel stdctx.CancelFunc

	fallbackP2C *PowerOfTwoChoices

	atomicState atomic.Pointer[stateSnapshot]

	droppedExperience atomic.Int64
	inflight          atomic.Pointer[[]atomic.Int64]

	cachedProbs   atomic.Pointer[[]float64]
	cachedArgmax  atomic.Int32
	deterministic bool

	prevSnap atomic.Pointer[policySnap]
	currSnap atomic.Pointer[policySnap]

	experiencesSeen atomic.Int64
	trainingSteps   atomic.Int64

	nStep *nStepBuffer

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
	prevLoads      atomic.Pointer[[]float64]
	smoothedCV     atomic.Uint64 // float64 stored as bits

	// Noise dimensions for NoisyLinearExternal inference.
	// Computed once during initGoMLX from hiddenSize/numPartitions.
	noiseValueIn  int // hiddenSize / 4  (input features of value NoisyLinear)
	noiseValueOut int // 1               (output features of value NoisyLinear)
	noiseAdvIn    int // hiddenSize / 4  (input features of advantage NoisyLinear)
	noiseAdvOut   int // numPartitions   (output features of advantage NoisyLinear)
}

// DQNOption configures a DQNBalancer.
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
		fallbackP2C:    NewPowerOfTwoChoices(),
		expCh:          make(chan nn.Transition, DefaultDQNExpChannelSize),
	}
	d.atomicState.Store(&stateSnapshot{
		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	})

	for _, opt := range opts {
		opt(d)
	}

	d.stateSize = numPartitions*5 + 4
	d.hiddenSize = max(DefaultDQNHiddenSize, d.stateSize*2)
	d.policyStateBuf = make([]float64, d.stateSize)
	d.betaSchedule = nn.NewBetaSchedule(0.4, 1.0, DefaultDQNReplayBufSize/d.batchSize)
	d.normalizer = nn.NewRunningNormalizer(d.stateSize)
	d.nStep = newNStepBuffer(5, d.gamma)
	d.initGoMLX()

	ctx, cancel := stdctx.WithCancel(stdctx.Background())
	d.cancel = cancel
	d.eg, _ = errgroup.WithContext(ctx)
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-balancer] trainLoop panic: %v", r)
			}
		}()
		d.trainLoop(ctx)
		return nil
	})
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-balancer] policyLoop panic: %v", r)
			}
		}()
		d.policyLoop(ctx)
		return nil
	})

	return d
}

func (d *DQNBalancer) initGoMLX() {
	d.backend = backends.MustNew()
	d.ctx = context.New()
	d.ctx.SetParams(map[string]any{
		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    d.lr,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamWeightDecay: DefaultDQNWeightDecay,
	})

	// Phase 1: create ALL variables (including NoisyLinear's sigma + RNG state)
	// by building one noisy graph on the non-Reuse context.
	initExec := context.MustNewExec(d.backend, d.ctx,
		func(ctx *context.Context, state *Node) *Node {
			return d.qNetworkGraph(ctx, state)
		})
	dummyState := make([]float64, d.stateSize)
	initExec.MustExec(dummyState)
	initExec.Finalize()

	// Compute noise dimensions from the dueling architecture.
	d.noiseValueIn = d.hiddenSize / 4
	d.noiseValueOut = 1
	d.noiseAdvIn = d.hiddenSize / 4
	d.noiseAdvOut = d.numPartitions

	// Phase 2: inference exec uses NoisyLinearExternal — noise is sampled in
	// Go (rand/v2.NormFloat64) and passed as graph inputs. This preserves
	// NoisyNet exploration while avoiding the shared #rngState concurrency bug.
	d.fwdExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, state, epsValueIn, epsValueOut, epsAdvIn, epsAdvOut *Node) *Node {
			return d.qNetworkGraphExternalNoise(ctx, state, epsValueIn, epsValueOut, epsAdvIn, epsAdvOut)
		})

	d.targetNet = nn.NewTargetNetwork(d.ctx)

	// Phase 3: training execs use full NoisyLinear (with noise via ctx.RandomNormal).
	// All of these run sequentially in the trainLoop goroutine.
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
	v := nn.NoisyLinear(ctx.In("value_out"), vStream, 1, nn.DefaultNoisySigmaInit)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = layers.LayerNormalization(ctx.In("ln_ah"), aStream, -1).Done()
	aStream = activations.Relu(aStream)
	a := nn.NoisyLinear(ctx.In("adv_out"), aStream, d.numPartitions, nn.DefaultNoisySigmaInit)

	aMean := ReduceMean(a, -1)
	aMean = ExpandDims(aMean, -1)
	return Add(v, Sub(a, aMean))
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
	v := nn.NoisyLinearExternal(ctx.In("value_out"), vStream, epsValueIn, epsValueOut, 1, nn.DefaultNoisySigmaInit)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = layers.LayerNormalization(ctx.In("ln_ah"), aStream, -1).Done()
	aStream = activations.Relu(aStream)
	a := nn.NoisyLinearExternal(ctx.In("adv_out"), aStream, epsAdvIn, epsAdvOut, d.numPartitions, nn.DefaultNoisySigmaInit)

	aMean := ReduceMean(a, -1)
	aMean = ExpandDims(aMean, -1)
	return Add(v, Sub(a, aMean))
}

func (d *DQNBalancer) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	q := d.qNetworkBatch(ctx, x)
	return Reshape(q, d.numPartitions)
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

	epsVI := sampleNoise(d.noiseValueIn)
	epsVO := sampleNoise(d.noiseValueOut)
	epsAI := sampleNoise(d.noiseAdvIn)
	epsAO := sampleNoise(d.noiseAdvOut)

	result := d.fwdExec.MustExec1(norm, epsVI, epsVO, epsAI, epsAO)
	float64BufPool.Put(buf)
	return tensorToFloat64Slice(result)
}

func (d *DQNBalancer) shouldRecordExperience() bool {
	return d.cachedProbs.Load() != nil
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
	inf := d.ensureInflight(numPartitions)

	cached := d.cachedProbs.Load()
	if cached == nil || len(*cached) < numPartitions {
		idx := d.fallbackP2C.SelectPartition(topic, key, numPartitions)
		(*inf)[idx].Add(1)
		return idx
	}

	if d.deterministic {
		idx := int(d.cachedArgmax.Load())
		if idx >= numPartitions {
			idx = 0
		}
		(*inf)[idx].Add(1)
		return idx
	}

	probs := (*cached)[:numPartitions]

	var totalInflight float64
	for i := 0; i < numPartitions; i++ {
		totalInflight += float64((*inf)[i].Load())
	}

	if totalInflight > 0 {
		corrected := make([]float64, numPartitions)
		for i := range corrected {
			logit := math.Log(probs[i] + 1e-10)
			logit -= dqnBoltzmannInflightAlpha * float64((*inf)[i].Load()) / totalInflight
			corrected[i] = logit
		}
		maxLogit := corrected[0]
		for _, v := range corrected[1:] {
			if v > maxLogit {
				maxLogit = v
			}
		}
		var sum float64
		for i := range corrected {
			corrected[i] = math.Exp(corrected[i] - maxLogit)
			sum += corrected[i]
		}
		for i := range corrected {
			corrected[i] /= sum
		}
		idx := sampleCDF(corrected)
		(*inf)[idx].Add(1)
		return idx
	}

	idx := sampleCDF(probs)
	(*inf)[idx].Add(1)
	return idx
}

func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	reward := d.computeReward(m)

	d.fallbackP2C.OnMetrics(m)

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

func (d *DQNBalancer) BalancerType() string { return "DQN" }
func (d *DQNBalancer) NumPartitions() int   { return d.numPartitions }

func (d *DQNBalancer) DroppedExperience() int64 {
	return d.droppedExperience.Load()
}

func (d *DQNBalancer) Temperature() float64 {
	return currentTemperature(d.experiencesSeen.Load())
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

func (d *DQNBalancer) Stop() {
	d.cancel()
	_ = d.eg.Wait()
}

func (d *DQNBalancer) trainLoop(ctx stdctx.Context) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	steps := 0
	var lastLoss float64

	for {
		select {
		case t := <-d.expCh:
			d.replayBuffer.Push(t)
			steps++
			if steps%d.trainEvery == 0 {
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

func (d *DQNBalancer) policyInterval() time.Duration {
	snap := d.atomicState.Load()
	if snap != nil && snap.msgRate > 0 && snap.msgRate < dqnLowRateThreshold {
		return dqnSlowPolicyTickInterval
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

			tau := currentTemperature(d.experiencesSeen.Load())
			probs := softmaxFromQ(qValues[:d.numPartitions], tau)
			d.cachedProbs.Store(&probs)
			d.cachedArgmax.Store(int32(argmax(qValues[:d.numPartitions])))

			stateCopy := make([]float64, len(d.policyStateBuf))
			copy(stateCopy, d.policyStateBuf)
			snap := &policySnap{
				state:  stateCopy,
				action: argmax(qValues[:d.numPartitions]),
			}

			d.prevSnap.Store(d.currSnap.Load())
			d.currSnap.Store(snap)

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

// computeReward produces a direct shaped reward signal for the DQN.
//
// Base reward combines two orthogonal signals:
//
//	balance  = -CV(loads) ∈ [-1, 0]
//	utility  = log1p(throughput/scale) / log1p(max/scale) ∈ [0, 1]
//
// No EWMA-diff: raw ∈ [-1.3, 1.0] — already scaled for learning.
const dqnCVEWMAAlpha = 0.3

func (d *DQNBalancer) computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}

	meanLoad := stat.Mean(m.PartitionLoads, nil)

	var balancePenalty float64
	if meanLoad > 0 {
		instantCV := stat.StdDev(m.PartitionLoads, nil) / meanLoad
		prevBits := d.smoothedCV.Load()
		prevCV := math.Float64frombits(prevBits)
		smoothed := dqnCVEWMAAlpha*instantCV + (1-dqnCVEWMAAlpha)*prevCV
		d.smoothedCV.Store(math.Float64bits(smoothed))
		balancePenalty = -math.Min(smoothed, 1.0)
	}

	var utilityReward float64
	if m.Throughput > 0 {
		utilityReward = math.Log1p(m.Throughput/dqnThroughputScale) / dqnThroughputLogNorm
	}

	pressure := math.Min(meanLoad/dqnOverloadThreshold, 1.0)
	balanceW := dqnBaseBalanceWeight + (dqnMaxBalanceWeight-dqnBaseBalanceWeight)*pressure

	raw := balanceW*balancePenalty + (1-balanceW)*utilityReward

	maxLoad := 0.0
	for _, l := range m.PartitionLoads {
		if l > maxLoad {
			maxLoad = l
		}
	}
	if maxLoad > 1.0 {
		excess := maxLoad - 1.0
		raw -= dqnMaxLoadWeight * excess * excess
	}

	return raw
}

// ResetForNewScenario resets accumulated statistics for a clean training signal.
// Should be called before each new scenario warmup to prevent cross-scenario pollution.
func (d *DQNBalancer) ResetForNewScenario() {
	d.normalizer.Reset()
	d.nStep.Reset()
	d.popArt.mean = 0
	d.popArt.m2 = 0
	d.popArt.count = 0
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

	bestActionsT := d.bestActionsExec.MustExec1(nextStatesT)
	nextQT := d.targetQForActionsExec.MustExec1(nextStatesT, bestActionsT)
	actionsT := tensors.FromFlatDataAndDimensions(actions, bs)
	currentQT := d.batchQExec.MustExec1(statesT, actionsT)

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
	lossT := d.trainExec.MustExec1(statesT, targetsT, actionsT, isWeightsT)

	d.targetNet.Step()

	lossVals := tensorToFloat64Slice(lossT)
	if len(lossVals) > 0 {
		return lossVals[0]
	}
	return 0
}

func sampleNoise(n int) []float64 {
	eps := make([]float64, n)
	for i := range eps {
		eps[i] = rand.NormFloat64()
	}
	return eps
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
