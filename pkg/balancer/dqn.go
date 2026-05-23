package balancer

import (
	stdctx "context"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
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
	"github.com/puzpuzpuz/xsync/v3"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/stat"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

// DQNBalancer selects partitions via a DQN with softmax (Boltzmann) routing.
// State: [partition_loads, predicted_loads, inflight, msg_rate, avg_msg_size]; Action: partition index.
type policySnap struct {
	state  []float64
	action int
	gen    int64
}

type DQNBalancer struct {
	weightsMu *xsync.RBMutex

	numPartitions int
	hiddenSize    int
	gamma         float64
	lr            float64
	huberDelta    float64
	batchSize     int
	minReplay     int
	trainEvery    int

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

	stateMu        *xsync.RBMutex
	loads          []float64
	predictedLoads []float64
	msgRate        float64
	avgMsgSize     float64

	droppedExperience atomic.Int64
	inflight          atomic.Pointer[[]atomic.Int64]

	cachedQValues atomic.Pointer[[]float64]

	prevSnap atomic.Pointer[policySnap]
	currSnap atomic.Pointer[policySnap]
	snapGen  atomic.Int64

	lastProcessedGen atomic.Int64

	trainingSteps atomic.Int64
	ewmaReward    atomic.Uint64 // float64 bits — differential reward baseline
	peakReward    atomic.Uint64 // float64 bits — best EWMA seen (drift detection)

	boosterMu *xsync.RBMutex
	booster   atomic.Pointer[dqnBooster]

	policyStateBuf []float64
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

func NewDQNBalancer(numPartitions int, opts ...DQNOption) *DQNBalancer {
	d := &DQNBalancer{
		weightsMu:     xsync.NewRBMutex(),
		stateMu:       xsync.NewRBMutex(),
		boosterMu:     xsync.NewRBMutex(),
		numPartitions: numPartitions,
		hiddenSize:    DefaultDQNHiddenSize,
		gamma:         DefaultDQNGamma,
		lr:            DefaultDQNLearningRate,
		huberDelta:    1.0,
		batchSize:     DefaultDQNBatchSize,
		minReplay:     DefaultDQNMinReplay,
		trainEvery:    DefaultDQNTrainEvery,
		replayBuffer:  nn.NewPrioritizedReplayBuffer(DefaultDQNReplayBufSize, 0.6),
		fallbackP2C:   NewPowerOfTwoChoices(),
		expCh:         make(chan nn.Transition, DefaultDQNExpChannelSize),

		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.stateSize = numPartitions*3 + 2
	d.hiddenSize = max(DefaultDQNHiddenSize, d.stateSize*3)
	d.policyStateBuf = make([]float64, d.stateSize)
	d.betaSchedule = nn.NewBetaSchedule(0.4, 1.0, DefaultDQNReplayBufSize/d.batchSize)
	d.normalizer = nn.NewRunningNormalizer(d.stateSize)
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

	d.fwdExec = context.MustNewExec(d.backend, d.ctx, func(ctx *context.Context, state *Node) *Node {
		return d.qNetworkGraph(ctx, state)
	})
	dummyState := make([]float64, d.stateSize)
	d.fwdExec.MustExec(dummyState)

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
	v := layers.Dense(ctx.In("value_out"), vStream, true, 1)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = layers.LayerNormalization(ctx.In("ln_ah"), aStream, -1).Done()
	aStream = activations.Relu(aStream)
	a := layers.Dense(ctx.In("adv_out"), aStream, true, d.numPartitions)

	aMean := ReduceMean(a, -1)
	aMean = ExpandDims(aMean, -1)
	return Add(v, Sub(a, aMean))
}

func (d *DQNBalancer) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	q := d.qNetworkBatch(ctx, x)
	return Reshape(q, d.numPartitions)
}

func (d *DQNBalancer) forward(state []float64) []float64 {
	result := d.fwdExec.MustExec1(state)
	return tensorToFloat64Slice(result)
}

// temperature returns the current Boltzmann temperature based on training progress.
func (d *DQNBalancer) temperature() float64 {
	steps := d.trainingSteps.Load()
	if steps < dqnMinStepsForGreedy {
		return dqnInitialTemperature
	}
	progress := float64(steps-dqnMinStepsForGreedy) / float64(dqnTempDecaySteps)
	progress = math.Min(progress, 1.0)
	return dqnInitialTemperature * math.Pow(dqnFinalTemperature/dqnInitialTemperature, progress)
}

// softmaxSelect samples a partition from Q-values using Boltzmann exploration.
// At high τ the distribution is ~uniform; as τ→0 it converges to argmax.
func (d *DQNBalancer) softmaxSelect(qValues []float64) int {
	tau := d.temperature()
	maxQ := slices.Max(qValues)

	probs := make([]float64, len(qValues))
	sum := 0.0
	for i, q := range qValues {
		probs[i] = math.Exp((q - maxQ) / tau)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return len(qValues) - 1
}

func (d *DQNBalancer) shouldRecordExperience() bool {
	return d.temperature() < dqnExperienceRecordThreshold
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

	if cached := d.cachedQValues.Load(); cached != nil && len(*cached) >= numPartitions {
		qv := (*cached)[:numPartitions]
		if b := d.booster.Load(); b != nil {
			qv = b.addResidual(qv)
		}
		idx := d.softmaxSelect(qv)
		(*inf)[idx].Add(1)
		return idx
	}

	idx := d.fallbackP2C.SelectPartition(topic, key, numPartitions)
	(*inf)[idx].Add(1)
	return idx
}

func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	reward := d.computeReward(m)

	d.fallbackP2C.OnMetrics(m)

	d.stateMu.Lock()
	if len(m.PartitionLoads) > 0 {
		if cap(d.loads) < len(m.PartitionLoads) {
			d.loads = make([]float64, len(m.PartitionLoads))
		}
		d.loads = d.loads[:len(m.PartitionLoads)]
		copy(d.loads, m.PartitionLoads)
	}
	if len(m.PredictedLoads) > 0 {
		if cap(d.predictedLoads) < len(m.PredictedLoads) {
			d.predictedLoads = make([]float64, len(m.PredictedLoads))
		}
		d.predictedLoads = d.predictedLoads[:len(m.PredictedLoads)]
		copy(d.predictedLoads, m.PredictedLoads)
	}
	d.msgRate = m.MsgRate
	d.avgMsgSize = m.AvgMsgSize
	d.stateMu.Unlock()

	if !d.shouldRecordExperience() {
		return
	}

	prev := d.prevSnap.Load()
	curr := d.currSnap.Load()
	if prev != nil && curr != nil && curr.gen > d.lastProcessedGen.Load() {
		d.lastProcessedGen.Store(curr.gen)
		t := nn.Transition{
			State:     append([]float64(nil), prev.state...),
			Action:    []float64{float64(prev.action)},
			Reward:    reward,
			NextState: append([]float64(nil), curr.state...),
			Done:      false,
		}
		select {
		case d.expCh <- t:
		default:
			d.droppedExperience.Add(1)
		}
	}
}

func (d *DQNBalancer) SetPredictedLoads(predicted []float64) {
	d.stateMu.Lock()
	if cap(d.predictedLoads) < len(predicted) {
		d.predictedLoads = make([]float64, len(predicted))
	}
	d.predictedLoads = d.predictedLoads[:len(predicted)]
	copy(d.predictedLoads, predicted)
	d.stateMu.Unlock()
}

func (d *DQNBalancer) DroppedExperience() int64 {
	return d.droppedExperience.Load()
}

func (d *DQNBalancer) Temperature() float64 {
	return d.temperature()
}

func (d *DQNBalancer) TrainingSteps() int64 {
	return d.trainingSteps.Load()
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
				lastLoss = d.trainStep()
				d.trainingSteps.Add(1)
			}
			if steps%100 == 0 {
				log.Printf("[dqn-bal] step=%d loss=%.4f τ=%.3f buf=%d/%d dropped=%d",
					steps, lastLoss, d.temperature(), d.replayBuffer.Len(), d.replayBuffer.Cap(),
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
	srt := d.stateMu.RLock()
	rate := d.msgRate
	d.stateMu.RUnlock(srt)

	if rate > 0 && rate < dqnLowRateThreshold {
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
			srt := d.stateMu.RLock()
			d.buildStateInto(d.policyStateBuf)
			d.stateMu.RUnlock(srt)

			rt := d.weightsMu.RLock()
			qValues := d.forward(d.policyStateBuf)
			d.weightsMu.RUnlock(rt)

			d.cachedQValues.Store(&qValues)

			gen := d.snapGen.Add(1)
			stateCopy := make([]float64, len(d.policyStateBuf))
			copy(stateCopy, d.policyStateBuf)
			snap := &policySnap{
				state:  stateCopy,
				action: argmax(qValues[:d.numPartitions]),
				gen:    gen,
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
	n := copy(dst, d.loads)
	if n < d.numPartitions {
		n = d.numPartitions
	}
	copy(dst[d.numPartitions:], d.predictedLoads)

	inflightBase := d.numPartitions * 2
	if ptr := d.inflight.Load(); ptr != nil {
		for i := 0; i < d.numPartitions && i < len(*ptr); i++ {
			dst[inflightBase+i] = float64((*ptr)[i].Load()) / dqnInflightScale
		}
	}

	idx := d.numPartitions * 3
	if idx < len(dst) {
		dst[idx] = d.msgRate / dqnMsgRateScale
	}
	if idx+1 < len(dst) {
		dst[idx+1] = d.avgMsgSize / dqnMsgSizeScale
	}

	d.normalizer.Observe(dst)
	d.normalizer.NormalizeInPlace(dst)
}

// computeReward produces a differential reward signal for the DQN.
//
// Base reward combines two orthogonal signals:
//
//	balance  = -CV(loads) ∈ [-1, 0]
//	utility  = log1p(throughput/scale) / log1p(max/scale) ∈ [0, 1]
//
// Differential: reward = raw - ewma(raw). Positive = improving, negative = degrading.
func (d *DQNBalancer) computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}

	meanLoad := stat.Mean(m.PartitionLoads, nil)

	var balancePenalty float64
	if meanLoad > 0 {
		cv := stat.StdDev(m.PartitionLoads, nil) / meanLoad
		balancePenalty = -math.Min(cv, 1.0)
	}

	var utilityReward float64
	if m.Throughput > 0 {
		utilityReward = math.Log1p(m.Throughput/dqnThroughputScale) / dqnThroughputLogNorm
	}

	pressure := math.Min(meanLoad/dqnOverloadThreshold, 1.0)
	balanceW := dqnBaseBalanceWeight + (dqnMaxBalanceWeight-dqnBaseBalanceWeight)*pressure

	rawReward := balanceW*balancePenalty + (1-balanceW)*utilityReward

	var totalDepth int
	for _, dep := range m.QueueDepth {
		totalDepth += dep
	}
	if totalDepth > 0 {
		normalized := float64(totalDepth) / dqnDepthSoftCap
		depthPenalty := -math.Min(normalized*normalized, 1.0)
		rawReward += dqnDepthWeight * depthPenalty
	}

	prevEWMA := math.Float64frombits(d.ewmaReward.Load())
	diff := rawReward - prevEWMA
	newEWMA := prevEWMA + dqnRewardEWMAAlpha*(rawReward-prevEWMA)
	d.ewmaReward.Store(math.Float64bits(newEWMA))

	peak := math.Float64frombits(d.peakReward.Load())
	if newEWMA > peak {
		d.peakReward.Store(math.Float64bits(newEWMA))
	} else if peak > 0 && newEWMA < peak*dqnDriftThreshold {
		d.handleDrift()
	}

	return diff
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

	statesBuf := make([]float64, bs*d.stateSize)
	nextStatesBuf := make([]float64, bs*d.stateSize)
	actions := make([]float64, bs)
	rewards := make([]float64, bs)
	dones := make([]float64, bs)

	for i, t := range batch {
		copy(statesBuf[i*d.stateSize:], padOrTrunc(t.State, d.stateSize))
		copy(nextStatesBuf[i*d.stateSize:], padOrTrunc(t.NextState, d.stateSize))
		if len(t.Action) > 0 {
			actions[i] = t.Action[0]
		}
		rewards[i] = t.Reward
		if t.Done {
			dones[i] = 1.0
		}
	}

	statesT := tensors.FromFlatDataAndDimensions(statesBuf, bs, d.stateSize)
	nextStatesT := tensors.FromFlatDataAndDimensions(nextStatesBuf, bs, d.stateSize)

	rt := d.weightsMu.RLock()
	bestActionsT := d.bestActionsExec.MustExec1(nextStatesT)
	nextQT := d.targetQForActionsExec.MustExec1(nextStatesT, bestActionsT)
	actionsT := tensors.FromFlatDataAndDimensions(actions, bs)
	currentQT := d.batchQExec.MustExec1(statesT, actionsT)
	d.weightsMu.RUnlock(rt)

	maxNextQ := tensorToFloat64Slice(nextQT)
	currentQ := tensorToFloat64Slice(currentQT)
	targets := make([]float64, bs)
	for i := range batch {
		targets[i] = rewards[i] + d.gamma*maxNextQ[i]*(1-dones[i])
	}

	for i, idx := range indices {
		tdError := targets[i] - currentQ[i]
		d.replayBuffer.UpdatePriority(idx, tdError)
	}

	d.weightsMu.Lock()
	defer d.weightsMu.Unlock()

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
