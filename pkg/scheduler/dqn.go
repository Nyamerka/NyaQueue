package scheduler

import (
	stdctx "context"
	"log"
	"math"
	"math/rand/v2"
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
	"github.com/puzpuzpuz/xsync/v3"
	"github.com/samber/oops"
	"golang.org/x/sync/errgroup"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

// DQNScheduler adaptively sets the priority threshold via DQN.
// State: [level_distribution, avg_wait_per_level, queue_depth, consumer_lag]; Action: threshold.
type schedPolicySnap struct {
	state     []float64
	threshold int
}

type DQNScheduler struct {
	weightsMu *xsync.RBMutex

	hiddenSize int
	stateSize  int
	numActions int
	epsilon    float64
	gamma      float64
	lr         float64
	huberDelta float64
	batchSize  int
	minReplay  int
	trainEvery int

	backend               backends.Backend
	ctx                   *context.Context
	targetNet             *nn.TargetNetwork
	fwdExec               *context.Exec
	bestActionsExec       *context.Exec
	targetQForActionsExec *context.Exec
	batchQExec            *context.Exec
	trainExec             *context.Exec

	replayBuffer *nn.PrioritizedReplayBuffer
	crisisBuffer *nn.ReplayBuffer
	betaSchedule *nn.BetaSchedule
	normalizer   *nn.RunningNormalizer

	expCh  chan nn.Transition
	eg     *errgroup.Group
	cancel stdctx.CancelFunc

	stateMu   *xsync.RBMutex
	lastState []float64
	lastPI    atomic.Pointer[broker.PriorityIndex]
	threshold int

	cachedThreshold atomic.Int32

	prevSnap atomic.Pointer[schedPolicySnap]
	currSnap atomic.Pointer[schedPolicySnap]

	latencySums   [broker.MaxPriority]atomic.Uint64
	latencyCounts [broker.MaxPriority]atomic.Uint64

	prevTotalDepth  atomic.Int64
	prevRewardDepth atomic.Int64
	produceRate     float64
	consumeRate     float64

	lastPolicyDepth int // protected by stateMu
}

const (
	DefaultDQNSchedTrainEvery = 4
	DefaultDQNSchedExpChSize  = 2048
)

type DQNSchedOption func(*DQNScheduler)

func WithDQNSchedEpsilon(e float64) DQNSchedOption       { return func(d *DQNScheduler) { d.epsilon = e } }
func WithDQNSchedGamma(g float64) DQNSchedOption         { return func(d *DQNScheduler) { d.gamma = g } }
func WithDQNSchedLearningRate(lr float64) DQNSchedOption { return func(d *DQNScheduler) { d.lr = lr } }
func WithDQNSchedHiddenSize(n int) DQNSchedOption        { return func(d *DQNScheduler) { d.hiddenSize = n } }
func WithDQNSchedHuberDelta(delta float64) DQNSchedOption {
	return func(d *DQNScheduler) { d.huberDelta = delta }
}
func WithDQNSchedReplayBufSize(n int) DQNSchedOption {
	return func(d *DQNScheduler) { d.replayBuffer = nn.NewPrioritizedReplayBuffer(n, 0.6) }
}
func WithDQNSchedBatchSize(n int) DQNSchedOption { return func(d *DQNScheduler) { d.batchSize = n } }

func NewDQNScheduler(opts ...DQNSchedOption) *DQNScheduler {
	d := &DQNScheduler{
		weightsMu:    xsync.NewRBMutex(),
		stateMu:      xsync.NewRBMutex(),
		hiddenSize:   DefaultDQNSchedHiddenSize,
		stateSize:    DefaultDQNSchedStateSize,
		numActions:   DefaultDQNSchedActions,
		epsilon:      DefaultDQNSchedEpsilon,
		gamma:        DefaultDQNSchedGamma,
		lr:           DefaultDQNSchedLearningRate,
		huberDelta:   1.0,
		batchSize:    DefaultDQNSchedBatchSize,
		minReplay:    DefaultDQNSchedMinReplay,
		trainEvery:   DefaultDQNSchedTrainEvery,
		replayBuffer: nn.NewPrioritizedReplayBuffer(DefaultDQNSchedReplayBufSize, 0.6),
		crisisBuffer: nn.NewReplayBuffer(dqnSchedCrisisBufSize),
		threshold:    DefaultDQNSchedThreshold,
		expCh:        make(chan nn.Transition, DefaultDQNSchedExpChSize),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.cachedThreshold.Store(int32(d.threshold))
	d.betaSchedule = nn.NewBetaSchedule(0.4, 1.0, DefaultDQNSchedReplayBufSize/d.batchSize)
	d.normalizer = nn.NewRunningNormalizer(d.stateSize)
	d.initGoMLX()

	ctx, cancel := stdctx.WithCancel(stdctx.Background())
	d.cancel = cancel
	d.eg, _ = errgroup.WithContext(ctx)
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-scheduler] trainLoop panic: %v", r)
			}
		}()
		d.trainLoop(ctx)
		return nil
	})
	d.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[dqn-scheduler] policyLoop panic: %v", r)
			}
		}()
		d.policyLoop(ctx)
		return nil
	})

	d.seedExpertTransitions()

	return d
}

func (d *DQNScheduler) seedExpertTransitions() {
	depthIdx := broker.MaxPriority * 2
	for i := 0; i < dqnSchedExpertSeeds; i++ {
		state := make([]float64, d.stateSize)
		state[depthIdx] = 0.8 + rand.Float64()*0.2
		if depthIdx+1 < d.stateSize {
			state[depthIdx+1] = 0.5 + rand.Float64()*0.5
		}
		if depthIdx+2 < d.stateSize {
			state[depthIdx+2] = 3.0 + rand.Float64()*2.0
		}

		next := make([]float64, d.stateSize)
		copy(next, state)
		next[depthIdx] *= 0.6

		d.replayBuffer.Push(nn.Transition{
			State:     state,
			Action:    []float64{0},
			Reward:    2.0,
			NextState: next,
			Done:      false,
		})
	}
}

func (d *DQNScheduler) initGoMLX() {
	d.backend = backends.MustNew()
	d.ctx = context.New()

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
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numActions, qAll.DType())
			return ReduceSum(Mul(qAll, oneHot), -1)
		})

	d.batchQExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, actionIdx *Node) *Node {
			qAll := d.qNetworkBatch(ctx, states)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numActions, qAll.DType())
			return ReduceSum(Mul(qAll, oneHot), -1)
		})

	huber := losses.MakeHuberLoss(d.huberDelta)
	d.trainExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, targetQ, actionIdx, isWeights *Node) *Node {
			g := states.Graph()
			qAll := d.qNetworkBatch(ctx, states)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numActions, qAll.DType())
			qSelected := ReduceSum(Mul(qAll, oneHot), -1)
			loss := huber([]*Node{targetQ, isWeights}, []*Node{qSelected})
			grads := ctx.BuildTrainableVariablesGradientsGraph(loss)
			lr := Const(g, d.lr)
			idx := 0
			ctx.EnumerateVariables(func(v *context.Variable) {
				if v.Trainable && v.InUseByGraph(g) {
					w := v.ValueGraph(g)
					v.SetValueGraph(Sub(w, Mul(lr, grads[idx])))
					idx++
				}
			})
			return loss
		})
}

func (d *DQNScheduler) qNetworkBatch(ctx *context.Context, x *Node) *Node {
	h := layers.Dense(ctx.In("feature1"), x, true, d.hiddenSize)
	h = activations.Relu(h)
	h = layers.Dense(ctx.In("feature2"), h, true, d.hiddenSize/2)
	h = activations.Relu(h)

	vStream := layers.Dense(ctx.In("value_hidden"), h, true, d.hiddenSize/4)
	vStream = activations.Relu(vStream)
	v := layers.Dense(ctx.In("value_out"), vStream, true, 1)

	aStream := layers.Dense(ctx.In("adv_hidden"), h, true, d.hiddenSize/4)
	aStream = activations.Relu(aStream)
	a := layers.Dense(ctx.In("adv_out"), aStream, true, d.numActions)

	aMean := ReduceMean(a, -1)
	aMean = ExpandDims(aMean, -1)
	return Add(v, Sub(a, aMean))
}

func (d *DQNScheduler) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	q := d.qNetworkBatch(ctx, x)
	return Reshape(q, d.numActions)
}

func (d *DQNScheduler) forward(state []float64) []float64 {
	result := d.fwdExec.MustExec1(state)
	return tensorToFloat64Sched(result)
}

func (d *DQNScheduler) Next(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	pi := partition.PriorityIndex()
	if pi == nil {
		return nil, consumerOffset, oops.Errorf("partition %d has no PriorityIndex", partition.ID())
	}

	d.lastPI.Store(pi)

	srt := d.stateMu.RLock()
	eps := d.epsilon
	d.stateMu.RUnlock(srt)

	explore := rand.Float64() < eps

	var threshold int
	if explore {
		threshold = rand.IntN(d.numActions)
	} else {
		threshold = int(d.cachedThreshold.Load())
	}

	entry, ok := pi.PopWithThreshold(threshold)
	if !ok {
		return nil, consumerOffset, broker.ErrNoMessages
	}

	msg, err := partition.Read(uint64(entry.WalOffset))
	if err != nil {
		return nil, 0, err
	}

	latencyNs := uint64(time.Since(entry.ArrivedAt).Nanoseconds())
	level := int(msg.Header.Priority)
	if level >= broker.MaxPriority {
		level = broker.MaxPriority - 1
	}
	d.latencySums[level].Add(latencyNs)
	d.latencyCounts[level].Add(1)

	return msg, uint64(entry.WalOffset), nil
}

func (d *DQNScheduler) Enqueue(_ *broker.Message, _ int64) {}

func (d *DQNScheduler) OnMetrics(m broker.Metrics) {
	d.stateMu.Lock()
	d.produceRate = m.MsgRate
	d.consumeRate = m.ConsumeRate
	d.stateMu.Unlock()

	prev := d.prevSnap.Load()
	curr := d.currSnap.Load()
	if prev == nil || curr == nil {
		return
	}

	reward := d.computePerPriorityReward()

	t := nn.Transition{
		State:     append([]float64(nil), prev.state...),
		Action:    []float64{float64(prev.threshold)},
		Reward:    reward,
		NextState: append([]float64(nil), curr.state...),
		Done:      false,
	}

	select {
	case d.expCh <- t:
	default:
	}
}

func (d *DQNScheduler) computePerPriorityReward() float64 {
	var sums [broker.MaxPriority]uint64
	var counts [broker.MaxPriority]uint64
	for i := 0; i < broker.MaxPriority; i++ {
		sums[i] = d.latencySums[i].Swap(0)
		counts[i] = d.latencyCounts[i].Swap(0)
	}

	var reward float64
	var totalWeight float64
	avgLatencies := make([]float64, broker.MaxPriority)
	for level := 0; level < broker.MaxPriority; level++ {
		if counts[level] == 0 {
			continue
		}
		meanLatencySec := float64(sums[level]) / float64(counts[level]) / 1e9
		avgLatencies[level] = meanLatencySec
		weight := float64(broker.MaxPriority - level) // P0 → weight=10, P9 → weight=1
		reward -= meanLatencySec * weight
		totalWeight += weight
	}
	if totalWeight > 0 {
		reward /= totalWeight
	}

	if pi := d.lastPI.Load(); pi != nil {
		currentDepth := int64(pi.Len())
		prevDepth := d.prevRewardDepth.Swap(currentDepth)
		growth := float64(currentDepth - prevDepth)
		if growth > 0 {
			reward -= dqnSchedGrowthPenaltyW * math.Log1p(growth)
		}
	}

	var highPrioActive, lowPrioActive bool
	var highPrioLatency, lowPrioLatency float64
	for level := 0; level < broker.MaxPriority; level++ {
		if counts[level] > 0 && !highPrioActive {
			highPrioLatency = avgLatencies[level]
			highPrioActive = true
			break
		}
	}
	for level := broker.MaxPriority - 1; level >= 0; level-- {
		if counts[level] > 0 && !lowPrioActive {
			lowPrioLatency = avgLatencies[level]
			lowPrioActive = true
			break
		}
	}
	if highPrioActive && lowPrioActive && highPrioLatency < lowPrioLatency {
		reward += dqnSchedPriorityOrderBonus
	}

	return reward
}

func (d *DQNScheduler) Stop() {
	d.cancel()
	_ = d.eg.Wait()
}

func (d *DQNScheduler) policyLoop(ctx stdctx.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pi := d.lastPI.Load()
			if pi == nil {
				continue
			}

			state := d.buildState(pi)

			d.stateMu.Lock()
			d.lastState = state
			d.stateMu.Unlock()

			rt := d.weightsMu.RLock()
			q := d.forward(state)
			d.weightsMu.RUnlock(rt)

			threshold := argmaxSched(q)
			d.cachedThreshold.Store(int32(threshold))

			snap := &schedPolicySnap{
				state:     state,
				threshold: threshold,
			}
			d.prevSnap.Store(d.currSnap.Load())
			d.currSnap.Store(snap)

			d.stateMu.Lock()
			d.threshold = threshold

			depth := pi.Len()
			velocity := depth - d.lastPolicyDepth
			d.lastPolicyDepth = depth

			if depth > dqnSchedOverloadDepth && velocity > 0 {
				d.epsilon = 0
			} else {
				d.epsilon = DefaultDQNSchedEpsilon
			}
			d.stateMu.Unlock()

		case <-ctx.Done():
			return
		}
	}
}

func (d *DQNScheduler) trainLoop(ctx stdctx.Context) {
	steps := 0

	for {
		select {
		case t := <-d.expCh:
			d.replayBuffer.Push(t)
			if d.isOverloadTransition(t) {
				d.crisisBuffer.Push(t)
			}
			steps++
			if steps%d.trainEvery == 0 {
				d.trainStep()
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

func (d *DQNScheduler) isOverloadTransition(t nn.Transition) bool {
	depthIdx := broker.MaxPriority * 2
	velocityIdx := depthIdx + 1
	if depthIdx >= len(t.State) {
		return false
	}
	depth := t.State[depthIdx]
	var velocity float64
	if velocityIdx < len(t.State) {
		velocity = t.State[velocityIdx]
	}
	return depth > float64(dqnSchedOverloadDepth) && velocity > 0
}

func (d *DQNScheduler) buildState(pi *broker.PriorityIndex) []float64 {
	dist := pi.LevelDistribution()

	state := make([]float64, d.stateSize)
	levels := len(dist)
	totalDepth := 0
	for i := 0; i < levels && i < d.stateSize; i++ {
		state[i] = float64(dist[i])
		totalDepth += dist[i]
	}

	for i := 0; i < levels; i++ {
		if waitIdx := levels + i; waitIdx < d.stateSize {
			cnt := d.latencyCounts[i].Load()
			if cnt > 0 {
				sumNs := d.latencySums[i].Load()
				state[waitIdx] = float64(sumNs) / float64(cnt) / 1e9
			}
		}
	}

	depthIdx := levels * 2
	if depthIdx < d.stateSize {
		state[depthIdx] = float64(totalDepth)
	}

	velocityIdx := depthIdx + 1
	if velocityIdx < d.stateSize {
		prevDepth := d.prevTotalDepth.Swap(int64(totalDepth))
		velocity := float64(int64(totalDepth) - prevDepth)
		state[velocityIdx] = velocity / dqnSchedVelocityScale
	}

	ratioIdx := velocityIdx + 1
	if ratioIdx < d.stateSize {
		d.stateMu.Lock()
		if d.consumeRate > 0 {
			state[ratioIdx] = d.produceRate / d.consumeRate
		}
		d.stateMu.Unlock()
	}

	d.normalizer.Observe(state)
	d.normalizer.NormalizeInPlace(state)
	return state
}

func (d *DQNScheduler) trainStep() {
	if d.replayBuffer.Len() < d.minReplay {
		return
	}

	mainSize := d.batchSize
	crisisSize := 0
	if d.crisisBuffer.Len() > 0 {
		crisisSize = int(float64(d.batchSize) * dqnSchedCrisisRatio)
		mainSize = d.batchSize - crisisSize
	}

	beta := d.betaSchedule.Next()
	mainBatch, indices, isWeights := d.replayBuffer.Sample(mainSize, beta)
	if len(mainBatch) < mainSize {
		return
	}

	crisisBatch := d.crisisBuffer.Sample(crisisSize)
	batch := append(mainBatch, crisisBatch...)

	for range crisisBatch {
		isWeights = append(isWeights, 1.0)
	}
	bs := len(batch)

	statesBuf := make([]float64, bs*d.stateSize)
	nextStatesBuf := make([]float64, bs*d.stateSize)
	actions := make([]float64, bs)
	rewards := make([]float64, bs)
	dones := make([]float64, bs)

	for i, t := range batch {
		copy(statesBuf[i*d.stateSize:], padOrTruncSched(t.State, d.stateSize))
		copy(nextStatesBuf[i*d.stateSize:], padOrTruncSched(t.NextState, d.stateSize))
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

	maxNextQ := tensorToFloat64Sched(nextQT)
	currentQ := tensorToFloat64Sched(currentQT)
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
	d.trainExec.MustExec(statesT, targetsT, actionsT, isWeightsT)

	d.targetNet.Step()
}

func padOrTruncSched(s []float64, n int) []float64 {
	if len(s) == n {
		return s
	}
	out := make([]float64, n)
	copy(out, s)
	return out
}

func argmaxSched(s []float64) int {
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

func tensorToFloat64Sched(t *tensors.Tensor) []float64 {
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
