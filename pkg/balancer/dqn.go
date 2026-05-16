package balancer

import (
	stdctx "context"
	"log"
	"math"
	"math/rand"
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
	"github.com/puzpuzpuz/xsync/v3"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/stat"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

// DQNBalancer selects partitions via a DQN.
// State: [partition_loads, predicted_loads, msg_rate, avg_msg_size]; Action: partition index.
type policySnap struct {
	state  []float64
	action int
	gen    int64
}

type DQNBalancer struct {
	weightsMu *xsync.RBMutex

	numPartitions int
	hiddenSize    int
	epsilon       float64
	gamma         float64
	lr            float64
	batchSize     int
	minReplay     int
	trainEvery    int

	backend   backends.Backend
	ctx       *context.Context
	fwdExec   *context.Exec
	nextQExec *context.Exec
	trainExec *context.Exec
	stateSize int

	replayBuffer *nn.ReplayBuffer

	expCh  chan nn.Transition
	eg     *errgroup.Group
	cancel stdctx.CancelFunc

	fallbackRR     *RoundRobin
	fallbackRatio  float64
	loadThreshold  float64
	baseThroughput atomic.Int64
	fallbackActive atomic.Bool

	stateMu        *xsync.RBMutex
	loads          []float64
	predictedLoads []float64
	msgRate        float64
	avgMsgSize     float64

	droppedExperience atomic.Int64

	cachedQValues atomic.Pointer[[]float64]

	prevSnap atomic.Pointer[policySnap]
	currSnap atomic.Pointer[policySnap]
	snapGen  atomic.Int64

	lastProcessedGen atomic.Int64

	rngMu sync.Mutex
	rng   *rand.Rand

	consecutiveFallbackTicks int
	consecutiveRecoveryTicks int

	policyStateBuf []float64
}

// DQNOption configures a DQNBalancer.
type DQNOption func(*DQNBalancer)

func WithDQNEpsilon(e float64) DQNOption       { return func(d *DQNBalancer) { d.epsilon = e } }
func WithDQNGamma(g float64) DQNOption         { return func(d *DQNBalancer) { d.gamma = g } }
func WithDQNLearningRate(lr float64) DQNOption { return func(d *DQNBalancer) { d.lr = lr } }
func WithDQNHiddenSize(n int) DQNOption        { return func(d *DQNBalancer) { d.hiddenSize = n } }
func WithDQNReplayBufSize(n int) DQNOption {
	return func(d *DQNBalancer) { d.replayBuffer = nn.NewReplayBuffer(n) }
}
func WithDQNBatchSize(n int) DQNOption         { return func(d *DQNBalancer) { d.batchSize = n } }
func WithDQNMinReplay(n int) DQNOption         { return func(d *DQNBalancer) { d.minReplay = n } }
func WithDQNFallbackRatio(r float64) DQNOption { return func(d *DQNBalancer) { d.fallbackRatio = r } }
func WithDQNLoadThreshold(t float64) DQNOption { return func(d *DQNBalancer) { d.loadThreshold = t } }
func WithDQNTrainEvery(n int) DQNOption        { return func(d *DQNBalancer) { d.trainEvery = n } }

func NewDQNBalancer(numPartitions int, opts ...DQNOption) *DQNBalancer {
	d := &DQNBalancer{
		weightsMu:     xsync.NewRBMutex(),
		stateMu:       xsync.NewRBMutex(),
		numPartitions: numPartitions,
		hiddenSize:    DefaultDQNHiddenSize,
		epsilon:       DefaultDQNEpsilon,
		gamma:         DefaultDQNGamma,
		lr:            DefaultDQNLearningRate,
		batchSize:     DefaultDQNBatchSize,
		minReplay:     DefaultDQNMinReplay,
		trainEvery:    DefaultDQNTrainEvery,
		replayBuffer:  nn.NewReplayBuffer(DefaultDQNReplayBufSize),
		fallbackRR:    NewRoundRobin(),
		fallbackRatio: DefaultDQNFallbackRatio,
		loadThreshold: DefaultDQNLoadThreshold,
		expCh:         make(chan nn.Transition, DefaultDQNExpChannelSize),

		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	d.stateSize = numPartitions*2 + 2
	d.policyStateBuf = make([]float64, d.stateSize)
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

	d.fwdExec = context.MustNewExec(d.backend, d.ctx, func(ctx *context.Context, state *Node) *Node {
		return d.qNetworkGraph(ctx, state)
	})
	dummyState := make([]float64, d.stateSize)
	d.fwdExec.MustExec(dummyState)

	d.nextQExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states *Node) *Node {
			h := layers.Dense(ctx.In("hidden"), states, true, d.hiddenSize)
			h = activations.Relu(h)
			q := layers.Dense(ctx.In("output"), h, true, d.numPartitions)
			return ReduceMax(q, -1)
		})

	d.trainExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, targetQ, actionIdx *Node) *Node {
			g := states.Graph()
			h := layers.Dense(ctx.In("hidden"), states, true, d.hiddenSize)
			h = activations.Relu(h)
			qAll := layers.Dense(ctx.In("output"), h, true, d.numPartitions)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numPartitions, qAll.DType())
			qSelected := ReduceSum(Mul(qAll, oneHot), -1)
			diff := Sub(qSelected, targetQ)
			loss := ReduceAllMean(Mul(diff, diff))
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

func (d *DQNBalancer) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0) // [1, stateSize]
	h := layers.Dense(ctx.In("hidden"), x, true, d.hiddenSize)
	h = activations.Relu(h)
	q := layers.Dense(ctx.In("output"), h, true, d.numPartitions)
	q = Reshape(q, d.numPartitions) // flatten to [numPartitions]
	return q
}

func (d *DQNBalancer) forward(state []float64) []float64 {
	result := d.fwdExec.MustExec1(state)
	return tensorToFloat64Slice(result)
}

func (d *DQNBalancer) SelectPartition(topic string, key []byte, numPartitions int) int {
	if d.fallbackActive.Load() {
		return d.fallbackRR.SelectPartition(topic, key, numPartitions)
	}

	d.rngMu.Lock()
	explore := d.rng.Float64() < d.epsilon
	var randomPart int
	if explore {
		randomPart = d.rng.Intn(numPartitions)
	}
	d.rngMu.Unlock()

	if explore {
		return randomPart
	}

	if cached := d.cachedQValues.Load(); cached != nil && len(*cached) >= numPartitions {
		return argmax((*cached)[:numPartitions])
	}

	return d.fallbackRR.SelectPartition(topic, key, numPartitions)
}

func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	reward := computeReward(m)

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

	shouldFallback := false

	baseTP := float64(d.baseThroughput.Load())
	if baseTP > 0 && m.Throughput < d.fallbackRatio*baseTP {
		shouldFallback = true
	}

	if d.loadThreshold > 0 && len(m.PartitionLoads) > 0 {
		meanLoad := 0.0
		for _, l := range m.PartitionLoads {
			meanLoad += l
		}
		meanLoad /= float64(len(m.PartitionLoads))
		if meanLoad > d.loadThreshold {
			shouldFallback = true
		}
	}

	if shouldFallback {
		d.consecutiveRecoveryTicks = 0
		d.consecutiveFallbackTicks++
		if d.consecutiveFallbackTicks >= fallbackEnterTicks {
			d.fallbackActive.Store(true)
		}
	} else {
		d.consecutiveFallbackTicks = 0
		d.consecutiveRecoveryTicks++
		if d.consecutiveRecoveryTicks >= fallbackExitTicks {
			d.fallbackActive.Store(false)
		}
	}
	d.stateMu.Unlock()

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

func (d *DQNBalancer) SetBaseThroughput(t float64) {
	d.baseThroughput.Store(int64(t))
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

func (d *DQNBalancer) IsFallbackActive() bool {
	return d.fallbackActive.Load()
}

func (d *DQNBalancer) DroppedExperience() int64 {
	return d.droppedExperience.Load()
}

func (d *DQNBalancer) Stop() {
	d.cancel()
	_ = d.eg.Wait()
}

func (d *DQNBalancer) trainLoop(ctx stdctx.Context) {
	steps := 0

	for {
		select {
		case t := <-d.expCh:
			d.replayBuffer.Push(t)
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

func (d *DQNBalancer) policyLoop(ctx stdctx.Context) {
	ticker := newTicker100ms()
	defer ticker.Stop()

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
	idx := d.numPartitions * 2
	if idx < len(dst) {
		dst[idx] = d.msgRate / 100000
	}
	if idx+1 < len(dst) {
		dst[idx+1] = d.avgMsgSize / 1024
	}
}

func computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}
	imbalancePenalty := -stat.StdDev(m.PartitionLoads, nil)

	var throughputReward float64
	if m.Throughput > 0 {
		throughputReward = math.Log1p(m.Throughput/10_000.0) / dqnThroughputLogNorm
	}

	meanLoad := 0.0
	for _, l := range m.PartitionLoads {
		meanLoad += l
	}
	meanLoad /= float64(len(m.PartitionLoads))

	if meanLoad > 0.7 {
		return 1.5*imbalancePenalty + 0.1*throughputReward
	}
	return 0.5*imbalancePenalty + 0.5*throughputReward
}

func (d *DQNBalancer) trainStep() {
	if d.replayBuffer.Len() < d.minReplay {
		return
	}

	batch := d.replayBuffer.Sample(d.batchSize)
	bs := len(batch)
	if bs < d.batchSize {
		return
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
	maxNextQT := d.nextQExec.MustExec1(nextStatesT)
	d.weightsMu.RUnlock(rt)

	maxNextQ := tensorToFloat64Slice(maxNextQT)
	targets := make([]float64, bs)
	for i := range batch {
		targets[i] = rewards[i] + d.gamma*maxNextQ[i]*(1-dones[i])
	}

	d.weightsMu.Lock()
	defer d.weightsMu.Unlock()

	actionsT := tensors.FromFlatDataAndDimensions(actions, bs)
	targetsT := tensors.FromFlatDataAndDimensions(targets, bs)
	d.trainExec.MustExec(statesT, targetsT, actionsT)
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

func newTicker100ms() *time.Ticker {
	return time.NewTicker(100 * time.Millisecond)
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
