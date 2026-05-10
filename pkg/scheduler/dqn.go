package scheduler

import (
	"math/rand/v2"
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
	"github.com/samber/oops"

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
	batchSize  int
	minReplay  int
	trainEvery int

	backend   backends.Backend
	ctx       *context.Context
	fwdExec   *context.Exec
	nextQExec *context.Exec
	trainExec *context.Exec

	replayBuffer *nn.ReplayBuffer

	expCh  chan nn.Transition
	stopCh chan struct{}
	done   chan struct{}

	stateMu   sync.Mutex
	lastState []float64
	lastPI    atomic.Pointer[broker.PriorityIndex]
	threshold int

	fallbackFIFO bool

	cachedThreshold atomic.Int32

	// prevSnap / currSnap: policyLoop writes, OnMetrics reads.
	prevSnap atomic.Pointer[schedPolicySnap]
	currSnap atomic.Pointer[schedPolicySnap]

	latencyMu     sync.Mutex
	latencySums   [broker.MaxPriority]float64
	latencyCounts [broker.MaxPriority]int

	rng   *rand.Rand
	rngMu sync.Mutex
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
func WithDQNSchedReplayBufSize(n int) DQNSchedOption {
	return func(d *DQNScheduler) { d.replayBuffer = nn.NewReplayBuffer(n) }
}
func WithDQNSchedBatchSize(n int) DQNSchedOption { return func(d *DQNScheduler) { d.batchSize = n } }

func NewDQNScheduler(opts ...DQNSchedOption) *DQNScheduler {
	d := &DQNScheduler{
		weightsMu:    xsync.NewRBMutex(),
		hiddenSize:   DefaultDQNSchedHiddenSize,
		stateSize:    DefaultDQNSchedStateSize,
		numActions:   DefaultDQNSchedActions,
		epsilon:      DefaultDQNSchedEpsilon,
		gamma:        DefaultDQNSchedGamma,
		lr:           DefaultDQNSchedLearningRate,
		batchSize:    DefaultDQNSchedBatchSize,
		minReplay:    DefaultDQNSchedMinReplay,
		trainEvery:   DefaultDQNSchedTrainEvery,
		replayBuffer: nn.NewReplayBuffer(DefaultDQNSchedReplayBufSize),
		threshold:    DefaultDQNSchedThreshold,
		expCh:        make(chan nn.Transition, DefaultDQNSchedExpChSize),
		stopCh:       make(chan struct{}),
		done:         make(chan struct{}),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	d.cachedThreshold.Store(int32(d.threshold))
	d.initGoMLX()

	go d.trainLoop()
	go d.policyLoop()

	return d
}

func (d *DQNScheduler) initGoMLX() {
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
			q := layers.Dense(ctx.In("output"), h, true, d.numActions)
			return ReduceMax(q, -1)
		})

	d.trainExec = context.MustNewExec(d.backend, d.ctx.Reuse(),
		func(ctx *context.Context, states, targetQ, actionIdx *Node) *Node {
			g := states.Graph()
			h := layers.Dense(ctx.In("hidden"), states, true, d.hiddenSize)
			h = activations.Relu(h)
			qAll := layers.Dense(ctx.In("output"), h, true, d.numActions)
			oneHot := OneHot(ConvertDType(actionIdx, dtypes.Int32), d.numActions, qAll.DType())
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

func (d *DQNScheduler) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	h := layers.Dense(ctx.In("hidden"), x, true, d.hiddenSize)
	h = activations.Relu(h)
	q := layers.Dense(ctx.In("output"), h, true, d.numActions)
	q = Reshape(q, d.numActions)
	return q
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

	d.stateMu.Lock()
	fallback := d.fallbackFIFO
	d.stateMu.Unlock()

	if fallback {
		return d.fifoFallback(partition, consumerOffset)
	}

	d.rngMu.Lock()
	explore := d.rng.Float64() < d.epsilon
	var randThreshold int
	if explore {
		randThreshold = d.rng.IntN(d.numActions)
	}
	d.rngMu.Unlock()

	var threshold int
	if explore {
		threshold = randThreshold
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

	latency := time.Since(entry.ArrivedAt).Seconds()
	level := int(msg.Header.Priority)
	if level >= broker.MaxPriority {
		level = broker.MaxPriority - 1
	}
	d.latencyMu.Lock()
	d.latencySums[level] += latency
	d.latencyCounts[level]++
	d.latencyMu.Unlock()

	return msg, uint64(entry.WalOffset), nil
}

func (d *DQNScheduler) Enqueue(_ *broker.Message, _ int64) {}

func (d *DQNScheduler) OnMetrics(m broker.Metrics) {
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
	d.latencyMu.Lock()
	var sums [broker.MaxPriority]float64
	var counts [broker.MaxPriority]int
	copy(sums[:], d.latencySums[:])
	copy(counts[:], d.latencyCounts[:])
	// Reset windows for next interval.
	d.latencySums = [broker.MaxPriority]float64{}
	d.latencyCounts = [broker.MaxPriority]int{}
	d.latencyMu.Unlock()

	var reward float64
	var totalWeight float64
	for level := 0; level < broker.MaxPriority; level++ {
		if counts[level] == 0 {
			continue
		}
		meanLatency := sums[level] / float64(counts[level])
		// Weight: higher priority (higher index) = higher weight.
		weight := float64(level + 1)
		reward -= meanLatency * weight
		totalWeight += weight
	}
	if totalWeight > 0 {
		reward /= totalWeight
	}
	return reward
}

func (d *DQNScheduler) SetFallbackFIFO(on bool) {
	d.stateMu.Lock()
	defer d.stateMu.Unlock()
	d.fallbackFIFO = on
}

func (d *DQNScheduler) Stop() {
	close(d.stopCh)
	<-d.done
}

func (d *DQNScheduler) policyLoop() {
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
			d.stateMu.Unlock()
		case <-d.stopCh:
			return
		}
	}
}

func (d *DQNScheduler) trainLoop() {
	defer close(d.done)
	steps := 0

	for {
		select {
		case t := <-d.expCh:
			d.replayBuffer.Push(t)
			steps++
			if steps%d.trainEvery == 0 {
				d.trainStep()
			}
		case <-d.stopCh:
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

func (d *DQNScheduler) fifoFallback(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	hwm := partition.HighWaterMark()
	if consumerOffset > hwm {
		return nil, consumerOffset, broker.ErrNoMessages
	}
	msg, err := partition.Read(consumerOffset)
	if err != nil {
		return nil, consumerOffset, err
	}
	return msg, consumerOffset + 1, nil
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

	d.latencyMu.Lock()
	for i := 0; i < levels; i++ {
		if waitIdx := levels + i; waitIdx < d.stateSize {
			if d.latencyCounts[i] > 0 {
				state[waitIdx] = d.latencySums[i] / float64(d.latencyCounts[i])
			}
		}
	}
	d.latencyMu.Unlock()

	if depthIdx := levels * 2; depthIdx < d.stateSize {
		state[depthIdx] = float64(totalDepth)
	}
	if lagIdx := levels*2 + 1; lagIdx < d.stateSize {
		state[lagIdx] = 0
	}

	return state
}

func (d *DQNScheduler) trainStep() {
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
	maxNextQT := d.nextQExec.MustExec1(nextStatesT)
	d.weightsMu.RUnlock(rt)

	maxNextQ := tensorToFloat64Sched(maxNextQT)
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
