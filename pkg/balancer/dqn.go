package balancer

import (
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"gonum.org/v1/gonum/stat"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

// DQNBalancer uses a Deep Q-Network (GoMLX-backed) to select partitions based
// on current and predicted loads.
//
// Architecture: inference and training are separated to eliminate hot-path
// contention. SelectPartition() takes a RLock on model weights for a single
// forward pass. A background training goroutine drains experience from a
// non-blocking channel and periodically runs backprop via GoMLX autograd.
//
// State vector: [partition_loads..., predicted_loads..., msg_rate, avg_msg_size]
// Action: partition index (discrete)
// Reward: -load_imbalance_stddev (lower imbalance = higher reward)
type DQNBalancer struct {
	weightsMu sync.RWMutex

	numPartitions int
	hiddenSize    int
	epsilon       float64
	gamma         float64
	lr            float64
	batchSize     int
	minReplay     int
	trainEvery    int
	weightInit    float64

	backend   backends.Backend
	ctx       *context.Context
	fwdExec   *context.Exec
	nextQExec *context.Exec
	trainExec *context.Exec
	stateSize int

	replayBuffer *nn.ReplayBuffer

	expCh  chan nn.Transition
	stopCh chan struct{}
	done   chan struct{}

	fallbackRR     *RoundRobin
	fallbackRatio  float64
	loadThreshold  float64
	baseThroughput atomic.Int64
	fallbackActive atomic.Bool

	stateMu        sync.Mutex
	loads          []float64
	predictedLoads []float64
	msgRate        float64
	avgMsgSize     float64
	lastState      []float64
	lastAction     int

	droppedExperience atomic.Int64
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
func WithDQNWeightInit(s float64) DQNOption    { return func(d *DQNBalancer) { d.weightInit = s } }
func WithDQNTrainEvery(n int) DQNOption        { return func(d *DQNBalancer) { d.trainEvery = n } }

func NewDQNBalancer(numPartitions int, opts ...DQNOption) *DQNBalancer {
	d := &DQNBalancer{
		numPartitions: numPartitions,
		hiddenSize:    DefaultDQNHiddenSize,
		epsilon:       DefaultDQNEpsilon,
		gamma:         DefaultDQNGamma,
		lr:            DefaultDQNLearningRate,
		batchSize:     DefaultDQNBatchSize,
		minReplay:     DefaultDQNMinReplay,
		trainEvery:    DefaultDQNTrainEvery,
		weightInit:    DefaultDQNWeightInit,
		replayBuffer:  nn.NewReplayBuffer(DefaultDQNReplayBufSize),
		fallbackRR:    NewRoundRobin(),
		fallbackRatio: DefaultDQNFallbackRatio,
		loadThreshold: DefaultDQNLoadThreshold,
		expCh:         make(chan nn.Transition, DefaultDQNExpChannelSize),
		stopCh:        make(chan struct{}),
		done:          make(chan struct{}),

		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.stateSize = numPartitions*2 + 2
	d.initGoMLX()
	go d.trainLoop()

	return d
}

func (d *DQNBalancer) initGoMLX() {
	d.backend = backends.MustNew()
	d.ctx = context.New()

	// fwdExec initializes variables; built first so Reuse() finds them below.
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

// qNetworkGraph builds the Q-network: Dense(hiddenSize) → ReLU → Dense(numPartitions).
func (d *DQNBalancer) qNetworkGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0) // [1, stateSize]
	h := layers.Dense(ctx.In("hidden"), x, true, d.hiddenSize)
	h = activations.Relu(h)
	q := layers.Dense(ctx.In("output"), h, true, d.numPartitions)
	q = Reshape(q, d.numPartitions) // flatten to [numPartitions]
	return q
}

// forward performs a single forward pass returning Q-values for the state.
// Caller must hold at least weightsMu.RLock.
func (d *DQNBalancer) forward(state []float64) []float64 {
	result := d.fwdExec.MustExec1(state)
	return tensorToFloat64Slice(result)
}

// SelectPartition picks a partition via DQN inference. Only takes a RLock on
// model weights, so it never blocks on training.
func (d *DQNBalancer) SelectPartition(topic string, key []byte, numPartitions int) int {
	if d.fallbackActive.Load() {
		return d.fallbackRR.SelectPartition(topic, key, numPartitions)
	}

	d.stateMu.Lock()
	state := d.buildStateLocked()
	d.stateMu.Unlock()

	if rand.Float64() < d.epsilon {
		action := rand.Intn(numPartitions)
		d.stateMu.Lock()
		d.lastState = state
		d.lastAction = action
		d.stateMu.Unlock()
		return action
	}

	d.weightsMu.RLock()
	qValues := d.forward(state)
	d.weightsMu.RUnlock()

	action := argmax(qValues[:numPartitions])

	d.stateMu.Lock()
	d.lastState = state
	d.lastAction = action
	d.stateMu.Unlock()

	return action
}

// OnMetrics updates loads, evaluates watchdog, and pushes experience to the
// training goroutine via a non-blocking channel send.
// computeReward runs outside the lock to avoid blocking SelectPartition.
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

	d.fallbackActive.Store(shouldFallback)

	var t *nn.Transition
	if d.lastState != nil {
		nextState := d.buildStateLocked()
		t = &nn.Transition{
			State:     append([]float64(nil), d.lastState...),
			Action:    []float64{float64(d.lastAction)},
			Reward:    reward,
			NextState: nextState,
			Done:      false,
		}
	}

	d.stateMu.Unlock()

	if t != nil {
		select {
		case d.expCh <- *t:
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
	defer d.stateMu.Unlock()
	if cap(d.predictedLoads) < len(predicted) {
		d.predictedLoads = make([]float64, len(predicted))
	}
	d.predictedLoads = d.predictedLoads[:len(predicted)]
	copy(d.predictedLoads, predicted)
}

func (d *DQNBalancer) IsFallbackActive() bool {
	return d.fallbackActive.Load()
}

func (d *DQNBalancer) DroppedExperience() int64 {
	return d.droppedExperience.Load()
}

// Stop terminates the background training goroutine.
func (d *DQNBalancer) Stop() {
	close(d.stopCh)
	<-d.done
}

func (d *DQNBalancer) trainLoop() {
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

func (d *DQNBalancer) buildStateLocked() []float64 {
	state := make([]float64, 0, d.numPartitions*2+2)
	state = append(state, d.loads...)
	for len(state) < d.numPartitions {
		state = append(state, 0)
	}
	state = append(state, d.predictedLoads...)
	for len(state) < d.numPartitions*2 {
		state = append(state, 0)
	}
	state = append(state, d.msgRate/100000, d.avgMsgSize/1024)
	return state
}

func computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}
	return -stat.StdDev(m.PartitionLoads, nil)
}

// trainStep performs one DQN training step using GoMLX autograd.
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

	// Compute target Q-values: r + gamma * max(Q(s', a')) * (1-done).
	// Uses pre-built exec; RLock allows concurrent inference during this read.
	d.weightsMu.RLock()
	maxNextQT := d.nextQExec.MustExec1(nextStatesT)
	d.weightsMu.RUnlock()

	maxNextQ := tensorToFloat64Slice(maxNextQT)
	targets := make([]float64, bs)
	for i := range batch {
		targets[i] = rewards[i] + d.gamma*maxNextQ[i]*(1-dones[i])
	}

	// Weight update: exclusive lock so inference sees a consistent snapshot.
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
