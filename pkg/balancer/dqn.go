package balancer

import (
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// DQNBalancer uses a Deep Q-Network to select partitions based on current
// and predicted loads.
//
// Architecture: inference and training are separated to eliminate hot-path
// contention. SelectPartition() takes a RLock on model weights for a single
// forward pass. A background training goroutine drains experience from a
// non-blocking channel, pushes it into the replay buffer, and periodically
// runs backprop under a full Lock.
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

	w1 *mat.Dense // [hiddenSize x stateSize]
	b1 []float64  // [hiddenSize]
	w2 *mat.Dense // [numActions x hiddenSize]
	b2 []float64  // [numActions]

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
	lastState      []float64
	lastAction     int
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

	stateSize := numPartitions*2 + 2
	d.initWeights(stateSize, numPartitions)

	go d.trainLoop()

	return d
}

func (d *DQNBalancer) initWeights(stateSize, numActions int) {
	w1Data := make([]float64, d.hiddenSize*stateSize)
	for i := range w1Data {
		w1Data[i] = rand.NormFloat64() * d.weightInit
	}
	d.w1 = mat.NewDense(d.hiddenSize, stateSize, w1Data)
	d.b1 = make([]float64, d.hiddenSize)

	w2Data := make([]float64, numActions*d.hiddenSize)
	for i := range w2Data {
		w2Data[i] = rand.NormFloat64() * d.weightInit
	}
	d.w2 = mat.NewDense(numActions, d.hiddenSize, w2Data)
	d.b2 = make([]float64, numActions)
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
	qValues, _ := d.forward(state)
	d.weightsMu.RUnlock()

	action := floats.MaxIdx(qValues[:numPartitions])

	d.stateMu.Lock()
	d.lastState = state
	d.lastAction = action
	d.stateMu.Unlock()

	return action
}

// OnMetrics updates loads, evaluates watchdog, and pushes experience to the
// training goroutine via a non-blocking channel send.
func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	d.stateMu.Lock()

	if len(m.PartitionLoads) > 0 {
		d.loads = m.PartitionLoads
	}

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

	if d.lastState != nil {
		reward := computeReward(m)
		nextState := d.buildStateLocked()

		t := nn.Transition{
			State:     d.lastState,
			Action:    []float64{float64(d.lastAction)},
			Reward:    reward,
			NextState: nextState,
			Done:      false,
		}

		d.stateMu.Unlock()

		select {
		case d.expCh <- t:
		default:
		}
		return
	}

	d.stateMu.Unlock()
}

func (d *DQNBalancer) SetBaseThroughput(t float64) {
	d.baseThroughput.Store(int64(t))
}

func (d *DQNBalancer) SetPredictedLoads(predicted []float64) {
	d.stateMu.Lock()
	defer d.stateMu.Unlock()
	d.predictedLoads = predicted
}

func (d *DQNBalancer) IsFallbackActive() bool {
	return d.fallbackActive.Load()
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
	state = append(state, 0, 0)
	return state
}

func (d *DQNBalancer) forward(state []float64) ([]float64, []float64) {
	_, stateSize := d.w1.Dims()
	s := state
	if len(s) < stateSize {
		padded := make([]float64, stateSize)
		copy(padded, s)
		s = padded
	} else if len(s) > stateSize {
		s = s[:stateSize]
	}

	sv := mat.NewVecDense(stateSize, s)
	hv := mat.NewVecDense(d.hiddenSize, nil)
	hv.MulVec(d.w1, sv)

	hidden := make([]float64, d.hiddenSize)
	copy(hidden, hv.RawVector().Data)
	floats.Add(hidden, d.b1)
	for i, v := range hidden {
		if v < 0 {
			hidden[i] = 0
		}
	}

	hv2 := mat.NewVecDense(d.hiddenSize, hidden)
	qv := mat.NewVecDense(d.numPartitions, nil)
	qv.MulVec(d.w2, hv2)

	q := make([]float64, d.numPartitions)
	copy(q, qv.RawVector().Data)
	floats.Add(q, d.b2)
	return q, hidden
}

func computeReward(m broker.Metrics) float64 {
	if len(m.PartitionLoads) == 0 {
		return 0
	}
	return -stat.StdDev(m.PartitionLoads, nil)
}

func (d *DQNBalancer) trainStep() {
	if d.replayBuffer.Len() < d.minReplay {
		return
	}

	batch := d.replayBuffer.Sample(d.batchSize)

	d.weightsMu.Lock()
	defer d.weightsMu.Unlock()

	for _, t := range batch {
		qValues, hidden := d.forward(t.State)
		nextQ, _ := d.forward(t.NextState)
		maxNextQ := floats.Max(nextQ)

		action := 0
		if len(t.Action) > 0 {
			action = int(t.Action[0])
		}
		if action >= len(qValues) {
			continue
		}

		target := t.Reward + d.gamma*maxNextQ
		if t.Done {
			target = t.Reward
		}

		tdError := target - qValues[action]
		d.updateWeights(t.State, action, tdError, hidden)
	}
}

func (d *DQNBalancer) updateWeights(state []float64, action int, tdError float64, hidden []float64) {
	if action >= d.numPartitions {
		return
	}

	w2Row := d.w2.RawRowView(action)
	w2Snap := make([]float64, len(w2Row))
	copy(w2Snap, w2Row)

	floats.AddScaled(w2Row, d.lr*tdError, hidden)
	d.b2[action] += d.lr * tdError

	_, stateSize := d.w1.Dims()
	sLen := len(state)
	if sLen > stateSize {
		sLen = stateSize
	}
	for i := 0; i < d.hiddenSize; i++ {
		if hidden[i] <= 0 {
			continue
		}
		dH := tdError * w2Snap[i]
		w1Row := d.w1.RawRowView(i)
		floats.AddScaled(w1Row[:sLen], d.lr*dH, state[:sLen])
		d.b1[i] += d.lr * dH
	}
}
