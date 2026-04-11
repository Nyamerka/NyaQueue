package balancer

import (
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

// DQNBalancer uses a Deep Q-Network to select partitions based on current
// and predicted loads. Uses GoMLX FC layers for the Q-network.
//
// State vector: [partition_loads..., predicted_loads..., msg_rate, avg_msg_size]
// Action: partition index (discrete)
// Reward: -load_imbalance_stddev (lower imbalance = higher reward)
type DQNBalancer struct {
	mu sync.Mutex

	numPartitions int
	hiddenSize    int
	epsilon       float64
	gamma         float64
	lr            float64
	batchSize     int
	minReplay     int
	weightInit    float64

	w1 [][]float64 // [hiddenSize][stateSize]
	b1 []float64   // [hiddenSize]
	w2 [][]float64 // [numActions][hiddenSize]
	b2 []float64   // [numActions]

	replayBuffer *nn.ReplayBuffer
	lastState    []float64
	lastAction   int

	fallbackRR     *RoundRobin
	fallbackRatio  float64
	baseThroughput float64
	fallbackActive bool

	loads          []float64
	predictedLoads []float64
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
func WithDQNWeightInit(s float64) DQNOption    { return func(d *DQNBalancer) { d.weightInit = s } }

func NewDQNBalancer(numPartitions int, opts ...DQNOption) *DQNBalancer {
	d := &DQNBalancer{
		numPartitions:  numPartitions,
		hiddenSize:     DefaultDQNHiddenSize,
		epsilon:        DefaultDQNEpsilon,
		gamma:          DefaultDQNGamma,
		lr:             DefaultDQNLearningRate,
		batchSize:      DefaultDQNBatchSize,
		minReplay:      DefaultDQNMinReplay,
		weightInit:     DefaultDQNWeightInit,
		replayBuffer:   nn.NewReplayBuffer(DefaultDQNReplayBufSize),
		fallbackRR:     NewRoundRobin(),
		fallbackRatio:  DefaultDQNFallbackRatio,
		loads:          make([]float64, numPartitions),
		predictedLoads: make([]float64, numPartitions),
	}

	for _, opt := range opts {
		opt(d)
	}

	stateSize := numPartitions*2 + 2
	d.initWeights(stateSize, numPartitions)
	return d
}

func (d *DQNBalancer) initWeights(stateSize, numActions int) {
	d.w1 = make([][]float64, d.hiddenSize)
	d.b1 = make([]float64, d.hiddenSize)
	for i := range d.w1 {
		d.w1[i] = make([]float64, stateSize)
		for j := range d.w1[i] {
			d.w1[i][j] = rand.NormFloat64() * d.weightInit
		}
	}

	d.w2 = make([][]float64, numActions)
	d.b2 = make([]float64, numActions)
	for i := range d.w2 {
		d.w2[i] = make([]float64, d.hiddenSize)
		for j := range d.w2[i] {
			d.w2[i][j] = rand.NormFloat64() * d.weightInit
		}
	}
}

func (d *DQNBalancer) SelectPartition(topic string, key []byte, numPartitions int) int {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Watchdog: if throughput fell below threshold, fallback to RR.
	if d.fallbackActive {
		return d.fallbackRR.SelectPartition(topic, key, numPartitions)
	}

	state := d.buildState()

	if rand.Float64() < d.epsilon {
		action := rand.Intn(numPartitions)
		d.lastState = state
		d.lastAction = action
		return action
	}

	qValues := d.forward(state)
	action := floats.MaxIdx(qValues[:numPartitions])

	d.lastState = state
	d.lastAction = action
	return action
}

func (d *DQNBalancer) OnMetrics(m broker.Metrics) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if len(m.PartitionLoads) > 0 {
		d.loads = m.PartitionLoads
	}

	// Watchdog: check if throughput fell below the fallback threshold.
	if d.baseThroughput > 0 {
		if m.Throughput < d.fallbackRatio*d.baseThroughput {
			d.fallbackActive = true
		} else {
			d.fallbackActive = false
		}
	}

	if d.lastState != nil {
		reward := d.computeReward(m)
		nextState := d.buildState()

		d.replayBuffer.Push(nn.Transition{
			State:     d.lastState,
			Action:    []float64{float64(d.lastAction)},
			Reward:    reward,
			NextState: nextState,
			Done:      false,
		})

		d.trainStep()
	}
}

// SetBaseThroughput stores the baseline (e.g., from Phase 1 with RR) for the watchdog.
func (d *DQNBalancer) SetBaseThroughput(t float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.baseThroughput = t
}

// SetPredictedLoads is called by LoadPredictor to feed predictions into the DQN state.
func (d *DQNBalancer) SetPredictedLoads(predicted []float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.predictedLoads = predicted
}

// IsFallbackActive reports whether the DQN has fallen back to RR.
func (d *DQNBalancer) IsFallbackActive() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.fallbackActive
}

func (d *DQNBalancer) buildState() []float64 {
	state := make([]float64, 0, d.numPartitions*2+2)
	state = append(state, d.loads...)
	for len(state) < d.numPartitions {
		state = append(state, 0)
	}
	state = append(state, d.predictedLoads...)
	for len(state) < d.numPartitions*2 {
		state = append(state, 0)
	}
	state = append(state, 0, 0) // rate, avg_msg_size (fed via metrics)
	return state
}

// forward computes Q-values: hidden = ReLU(W1·state + b1), Q = W2·hidden + b2.
// Uses gonum/floats.Dot for vectorised inner products.
func (d *DQNBalancer) forward(state []float64) []float64 {
	hidden := make([]float64, d.hiddenSize)
	sLen := len(state)
	for i := 0; i < d.hiddenSize; i++ {
		wRow := d.w1[i]
		n := len(wRow)
		if sLen < n {
			n = sLen
		}
		hidden[i] = floats.Dot(wRow[:n], state[:n]) + d.b1[i]
		if hidden[i] < 0 {
			hidden[i] = 0 // ReLU
		}
	}

	qValues := make([]float64, d.numPartitions)
	for i := 0; i < d.numPartitions && i < len(d.w2); i++ {
		qValues[i] = floats.Dot(d.w2[i], hidden) + d.b2[i]
	}
	return qValues
}

// computeReward returns -stddev(partition_loads). Lower imbalance = higher reward.
func (d *DQNBalancer) computeReward(m broker.Metrics) float64 {
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
	for _, t := range batch {
		qValues := d.forward(t.State)
		nextQ := d.forward(t.NextState)
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
		d.updateWeights(t.State, action, tdError)
	}
}

func (d *DQNBalancer) updateWeights(state []float64, action int, tdError float64) {
	hidden := make([]float64, d.hiddenSize)
	sLen := len(state)
	for i := 0; i < d.hiddenSize; i++ {
		wRow := d.w1[i]
		n := len(wRow)
		if sLen < n {
			n = sLen
		}
		hidden[i] = floats.Dot(wRow[:n], state[:n]) + d.b1[i]
		if hidden[i] < 0 {
			hidden[i] = 0
		}
	}

	if action < len(d.w2) {
		for j := 0; j < d.hiddenSize; j++ {
			d.w2[action][j] += d.lr * tdError * hidden[j]
		}
		d.b2[action] += d.lr * tdError
	}

	for i := 0; i < d.hiddenSize; i++ {
		if hidden[i] <= 0 {
			continue
		}
		if action < len(d.w2) {
			dHidden := tdError * d.w2[action][i]
			for j := 0; j < len(state) && j < len(d.w1[i]); j++ {
				d.w1[i][j] += d.lr * dHidden * state[j]
			}
			d.b1[i] += d.lr * dHidden
		}
	}
}
