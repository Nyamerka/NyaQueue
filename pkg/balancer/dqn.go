package balancer

import (
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
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

	w1 *mat.Dense // [hiddenSize x stateSize]
	b1 []float64  // [hiddenSize]
	w2 *mat.Dense // [numActions x hiddenSize]
	b2 []float64  // [numActions]

	replayBuffer *nn.ReplayBuffer
	lastState    []float64
	lastAction   int

	fallbackRR     *RoundRobin
	fallbackRatio  float64
	loadThreshold  float64
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
func WithDQNLoadThreshold(t float64) DQNOption { return func(d *DQNBalancer) { d.loadThreshold = t } }
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
		loadThreshold:  DefaultDQNLoadThreshold,
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

func (d *DQNBalancer) SelectPartition(topic string, key []byte, numPartitions int) int {
	d.mu.Lock()
	defer d.mu.Unlock()

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

	qValues, _ := d.forward(state)
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

	shouldFallback := false

	if d.baseThroughput > 0 && m.Throughput < d.fallbackRatio*d.baseThroughput {
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

	d.fallbackActive = shouldFallback

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

func (d *DQNBalancer) SetBaseThroughput(t float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.baseThroughput = t
}

func (d *DQNBalancer) SetPredictedLoads(predicted []float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.predictedLoads = predicted
}

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

	hidden := hv.RawVector().Data
	floats.Add(hidden, d.b1)
	for i, v := range hidden {
		if v < 0 {
			hidden[i] = 0
		}
	}

	qv := mat.NewVecDense(d.numPartitions, nil)
	qv.MulVec(d.w2, hv)

	q := qv.RawVector().Data
	floats.Add(q, d.b2)
	return q, hidden
}

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
