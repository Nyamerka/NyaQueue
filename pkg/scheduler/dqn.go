package scheduler

import (
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/samber/oops"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// DQNScheduler uses a Deep Q-Network to adaptively set the priority threshold.
//
// State: [level_distribution(10), avg_wait_per_level(10), queue_depth, consumer_lag]
// Action: threshold (0-9) — priorities >= threshold go by priority order, below by FIFO
// Reward: weighted_latency_reduction - starvation_penalty
type DQNScheduler struct {
	mu sync.Mutex

	hiddenSize int
	stateSize  int
	numActions int
	epsilon    float64
	gamma      float64
	lr         float64
	batchSize  int
	minReplay  int
	weightInit float64

	w1 *mat.Dense
	b1 []float64
	w2 *mat.Dense
	b2 []float64

	replayBuffer *nn.ReplayBuffer
	lastState    []float64
	lastAction   int
	threshold    int

	throttleOnLoad float64
	fallbackFIFO   bool
}

type DQNSchedOption func(*DQNScheduler)

func WithDQNSchedEpsilon(e float64) DQNSchedOption       { return func(d *DQNScheduler) { d.epsilon = e } }
func WithDQNSchedGamma(g float64) DQNSchedOption         { return func(d *DQNScheduler) { d.gamma = g } }
func WithDQNSchedLearningRate(lr float64) DQNSchedOption { return func(d *DQNScheduler) { d.lr = lr } }
func WithDQNSchedHiddenSize(n int) DQNSchedOption        { return func(d *DQNScheduler) { d.hiddenSize = n } }
func WithDQNSchedReplayBufSize(n int) DQNSchedOption {
	return func(d *DQNScheduler) { d.replayBuffer = nn.NewReplayBuffer(n) }
}
func WithDQNSchedBatchSize(n int) DQNSchedOption { return func(d *DQNScheduler) { d.batchSize = n } }
func WithDQNSchedThrottleOnLoad(v float64) DQNSchedOption {
	return func(d *DQNScheduler) { d.throttleOnLoad = v }
}

func NewDQNScheduler(opts ...DQNSchedOption) *DQNScheduler {
	d := &DQNScheduler{
		hiddenSize:   DefaultDQNSchedHiddenSize,
		stateSize:    DefaultDQNSchedStateSize,
		numActions:   DefaultDQNSchedActions,
		epsilon:      DefaultDQNSchedEpsilon,
		gamma:        DefaultDQNSchedGamma,
		lr:           DefaultDQNSchedLearningRate,
		batchSize:    DefaultDQNSchedBatchSize,
		minReplay:    DefaultDQNSchedMinReplay,
		weightInit:   DefaultDQNSchedWeightInit,
		replayBuffer: nn.NewReplayBuffer(DefaultDQNSchedReplayBufSize),
		threshold:    DefaultDQNSchedThreshold,
	}

	for _, opt := range opts {
		opt(d)
	}

	d.initWeights()
	return d
}

func (d *DQNScheduler) initWeights() {
	w1Data := make([]float64, d.hiddenSize*d.stateSize)
	for i := range w1Data {
		w1Data[i] = rand.NormFloat64() * d.weightInit
	}
	d.w1 = mat.NewDense(d.hiddenSize, d.stateSize, w1Data)
	d.b1 = make([]float64, d.hiddenSize)

	w2Data := make([]float64, d.numActions*d.hiddenSize)
	for i := range w2Data {
		w2Data[i] = rand.NormFloat64() * d.weightInit
	}
	d.w2 = mat.NewDense(d.numActions, d.hiddenSize, w2Data)
	d.b2 = make([]float64, d.numActions)
}

func (d *DQNScheduler) Next(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	pi := partition.PriorityIndex()
	if pi == nil {
		return nil, consumerOffset, oops.Errorf("partition %d has no PriorityIndex", partition.ID())
	}

	d.mu.Lock()
	if d.fallbackFIFO {
		d.mu.Unlock()
		return d.fifoFallback(partition, consumerOffset)
	}

	state := d.buildState(pi)
	threshold := d.selectAction(state)
	d.threshold = threshold
	d.lastState = state
	d.lastAction = threshold
	d.mu.Unlock()

	entry, ok := pi.PopWithThreshold(threshold)
	if !ok {
		return nil, consumerOffset, oops.Errorf("no pending messages")
	}

	msg, err := partition.Read(uint64(entry.WalOffset))
	if err != nil {
		return nil, 0, err
	}

	return msg, uint64(entry.WalOffset), nil
}

func (d *DQNScheduler) Enqueue(_ *broker.Message, _ int64) {}

func (d *DQNScheduler) OnMetrics(m broker.Metrics) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.lastState == nil {
		return
	}

	reward := -m.AvgLatency
	nextState := d.lastState

	d.replayBuffer.Push(nn.Transition{
		State:     d.lastState,
		Action:    []float64{float64(d.lastAction)},
		Reward:    reward,
		NextState: nextState,
		Done:      false,
	})

	d.trainStep()
}

func (d *DQNScheduler) SetFallbackFIFO(on bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.fallbackFIFO = on
}

func (d *DQNScheduler) fifoFallback(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	hwm := partition.HighWaterMark()
	if consumerOffset > hwm {
		return nil, consumerOffset, oops.Errorf("no new messages")
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
	totalDepth := 0
	for i, cnt := range dist {
		state[i] = float64(cnt)
		totalDepth += cnt
		state[10+i] = 0
	}
	state[20] = float64(totalDepth)
	state[21] = 0

	return state
}

func (d *DQNScheduler) selectAction(state []float64) int {
	if rand.Float64() < d.epsilon {
		return rand.Intn(d.numActions)
	}
	q, _ := d.forward(state)
	return floats.MaxIdx(q)
}

func (d *DQNScheduler) forward(state []float64) ([]float64, []float64) {
	s := state
	if len(s) < d.stateSize {
		padded := make([]float64, d.stateSize)
		copy(padded, s)
		s = padded
	} else if len(s) > d.stateSize {
		s = s[:d.stateSize]
	}

	sv := mat.NewVecDense(d.stateSize, s)
	hv := mat.NewVecDense(d.hiddenSize, nil)
	hv.MulVec(d.w1, sv)

	hidden := hv.RawVector().Data
	floats.Add(hidden, d.b1)
	for i, v := range hidden {
		if v < 0 {
			hidden[i] = 0
		}
	}

	qv := mat.NewVecDense(d.numActions, nil)
	qv.MulVec(d.w2, hv)

	q := qv.RawVector().Data
	floats.Add(q, d.b2)
	return q, hidden
}

func (d *DQNScheduler) trainStep() {
	if d.replayBuffer.Len() < d.minReplay {
		return
	}

	batch := d.replayBuffer.Sample(d.batchSize)
	for _, t := range batch {
		qValues, hidden := d.forward(t.State)
		nextQ, _ := d.forward(t.NextState)

		action := 0
		if len(t.Action) > 0 {
			action = int(t.Action[0])
		}
		if action >= len(qValues) {
			continue
		}

		maxNext := floats.Max(nextQ)
		target := t.Reward + d.gamma*maxNext
		if t.Done {
			target = t.Reward
		}

		tdError := target - qValues[action]
		d.updateWeights(t.State, action, tdError, hidden)
	}
}

func (d *DQNScheduler) updateWeights(state []float64, action int, tdError float64, hidden []float64) {
	if action >= d.numActions {
		return
	}

	w2Row := d.w2.RawRowView(action)
	w2Snap := make([]float64, len(w2Row))
	copy(w2Snap, w2Row)

	floats.AddScaled(w2Row, d.lr*tdError, hidden)
	d.b2[action] += d.lr * tdError

	sLen := len(state)
	if sLen > d.stateSize {
		sLen = d.stateSize
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
