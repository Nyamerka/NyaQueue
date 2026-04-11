package scheduler

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
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

	w1 [][]float64
	b1 []float64
	w2 [][]float64
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
	d.w1 = make([][]float64, d.hiddenSize)
	d.b1 = make([]float64, d.hiddenSize)
	for i := range d.w1 {
		d.w1[i] = make([]float64, d.stateSize)
		for j := range d.w1[i] {
			d.w1[i][j] = rand.NormFloat64() * d.weightInit
		}
	}

	d.w2 = make([][]float64, d.numActions)
	d.b2 = make([]float64, d.numActions)
	for i := range d.w2 {
		d.w2[i] = make([]float64, d.hiddenSize)
		for j := range d.w2[i] {
			d.w2[i][j] = rand.NormFloat64() * d.weightInit
		}
	}
}

func (d *DQNScheduler) Next(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	pi := partition.PriorityIndex()
	if pi == nil {
		return nil, consumerOffset, fmt.Errorf("partition %d has no PriorityIndex", partition.ID())
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
		return nil, consumerOffset, fmt.Errorf("no pending messages")
	}

	msg, err := partition.Read(uint64(entry.WalOffset))
	if err != nil {
		return nil, 0, err
	}

	return msg, uint64(entry.WalOffset), nil
}

func (d *DQNScheduler) Enqueue(_ *broker.Message, _ int64) {}

// OnMetrics feeds reward signal back to the DQN.
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

// SetFallbackFIFO enables FIFO fallback under high load.
func (d *DQNScheduler) SetFallbackFIFO(on bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.fallbackFIFO = on
}

func (d *DQNScheduler) fifoFallback(partition *broker.Partition, consumerOffset uint64) (*broker.Message, uint64, error) {
	hwm := partition.HighWaterMark()
	if consumerOffset > hwm {
		return nil, consumerOffset, fmt.Errorf("no new messages")
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
		state[10+i] = 0 // wait time placeholder
	}
	state[20] = float64(totalDepth)
	state[21] = 0 // consumer lag placeholder

	return state
}

func (d *DQNScheduler) selectAction(state []float64) int {
	if rand.Float64() < d.epsilon {
		return rand.Intn(d.numActions)
	}
	qValues := d.forward(state)
	return floats.MaxIdx(qValues)
}

// forward computes Q-values using gonum/floats.Dot.
func (d *DQNScheduler) forward(state []float64) []float64 {
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

	q := make([]float64, d.numActions)
	for i := 0; i < d.numActions; i++ {
		q[i] = floats.Dot(d.w2[i], hidden) + d.b2[i]
	}
	return q
}

func (d *DQNScheduler) trainStep() {
	if d.replayBuffer.Len() < d.minReplay {
		return
	}

	batch := d.replayBuffer.Sample(d.batchSize)
	for _, t := range batch {
		qValues := d.forward(t.State)
		nextQ := d.forward(t.NextState)

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

		hidden := make([]float64, d.hiddenSize)
		sLen := len(t.State)
		for i := 0; i < d.hiddenSize; i++ {
			wRow := d.w1[i]
			n := len(wRow)
			if sLen < n {
				n = sLen
			}
			hidden[i] = floats.Dot(wRow[:n], t.State[:n]) + d.b1[i]
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
				dH := tdError * d.w2[action][i]
				for j := 0; j < len(t.State) && j < len(d.w1[i]); j++ {
					d.w1[i][j] += d.lr * dH * t.State[j]
				}
				d.b1[i] += d.lr * dH
			}
		}
	}
}
