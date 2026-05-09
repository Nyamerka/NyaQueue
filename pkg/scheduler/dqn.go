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
// Architecture mirrors DQN Balancer: inference uses RLock on model weights,
// training runs in a separate goroutine fed via a non-blocking channel.
//
// State: [level_distribution(10), avg_wait_per_level(10), queue_depth, consumer_lag]
// Action: threshold (0-9) — priorities >= threshold go by priority order, below by FIFO
// Reward: weighted_latency_reduction - starvation_penalty
type DQNScheduler struct {
	weightsMu sync.RWMutex

	hiddenSize int
	stateSize  int
	numActions int
	epsilon    float64
	gamma      float64
	lr         float64
	batchSize  int
	minReplay  int
	trainEvery int
	weightInit float64

	w1 *mat.Dense
	b1 []float64
	w2 *mat.Dense
	b2 []float64

	replayBuffer *nn.ReplayBuffer

	expCh  chan nn.Transition
	stopCh chan struct{}
	done   chan struct{}

	stateMu    sync.Mutex
	lastState  []float64
	lastAction int
	threshold  int

	throttleOnLoad float64
	fallbackFIFO   bool
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
		trainEvery:   DefaultDQNSchedTrainEvery,
		weightInit:   DefaultDQNSchedWeightInit,
		replayBuffer: nn.NewReplayBuffer(DefaultDQNSchedReplayBufSize),
		threshold:    DefaultDQNSchedThreshold,
		expCh:        make(chan nn.Transition, DefaultDQNSchedExpChSize),
		stopCh:       make(chan struct{}),
		done:         make(chan struct{}),
	}

	for _, opt := range opts {
		opt(d)
	}

	d.initWeights()

	go d.trainLoop()

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

	d.stateMu.Lock()
	if d.fallbackFIFO {
		d.stateMu.Unlock()
		return d.fifoFallback(partition, consumerOffset)
	}

	state := d.buildState(pi)

	var threshold int
	if rand.Float64() < d.epsilon {
		threshold = rand.Intn(d.numActions)
	} else {
		d.weightsMu.RLock()
		q, _ := d.forward(state)
		d.weightsMu.RUnlock()
		threshold = floats.MaxIdx(q)
	}

	d.threshold = threshold
	d.lastState = state
	d.lastAction = threshold
	d.stateMu.Unlock()

	entry, ok := pi.PopWithThreshold(threshold)
	if !ok {
		return nil, consumerOffset, broker.ErrNoMessages
	}

	msg, err := partition.Read(uint64(entry.WalOffset))
	if err != nil {
		return nil, 0, err
	}

	return msg, uint64(entry.WalOffset), nil
}

func (d *DQNScheduler) Enqueue(_ *broker.Message, _ int64) {}

func (d *DQNScheduler) OnMetrics(m broker.Metrics) {
	d.stateMu.Lock()
	if d.lastState == nil {
		d.stateMu.Unlock()
		return
	}

	t := nn.Transition{
		State:     d.lastState,
		Action:    []float64{float64(d.lastAction)},
		Reward:    -m.AvgLatency,
		NextState: d.lastState,
		Done:      false,
	}
	d.stateMu.Unlock()

	select {
	case d.expCh <- t:
	default:
	}
}

func (d *DQNScheduler) SetFallbackFIFO(on bool) {
	d.stateMu.Lock()
	defer d.stateMu.Unlock()
	d.fallbackFIFO = on
}

// Stop terminates the background training goroutine.
func (d *DQNScheduler) Stop() {
	close(d.stopCh)
	<-d.done
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
		if waitIdx := levels + i; waitIdx < d.stateSize {
			state[waitIdx] = 0
		}
	}
	if depthIdx := levels * 2; depthIdx < d.stateSize {
		state[depthIdx] = float64(totalDepth)
	}
	if lagIdx := levels*2 + 1; lagIdx < d.stateSize {
		state[lagIdx] = 0
	}

	return state
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

	d.weightsMu.Lock()
	defer d.weightsMu.Unlock()

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
