package optimizer

import (
	"log"
	"sync"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

// Optimizer runs the DDPG training loop: metrics -> state -> action -> apply -> reward.
// Operates on the active (Lasso-selected) parameters of the broker config.
type Optimizer struct {
	mu sync.Mutex

	ddpg      *DDPG
	params    []TunableParam
	broker    *broker.Broker
	interval  time.Duration
	batchSize int

	currentVals []float64
	prevMetrics *broker.Metrics
	stopCh      chan struct{}
}

func NewOptimizer(b *broker.Broker, params []TunableParam, interval time.Duration) *Optimizer {
	active := ActiveParams(params)
	stateSize := len(active) + 3 // params + throughput + latency + success_rate
	actionSize := len(active)

	currentVals := make([]float64, len(active))
	for i := range currentVals {
		currentVals[i] = 0.5 // start at midpoint of each range
	}

	return &Optimizer{
		ddpg:        NewDDPG(stateSize, actionSize, 0.0001),
		params:      active,
		broker:      b,
		interval:    interval,
		batchSize:   64,
		currentVals: currentVals,
		stopCh:      make(chan struct{}),
	}
}

func (o *Optimizer) Start() {
	go o.loop()
}

func (o *Optimizer) Stop() {
	close(o.stopCh)
}

func (o *Optimizer) loop() {
	ticker := time.NewTicker(o.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.step()
		case <-o.stopCh:
			return
		}
	}
}

func (o *Optimizer) step() {
	o.mu.Lock()
	defer o.mu.Unlock()

	metrics := o.broker.Metrics()

	state := o.buildState(&metrics)
	action := o.ddpg.Act(state)

	// Apply action: delta to each parameter
	for i, a := range action {
		if i >= len(o.params) {
			break
		}
		scale := (o.params[i].Max - o.params[i].Min) * 0.05 // 5% of range per step
		delta := a * scale
		rawVal := Denormalize(o.currentVals[i], o.params[i].Min, o.params[i].Max)
		newVal := ClipAction(rawVal, delta, o.params[i].Min, o.params[i].Max)
		o.currentVals[i] = Normalize(newVal, o.params[i].Min, o.params[i].Max)
	}

	if o.prevMetrics != nil {
		reward := metrics.Throughput - o.prevMetrics.Throughput
		prevState := o.buildState(o.prevMetrics)

		o.ddpg.Store(prevState, action, reward, state, false)
		o.ddpg.Train(o.batchSize)
	}

	o.prevMetrics = &metrics

	log.Printf("[optimizer] throughput=%.1f latency=%.2fms params_snapshot=%v",
		metrics.Throughput, metrics.AvgLatency, o.currentVals[:min(3, len(o.currentVals))])
}

func (o *Optimizer) buildState(m *broker.Metrics) []float64 {
	state := make([]float64, 0, len(o.currentVals)+3)
	state = append(state, o.currentVals...)
	state = append(state, m.Throughput/100000) // normalize
	state = append(state, m.AvgLatency/1000)
	state = append(state, m.SuccessRate)
	return state
}
