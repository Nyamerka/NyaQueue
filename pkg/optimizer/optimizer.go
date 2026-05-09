package optimizer

import (
	"log"
	"sync"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

// PilotData holds results from pilot runs used for Lasso calibration.
type PilotData struct {
	Configs     [][]float64 // each row: one config vector (len == NumTunableParams)
	Throughputs []float64   // observed throughput per config
	Alpha       float64     // L1 regularisation strength
}

// Optimizer runs the DDPG training loop on Lasso-selected broker parameters.
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

func NewOptimizer(b *broker.Broker, params []TunableParam, interval time.Duration, pilot ...PilotData) *Optimizer {
	calibrated := params
	if len(pilot) > 0 && len(pilot[0].Configs) > 0 {
		calibrated = CalibrateWeights(params, pilot[0].Configs, pilot[0].Throughputs, pilot[0].Alpha)
		log.Printf("[optimizer] Lasso calibration: %d/%d params active (alpha=%.4f)",
			len(ActiveParams(calibrated)), len(params), pilot[0].Alpha)
	}

	active := ActiveParams(calibrated)
	stateSize := len(active) + 3 // params + throughput + latency + success_rate
	actionSize := len(active)

	currentVals := make([]float64, len(active))
	for i := range currentVals {
		currentVals[i] = 0.5
	}

	effectiveInterval := interval
	if effectiveInterval == 0 {
		effectiveInterval = 1100 * time.Millisecond
	}

	return &Optimizer{
		ddpg:        NewDDPG(stateSize, actionSize, 0.0001),
		params:      active,
		broker:      b,
		interval:    effectiveInterval,
		batchSize:   256,
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

	for i, a := range action {
		if i >= len(o.params) {
			break
		}
		scale := (o.params[i].Max - o.params[i].Min) * 0.05
		delta := a * scale
		rawVal := Denormalize(o.currentVals[i], o.params[i].Min, o.params[i].Max)
		newVal := ClipAction(rawVal, delta, o.params[i].Min, o.params[i].Max)
		o.currentVals[i] = Normalize(newVal, o.params[i].Min, o.params[i].Max)
	}

	newCfg := o.broker.Config()
	for i, p := range o.params {
		broker.SetParamByName(&newCfg, p.Name, Denormalize(o.currentVals[i], p.Min, p.Max))
	}
	if err := o.broker.ApplyConfig(newCfg); err != nil {
		log.Printf("[optimizer] ApplyConfig failed: %v", err)
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
