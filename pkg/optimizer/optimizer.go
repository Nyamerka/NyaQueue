package optimizer

import (
	"context"
	"log"
	"math"
	"sync"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"golang.org/x/sync/errgroup"
)

// PilotData holds results from pilot runs used for Lasso calibration.
type PilotData struct {
	Configs     [][]float64 // each row: one config vector (len == NumTunableParams)
	Throughputs []float64   // observed throughput per config
	Alpha       float64     // L1 regularisation strength
}

type OptimizerConfig struct {
	Interval     time.Duration
	BatchSize    int
	WarmupTicks  int
	LearningRate float64
	Hysteresis   float64
	NoiseDecay   float64 // per-step multiplicative decay for OUNoise sigma (default 0.999)
	NoiseFloor   float64 // minimum sigma value (default 0.05)
}

func DefaultOptimizerConfig() OptimizerConfig {
	return OptimizerConfig{
		Interval:     250 * time.Millisecond,
		BatchSize:    32,
		WarmupTicks:  64,
		LearningRate: 0.0003,
		Hysteresis:   0.10,
		NoiseDecay:   0.999,
		NoiseFloor:   0.05,
	}
}

// Optimizer runs the DDPG training loop on Lasso-selected broker parameters.
type Optimizer struct {
	mu sync.Mutex

	ddpg       *DDPG
	params     []TunableParam
	broker     *broker.Broker
	interval   time.Duration
	batchSize  int
	warmup     int
	ticks      int
	hysteresis float64

	currentVals []float64
	prevMetrics *broker.Metrics
	eg          *errgroup.Group
	cancel      context.CancelFunc

	throughputWindow []float64
	windowCap        int
}

func NewOptimizer(b *broker.Broker, params []TunableParam, optCfg OptimizerConfig, pilot ...PilotData) *Optimizer {
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

	if optCfg.Interval == 0 {
		optCfg.Interval = 250 * time.Millisecond
	}
	if optCfg.BatchSize == 0 {
		optCfg.BatchSize = 32
	}
	if optCfg.WarmupTicks == 0 {
		optCfg.WarmupTicks = optCfg.BatchSize * 2
	}
	if optCfg.LearningRate == 0 {
		optCfg.LearningRate = 0.0003
	}
	if optCfg.Hysteresis == 0 {
		optCfg.Hysteresis = 0.10
	}
	if optCfg.NoiseDecay == 0 {
		optCfg.NoiseDecay = 0.999
	}
	if optCfg.NoiseFloor == 0 {
		optCfg.NoiseFloor = 0.05
	}

	const rewardWindowCap = 20

	ddpg := NewDDPG(stateSize, actionSize, optCfg.LearningRate, optCfg.BatchSize)
	ddpg.SetNoiseDecay(optCfg.NoiseDecay, optCfg.NoiseFloor)

	return &Optimizer{
		ddpg:             ddpg,
		params:           active,
		broker:           b,
		interval:         optCfg.Interval,
		batchSize:        optCfg.BatchSize,
		warmup:           optCfg.WarmupTicks,
		hysteresis:       optCfg.Hysteresis,
		currentVals:      currentVals,
		windowCap:        rewardWindowCap,
		throughputWindow: make([]float64, 0, rewardWindowCap),
	}
}

func (o *Optimizer) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	o.cancel = cancel
	o.eg, _ = errgroup.WithContext(ctx)
	o.eg.Go(func() error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[optimizer] loop panic: %v", r)
			}
		}()
		o.loop(ctx)
		return nil
	})
}

func (o *Optimizer) Stop() {
	if o.cancel != nil {
		o.cancel()
	}
	if o.eg != nil {
		_ = o.eg.Wait()
	}
}

func (o *Optimizer) loop(ctx context.Context) {
	ticker := time.NewTicker(o.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.step()
		case <-ctx.Done():
			return
		}
	}
}

func (o *Optimizer) step() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.ticks++
	metrics := o.broker.Metrics()

	state := o.buildState(&metrics)
	action := o.ddpg.Act(state)

	if o.ticks > o.warmup {
		safe := true
		if metrics.MsgRate > 0 && metrics.ConsumeRate > 0 {
			if metrics.ConsumeRate/metrics.MsgRate < 0.7 {
				safe = false
			}
		}
		if safe {
			o.applyAction(action)
		}
	}

	if o.prevMetrics != nil {
		reward := o.computeReward(&metrics)
		prevState := o.buildState(o.prevMetrics)

		o.ddpg.Store(prevState, action, reward, state, false)
		o.ddpg.Train(o.batchSize)
	}

	o.prevMetrics = &metrics

	if o.ticks%20 == 0 {
		log.Printf("[optimizer] tick=%d throughput=%.1f latency=%.2fms params_snapshot=%v",
			o.ticks, metrics.Throughput, metrics.AvgLatency, o.currentVals[:min(3, len(o.currentVals))])
	}
}

func (o *Optimizer) applyAction(action []float64) {
	changed := false
	for i, a := range action {
		if i >= len(o.params) {
			break
		}
		newNorm := (a + 1.0) / 2.0
		if newNorm < 0 {
			newNorm = 0
		}
		if newNorm > 1 {
			newNorm = 1
		}

		if math.Abs(newNorm-o.currentVals[i]) < o.hysteresis {
			continue
		}
		o.currentVals[i] = newNorm
		changed = true
	}

	if !changed {
		return
	}

	newCfg := o.broker.Config()
	for i, p := range o.params {
		broker.SetParamByName(&newCfg, p.Name, Denormalize(o.currentVals[i], p.Min, p.Max))
	}
	if err := o.broker.ApplyConfig(newCfg); err != nil {
		log.Printf("[optimizer] ApplyConfig failed: %v", err)
	}
}

func (o *Optimizer) computeReward(m *broker.Metrics) float64 {
	delivered := m.ConsumeRate
	if delivered == 0 {
		delivered = m.Throughput
	}

	o.throughputWindow = append(o.throughputWindow, delivered)
	if len(o.throughputWindow) > o.windowCap {
		o.throughputWindow = o.throughputWindow[1:]
	}

	baseline := 0.0
	for _, t := range o.throughputWindow {
		baseline += t
	}
	if len(o.throughputWindow) > 0 {
		baseline /= float64(len(o.throughputWindow))
	}

	var throughputReward float64
	if baseline > 1 {
		throughputReward = (delivered - baseline) / baseline
	}

	latencyPenalty := 0.0
	if m.AvgLatency > 0 {
		latencyPenalty = -m.AvgLatency / 1000.0
	}

	queuePenalty := 0.0
	totalDepth := 0
	for _, d := range m.QueueDepth {
		totalDepth += d
	}
	if totalDepth > 1000 {
		queuePenalty = -float64(totalDepth) / 10000.0
	}

	reward := throughputReward + 0.3*latencyPenalty + 1.5*queuePenalty

	if reward < -2.0 {
		reward = -2.0
	}
	if reward > 2.0 {
		reward = 2.0
	}

	return reward
}

func (o *Optimizer) buildState(m *broker.Metrics) []float64 {
	state := make([]float64, 0, len(o.currentVals)+3)
	state = append(state, o.currentVals...)
	state = append(state, m.Throughput/100000) // normalize
	state = append(state, m.AvgLatency/1000)
	state = append(state, m.SuccessRate)
	return state
}
