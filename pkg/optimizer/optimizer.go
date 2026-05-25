package optimizer

import (
	"context"
	"log"
	"math"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/puzpuzpuz/xsync/v3"
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
	NoiseDecay   float64 // per-step multiplicative decay for Gaussian sigma (default 0.999)
	NoiseFloor   float64 // minimum sigma value (default 0.05)
}

func DefaultOptimizerConfig() OptimizerConfig {
	return OptimizerConfig{
		Interval:     250 * time.Millisecond,
		BatchSize:    32,
		WarmupTicks:  DefaultOptimizerWarmupTicks,
		LearningRate: 0.0003,
		Hysteresis:   0.10,
		NoiseDecay:   0.999,
		NoiseFloor:   0.05,
	}
}

// Optimizer runs the TD3 training loop on Lasso-selected broker parameters.
type Optimizer struct {
	mu xsync.RBMutex

	td3        *TD3
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

	lastApplyTime          time.Time
	minApplyInterval       time.Duration
	startConfig            broker.Config
	lowDeliveryTicksInARow int
	cooldownTicksLeft      int

	rewardMean  float64
	rewardM2    float64
	rewardCount int64

	normalizer *nn.RunningNormalizer
}

func NewOptimizer(b *broker.Broker, params []TunableParam, optCfg OptimizerConfig, pilot ...PilotData) *Optimizer {
	calibrated := params
	if len(pilot) > 0 && len(pilot[0].Configs) > 0 {
		calibrated = CalibrateWeights(params, pilot[0].Configs, pilot[0].Throughputs, pilot[0].Alpha)
		log.Printf("[optimizer] Lasso calibration: %d/%d params active (alpha=%.4f)",
			len(ActiveParams(calibrated)), len(params), pilot[0].Alpha)
	}

	active := ActiveParams(calibrated)
	stateSize := len(active) + 10 // params + throughput + latency + dr + max_load + load_stddev + max_depth + avg_depth + msg_rate + consume_rate + rate_ratio
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

	td3 := NewTD3(stateSize, actionSize, optCfg.LearningRate, optCfg.BatchSize)
	td3.SetNoiseDecay(optCfg.NoiseDecay, optCfg.NoiseFloor)

	var startConfig broker.Config
	if b != nil {
		startConfig = b.Config()
	}

	return &Optimizer{
		td3:              td3,
		params:           active,
		broker:           b,
		interval:         optCfg.Interval,
		batchSize:        optCfg.BatchSize,
		warmup:           optCfg.WarmupTicks,
		hysteresis:       optCfg.Hysteresis,
		currentVals:      currentVals,
		windowCap:        DefaultOptimizerWindowCap,
		throughputWindow: make([]float64, 0, DefaultOptimizerWindowCap),
		minApplyInterval: time.Duration(DefaultMinApplyInterval),
		startConfig:      startConfig,
		normalizer:       nn.NewRunningNormalizer(stateSize),
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

	if o.ticks > o.warmup {
		o.checkEmergencyRollback(&metrics)
	}

	if o.cooldownTicksLeft > 0 {
		o.cooldownTicksLeft--
	}

	rawState := o.buildState(&metrics)
	o.normalizer.Observe(rawState)
	state := o.normalizer.Normalize(rawState)
	action := o.td3.Act(state)

	if o.ticks > o.warmup && o.cooldownTicksLeft == 0 {
		safe := true
		if metrics.MsgRate > 0 {
			ratio := metrics.ConsumeRate / metrics.MsgRate
			if ratio < DefaultSafetyConsumeRatio {
				safe = false
			}
		}
		if safe {
			o.applyAction(action)
		}
	}

	if o.prevMetrics != nil {
		reward := o.computeReward(&metrics)
		prevState := o.normalizer.Normalize(o.buildState(o.prevMetrics))

		o.td3.Store(prevState, action, reward, state, false)
		o.td3.Train(o.batchSize)
		o.td3.DecayNoise()
	}

	o.prevMetrics = &metrics

	if o.ticks%DefaultLogEveryTicks == 0 {
		log.Printf("[optimizer] tick=%d throughput=%.1f latency=%.2fms cooldown=%d params_snapshot=%v",
			o.ticks, metrics.Throughput, metrics.AvgLatency, o.cooldownTicksLeft,
			o.currentVals[:min(3, len(o.currentVals))])
	}
}

func (o *Optimizer) applyAction(action []float64) {
	now := time.Now()
	if !o.lastApplyTime.IsZero() && now.Sub(o.lastApplyTime) < o.minApplyInterval {
		return
	}

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

	o.lastApplyTime = now

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
		latencyPenalty = -m.AvgLatency / optimizerLatencyScale
	}

	totalDepth := 0
	for _, d := range m.QueueDepth {
		totalDepth += d
	}
	normalized := float64(totalDepth) / DefaultQueueSoftThreshold
	queuePenalty := -(normalized * normalized)
	if queuePenalty < DefaultQueuePenaltyClamp {
		queuePenalty = DefaultQueuePenaltyClamp
	}

	reward := throughputReward + DefaultLatencyPenaltyWeight*latencyPenalty + DefaultQueuePenaltyWeight*queuePenalty

	if m.ConsumeRate > 0 && m.MsgRate > 0 {
		errorRatio := 1.0 - m.DeliveryRatio
		if errorRatio > DefaultErrorPenaltyThresh {
			reward -= DefaultErrorPenaltyWeight * errorRatio
		}
	}

	if reward < -DefaultRewardClamp {
		reward = -DefaultRewardClamp
	}
	if reward > DefaultRewardClamp {
		reward = DefaultRewardClamp
	}

	o.rewardCount++
	delta := reward - o.rewardMean
	o.rewardMean += delta / float64(o.rewardCount)
	o.rewardM2 += delta * (reward - o.rewardMean)
	if o.rewardCount > 1 {
		sigma := math.Sqrt(o.rewardM2/float64(o.rewardCount-1)) + 1e-8
		reward = (reward - o.rewardMean) / sigma
	}

	return reward
}

func (o *Optimizer) checkEmergencyRollback(m *broker.Metrics) {
	if o.cooldownTicksLeft > 0 {
		return
	}
	if m.DeliveryRatio < DefaultEmergencyRollbackDR && m.MsgRate >= DefaultEmergencyMinMsgRate {
		o.lowDeliveryTicksInARow++
	} else {
		o.lowDeliveryTicksInARow = 0
	}
	if o.lowDeliveryTicksInARow >= DefaultEmergencyRollbackN {
		log.Printf("[optimizer] emergency rollback: DeliveryRatio %.2f < %.2f for %d ticks → cooldown %d",
			m.DeliveryRatio, DefaultEmergencyRollbackDR, o.lowDeliveryTicksInARow,
			DefaultEmergencyRollbackCooldown)
		if o.broker != nil {
			if err := o.broker.ApplyConfig(o.startConfig); err != nil {
				log.Printf("[optimizer] emergency rollback ApplyConfig failed: %v", err)
			}
		}
		o.cooldownTicksLeft = DefaultEmergencyRollbackCooldown
		o.lowDeliveryTicksInARow = 0
		if o.td3 != nil {
			o.td3.BoostNoise(3.0)
		}
	}
}

func (o *Optimizer) buildState(m *broker.Metrics) []float64 {
	state := make([]float64, 0, len(o.currentVals)+10)
	state = append(state, o.currentVals...)
	state = append(state, m.Throughput/optimizerThroughputScale)
	state = append(state, m.AvgLatency/optimizerLatencyScale)
	state = append(state, m.DeliveryRatio)

	var maxLoad, sumLoad, sumLoadSq float64
	for _, l := range m.PartitionLoads {
		sumLoad += l
		sumLoadSq += l * l
		if l > maxLoad {
			maxLoad = l
		}
	}
	nLoads := float64(max(len(m.PartitionLoads), 1))
	meanLoad := sumLoad / nLoads
	loadStddev := math.Sqrt(sumLoadSq/nLoads - meanLoad*meanLoad)
	state = append(state, maxLoad)
	state = append(state, loadStddev)

	var maxDepth, sumDepth float64
	for _, d := range m.QueueDepth {
		df := float64(d)
		sumDepth += df
		if df > maxDepth {
			maxDepth = df
		}
	}
	nDepth := float64(max(len(m.QueueDepth), 1))
	state = append(state, maxDepth/optimizerDepthScale)
	state = append(state, (sumDepth/nDepth)/optimizerDepthScale)

	state = append(state, m.MsgRate/optimizerThroughputScale)
	state = append(state, m.ConsumeRate/optimizerThroughputScale)
	if m.ConsumeRate > 0 {
		state = append(state, m.MsgRate/m.ConsumeRate)
	} else {
		state = append(state, 0)
	}

	return state
}
