package optimizer

import (
	"math"
	"sync"
	"testing"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type TD3Suite struct {
	suite.Suite
}

func TestTD3Suite(t *testing.T) { suite.Run(t, new(TD3Suite)) }

func (s *TD3Suite) TestAct() {
	tests := []struct {
		name       string
		stateSize  int
		actionSize int
	}{
		{"small", 5, 3},
		{"medium", 20, 10},
		{"single_action", 10, 1},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			d := NewTD3(tc.stateSize, tc.actionSize, 0.001, 32)
			state := make([]float64, tc.stateSize)
			for i := range state {
				state[i] = float64(i) * 0.1
			}

			action := d.Act(state)
			require.Len(s.T(), action, tc.actionSize)
			for _, a := range action {
				require.GreaterOrEqual(s.T(), a, -1.0)
				require.LessOrEqual(s.T(), a, 1.0)
			}
		})
	}
}

func (s *TD3Suite) TestStoreAndTrain() {
	d := NewTD3(5, 3, 0.001, 32)

	for i := 0; i < 200; i++ {
		state := []float64{float64(i), 0.1, 0.2, 0.3, 0.4}
		action := []float64{0.1, -0.2, 0.3}
		nextState := []float64{float64(i + 1), 0.2, 0.3, 0.4, 0.5}
		d.Store(state, action, 1.0, nextState, false)
	}

	require.NotPanics(s.T(), func() { d.Train(32) })
}

func (s *TD3Suite) TestTrainInsufficientData() {
	d := NewTD3(5, 3, 0.001, 32)
	d.Store([]float64{1, 2, 3, 4, 5}, []float64{0.1, 0.2, 0.3}, 1.0, []float64{2, 3, 4, 5, 6}, false)

	require.NotPanics(s.T(), func() { d.Train(64) })
}

func (s *TD3Suite) TestResetNoise() {
	d := NewTD3(5, 3, 0.001, 32)
	d.sigma = 0.01
	d.ResetNoise()
	require.Equal(s.T(), td3ExploreNoiseInit, d.sigma)
}

func (s *TD3Suite) TestActorForwardOutputInRange() {
	d := NewTD3(5, 3, 0.001, 32)
	state := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

	result := d.actorFwdExec.MustExec1(state)

	out := tensorToFloat64(result)
	require.Len(s.T(), out, 3)
	for _, v := range out {
		require.GreaterOrEqual(s.T(), v, -1.0)
		require.LessOrEqual(s.T(), v, 1.0)
	}
}

func (s *TD3Suite) TestTwinCriticsBothReturnCorrectShape() {
	d := NewTD3(5, 3, 0.001, 32)

	states := tensors.FromFlatDataAndDimensions(
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1}, 2, 5)
	actions := tensors.FromFlatDataAndDimensions(
		[]float64{0.1, 0.2, 0.3, 0.3, 0.2, 0.1}, 2, 3)

	q1 := tensorToFloat64(d.targetCritic1Exec.MustExec1(states, actions))
	q2 := tensorToFloat64(d.targetCritic2Exec.MustExec1(states, actions))

	require.Len(s.T(), q1, 2, "critic1 should return one Q per sample")
	require.Len(s.T(), q2, 2, "critic2 should return one Q per sample")
}

func (s *TD3Suite) TestTwinCriticsAreDifferent() {
	d := NewTD3(5, 3, 0.001, 32)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}
	for i := 0; i < 5; i++ {
		d.Train(32)
	}

	states := tensors.FromFlatDataAndDimensions(
		[]float64{0.5, 0.5, 0.5, 0.5, 0.5}, 1, 5)
	actions := tensors.FromFlatDataAndDimensions(
		[]float64{0.1, 0.2, 0.3}, 1, 3)

	q1 := tensorToFloat64(d.targetCritic1Exec.MustExec1(states, actions))
	q2 := tensorToFloat64(d.targetCritic2Exec.MustExec1(states, actions))

	require.NotEqual(s.T(), q1[0], q2[0],
		"twin critics should have different Q-values after independent training")
}

func (s *TD3Suite) TestDelayedActorUpdate() {
	d := NewTD3(5, 3, 0.01, 32)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}

	d.Train(32)
	require.Equal(s.T(), 1, d.criticStep)

	d.Train(32)
	require.Equal(s.T(), 2, d.criticStep,
		"after 2 critic steps, delayed actor update should have triggered")
}

func (s *TD3Suite) TestTrainUpdatesActorWeights() {
	d := NewTD3(5, 3, 0.01, 32)

	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	actionBefore := d.Act(state)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}

	for i := 0; i < 4; i++ {
		d.Train(32)
	}

	actionAfter := d.Act(state)

	changed := false
	for i := range actionBefore {
		if actionAfter[i] != actionBefore[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "actor should produce different actions after training")
}

func (s *TD3Suite) TestSoftUpdateChangesTarget() {
	d := NewTD3(5, 3, 0.001, 32)

	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	mainAction := d.Act(state)
	require.Len(s.T(), mainAction, 3)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}

	for i := 0; i < 4; i++ {
		d.Train(32)
	}

	actionAfter := d.Act(state)

	changed := false
	for i := range mainAction {
		if actionAfter[i] != mainAction[i] {
			changed = true
			break
		}
	}
	require.True(s.T(), changed, "target network should update via soft update")
}

func (s *TD3Suite) TestBatchedTrainMultipleSteps() {
	d := NewTD3(5, 3, 0.001, 32)

	for i := 0; i < 300; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			float64(i%10)*0.1,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			i%50 == 0,
		)
	}

	require.NotPanics(s.T(), func() {
		for step := 0; step < 5; step++ {
			d.Train(32)
		}
	})
}

func (s *TD3Suite) TestSoftUpdateDtypeRobust() {
	d := NewTD3(5, 3, 0.001, 32)
	require.NotPanics(s.T(), func() {
		d.softUpdate()
	})
	require.Greater(s.T(), len(d.varPairs), 0, "varPairs should be populated")
}

func (s *TD3Suite) TestSoftUpdateBlendedWeights() {
	d := NewTD3(5, 3, 0.001, 32)

	require.Greater(s.T(), len(d.varPairs), 0, "should have variable pairs")

	p := d.varPairs[0]
	srcVal := p.src.MustValue()
	tgtVal := p.tgt.MustValue()

	srcFlat := flattenTensorValue(srcVal.Value())
	tgtFlat := flattenTensorValue(tgtVal.Value())
	require.Equal(s.T(), len(srcFlat), len(tgtFlat))

	expectedBuf := make([]float64, len(tgtFlat))
	for i := range expectedBuf {
		expectedBuf[i] = (1-td3Tau)*tgtFlat[i] + td3Tau*srcFlat[i]
	}

	d.softUpdate()

	newTgtVal := p.tgt.MustValue()
	newTgtFlat := flattenTensorValue(newTgtVal.Value())
	for i := range expectedBuf {
		require.InDelta(s.T(), expectedBuf[i], newTgtFlat[i], 1e-9,
			"softUpdate must produce exact blended value (idx %d)", i)
	}
}

func (s *TD3Suite) TestStoreDeepCopies() {
	d := NewTD3(5, 3, 0.001, 32)
	state := []float64{1, 2, 3, 4, 5}
	action := []float64{0.1, 0.2, 0.3}
	nextState := []float64{2, 3, 4, 5, 6}
	d.Store(state, action, 1.0, nextState, false)

	state[0] = 999
	action[0] = 999
	nextState[0] = 999

	batch := d.replayBuffer.Sample(1)
	require.Equal(s.T(), 1.0, batch[0].State[0], "Store must deep-copy state")
	require.Equal(s.T(), 0.1, batch[0].Action[0], "Store must deep-copy action")
	require.Equal(s.T(), 2.0, batch[0].NextState[0], "Store must deep-copy nextState")
}

func (s *TD3Suite) TestPreAllocatedBuffersProperty() {
	batchSize := 32
	d := NewTD3(5, 3, 0.001, batchSize)
	require.GreaterOrEqual(s.T(), len(d.statesBuf), batchSize*d.stateSize)
	require.GreaterOrEqual(s.T(), len(d.actionsBuf), batchSize*d.actionSize)
	require.GreaterOrEqual(s.T(), len(d.rewardsBuf), batchSize)
	require.GreaterOrEqual(s.T(), len(d.targetQBuf), batchSize)
}

func (s *TD3Suite) TestNoiseDecay() {
	d := NewTD3(5, 3, 0.001, 32)
	d.SetNoiseDecay(0.9, 0.01)

	initial := d.Sigma()
	for i := 0; i < 50; i++ {
		d.DecayNoise()
	}
	after := d.Sigma()
	require.Less(s.T(), after, initial, "sigma should decay over steps")
	require.GreaterOrEqual(s.T(), after, 0.01, "sigma should not go below floor")
}

func (s *TD3Suite) TestConcurrentActAndTrain() {
	d := NewTD3(5, 3, 0.001, 32)

	for i := 0; i < 200; i++ {
		d.Store(
			[]float64{float64(i) * 0.01, 0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, -0.2, 0.3},
			1.0,
			[]float64{float64(i+1) * 0.01, 0.2, 0.3, 0.4, 0.5},
			false,
		)
	}

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
			for j := 0; j < 20; j++ {
				_ = d.Act(state)
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 10; j++ {
			d.Train(32)
		}
	}()
	wg.Wait()
}

func (s *TD3Suite) TestOptimizer_ResidualActionBounded() {
	b := setupMinimalBroker(s.T())
	defer b.Stop()

	params := []TunableParam{{Name: "NumIOGoroutines", Min: 1, Max: 32, Weight: 1.0}}
	opt := NewOptimizer(b, params, DefaultOptimizerConfig())

	opt.applyAction([]float64{10.0})
	require.LessOrEqual(s.T(), opt.currentVals[0], 0.95,
		"action should be clipped to safe range")
	require.GreaterOrEqual(s.T(), opt.currentVals[0], 0.05,
		"action should be clipped to safe range")
}

func flattenTensorValue(v interface{}) []float64 {
	switch d := v.(type) {
	case []float64:
		return d
	case [][]float64:
		var out []float64
		for _, row := range d {
			out = append(out, row...)
		}
		return out
	default:
		return nil
	}
}

func (s *TD3Suite) TestOptimizer_SafeByConstruction() {
	b := setupMinimalBroker(s.T())
	defer b.Stop()

	params := []TunableParam{{Name: "NumIOGoroutines", Min: 1, Max: 32, Weight: 1.0}}
	opt := NewOptimizer(b, params, DefaultOptimizerConfig())

	for i := 0; i < 100; i++ {
		opt.applyAction([]float64{float64(i-50) / 10.0})
		require.GreaterOrEqual(s.T(), opt.currentVals[0], 0.05)
		require.LessOrEqual(s.T(), opt.currentVals[0], 0.95)
	}
}

func (s *TD3Suite) TestTD3RewardBounded() {
	opt := &Optimizer{}

	highTP := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{
			Throughput:    100000,
			MsgRate:       100000,
			DeliveryRatio: 0.99,
			AvgLatency:    1,
		},
	}
	r1 := opt.computeReward(highTP)
	require.GreaterOrEqual(s.T(), r1, -1.0)
	require.LessOrEqual(s.T(), r1, 1.0)

	lowTP := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{
			Throughput:    100,
			MsgRate:       100,
			DeliveryRatio: 0.99,
			AvgLatency:    500,
		},
	}
	r2 := opt.computeReward(lowTP)
	require.GreaterOrEqual(s.T(), r2, -1.0)
	require.LessOrEqual(s.T(), r2, 1.0)
	require.Less(s.T(), r2, r1,
		"low throughput + high latency should give lower reward than high throughput + low latency")
}

func (s *TD3Suite) TestTD3StateNormalized() {
	b := setupMinimalBroker(s.T())
	defer b.Stop()

	params := []TunableParam{{Name: "MaxBatchSize", Min: 1, Max: 1000, Weight: 1.0}}
	opt := NewOptimizer(b, params, DefaultOptimizerConfig())

	for i := 0; i < 100; i++ {
		state := opt.buildState(&broker.Metrics{
			BusinessMetrics: broker.BusinessMetrics{
				Throughput:    float64(50000 + i*100),
				AvgLatency:    float64(i),
				MsgRate:       float64(50000 + i*50),
				ConsumeRate:   float64(50000 + i*40),
				DeliveryRatio: 0.99,
			},
			DerivedMetrics: broker.DerivedMetrics{
				PartitionLoads: []float64{0.3, 0.4, 0.5, 0.6},
				QueueDepth:     []int{100, 200, 300, 400},
			},
		})
		opt.normalizer.Observe(state)
	}

	raw := opt.buildState(&broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{
			Throughput:    55000,
			AvgLatency:    50,
			MsgRate:       52000,
			ConsumeRate:   52000,
			DeliveryRatio: 0.99,
		},
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.3, 0.4, 0.5, 0.6},
			QueueDepth:     []int{100, 200, 300, 400},
		},
	})
	normalized := opt.normalizer.Normalize(raw)
	for i, v := range normalized {
		require.Less(s.T(), math.Abs(v), 5.0,
			"normalized state[%d]=%.4f should be within [-5, 5]", i, v)
	}
}

func setupMinimalBroker(t *testing.T) *broker.Broker {
	t.Helper()
	dir := t.TempDir()
	store, err := broker.NewOffsetStore(dir)
	require.NoError(t, err)
	t.Cleanup(func() { store.Close() })
	return broker.New(broker.DefaultConfig(), dir, noopBal{}, store)
}

func (s *TD3Suite) TestGaussianExplorationNoise() {
	d := NewTD3(5, 3, 0.001, 32)
	d.sigma = 0.5
	state := []float64{0.5, 0.5, 0.5, 0.5, 0.5}

	var actions [][]float64
	for i := 0; i < 50; i++ {
		actions = append(actions, d.Act(state))
	}

	differ := false
	for i := 1; i < len(actions); i++ {
		for j := range actions[i] {
			if actions[i][j] != actions[0][j] {
				differ = true
				break
			}
		}
		if differ {
			break
		}
	}
	require.True(s.T(), differ,
		"Gaussian noise should produce variation in actions across calls")
}

func (s *TD3Suite) TestTD3EvalOnlyNoStore() {
	brk := setupMinimalBroker(s.T())
	defer brk.Stop()

	params, cfg := DefaultTunableParams(), DefaultOptimizerConfig()
	cfg.EvalOnly = true
	cfg.CollectInterval = 10 * time.Millisecond
	cfg.WarmupTicks = 0
	cfg.ApplyEvery = 1

	opt := NewOptimizer(brk, params, cfg)
	opt.Start()
	time.Sleep(200 * time.Millisecond)
	opt.Stop()

	require.Equal(s.T(), 0, opt.td3.replayBuffer.Len(),
		"EvalOnly optimizer must not Store() experiences")
}

func (s *TD3Suite) TestTD3NonEvalStores() {
	brk := setupMinimalBroker(s.T())
	defer brk.Stop()

	params, cfg := DefaultTunableParams(), DefaultOptimizerConfig()
	cfg.EvalOnly = false
	cfg.CollectInterval = 10 * time.Millisecond
	cfg.WarmupTicks = 0
	cfg.ApplyEvery = 100

	opt := NewOptimizer(brk, params, cfg)
	opt.Start()
	time.Sleep(200 * time.Millisecond)
	opt.Stop()

	require.Greater(s.T(), opt.td3.replayBuffer.Len(), 0,
		"non-EvalOnly optimizer must Store() experiences")
}
