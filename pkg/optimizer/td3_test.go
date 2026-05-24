package optimizer

import (
	"sync"
	"testing"

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

	d.mu.Lock()
	result := d.actorFwdExec.MustExec1(state)
	d.mu.Unlock()

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

	d.mu.Lock()
	q1 := tensorToFloat64(d.targetCritic1Exec.MustExec1(states, actions))
	q2 := tensorToFloat64(d.targetCritic2Exec.MustExec1(states, actions))
	d.mu.Unlock()

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

	d.mu.Lock()
	q1 := tensorToFloat64(d.targetCritic1Exec.MustExec1(states, actions))
	q2 := tensorToFloat64(d.targetCritic2Exec.MustExec1(states, actions))
	d.mu.Unlock()

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
		d.mu.Lock()
		d.softUpdate()
		d.mu.Unlock()
	})
	require.Greater(s.T(), len(d.varPairs), 0, "varPairs should be populated")
}

func (s *TD3Suite) TestSoftUpdateBlendedWeights() {
	d := NewTD3(5, 3, 0.001, 32)

	d.mu.Lock()
	defer d.mu.Unlock()

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

func (s *TD3Suite) TestOptimizer_EmergencyRollback() {
	opt := &Optimizer{
		minApplyInterval: DefaultMinApplyInterval,
	}

	healthy := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{DeliveryRatio: 0.9, MsgRate: 100},
	}
	opt.checkEmergencyRollback(healthy)
	require.Equal(s.T(), 0, opt.lowDeliveryTicksInARow)
	require.False(s.T(), opt.rolledBack)

	bad := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{DeliveryRatio: 0.3, MsgRate: 100},
	}

	opt.checkEmergencyRollback(bad)
	require.Equal(s.T(), 1, opt.lowDeliveryTicksInARow)
	require.False(s.T(), opt.rolledBack, "single bad tick should not rollback")

	opt.checkEmergencyRollback(bad)
	require.Equal(s.T(), 2, opt.lowDeliveryTicksInARow)
	require.True(s.T(), opt.rolledBack, "two consecutive bad ticks should trigger rollback")

	opt.checkEmergencyRollback(bad)
	require.True(s.T(), opt.rolledBack, "should stay rolled back")
	require.Equal(s.T(), 2, opt.lowDeliveryTicksInARow, "counter frozen after rollback")
}

func (s *TD3Suite) TestOptimizer_EmergencyRollback_Recovery() {
	opt := &Optimizer{
		minApplyInterval: DefaultMinApplyInterval,
	}

	bad := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{DeliveryRatio: 0.3, MsgRate: 100},
	}
	opt.checkEmergencyRollback(bad)
	require.Equal(s.T(), 1, opt.lowDeliveryTicksInARow)

	healthy := &broker.Metrics{
		BusinessMetrics: broker.BusinessMetrics{DeliveryRatio: 0.9, MsgRate: 100},
	}
	opt.checkEmergencyRollback(healthy)
	require.Equal(s.T(), 0, opt.lowDeliveryTicksInARow, "good tick should reset counter")
	require.False(s.T(), opt.rolledBack)
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
