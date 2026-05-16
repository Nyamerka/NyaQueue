package optimizer

import (
	"sync"
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type DDPGSuite struct {
	suite.Suite
}

func TestDDPGSuite(t *testing.T) { suite.Run(t, new(DDPGSuite)) }

func (s *DDPGSuite) TestAct() {
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
			d := NewDDPG(tc.stateSize, tc.actionSize, 0.001, 32)
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

func (s *DDPGSuite) TestStoreAndTrain() {
	d := NewDDPG(5, 3, 0.001, 32)

	for i := 0; i < 200; i++ {
		state := []float64{float64(i), 0.1, 0.2, 0.3, 0.4}
		action := []float64{0.1, -0.2, 0.3}
		nextState := []float64{float64(i + 1), 0.2, 0.3, 0.4, 0.5}
		d.Store(state, action, 1.0, nextState, false)
	}

	require.NotPanics(s.T(), func() { d.Train(32) })
}

func (s *DDPGSuite) TestTrainInsufficientData() {
	d := NewDDPG(5, 3, 0.001, 32)
	d.Store([]float64{1, 2, 3, 4, 5}, []float64{0.1, 0.2, 0.3}, 1.0, []float64{2, 3, 4, 5, 6}, false)

	require.NotPanics(s.T(), func() { d.Train(64) })
}

func (s *DDPGSuite) TestResetNoise() {
	d := NewDDPG(5, 3, 0.001, 32)
	require.NotPanics(s.T(), func() { d.ResetNoise() })
}

func (s *DDPGSuite) TestActorForwardOutputInRange() {
	d := NewDDPG(5, 3, 0.001, 32)
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

func (s *DDPGSuite) TestCriticBatchedReturnsCorrectShape() {
	d := NewDDPG(5, 3, 0.001, 32)

	states := tensors.FromFlatDataAndDimensions([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1}, 2, 5)
	actions := tensors.FromFlatDataAndDimensions([]float64{0.1, 0.2, 0.3, 0.3, 0.2, 0.1}, 2, 3)

	d.mu.Lock()
	result := d.targetCriticExec.MustExec1(states, actions)
	d.mu.Unlock()

	out := tensorToFloat64(result)
	require.Len(s.T(), out, 2, "batched critic should return one Q-value per sample")
}

func (s *DDPGSuite) TestTrainUpdatesActorWeights() {
	d := NewDDPG(5, 3, 0.01, 32)

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
	d.Train(32)

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

func (s *DDPGSuite) TestSoftUpdateChangesTarget() {
	d := NewDDPG(5, 3, 0.001, 32)

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
	d.Train(32)

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

func (s *DDPGSuite) TestBatchedTrainMultipleSteps() {
	d := NewDDPG(5, 3, 0.001, 32)

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

func (s *DDPGSuite) TestSoftUpdateDtypeRobust() {
	d := NewDDPG(5, 3, 0.001, 32)
	require.NotPanics(s.T(), func() {
		d.mu.Lock()
		d.softUpdate()
		d.mu.Unlock()
	})
	require.Greater(s.T(), len(d.varPairs), 0, "varPairs should be populated")
}

func (s *DDPGSuite) TestSoftUpdateBlendedWeights() {
	d := NewDDPG(5, 3, 0.001, 32)

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
		expectedBuf[i] = (1-ddpgTau)*tgtFlat[i] + ddpgTau*srcFlat[i]
	}

	d.softUpdate()

	newTgtVal := p.tgt.MustValue()
	newTgtFlat := flattenTensorValue(newTgtVal.Value())
	for i := range expectedBuf {
		require.InDelta(s.T(), expectedBuf[i], newTgtFlat[i], 1e-9,
			"softUpdate must produce exact blended value (idx %d)", i)
	}
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

func (s *DDPGSuite) TestStoreDeepCopies() {
	d := NewDDPG(5, 3, 0.001, 32)
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

func (s *DDPGSuite) TestPreAllocatedBuffersProperty() {
	batchSize := 32
	d := NewDDPG(5, 3, 0.001, batchSize)
	require.GreaterOrEqual(s.T(), len(d.statesBuf), batchSize*d.stateSize)
	require.GreaterOrEqual(s.T(), len(d.actionsBuf), batchSize*d.actionSize)
	require.GreaterOrEqual(s.T(), len(d.rewardsBuf), batchSize)
	require.GreaterOrEqual(s.T(), len(d.targetQBuf), batchSize)
}

func (s *DDPGSuite) TestNoiseDecay() {
	d := NewDDPG(5, 3, 0.001, 32)
	d.SetNoiseDecay(0.9, 0.01)

	initial := d.noise.Sigma()
	for i := 0; i < 50; i++ {
		d.noise.Sample()
	}
	after := d.noise.Sigma()
	require.Less(s.T(), after, initial, "sigma should decay over steps")
	require.GreaterOrEqual(s.T(), after, 0.01, "sigma should not go below floor")
}

func (s *DDPGSuite) TestConcurrentActAndTrain() {
	d := NewDDPG(5, 3, 0.001, 32)

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

func (s *DDPGSuite) TestOptimizer_EmergencyRollback() {
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

func (s *DDPGSuite) TestOptimizer_EmergencyRollback_Recovery() {
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
