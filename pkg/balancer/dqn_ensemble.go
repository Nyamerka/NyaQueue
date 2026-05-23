package balancer

import (
	"log"
	"math"
	"sync/atomic"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"

	_ "github.com/gomlx/gomlx/backends/simplego"
)

// dqnBooster is a lightweight residual Q-network added when environment drift
// is detected (GB-DQN style). It learns corrections on top of the frozen main Q.
type dqnBooster struct {
	numPartitions int
	hiddenSize    int
	stateSize     int

	backend backends.Backend
	ctx     *context.Context
	fwdExec *context.Exec

	frozenQ atomic.Pointer[[]float64]
}

func newBooster(numPartitions, stateSize int, lr float64) *dqnBooster {
	hiddenSize := max(64, stateSize*2)
	b := &dqnBooster{
		numPartitions: numPartitions,
		hiddenSize:    hiddenSize,
		stateSize:     stateSize,
	}

	b.backend = backends.MustNew()
	b.ctx = context.New()
	b.ctx.SetParams(map[string]any{
		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    lr * 2,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamWeightDecay: DefaultDQNWeightDecay,
	})

	b.fwdExec = context.MustNewExec(b.backend, b.ctx,
		func(ctx *context.Context, state *Node) *Node {
			return b.residualGraph(ctx, state)
		})
	dummy := make([]float64, stateSize)
	b.fwdExec.MustExec(dummy)

	return b
}

func (b *dqnBooster) residualGraph(ctx *context.Context, state *Node) *Node {
	x := InsertAxes(state, 0)
	h := layers.Dense(ctx.In("boost_l1"), x, true, b.hiddenSize)
	h = layers.LayerNormalization(ctx.In("boost_ln1"), h, -1).Done()
	h = activations.Relu(h)
	h = layers.Dense(ctx.In("boost_l2"), h, true, b.numPartitions)
	return Reshape(h, b.numPartitions)
}

func (b *dqnBooster) forward(state []float64) []float64 {
	result := b.fwdExec.MustExec1(state)
	return tensorToFloat64Slice(result)
}

// addResidual returns mainQ + boosterQ (frozen snapshot + residual correction).
func (b *dqnBooster) addResidual(mainQ []float64) []float64 {
	combined := make([]float64, len(mainQ))
	copy(combined, mainQ)
	if fq := b.frozenQ.Load(); fq != nil && len(*fq) == len(mainQ) {
		for i := range combined {
			combined[i] += (*fq)[i] - mainQ[i]
		}
	}
	return combined
}

// handleDrift is called when EWMA reward drops below dqnDriftThreshold of peak.
// Snapshots current Q-values into a booster and logs the event.
func (d *DQNBalancer) handleDrift() {
	cached := d.cachedQValues.Load()
	if cached == nil {
		return
	}

	b := d.booster.Load()
	if b != nil {
		return
	}

	snapshot := make([]float64, len(*cached))
	copy(snapshot, *cached)

	booster := newBooster(d.numPartitions, d.stateSize, d.lr)
	booster.frozenQ.Store(&snapshot)
	d.booster.Store(booster)

	log.Printf("[dqn-bal] drift detected, booster created (peak=%.4f, ewma=%.4f)",
		math.Float64frombits(d.peakReward.Load()), math.Float64frombits(d.ewmaReward.Load()))
}
