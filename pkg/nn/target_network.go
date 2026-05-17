package nn

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/puzpuzpuz/xsync/v3"
)

// TargetNetwork maintains a separate set of weights for computing stable target Q-values
// (Mnih 2015). It periodically performs Polyak averaging (soft update) from the main context.
type TargetNetwork struct {
	mu        *xsync.RBMutex
	mainCtx   *context.Context
	targetCtx *context.Context
	tau       float64
	syncEvery int
	steps     int
}

type TargetNetworkOption func(*TargetNetwork)

func WithTau(tau float64) TargetNetworkOption {
	return func(tn *TargetNetwork) { tn.tau = tau }
}

func WithSyncEvery(n int) TargetNetworkOption {
	return func(tn *TargetNetwork) { tn.syncEvery = n }
}

// NewTargetNetwork creates a target network by cloning all trainable variables from mainCtx.
// Must be called after mainCtx variables are initialized (i.e., after a dummy forward pass).
func NewTargetNetwork(mainCtx *context.Context, opts ...TargetNetworkOption) *TargetNetwork {
	tn := &TargetNetwork{
		mu:        xsync.NewRBMutex(),
		mainCtx:   mainCtx,
		targetCtx: context.New(),
		tau:       0.005,
		syncEvery: 10,
	}
	for _, opt := range opts {
		opt(tn)
	}

	mainCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			if _, err := v.CloneToContext(tn.targetCtx); err != nil {
				panic("target network: failed to clone variable " + v.ScopeAndName() + ": " + err.Error())
			}
		}
	})

	return tn
}

// TargetCtx returns the target context for building target network exec-s.
func (tn *TargetNetwork) TargetCtx() *context.Context {
	return tn.targetCtx
}

// Step increments the internal counter and performs a soft update every syncEvery steps.
func (tn *TargetNetwork) Step() {
	tn.steps++
	if tn.steps%tn.syncEvery == 0 {
		tn.SoftUpdate()
	}
}

// SoftUpdate performs Polyak averaging: target_w = τ * main_w + (1-τ) * target_w.
func (tn *TargetNetwork) SoftUpdate() {
	tn.mu.Lock()
	defer tn.mu.Unlock()

	tau := tn.tau
	oneMinusTau := 1.0 - tau

	tn.mainCtx.EnumerateVariables(func(mainVar *context.Variable) {
		if !mainVar.Trainable {
			return
		}

		var targetVar *context.Variable
		tn.targetCtx.EnumerateVariables(func(tv *context.Variable) {
			if tv.ScopeAndName() == mainVar.ScopeAndName() {
				targetVar = tv
			}
		})
		if targetVar == nil {
			return
		}

		mainVal, err := mainVar.Value()
		if err != nil || mainVal == nil {
			return
		}
		targetVal, err := targetVar.Value()
		if err != nil || targetVal == nil {
			return
		}

		mainVal.MustConstFlatData(func(mainFlat any) {
			targetVal.MustMutableFlatData(func(targetFlat any) {
				switch m := mainFlat.(type) {
				case []float32:
					t := targetFlat.([]float32)
					for i := range m {
						t[i] = float32(tau)*m[i] + float32(oneMinusTau)*t[i]
					}
				case []float64:
					t := targetFlat.([]float64)
					for i := range m {
						t[i] = tau*m[i] + oneMinusTau*t[i]
					}
				}
			})
		})
	})
}
