package nn

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/puzpuzpuz/xsync/v3"
)

// TargetNetwork maintains a separate set of weights for computing stable target Q-values
// (Mnih 2015). It periodically performs a hard copy from the main context.
type TargetNetwork struct {
	mu        *xsync.RBMutex
	mainCtx   *context.Context
	targetCtx *context.Context
	varMap    map[string]*context.Variable
	syncEvery int
	steps     int
}

type TargetNetworkOption func(*TargetNetwork)

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
		syncEvery: 25,
	}
	for _, opt := range opts {
		opt(tn)
	}

	tn.varMap = make(map[string]*context.Variable)
	mainCtx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable {
			cloned, err := v.CloneToContext(tn.targetCtx)
			if err != nil {
				panic("target network: failed to clone variable " + v.ScopeAndName() + ": " + err.Error())
			}
			tn.varMap[v.ScopeAndName()] = cloned
		}
	})

	return tn
}

// TargetCtx returns the target context for building target network exec-s.
func (tn *TargetNetwork) TargetCtx() *context.Context {
	return tn.targetCtx
}

// Step increments the internal counter and performs a hard sync every syncEvery steps.
func (tn *TargetNetwork) Step() {
	tn.steps++
	if tn.steps%tn.syncEvery == 0 {
		tn.HardSync()
	}
}

// HardSync copies all trainable variables from main to target network.
func (tn *TargetNetwork) HardSync() {
	tn.mu.Lock()
	defer tn.mu.Unlock()

	tn.mainCtx.EnumerateVariables(func(mainVar *context.Variable) {
		if !mainVar.Trainable {
			return
		}

		targetVar, ok := tn.varMap[mainVar.ScopeAndName()]
		if !ok {
			return
		}

		mainVal, err := mainVar.Value()
		if err != nil || mainVal == nil {
			return
		}

		mainVal.MustConstFlatData(func(mainFlat any) {
			targetVal, err := targetVar.Value()
			if err != nil || targetVal == nil {
				return
			}
			targetVal.MustMutableFlatData(func(targetFlat any) {
				switch m := mainFlat.(type) {
				case []float32:
					t := targetFlat.([]float32)
					copy(t, m)
				case []float64:
					t := targetFlat.([]float64)
					copy(t, m)
				}
			})
		})
	})
}
