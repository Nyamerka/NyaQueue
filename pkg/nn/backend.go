package nn

import (
	"sync"

	"github.com/gomlx/gomlx/backends"

	_ "github.com/gomlx/gomlx/backends/default"
)

// SharedBackend returns the process-global XLA backend, creating it on first call.
// All DQN balancers, schedulers, and TD3 optimizers share this single backend
// to avoid native-memory leaks from per-instance churn (PJRT CPU client + device
// buffer arena + compiled program cache are expensive to allocate/destroy).
var SharedBackend = sync.OnceValue(func() backends.Backend {
	return backends.MustNew()
})
