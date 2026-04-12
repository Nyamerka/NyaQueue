package optimizer

import (
	"math"
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
)

const (
	ddpgHiddenSize = 256
	ddpgTau        = 0.005 // soft update coefficient
)

// DDPG implements Deep Deterministic Policy Gradient with manual backprop.
// Actor: state -> action (tanh-scaled), Critic: (state, action) -> Q-value.
type DDPG struct {
	mu sync.Mutex

	stateSize  int
	actionSize int
	lr         float64
	gamma      float64

	actorW1, actorW2, actorW3 [][]float64
	actorB1, actorB2, actorB3 []float64

	criticW1, criticW2, criticW3 [][]float64
	criticB1, criticB2, criticB3 []float64

	targetActorW1, targetActorW2, targetActorW3    [][]float64
	targetActorB1, targetActorB2, targetActorB3    []float64
	targetCriticW1, targetCriticW2, targetCriticW3 [][]float64
	targetCriticB1, targetCriticB2, targetCriticB3 []float64

	replayBuffer *nn.ReplayBuffer
	noise        *nn.OUNoise
}

func NewDDPG(stateSize, actionSize int, lr float64) *DDPG {
	d := &DDPG{
		stateSize:    stateSize,
		actionSize:   actionSize,
		lr:           lr,
		gamma:        0.99,
		replayBuffer: nn.NewReplayBuffer(100000),
		noise:        nn.NewOUNoise(actionSize, 0, 0.15, 0.2),
	}

	d.actorW1, d.actorB1 = initLayer(ddpgHiddenSize, stateSize)
	d.actorW2, d.actorB2 = initLayer(ddpgHiddenSize, ddpgHiddenSize)
	d.actorW3, d.actorB3 = initLayer(actionSize, ddpgHiddenSize)

	criticInput := stateSize + actionSize
	d.criticW1, d.criticB1 = initLayer(ddpgHiddenSize, criticInput)
	d.criticW2, d.criticB2 = initLayer(ddpgHiddenSize, ddpgHiddenSize)
	d.criticW3, d.criticB3 = initLayer(1, ddpgHiddenSize)

	d.targetActorW1, d.targetActorB1 = cloneLayer(d.actorW1, d.actorB1)
	d.targetActorW2, d.targetActorB2 = cloneLayer(d.actorW2, d.actorB2)
	d.targetActorW3, d.targetActorB3 = cloneLayer(d.actorW3, d.actorB3)
	d.targetCriticW1, d.targetCriticB1 = cloneLayer(d.criticW1, d.criticB1)
	d.targetCriticW2, d.targetCriticB2 = cloneLayer(d.criticW2, d.criticB2)
	d.targetCriticW3, d.targetCriticB3 = cloneLayer(d.criticW3, d.criticB3)

	return d
}

func (d *DDPG) Act(state []float64) []float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	action := d.actorForward(state, d.actorW1, d.actorB1, d.actorW2, d.actorB2, d.actorW3, d.actorB3)
	noise := d.noise.Sample()

	for i := range action {
		action[i] += noise[i]
		action[i] = math.Max(-1, math.Min(1, action[i]))
	}

	return action
}

func (d *DDPG) Store(state []float64, action []float64, reward float64, nextState []float64, done bool) {
	actionCopy := make([]float64, len(action))
	copy(actionCopy, action)
	t := nn.Transition{
		State:     state,
		Action:    actionCopy,
		Reward:    reward,
		NextState: nextState,
		Done:      done,
	}
	d.replayBuffer.Push(t)
}

func (d *DDPG) Train(batchSize int) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.replayBuffer.Len() < batchSize {
		return
	}

	batch := d.replayBuffer.Sample(batchSize)

	for _, t := range batch {
		action := t.Action
		if len(action) < d.actionSize {
			action = make([]float64, d.actionSize)
			copy(action, t.Action)
		}

		nextAction := d.actorForward(t.NextState,
			d.targetActorW1, d.targetActorB1,
			d.targetActorW2, d.targetActorB2,
			d.targetActorW3, d.targetActorB3)
		nextQ := d.criticForward(t.NextState, nextAction,
			d.targetCriticW1, d.targetCriticB1,
			d.targetCriticW2, d.targetCriticB2,
			d.targetCriticW3, d.targetCriticB3)

		targetQ := t.Reward
		if !t.Done {
			targetQ += d.gamma * nextQ
		}

		currentQ := d.criticForward(t.State, action,
			d.criticW1, d.criticB1,
			d.criticW2, d.criticB2,
			d.criticW3, d.criticB3)
		criticError := targetQ - currentQ
		d.updateCritic(t.State, action, criticError)

		d.updateActor(t.State)
	}

	softUpdate(d.actorW1, d.targetActorW1, ddpgTau)
	softUpdate(d.actorW2, d.targetActorW2, ddpgTau)
	softUpdate(d.actorW3, d.targetActorW3, ddpgTau)
	softUpdateBias(d.actorB1, d.targetActorB1, ddpgTau)
	softUpdateBias(d.actorB2, d.targetActorB2, ddpgTau)
	softUpdateBias(d.actorB3, d.targetActorB3, ddpgTau)

	softUpdate(d.criticW1, d.targetCriticW1, ddpgTau)
	softUpdate(d.criticW2, d.targetCriticW2, ddpgTau)
	softUpdate(d.criticW3, d.targetCriticW3, ddpgTau)
	softUpdateBias(d.criticB1, d.targetCriticB1, ddpgTau)
	softUpdateBias(d.criticB2, d.targetCriticB2, ddpgTau)
	softUpdateBias(d.criticB3, d.targetCriticB3, ddpgTau)
}

func (d *DDPG) ResetNoise() {
	d.noise.Reset()
}

func (d *DDPG) actorForward(state []float64,
	w1 [][]float64, b1 []float64,
	w2 [][]float64, b2 []float64,
	w3 [][]float64, b3 []float64,
) []float64 {
	h1 := linearReLU(state, w1, b1)
	h2 := linearReLU(h1, w2, b2)
	out := linearForward(h2, w3, b3)
	for i := range out {
		out[i] = math.Tanh(out[i])
	}
	return out
}

func (d *DDPG) criticForward(state, action []float64,
	w1 [][]float64, b1 []float64,
	w2 [][]float64, b2 []float64,
	w3 [][]float64, b3 []float64,
) float64 {
	input := append(state, action...)
	h1 := linearReLU(input, w1, b1)
	h2 := linearReLU(h1, w2, b2)
	out := linearForward(h2, w3, b3)
	if len(out) > 0 {
		return out[0]
	}
	return 0
}

func (d *DDPG) updateCritic(state, action []float64, tdError float64) {
	input := append(state, action...)
	h1 := linearReLU(input, d.criticW1, d.criticB1)
	h2 := linearReLU(h1, d.criticW2, d.criticB2)

	for j := 0; j < len(h2) && j < len(d.criticW3[0]); j++ {
		d.criticW3[0][j] += d.lr * tdError * h2[j]
	}
	d.criticB3[0] += d.lr * tdError

	dH2 := make([]float64, len(h2))
	for j := range dH2 {
		if j < len(d.criticW3[0]) {
			dH2[j] = tdError * d.criticW3[0][j]
		}
		if h2[j] <= 0 {
			dH2[j] = 0
		}
	}
	updateLinear(h1, dH2, d.criticW2, d.criticB2, d.lr)

	dH1 := make([]float64, len(h1))
	for j := range dH1 {
		for k := range dH2 {
			if j < len(d.criticW2[k]) {
				dH1[j] += dH2[k] * d.criticW2[k][j]
			}
		}
		if h1[j] <= 0 {
			dH1[j] = 0
		}
	}
	updateLinear(input, dH1, d.criticW1, d.criticB1, d.lr)
}

func (d *DDPG) updateActor(state []float64) {
	action := d.actorForward(state, d.actorW1, d.actorB1, d.actorW2, d.actorB2, d.actorW3, d.actorB3)

	eps := 0.001
	for i := range action {
		actionPlus := make([]float64, len(action))
		copy(actionPlus, action)
		actionPlus[i] += eps

		qPlus := d.criticForward(state, actionPlus,
			d.criticW1, d.criticB1,
			d.criticW2, d.criticB2,
			d.criticW3, d.criticB3)
		qOrig := d.criticForward(state, action,
			d.criticW1, d.criticB1,
			d.criticW2, d.criticB2,
			d.criticW3, d.criticB3)

		dQdA := (qPlus - qOrig) / eps

		h1 := linearReLU(state, d.actorW1, d.actorB1)
		h2 := linearReLU(h1, d.actorW2, d.actorB2)

		tanhDeriv := 1 - action[i]*action[i]
		grad := dQdA * tanhDeriv

		for j := range h2 {
			if j < len(d.actorW3[i]) {
				d.actorW3[i][j] += d.lr * grad * h2[j]
			}
		}
		d.actorB3[i] += d.lr * grad
	}
}

func initLayer(outSize, inSize int) ([][]float64, []float64) {
	scale := math.Sqrt(2.0 / float64(inSize))
	w := make([][]float64, outSize)
	b := make([]float64, outSize)
	for i := range w {
		w[i] = make([]float64, inSize)
		for j := range w[i] {
			w[i][j] = rand.NormFloat64() * scale
		}
	}
	return w, b
}

func cloneLayer(w [][]float64, b []float64) ([][]float64, []float64) {
	wc := make([][]float64, len(w))
	for i := range w {
		wc[i] = make([]float64, len(w[i]))
		copy(wc[i], w[i])
	}
	bc := make([]float64, len(b))
	copy(bc, b)
	return wc, bc
}

func linearForward(input []float64, w [][]float64, b []float64) []float64 {
	out := make([]float64, len(w))
	for i := range w {
		n := len(input)
		if len(w[i]) < n {
			n = len(w[i])
		}
		out[i] = b[i] + floats.Dot(w[i][:n], input[:n])
	}
	return out
}

func linearReLU(input []float64, w [][]float64, b []float64) []float64 {
	out := linearForward(input, w, b)
	for i := range out {
		if out[i] < 0 {
			out[i] = 0
		}
	}
	return out
}

func updateLinear(input, grad []float64, w [][]float64, b []float64, lr float64) {
	scaled := make([]float64, len(input))
	for i := range grad {
		if i >= len(w) {
			break
		}
		n := len(input)
		if len(w[i]) < n {
			n = len(w[i])
		}
		step := lr * grad[i]
		floats.ScaleTo(scaled[:n], step, input[:n])
		floats.Add(w[i][:n], scaled[:n])
		b[i] += step
	}
}

func softUpdate(src, dst [][]float64, tau float64) {
	for i := range src {
		floats.Scale(1-tau, dst[i])
		floats.AddScaled(dst[i], tau, src[i])
	}
}

func softUpdateBias(src, dst []float64, tau float64) {
	floats.Scale(1-tau, dst)
	floats.AddScaled(dst, tau, src)
}
