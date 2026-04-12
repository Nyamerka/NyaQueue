package optimizer

import (
	"math"
	"math/rand"
	"sync"

	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	ddpgHiddenSize    = 256
	ddpgTau           = 0.005
	ddpgFiniteDiffEps = 1e-3
)

type ddpgLayer struct {
	W *mat.Dense
	B []float64
}

func newDDPGLayer(out, in int) ddpgLayer {
	scale := math.Sqrt(2.0 / float64(in))
	data := make([]float64, out*in)
	for i := range data {
		data[i] = rand.NormFloat64() * scale
	}
	return ddpgLayer{
		W: mat.NewDense(out, in, data),
		B: make([]float64, out),
	}
}

func (l ddpgLayer) clone() ddpgLayer {
	r, c := l.W.Dims()
	data := make([]float64, r*c)
	copy(data, l.W.RawMatrix().Data)
	bc := make([]float64, len(l.B))
	copy(bc, l.B)
	return ddpgLayer{
		W: mat.NewDense(r, c, data),
		B: bc,
	}
}

func (l ddpgLayer) forward(input []float64) []float64 {
	rows, cols := l.W.Dims()
	s := input
	if len(s) < cols {
		padded := make([]float64, cols)
		copy(padded, s)
		s = padded
	} else if len(s) > cols {
		s = s[:cols]
	}
	sv := mat.NewVecDense(cols, s)
	rv := mat.NewVecDense(rows, nil)
	rv.MulVec(l.W, sv)
	out := rv.RawVector().Data
	floats.Add(out, l.B)
	return out
}

func (l ddpgLayer) forwardReLU(input []float64) []float64 {
	out := l.forward(input)
	for i, v := range out {
		if v < 0 {
			out[i] = 0
		}
	}
	return out
}

// DDPG implements Deep Deterministic Policy Gradient with manual backprop.
// Actor: state -> action (tanh-scaled), Critic: (state, action) -> Q-value.
type DDPG struct {
	mu sync.Mutex

	stateSize  int
	actionSize int
	lr         float64
	gamma      float64

	actor1, actor2, actor3                      ddpgLayer
	critic1, critic2, critic3                   ddpgLayer
	targetActor1, targetActor2, targetActor3    ddpgLayer
	targetCritic1, targetCritic2, targetCritic3 ddpgLayer

	replayBuffer *nn.ReplayBuffer
	noise        *nn.OUNoise

	criticBuf []float64
}

func NewDDPG(stateSize, actionSize int, lr float64) *DDPG {
	d := &DDPG{
		stateSize:    stateSize,
		actionSize:   actionSize,
		lr:           lr,
		gamma:        0.99,
		replayBuffer: nn.NewReplayBuffer(100000),
		noise:        nn.NewOUNoise(actionSize, 0, 0.15, 0.2),
		criticBuf:    make([]float64, stateSize+actionSize),
	}

	d.actor1 = newDDPGLayer(ddpgHiddenSize, stateSize)
	d.actor2 = newDDPGLayer(ddpgHiddenSize, ddpgHiddenSize)
	d.actor3 = newDDPGLayer(actionSize, ddpgHiddenSize)

	criticIn := stateSize + actionSize
	d.critic1 = newDDPGLayer(ddpgHiddenSize, criticIn)
	d.critic2 = newDDPGLayer(ddpgHiddenSize, ddpgHiddenSize)
	d.critic3 = newDDPGLayer(1, ddpgHiddenSize)

	d.targetActor1 = d.actor1.clone()
	d.targetActor2 = d.actor2.clone()
	d.targetActor3 = d.actor3.clone()
	d.targetCritic1 = d.critic1.clone()
	d.targetCritic2 = d.critic2.clone()
	d.targetCritic3 = d.critic3.clone()

	return d
}

func (d *DDPG) Act(state []float64) []float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	action := d.actorFwd(state, d.actor1, d.actor2, d.actor3)
	noise := d.noise.Sample()
	for i := range action {
		action[i] = math.Max(-1, math.Min(1, action[i]+noise[i]))
	}
	return action
}

func (d *DDPG) Store(state, action []float64, reward float64, nextState []float64, done bool) {
	ac := make([]float64, len(action))
	copy(ac, action)
	d.replayBuffer.Push(nn.Transition{
		State: state, Action: ac, Reward: reward,
		NextState: nextState, Done: done,
	})
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
			padded := make([]float64, d.actionSize)
			copy(padded, action)
			action = padded
		}

		nextAction := d.actorFwd(t.NextState, d.targetActor1, d.targetActor2, d.targetActor3)
		nextQ := d.criticFwd(t.NextState, nextAction, d.targetCritic1, d.targetCritic2, d.targetCritic3)

		targetQ := t.Reward
		if !t.Done {
			targetQ += d.gamma * nextQ
		}

		currentQ := d.criticFwd(t.State, action, d.critic1, d.critic2, d.critic3)
		d.updateCritic(t.State, action, targetQ-currentQ)
		d.updateActor(t.State)
	}

	softUpdateLayer(d.actor1, d.targetActor1, ddpgTau)
	softUpdateLayer(d.actor2, d.targetActor2, ddpgTau)
	softUpdateLayer(d.actor3, d.targetActor3, ddpgTau)
	softUpdateLayer(d.critic1, d.targetCritic1, ddpgTau)
	softUpdateLayer(d.critic2, d.targetCritic2, ddpgTau)
	softUpdateLayer(d.critic3, d.targetCritic3, ddpgTau)
}

func (d *DDPG) ResetNoise() {
	d.noise.Reset()
}

func (d *DDPG) actorFwd(state []float64, l1, l2, l3 ddpgLayer) []float64 {
	h1 := l1.forwardReLU(state)
	h2 := l2.forwardReLU(h1)
	out := l3.forward(h2)
	for i := range out {
		out[i] = math.Tanh(out[i])
	}
	return out
}

func (d *DDPG) fillCriticBuf(state, action []float64) []float64 {
	copy(d.criticBuf, state)
	copy(d.criticBuf[d.stateSize:], action)
	return d.criticBuf[:d.stateSize+len(action)]
}

func (d *DDPG) criticFwd(state, action []float64, l1, l2, l3 ddpgLayer) float64 {
	input := d.fillCriticBuf(state, action)
	h1 := l1.forwardReLU(input)
	h2 := l2.forwardReLU(h1)
	out := l3.forward(h2)
	if len(out) > 0 {
		return out[0]
	}
	return 0
}

func (d *DDPG) updateCritic(state, action []float64, tdError float64) {
	input := d.fillCriticBuf(state, action)
	h1 := d.critic1.forwardReLU(input)
	h2 := d.critic2.forwardReLU(h1)

	w3Row := d.critic3.W.RawRowView(0)
	w3Snap := make([]float64, len(w3Row))
	copy(w3Snap, w3Row)

	floats.AddScaled(w3Row, d.lr*tdError, h2)
	d.critic3.B[0] += d.lr * tdError

	dH2 := make([]float64, len(h2))
	for j := range dH2 {
		if j < len(w3Snap) {
			dH2[j] = tdError * w3Snap[j]
		}
		if h2[j] <= 0 {
			dH2[j] = 0
		}
	}

	w2Snap := cloneMatData(d.critic2.W)
	ddpgUpdateLayer(h1, dH2, d.critic2, d.lr)

	dH1 := matTransposeVecMul(w2Snap, dH2)
	for j := range dH1 {
		if j < len(h1) && h1[j] <= 0 {
			dH1[j] = 0
		}
	}
	ddpgUpdateLayer(input, dH1, d.critic1, d.lr)
}

func (d *DDPG) updateActor(state []float64) {
	h1 := d.actor1.forwardReLU(state)
	h2 := d.actor2.forwardReLU(h1)
	preAct := d.actor3.forward(h2)
	action := make([]float64, len(preAct))
	for i := range preAct {
		action[i] = math.Tanh(preAct[i])
	}

	qOrig := d.criticFwd(state, action, d.critic1, d.critic2, d.critic3)
	dQdA := make([]float64, d.actionSize)
	actionBuf := make([]float64, d.actionSize)
	for i := range action {
		copy(actionBuf, action)
		actionBuf[i] += ddpgFiniteDiffEps
		qPlus := d.criticFwd(state, actionBuf, d.critic1, d.critic2, d.critic3)
		dQdA[i] = (qPlus - qOrig) / ddpgFiniteDiffEps
	}

	dOut := make([]float64, d.actionSize)
	for i := range dOut {
		dOut[i] = dQdA[i] * (1 - action[i]*action[i])
	}

	w3Snap := cloneMatData(d.actor3.W)
	ddpgUpdateLayer(h2, dOut, d.actor3, d.lr)

	dH2 := matTransposeVecMul(w3Snap, dOut)
	for j := range dH2 {
		if j < len(h2) && h2[j] <= 0 {
			dH2[j] = 0
		}
	}

	w2Snap := cloneMatData(d.actor2.W)
	ddpgUpdateLayer(h1, dH2, d.actor2, d.lr)

	dH1 := matTransposeVecMul(w2Snap, dH2)
	for j := range dH1 {
		if j < len(h1) && h1[j] <= 0 {
			dH1[j] = 0
		}
	}

	ddpgUpdateLayer(state, dH1, d.actor1, d.lr)
}

func ddpgUpdateLayer(input, grad []float64, l ddpgLayer, lr float64) {
	rows, cols := l.W.Dims()
	for i := 0; i < rows && i < len(grad); i++ {
		row := l.W.RawRowView(i)
		n := len(input)
		if n > cols {
			n = cols
		}
		floats.AddScaled(row[:n], lr*grad[i], input[:n])
		l.B[i] += lr * grad[i]
	}
}

func softUpdateLayer(src, dst ddpgLayer, tau float64) {
	srcData := src.W.RawMatrix().Data
	dstData := dst.W.RawMatrix().Data
	floats.Scale(1-tau, dstData)
	floats.AddScaled(dstData, tau, srcData)
	floats.Scale(1-tau, dst.B)
	floats.AddScaled(dst.B, tau, src.B)
}

func cloneMatData(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	data := make([]float64, r*c)
	copy(data, m.RawMatrix().Data)
	return mat.NewDense(r, c, data)
}

// matTransposeVecMul computes W^T * v using BLAS.
func matTransposeVecMul(w *mat.Dense, v []float64) []float64 {
	rows, cols := w.Dims()
	n := len(v)
	if n > rows {
		n = rows
	}
	vv := mat.NewVecDense(rows, nil)
	for i := 0; i < n; i++ {
		vv.SetVec(i, v[i])
	}
	rv := mat.NewVecDense(cols, nil)
	rv.MulVec(w.T(), vv)
	return rv.RawVector().Data
}
