package nn

import (
	"math/rand/v2"
)

// OUNoise implements Ornstein-Uhlenbeck process for DDPG exploration
// with exponential sigma decay: sigma_t = max(sigmaFloor, sigma0 * decay^t).
type OUNoise struct {
	mu         float64
	theta      float64
	sigma0     float64
	sigma      float64
	sigmaDecay float64
	sigmaFloor float64
	state      []float64
	rng        *rand.Rand
	step       int
}

func NewOUNoise(size int, mu, theta, sigma float64) *OUNoise {
	state := make([]float64, size)
	for i := range state {
		state[i] = mu
	}
	return &OUNoise{
		mu:         mu,
		theta:      theta,
		sigma0:     sigma,
		sigma:      sigma,
		sigmaDecay: 1.0,
		sigmaFloor: 0.0,
		state:      state,
		rng:        rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
	}
}

func (ou *OUNoise) SetDecay(decay, floor float64) {
	ou.sigmaDecay = decay
	ou.sigmaFloor = floor
}

func (ou *OUNoise) Sample() []float64 {
	out := make([]float64, len(ou.state))
	for i := range ou.state {
		dx := ou.theta*(ou.mu-ou.state[i]) + ou.sigma*ou.rng.NormFloat64()
		ou.state[i] += dx
		out[i] = ou.state[i]
	}
	ou.step++
	if ou.sigmaDecay < 1.0 {
		ou.sigma = ou.sigma0 * pow(ou.sigmaDecay, ou.step)
		if ou.sigma < ou.sigmaFloor {
			ou.sigma = ou.sigmaFloor
		}
	}
	return out
}

func (ou *OUNoise) Reset() {
	for i := range ou.state {
		ou.state[i] = ou.mu
	}
}

func (ou *OUNoise) Size() int {
	return len(ou.state)
}

func (ou *OUNoise) Sigma() float64 {
	return ou.sigma
}

func pow(base float64, exp int) float64 {
	result := 1.0
	b := base
	e := exp
	for e > 0 {
		if e&1 == 1 {
			result *= b
		}
		b *= b
		e >>= 1
	}
	return result
}
