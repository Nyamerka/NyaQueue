package nn

import "math/rand"

// OUNoise implements Ornstein-Uhlenbeck process for DDPG exploration.
//
// dx = theta * (mu - x) * dt + sigma * dW
// Produces temporally correlated noise, better than i.i.d. Gaussian for continuous control.
type OUNoise struct {
	mu    float64
	theta float64
	sigma float64
	state []float64
}

func NewOUNoise(size int, mu, theta, sigma float64) *OUNoise {
	state := make([]float64, size)
	for i := range state {
		state[i] = mu
	}
	return &OUNoise{
		mu:    mu,
		theta: theta,
		sigma: sigma,
		state: state,
	}
}

func (ou *OUNoise) Sample() []float64 {
	out := make([]float64, len(ou.state))
	for i := range ou.state {
		dx := ou.theta*(ou.mu-ou.state[i]) + ou.sigma*rand.NormFloat64()
		ou.state[i] += dx
		out[i] = ou.state[i]
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
