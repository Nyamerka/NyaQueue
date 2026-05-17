package nn

import (
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type NormalizerSuite struct {
	suite.Suite
}

func TestNormalizerSuite(t *testing.T) { suite.Run(t, new(NormalizerSuite)) }

func (s *NormalizerSuite) TestNormalizerZeroMeanUnitVariance() {
	rn := NewRunningNormalizer(2)
	rng := rand.New(rand.NewPCG(42, 0))

	trueMean := []float64{5.0, -3.0}
	trueStd := []float64{3.0, 0.5}

	for i := 0; i < 10_000; i++ {
		x := []float64{
			rng.NormFloat64()*trueStd[0] + trueMean[0],
			rng.NormFloat64()*trueStd[1] + trueMean[1],
		}
		rn.Observe(x)
	}

	var sumNorm [2]float64
	var sum2Norm [2]float64
	const N = 5000
	for i := 0; i < N; i++ {
		x := []float64{
			rng.NormFloat64()*trueStd[0] + trueMean[0],
			rng.NormFloat64()*trueStd[1] + trueMean[1],
		}
		norm := rn.Normalize(x)
		for j := 0; j < 2; j++ {
			sumNorm[j] += norm[j]
			sum2Norm[j] += norm[j] * norm[j]
		}
	}

	for j := 0; j < 2; j++ {
		meanNorm := sumNorm[j] / N
		varNorm := sum2Norm[j]/N - meanNorm*meanNorm
		require.InDelta(s.T(), 0.0, meanNorm, 0.1, "normalized mean should be ~0")
		require.InDelta(s.T(), 1.0, math.Sqrt(varNorm), 0.15, "normalized std should be ~1")
	}
}

func (s *NormalizerSuite) TestNormalizerConcurrentSafe() {
	rn := NewRunningNormalizer(4)
	var wg sync.WaitGroup

	for g := 0; g < 50; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(id), 0))
			for i := 0; i < 200; i++ {
				x := make([]float64, 4)
				for j := range x {
					x[j] = rng.NormFloat64()
				}
				rn.Observe(x)
			}
		}(g)
	}

	for g := 0; g < 50; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(id+100), 0))
			for i := 0; i < 200; i++ {
				x := make([]float64, 4)
				for j := range x {
					x[j] = rng.NormFloat64()
				}
				_ = rn.Normalize(x)
			}
		}(g)
	}

	wg.Wait()
}

func (s *NormalizerSuite) TestNormalizerSingleObservation() {
	rn := NewRunningNormalizer(3)
	rn.Observe([]float64{1.0, 2.0, 3.0})

	result := rn.Normalize([]float64{5.0, 10.0, 15.0})
	for _, v := range result {
		require.Equal(s.T(), 0.0, v, "with single observation, normalize should return 0")
	}
}
