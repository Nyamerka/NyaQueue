package nn

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OUNoiseSuite struct {
	suite.Suite
}

func TestOUNoiseSuite(t *testing.T) { suite.Run(t, new(OUNoiseSuite)) }

func (s *OUNoiseSuite) TestSampleSize() {
	tests := []struct {
		name string
		size int
	}{
		{"single", 1},
		{"ten", 10},
		{"large", 100},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			ou := NewOUNoise(tc.size, 0, 0.15, 0.2)
			sample := ou.Sample()
			require.Len(s.T(), sample, tc.size)
		})
	}
}

func (s *OUNoiseSuite) TestReset() {
	ou := NewOUNoise(5, 0, 0.15, 0.2)

	for i := 0; i < 100; i++ {
		ou.Sample()
	}

	ou.Reset()
	for _, v := range ou.state {
		require.Equal(s.T(), 0.0, v)
	}
}

func (s *OUNoiseSuite) TestMeanReversion() {
	ou := NewOUNoise(1, 0, 0.5, 0.01)
	ou.state[0] = 10.0

	for i := 0; i < 1000; i++ {
		ou.Sample()
	}

	require.InDelta(s.T(), 0.0, ou.state[0], 1.0, "state should revert towards mu=0")
}

func (s *OUNoiseSuite) TestSizeMethod() {
	ou := NewOUNoise(7, 0, 0.15, 0.2)
	require.Equal(s.T(), 7, ou.Size())
}
