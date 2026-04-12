package scheduler

import (
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type FIFOSuite struct {
	suite.Suite
}

func TestFIFOSuite(t *testing.T) { suite.Run(t, new(FIFOSuite)) }

func (s *FIFOSuite) TestSequentialRead() {
	tests := []struct {
		name     string
		messages int
	}{
		{"single", 1},
		{"ten", 10},
		{"hundred", 100},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			dir := s.T().TempDir()
			p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
			require.NoError(s.T(), err)
			defer p.Close()

			for i := 0; i < tc.messages; i++ {
				_, err := p.Append(broker.NewMessage(0, []byte("k"), []byte("v")))
				require.NoError(s.T(), err)
			}

			fifo := NewFIFO()
			offset := uint64(1)
			for i := 0; i < tc.messages; i++ {
				msg, nextOff, err := fifo.Next(p, offset)
				require.NoError(s.T(), err)
				require.NotNil(s.T(), msg)
				require.Equal(s.T(), offset+1, nextOff)
				offset = nextOff
			}
		})
	}
}

func (s *FIFOSuite) TestNoMessages() {
	dir := s.T().TempDir()
	p, err := broker.NewPartition(0, "test", dir, broker.ModeFIFO)
	require.NoError(s.T(), err)
	defer p.Close()

	fifo := NewFIFO()
	_, _, err = fifo.Next(p, 1)
	require.Error(s.T(), err)
}

func (s *FIFOSuite) TestEnqueueNoop() {
	fifo := NewFIFO()
	require.NotPanics(s.T(), func() {
		fifo.Enqueue(&broker.Message{}, 0)
	})
}
