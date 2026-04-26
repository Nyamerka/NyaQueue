package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type PartitionSuite struct {
	suite.Suite
	dir string
}

func TestPartitionSuite(t *testing.T) { suite.Run(t, new(PartitionSuite)) }

func (s *PartitionSuite) SetupTest() {
	s.dir = s.T().TempDir()
}

func (s *PartitionSuite) TestAppendAndRead() {
	tests := []struct {
		name     string
		mode     ScheduleMode
		messages int
	}{
		{"fifo_single", ModeFIFO, 1},
		{"fifo_many", ModeFIFO, 100},
		{"priority_mode", ModeStrictPriority, 50},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			p, err := NewPartition(0, "test", s.T().TempDir(), tc.mode, SyncNone)
			require.NoError(s.T(), err)
			defer p.Close()

			for i := 0; i < tc.messages; i++ {
				msg := NewMessage(uint8(i%10), []byte("k"), []byte("v"))
				off, err := p.Append(msg)
				require.NoError(s.T(), err)
				require.Equal(s.T(), uint64(i+1), off) // WAL starts at 1

				got, err := p.Read(off)
				require.NoError(s.T(), err)
				require.Equal(s.T(), msg.Header.Priority, got.Header.Priority)
			}

			require.Equal(s.T(), uint64(tc.messages), p.HighWaterMark())
		})
	}
}

func (s *PartitionSuite) TestPriorityIndexPopulated() {
	p, err := NewPartition(0, "test", s.T().TempDir(), ModeStrictPriority, SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	require.NotNil(s.T(), p.PriorityIndex())

	_, _ = p.Append(NewMessage(5, []byte("k"), []byte("v")))
	_, _ = p.Append(NewMessage(9, []byte("k"), []byte("v")))

	require.Equal(s.T(), 2, p.PriorityIndex().Len())
}

func (s *PartitionSuite) TestFIFONoPriorityIndex() {
	p, err := NewPartition(0, "test", s.T().TempDir(), ModeFIFO, SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	require.Nil(s.T(), p.PriorityIndex())
}

func (s *PartitionSuite) TestAppendBatch() {
	p, err := NewPartition(0, "test", s.T().TempDir(), ModeFIFO, SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	msgs := make([]*Message, 50)
	for i := range msgs {
		msgs[i] = NewMessage(uint8(i%10), []byte("k"), []byte("v"))
	}

	offsets, err := p.AppendBatch(msgs)
	require.NoError(s.T(), err)
	require.Len(s.T(), offsets, 50)

	for i, off := range offsets {
		require.Equal(s.T(), uint64(i+1), off)
		got, err := p.Read(off)
		require.NoError(s.T(), err)
		require.Equal(s.T(), msgs[i].Header.Priority, got.Header.Priority)
	}
	require.Equal(s.T(), uint64(50), p.HighWaterMark())
}

func (s *PartitionSuite) TestAppendBatchEmpty() {
	p, err := NewPartition(0, "test", s.T().TempDir(), ModeFIFO, SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	offsets, err := p.AppendBatch(nil)
	require.NoError(s.T(), err)
	require.Nil(s.T(), offsets)
}

func (s *PartitionSuite) TestAppendBatchPriorityIndex() {
	p, err := NewPartition(0, "test", s.T().TempDir(), ModeStrictPriority, SyncNone)
	require.NoError(s.T(), err)
	defer p.Close()

	msgs := []*Message{
		NewMessage(1, []byte("k"), []byte("a")),
		NewMessage(9, []byte("k"), []byte("b")),
		NewMessage(5, []byte("k"), []byte("c")),
	}
	_, err = p.AppendBatch(msgs)
	require.NoError(s.T(), err)
	require.Equal(s.T(), 3, p.PriorityIndex().Len())
}

func (s *PartitionSuite) TestRebuild() {
	dir := s.T().TempDir()

	p, err := NewPartition(0, "test", dir, ModeStrictPriority, SyncNone)
	require.NoError(s.T(), err)

	for i := 0; i < 10; i++ {
		_, err := p.Append(NewMessage(uint8(i), []byte("k"), []byte("v")))
		require.NoError(s.T(), err)
	}

	// Simulate: drain the index
	pi := p.PriorityIndex()
	for pi.Len() > 0 {
		pi.PopHighest()
	}
	require.Equal(s.T(), 0, pi.Len())

	committed := map[uint64]bool{1: true, 2: true, 3: true}
	err = p.Rebuild(1, func(off uint64) bool { return committed[off] })
	require.NoError(s.T(), err)
	require.Equal(s.T(), 7, pi.Len()) // 10 - 3 committed

	p.Close()
}
