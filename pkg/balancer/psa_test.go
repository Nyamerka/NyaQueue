package balancer

import (
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type PSASuite struct {
	suite.Suite
}

func TestPSASuite(t *testing.T) { suite.Run(t, new(PSASuite)) }

func (s *PSASuite) TestKeyBinding() {
	psa := NewPSA(4)

	p1 := psa.SelectPartition("t", []byte("key-a"), 4)
	p2 := psa.SelectPartition("t", []byte("key-a"), 4)
	require.Equal(s.T(), p1, p2, "same key must route to same partition")
}

func (s *PSASuite) TestDifferentKeys() {
	psa := NewPSA(4)

	results := make(map[int]bool)
	for i := 0; i < 100; i++ {
		key := []byte{byte(i)}
		p := psa.SelectPartition("t", key, 4)
		results[p] = true
	}

	require.Greater(s.T(), len(results), 1, "different keys should spread across partitions")
}

func (s *PSASuite) TestReleaseBindingsOnEmptyQueue() {
	psa := NewPSA(4)

	psa.SelectPartition("t", []byte("k1"), 4)
	psa.SelectPartition("t", []byte("k2"), 4)

	psa.OnMetrics(broker.Metrics{
		PartitionLoads: []float64{0, 0, 0, 0},
		QueueDepth:     []int{0, 0, 0, 0},
	})

	require.Len(s.T(), psa.bindings, 0, "bindings should be released for empty partitions")
	require.Len(s.T(), psa.free, 4, "all partitions should be free")
}

func (s *PSASuite) TestNoFreePartitions() {
	psa := NewPSA(2)

	psa.SelectPartition("t", []byte("a"), 2)
	psa.SelectPartition("t", []byte("b"), 2)

	p := psa.SelectPartition("t", []byte("c"), 2)
	require.GreaterOrEqual(s.T(), p, 0)
	require.Less(s.T(), p, 2)
}

func (s *PSASuite) TestHashKeyDeterministic() {
	h1 := hashKey([]byte("test-key"))
	h2 := hashKey([]byte("test-key"))
	require.Equal(s.T(), h1, h2)
}
