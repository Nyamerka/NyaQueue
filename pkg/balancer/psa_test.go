package balancer

import (
	"sync"
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
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0, 0, 0, 0},
			QueueDepth:     []int{0, 0, 0, 0},
		},
	})

	require.Equal(s.T(), 0, psa.bindings.Len(), "bindings should be released for empty partitions")
	require.Equal(s.T(), 4, psa.free.Size(), "all partitions should be free")
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

func (s *PSASuite) TestConcurrentSelectAndOnMetrics() {
	psa := NewPSA(4)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				psa.SelectPartition("t", []byte{byte(i), byte(j)}, 4)
			}
		}(i)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 100; j++ {
			psa.OnMetrics(broker.Metrics{
				DerivedMetrics: broker.DerivedMetrics{
					PartitionLoads: []float64{0.1, 0.2, 0.3, 0.4},
					QueueDepth:     []int{10, 20, 30, 40},
				},
			})
		}
	}()

	wg.Wait()
}

func (s *PSASuite) TestEvictionCountIncrementsOnOverflow() {
	numParts := 128
	psa := NewPSA(numParts)

	total := defaultPSAMaxBindings + 500
	batchSize := numParts
	for i := 0; i < total; i += batchSize {
		for j := 0; j < batchSize && i+j < total; j++ {
			k := i + j
			psa.SelectPartition("t", []byte{byte(k >> 24), byte(k >> 16), byte(k >> 8), byte(k)}, numParts)
		}
		depths := make([]int, numParts)
		psa.OnMetrics(broker.Metrics{
			DerivedMetrics: broker.DerivedMetrics{
				PartitionLoads: make([]float64, numParts),
				QueueDepth:     depths,
			},
		})
	}

	require.Greater(s.T(), psa.EvictionCount(), int64(0))
}
