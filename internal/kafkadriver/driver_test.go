package kafkadriver

import (
	"context"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/twmb/franz-go/pkg/kgo"
)

// TestKafkaHarnessConcurrentMapAccess pins the fix for the "concurrent map
// writes" panic that crashed the experiment runner under overload. Many
// producer/consumer goroutines simultaneously hit ProduceBatch/Consume right
// after DeleteTopic clears the caches, racing on the consumer maps.
//
// We point the harness at an unreachable address so produce/consume fail
// quickly — the goal is to exercise only the cache-population path, which
// is where the race lived.
func TestKafkaHarnessConcurrentMapAccess(t *testing.T) {
	h := New([]string{"127.0.0.1:1"})
	defer h.Close()

	const goroutines = 64
	var wg sync.WaitGroup

	for i := 0; i < goroutines; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			_ = h.ProduceBatch(ctx, "race-topic", []kgo.Record{{Value: []byte("x")}})
		}()
		go func(partition int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			_, _ = h.Consume(ctx, "race-topic", "race-group", partition%4, 1024)
		}(i)
	}

	wg.Wait()
}

func kafkaBroker() string {
	if b := os.Getenv("KAFKA_BROKER"); b != "" {
		return b
	}
	return "127.0.0.1:9092"
}

func kafkaAvailable(addr string) bool {
	conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func TestKafkaFranzGoBatchedConsume(t *testing.T) {
	broker := kafkaBroker()
	if !kafkaAvailable(broker) {
		t.Skipf("kafka not available at %s, set KAFKA_BROKER to run", broker)
	}

	h := New([]string{broker})
	defer h.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	topic := "test-batched-consume-" + t.Name()
	_ = h.DeleteTopic(ctx, topic)
	require.NoError(t, h.CreateTopic(ctx, topic, 1))
	defer func() { _ = h.DeleteTopic(ctx, topic) }()

	const numMessages = 20
	recs := make([]kgo.Record, numMessages)
	for i := range recs {
		recs[i] = kgo.Record{Key: []byte("k"), Value: []byte("v")}
	}
	require.NoError(t, h.ProduceBatch(ctx, topic, recs))

	time.Sleep(500 * time.Millisecond)

	msgs, err := h.Consume(ctx, topic, "", 0, 1<<20)
	require.NoError(t, err)
	require.Greater(t, len(msgs), 1,
		"single Consume call should return > 1 message when multiple are available")
}

func TestKafkaHarnessConsumerCachedAcrossCalls(t *testing.T) {
	h := New([]string{"127.0.0.1:1"})
	defer h.Close()

	c1 := h.consumerFor("cache-topic", "g", 0)
	c2 := h.consumerFor("cache-topic", "g", 0)
	if c1 != c2 {
		t.Fatalf("expected cached consumer, got distinct instances: %p vs %p", c1, c2)
	}
}
