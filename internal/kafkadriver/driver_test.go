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

func TestKafkaProducerRecoversFromInitFailure(t *testing.T) {
	broker := kafkaBroker()
	if !kafkaAvailable(broker) {
		t.Skipf("kafka not available at %s, set KAFKA_BROKER to run", broker)
	}

	h := New([]string{"127.0.0.1:1"})
	defer h.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := h.Produce(ctx, "no-such-topic", []byte("k"), []byte("x"))
	require.Error(t, err, "produce to unreachable broker should fail")

	h.Close()

	h2 := New([]string{broker})
	defer h2.Close()

	topic := "test-recovery-" + t.Name()
	ctx2, cancel2 := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel2()
	_ = h2.DeleteTopic(ctx2, topic)
	require.NoError(t, h2.CreateTopic(ctx2, topic, 1))
	defer func() { _ = h2.DeleteTopic(ctx2, topic) }()

	err = h2.Produce(ctx2, topic, []byte("k"), []byte("ok"))
	require.NoError(t, err, "produce to valid broker after reconnect should succeed")
}

func TestKafkaSingleClientNoRebalance(t *testing.T) {
	broker := kafkaBroker()
	if !kafkaAvailable(broker) {
		t.Skipf("kafka not available at %s, set KAFKA_BROKER to run", broker)
	}

	h := New([]string{broker})
	defer h.Close()

	topic := "test-single-client-" + t.Name()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	_ = h.DeleteTopic(ctx, topic)
	require.NoError(t, h.CreateTopic(ctx, topic, 4))
	defer func() { _ = h.DeleteTopic(ctx, topic) }()

	recs := make([]kgo.Record, 20)
	for i := range recs {
		recs[i] = kgo.Record{Key: []byte("k"), Value: []byte("v"), Partition: int32(i % 4)}
	}
	require.NoError(t, h.ProduceBatch(ctx, topic, recs))

	time.Sleep(500 * time.Millisecond)

	c0 := h.consumerFor(topic, "", 0)
	c1 := h.consumerFor(topic, "", 1)
	c2 := h.consumerFor(topic, "", 2)
	c3 := h.consumerFor(topic, "", 3)

	require.Same(t, c0, c1, "partitions 0 and 1 should share the same kgo.Client")
	require.Same(t, c1, c2, "partitions 1 and 2 should share the same kgo.Client")
	require.Same(t, c2, c3, "partitions 2 and 3 should share the same kgo.Client")
}
