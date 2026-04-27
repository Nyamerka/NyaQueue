package kafkadriver

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/segmentio/kafka-go"
)

// TestKafkaHarnessConcurrentMapAccess pins the fix for the "concurrent map
// writes" panic that crashed the experiment runner under overload. Many
// producer/consumer goroutines simultaneously hit ProduceBatch/Consume right
// after DeleteTopic clears the caches, racing on the writers/readers maps.
//
// We point the harness at an unreachable address so WriteMessages/ReadMessage
// fail quickly — the goal is to exercise only the cache-population path, which
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
			_ = h.ProduceBatch(ctx, "race-topic", []kafka.Message{{Value: []byte("x")}})
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

func TestKafkaHarnessWriterCachedAcrossCalls(t *testing.T) {
	h := New([]string{"127.0.0.1:1"})
	defer h.Close()

	w1 := h.writerFor("cache-topic")
	w2 := h.writerFor("cache-topic")
	if w1 != w2 {
		t.Fatalf("expected cached writer, got distinct instances: %p vs %p", w1, w2)
	}
}

func TestKafkaHarnessReaderCachedAcrossCalls(t *testing.T) {
	h := New([]string{"127.0.0.1:1"})
	defer h.Close()

	r1 := h.readerFor("cache-topic", "g", 0, 1024)
	r2 := h.readerFor("cache-topic", "g", 0, 1024)
	if r1 != r2 {
		t.Fatalf("expected cached reader, got distinct instances: %p vs %p", r1, r2)
	}
}
