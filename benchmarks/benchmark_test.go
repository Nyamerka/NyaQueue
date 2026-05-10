package benchmarks

import (
	"context"
	"fmt"
	"testing"

	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/scheduler"
)

func BenchmarkPublishFIFO(b *testing.B) {
	brk := setupBroker(b, broker.ModeFIFO)
	defer brk.Stop()

	msg := broker.NewMessage(0, []byte("key"), GenerateMessage(256))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := brk.Publish(context.Background(), "bench", msg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPublishPriority(b *testing.B) {
	brk := setupBroker(b, broker.ModeStrictPriority)
	defer brk.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		priority := uint8(i % 10)
		msg := broker.NewMessage(priority, []byte("key"), GenerateMessage(256))
		_, _, err := brk.Publish(context.Background(), "bench", msg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConsumeFIFO(b *testing.B) {
	brk := setupBroker(b, broker.ModeFIFO)
	defer brk.Stop()

	for i := 0; i < b.N; i++ {
		msg := broker.NewMessage(0, []byte(fmt.Sprintf("k%d", i)), GenerateMessage(256))
		brk.Publish(context.Background(), "bench", msg)
	}

	fifo := scheduler.NewFIFO()
	brk.SetScheduler("bench", fifo)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := brk.Consume("bench", "group", 0)
		if err != nil {
			// expected: messages spread across partitions
			continue
		}
	}
}

func BenchmarkRoundRobinSelect(b *testing.B) {
	rr := balancer.NewRoundRobin()
	key := []byte("test-key")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rr.SelectPartition("topic", key, 8)
	}
}

func BenchmarkWRRSelect(b *testing.B) {
	wrr := balancer.NewWeightedRoundRobin()
	wrr.OnMetrics(broker.Metrics{
		DerivedMetrics: broker.DerivedMetrics{
			PartitionLoads: []float64{0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7},
		},
	})
	key := []byte("test-key")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wrr.SelectPartition("topic", key, 8)
	}
}

func BenchmarkMessageMarshalUnmarshal(b *testing.B) {
	msg := broker.NewMessage(5, []byte("my-key"), GenerateMessage(1024))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data := msg.Marshal()
		_, err := broker.UnmarshalMessage(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func setupBroker(b *testing.B, mode broker.ScheduleMode) *broker.Broker {
	b.Helper()

	dir := b.TempDir()
	offsetStore, err := broker.NewOffsetStore(dir)
	if err != nil {
		b.Fatal(err)
	}

	cfg := broker.DefaultConfig()
	bal := balancer.NewRoundRobin()
	brk := broker.New(cfg, dir, bal, offsetStore)

	topicCfg := broker.DefaultTopicConfig()
	topicCfg.ScheduleMode = mode
	topicCfg.NumPartitions = 4

	if err := brk.CreateTopic("bench", topicCfg); err != nil {
		b.Fatal(err)
	}

	return brk
}

func BenchmarkScenarioTable(b *testing.B) {
	for _, sc := range AllScenarios() {
		b.Run(sc.Name, func(b *testing.B) {
			brk := setupBroker(b, broker.ModeFIFO)
			defer brk.Stop()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				priority := sc.SamplePriority()
				msg := broker.NewMessage(priority, []byte("k"), GenerateMessage(sc.MsgSize))
				brk.Publish(context.Background(), "bench", msg)
			}
		})
	}
}
