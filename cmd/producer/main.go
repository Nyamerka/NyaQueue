package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

func main() {
	k := koanf.New(".")

	configPath := "config.yaml"
	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		configPath = os.Args[1]
	}

	_ = k.Load(file.Provider(configPath), yaml.Parser())
	_ = k.Load(env.Provider("NYAQUEUE_", ".", func(s string) string { return s }), nil)

	addr := k.String("producer.addr")
	if addr == "" {
		addr = "localhost:9090"
	}
	topic := k.String("producer.topic")
	if topic == "" {
		topic = "test"
	}
	n := k.Int("producer.messages")
	if n <= 0 {
		n = 100
	}
	priority := k.Int("producer.priority")

	client, err := transport.NewClient(addr)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	if err := client.CreateTopic(ctx, topic, 4, pb.ScheduleMode_FIFO); err != nil {
		log.Printf("create topic (may already exist): %v", err)
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		key := []byte(fmt.Sprintf("key-%d", i))
		value := []byte(fmt.Sprintf("message-%d", i))

		part, off, err := client.Produce(ctx, topic, key, value, uint32(priority))
		if err != nil {
			log.Printf("produce %d: %v", i, err)
			continue
		}

		if i%1000 == 0 || i == n-1 {
			log.Printf("sent %d/%d → partition=%d offset=%d", i+1, n, part, off)
		}
	}

	elapsed := time.Since(start)
	rate := float64(n) / elapsed.Seconds()
	log.Printf("done: %d messages in %v (%.0f msg/sec)", n, elapsed, rate)
}
