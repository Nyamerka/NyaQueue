package main

import (
	"context"
	"log"
	"os"
	"strings"
	"time"

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

	addr := k.String("consumer.addr")
	if addr == "" {
		addr = "localhost:9090"
	}
	topic := k.String("consumer.topic")
	if topic == "" {
		topic = "test"
	}
	group := k.String("consumer.group")
	if group == "" {
		group = "default"
	}
	partition := k.Int("consumer.partition")

	client, err := transport.NewClient(addr)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer client.Close()

	ctx := context.Background()
	total := 0

	for {
		msgs, err := client.Consume(ctx, topic, group, int32(partition), 1<<20)
		if err != nil {
			log.Printf("consume: %v (retrying in 1s)", err)
			time.Sleep(time.Second)
			continue
		}

		for _, m := range msgs {
			total++
			log.Printf("[%d] offset=%d priority=%d key=%s value=%s",
				total, m.Offset, m.Priority, string(m.Key), string(m.Value))

			if err := client.Commit(ctx, topic, group, int32(partition), m.Offset+1); err != nil {
				log.Printf("commit offset %d: %v", m.Offset+1, err)
			}
		}

		if len(msgs) == 0 {
			time.Sleep(100 * time.Millisecond)
		}
	}
}
