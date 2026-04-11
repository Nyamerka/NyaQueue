package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/internal/kafkadriver"
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

	target := k.String("loadgen.target")
	if target == "" {
		target = "nyaqueue"
	}
	addr := k.String("loadgen.addr")
	if addr == "" {
		addr = "localhost:9090"
	}
	scenarioName := k.String("loadgen.scenario")
	if scenarioName == "" {
		scenarioName = "uniform"
	}
	topic := k.String("loadgen.topic")
	if topic == "" {
		topic = "bench"
	}
	producers := k.Int("loadgen.producers")
	if producers <= 0 {
		producers = 4
	}
	durationStr := k.String("loadgen.duration")
	duration := 30 * time.Second
	if durationStr != "" {
		if d, err := time.ParseDuration(durationStr); err == nil {
			duration = d
		}
	}

	sc := selectScenario(scenarioName)
	if sc == nil {
		log.Fatalf("unknown scenario: %s", scenarioName)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		cancel()
	}()

	var produced int64
	start := time.Now()

	switch strings.ToLower(target) {
	case "nyaqueue":
		client, err := transport.NewClient(addr)
		if err != nil {
			log.Fatalf("connect nyaqueue: %v", err)
		}
		defer client.Close()

		deadline := time.After(duration)
		for i := 0; i < producers; i++ {
			go func(id int) {
				for {
					select {
					case <-deadline:
						return
					case <-ctx.Done():
						return
					default:
						key := []byte(fmt.Sprintf("p%d", id))
						value := benchmarks.GenerateMessage(sc.MsgSize)
						priority := sc.SamplePriority()
						_, _, err := client.Produce(ctx, topic, key, value, uint32(priority))
						if err == nil {
							atomic.AddInt64(&produced, 1)
						}
					}
				}
			}(i)
		}

		select {
		case <-deadline:
		case <-ctx.Done():
		}

	case "kafka":
		kfk := kafkadriver.New(strings.Split(addr, ","))
		defer kfk.Close()

		deadline := time.After(duration)
		for i := 0; i < producers; i++ {
			go func(id int) {
				for {
					select {
					case <-deadline:
						return
					case <-ctx.Done():
						return
					default:
						key := []byte(fmt.Sprintf("p%d", id))
						value := benchmarks.GenerateMessage(sc.MsgSize)
						err := kfk.Produce(ctx, topic, key, value)
						if err == nil {
							atomic.AddInt64(&produced, 1)
						}
					}
				}
			}(i)
		}

		select {
		case <-deadline:
		case <-ctx.Done():
		}

	default:
		log.Fatalf("unknown target: %s (use nyaqueue or kafka)", target)
	}

	elapsed := time.Since(start)
	total := atomic.LoadInt64(&produced)
	rate := float64(total) / elapsed.Seconds()
	log.Printf("done: %d messages in %v (%.0f msg/sec)", total, elapsed, rate)
}

func selectScenario(name string) *benchmarks.Scenario {
	for _, sc := range benchmarks.AllScenarios() {
		if sc.Name == name {
			return &sc
		}
	}
	return nil
}
