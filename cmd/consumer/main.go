package main

import (
	"context"
	"flag"
	"log"
	"os/signal"
	"syscall"
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/Nyamerka/NyaQueue/pkg/metrics"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	k := koanf.New(".")
	if err := k.Load(file.Provider(*configPath), yaml.Parser()); err != nil {
		log.Printf("config file not found (%s), using defaults", *configPath)
	}
	if err := k.Load(env.Provider("NYAQUEUE_", ".", func(s string) string { return s }), nil); err != nil {
		log.Printf("env load: %v", err)
	}

	cfg := DefaultConfig()
	if err := k.UnmarshalWithConf("consumer", &cfg, koanf.UnmarshalConf{Tag: "koanf"}); err != nil {
		log.Fatalf("config unmarshal: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config validation: %v", err)
	}

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := registerMetrics(reg)

	metricsSrv, err := metrics.Serve(cfg.MetricsAddr, reg)
	if err != nil {
		log.Fatalf("metrics serve: %v", err)
	}

	client, err := transport.NewClient(cfg.Addr)
	if err != nil {
		log.Fatalf("connect %s: %v", cfg.Addr, err)
	}
	defer client.Close()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	log.Printf("consumer: addr=%s topic=%s group=%s workers=%d metrics=%s",
		cfg.Addr, cfg.Topic, cfg.Group, cfg.Workers, metricsSrv.Addr())

	if err := Run(ctx, cfg, client, m); err != nil {
		log.Fatalf("run: %v", err)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := metricsSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("metrics shutdown: %v", err)
	}
	log.Println("bye")
}
