package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/Nyamerka/NyaQueue/internal/app"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	k := koanf.New(".")

	if err := k.Load(file.Provider(*configPath), yaml.Parser()); err != nil {
		log.Printf("config file not found (%s), using defaults", *configPath)
	}

	if err := k.Load(env.Provider("NYAQUEUE_", ".", func(s string) string {
		return s
	}), nil); err != nil {
		log.Printf("env load: %v", err)
	}

	addr := k.String("server.addr")
	if addr == "" {
		addr = ":9090"
	}
	dataDir := k.String("server.data_dir")
	if dataDir == "" {
		dataDir = "data"
	}

	cfg := broker.DefaultConfig()
	if err := k.UnmarshalWithConf("broker", &cfg, koanf.UnmarshalConf{Tag: "koanf"}); err != nil {
		log.Fatalf("config unmarshal: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config validation: %v", err)
	}

	httpAddr := k.String("server.http_addr")

	opts := []app.Option{
		app.WithDefaultBalancer(),
		app.WithGRPC(addr),
	}
	if httpAddr != "" {
		opts = append(opts, app.WithHTTP(httpAddr))
	}

	a, err := app.New(cfg, dataDir, opts...)
	if err != nil {
		log.Fatalf("init: %v", err)
	}

	if err := a.Start(); err != nil {
		log.Fatalf("start: %v", err)
	}

	log.Printf("NyaQueue broker gRPC listening on %s", a.Addr())
	if ha := a.HTTPAddr(); ha != "" {
		log.Printf("NyaQueue broker HTTP listening on %s", ha)
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	log.Println("shutting down...")
	a.Stop()
	log.Println("bye")
}
