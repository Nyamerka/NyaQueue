package main

import (
	"time"

	"github.com/nobl9/govy/pkg/govy"
	"github.com/nobl9/govy/pkg/rules"
)

type Config struct {
	Addr        string        `koanf:"addr"`
	Topic       string        `koanf:"topic"`
	Scenario    string        `koanf:"scenario"`
	Producers   int           `koanf:"producers"`
	Partitions  int           `koanf:"partitions"`
	Duration    time.Duration `koanf:"duration"`
	MetricsAddr string        `koanf:"metrics_addr"`
}

func DefaultConfig() Config {
	return Config{
		Addr:        "localhost:9090",
		Topic:       "bench",
		Scenario:    "uniform",
		Producers:   4,
		Partitions:  4,
		Duration:    0,
		MetricsAddr: ":8081",
	}
}

var configValidator = govy.New(
	govy.For(func(c Config) string { return c.Addr }).
		WithName("addr").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) string { return c.Topic }).
		WithName("topic").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) string { return c.Scenario }).
		WithName("scenario").
		Rules(rules.OneOf("uniform", "skewed", "bursty", "growing", "mixed_priority")),
	govy.For(func(c Config) int { return c.Producers }).
		WithName("producers").Rules(rules.GTE(1), rules.LTE(256)),
	govy.For(func(c Config) int { return c.Partitions }).
		WithName("partitions").Rules(rules.GTE(1), rules.LTE(1024)),
	govy.For(func(c Config) string { return c.MetricsAddr }).
		WithName("metrics_addr").Rules(rules.StringNotEmpty()),
)

func (c Config) Validate() error {
	return configValidator.WithName("producer.Config").Validate(c)
}
