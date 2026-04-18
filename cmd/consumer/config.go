package main

import (
	"github.com/nobl9/govy/pkg/govy"
	"github.com/nobl9/govy/pkg/rules"
)

type Config struct {
	Addr        string `koanf:"addr"`
	Topic       string `koanf:"topic"`
	Group       string `koanf:"group"`
	Partitions  []int  `koanf:"partitions"`
	Workers     int    `koanf:"workers"`
	MetricsAddr string `koanf:"metrics_addr"`
}

func DefaultConfig() Config {
	return Config{
		Addr:        "localhost:9090",
		Topic:       "bench",
		Group:       "bench-group",
		Partitions:  nil,
		Workers:     4,
		MetricsAddr: ":8082",
	}
}

var configValidator = govy.New(
	govy.For(func(c Config) string { return c.Addr }).
		WithName("addr").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) string { return c.Topic }).
		WithName("topic").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) string { return c.Group }).
		WithName("group").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) int { return c.Workers }).
		WithName("workers").Rules(rules.GTE(1), rules.LTE(256)),
	govy.For(func(c Config) string { return c.MetricsAddr }).
		WithName("metrics_addr").Rules(rules.StringNotEmpty()),
)

func (c Config) Validate() error {
	return configValidator.WithName("consumer.Config").Validate(c)
}
