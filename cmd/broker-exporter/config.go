package main

import (
	"time"

	"github.com/nobl9/govy/pkg/govy"
	"github.com/nobl9/govy/pkg/rules"
)

type Config struct {
	BrokerAddr     string        `koanf:"broker_addr"`
	MetricsAddr    string        `koanf:"metrics_addr"`
	ScrapeInterval time.Duration `koanf:"scrape_interval"`
}

func DefaultConfig() Config {
	return Config{
		BrokerAddr:     "localhost:9090",
		MetricsAddr:    ":8083",
		ScrapeInterval: time.Second,
	}
}

var configValidator = govy.New(
	govy.For(func(c Config) string { return c.BrokerAddr }).
		WithName("broker_addr").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) string { return c.MetricsAddr }).
		WithName("metrics_addr").Rules(rules.StringNotEmpty()),
	govy.For(func(c Config) time.Duration { return c.ScrapeInterval }).
		WithName("scrape_interval").Rules(rules.GTE(100*time.Millisecond), rules.LTE(time.Minute)),
)

func (c Config) Validate() error {
	return configValidator.WithName("broker_exporter.Config").Validate(c)
}
