package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/experiments"
	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/basicflag"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

func main() {
	fs := flag.NewFlagSet("experiment", flag.ContinueOnError)
	fs.String("experiment.mode", "inprocess", "comma-separated modes: inprocess,grpc,kafka")
	fs.String("experiment.scenarios", "all", "comma-separated scenarios or 'all'")
	fs.String("experiment.algorithms", "all", "comma-separated algorithms or 'all'")
	fs.String("experiment.kafka_brokers", "localhost:9092", "kafka broker addresses")
	fs.String("experiment.broker_addr", "", "external NyaQueue broker address for grpc mode (e.g. broker:9090)")
	fs.String("experiment.output", "experiments/results", "output directory for results")
	fs.String("experiment.duration", "", "per-scenario duration (e.g. 30s)")
	_ = fs.Parse(os.Args[1:])

	k := koanf.New(".")
	_ = k.Load(file.Provider("config.yaml"), yaml.Parser())
	_ = k.Load(env.Provider("NYAQUEUE_", ".", func(s string) string { return s }), nil)
	_ = k.Load(basicflag.Provider(fs, "."), nil)

	modeStr := k.String("experiment.mode")
	scenarioStr := k.String("experiment.scenarios")
	algorithmStr := k.String("experiment.algorithms")
	kafkaBrokersStr := k.String("experiment.kafka_brokers")
	brokerAddr := k.String("experiment.broker_addr")
	outputDir := k.String("experiment.output")
	durationStr := k.String("experiment.duration")
	duration := 10 * time.Second
	if durationStr != "" {
		if d, err := time.ParseDuration(durationStr); err == nil {
			duration = d
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		cancel()
	}()

	modes := parseModes(modeStr)
	scenarios := parseScenarios(scenarioStr)
	algorithms := parseAlgorithms(algorithmStr)

	runner := &experiments.Runner{
		Scenarios:    scenarios,
		Algorithms:   algorithms,
		Modes:        modes,
		KafkaBrokers: strings.Split(kafkaBrokersStr, ","),
		BrokerAddr:   brokerAddr,
		Duration:     duration,
	}

	nyaModes := 0
	hasKafka := false
	for _, m := range modes {
		if m == experiments.ModeKafka {
			hasKafka = true
		} else {
			nyaModes++
		}
	}

	totalRuns := nyaModes * len(scenarios) * len(algorithms)
	if hasKafka {
		totalRuns += len(scenarios)
	}

	kafkaNote := ""
	if hasKafka {
		kafkaNote = " + Kafka baseline"
	}
	log.Printf("Running %d total runs: %d scenarios × %d algorithms × %d NyaQueue mode(s)%s",
		totalRuns, len(scenarios), len(algorithms), nyaModes, kafkaNote)

	results, err := runner.RunAll(ctx)
	if err != nil {
		log.Fatalf("run: %v", err)
	}

	if err := experiments.ExportCSV(results, outputDir); err != nil {
		log.Printf("csv export: %v", err)
	}
	if err := experiments.ExportJSON(results, outputDir); err != nil {
		log.Printf("json export: %v", err)
	}

	log.Printf("Done. %d results written to %s", len(results), outputDir)
	printSummary(results)
}

func parseModes(s string) []experiments.Mode {
	if s == "all" {
		return []experiments.Mode{experiments.ModeInProcess, experiments.ModeGRPC, experiments.ModeKafka}
	}
	var modes []experiments.Mode
	for _, m := range strings.Split(s, ",") {
		switch strings.TrimSpace(m) {
		case "inprocess":
			modes = append(modes, experiments.ModeInProcess)
		case "grpc":
			modes = append(modes, experiments.ModeGRPC)
		case "kafka":
			modes = append(modes, experiments.ModeKafka)
		}
	}
	return modes
}

func parseScenarios(s string) []benchmarks.Scenario {
	if s == "all" {
		return benchmarks.AllScenarios()
	}
	all := benchmarks.AllScenarios()
	names := strings.Split(s, ",")
	var out []benchmarks.Scenario
	for _, name := range names {
		name = strings.TrimSpace(name)
		for _, sc := range all {
			if sc.Name == name {
				out = append(out, sc)
			}
		}
	}
	return out
}

func parseAlgorithms(s string) []experiments.AlgorithmConfig {
	if s == "all" {
		return experiments.AllAlgorithms()
	}
	all := experiments.AllAlgorithms()
	names := strings.Split(s, ",")
	var out []experiments.AlgorithmConfig
	for _, name := range names {
		name = strings.TrimSpace(name)
		for _, alg := range all {
			if alg.Name == name {
				out = append(out, alg)
			}
		}
	}
	return out
}

func printSummary(results []experiments.ExperimentResult) {
	fmt.Println()
	fmt.Printf("%-20s %-15s %-10s %12s %10s %10s %10s %10s\n",
		"SCENARIO", "ALGORITHM", "MODE", "THROUGHPUT", "P50(us)", "P95(us)", "P99(us)", "STDDEV")
	fmt.Println(strings.Repeat("-", 107))

	for _, r := range results {
		fmt.Printf("%-20s %-15s %-10s %12.0f %10.1f %10.1f %10.1f %10.6f\n",
			r.Scenario, r.Algorithm, r.Mode, r.Throughput,
			float64(r.LatencyP50)/1000, float64(r.LatencyP95)/1000,
			float64(r.LatencyP99)/1000, r.LoadStdDev,
		)
	}
}
