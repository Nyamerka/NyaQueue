package experiments

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/samber/oops"
)

// Runner orchestrates experiment runs across scenarios and algorithms.
type Runner struct {
	Scenarios    []benchmarks.Scenario
	Algorithms   []AlgorithmConfig
	Modes        []Mode
	KafkaBrokers []string
	Duration     time.Duration // override scenario duration if > 0
}

// RunAll executes every (scenario, algorithm, mode) combination and returns results.
func (r *Runner) RunAll(ctx context.Context) ([]ExperimentResult, error) {
	var results []ExperimentResult

	for _, sc := range r.Scenarios {
		dur := sc.Duration
		if r.Duration > 0 {
			dur = r.Duration
		}

		for _, mode := range r.Modes {
			if mode == ModeKafka {
				res, err := r.runKafka(ctx, sc, dur)
				if err != nil {
					log.Printf("SKIP kafka/%s: %v", sc.Name, err)
					continue
				}
				results = append(results, res)
				continue
			}

			for _, alg := range r.Algorithms {
				res, err := r.runNyaQueue(ctx, sc, alg, mode, dur)
				if err != nil {
					log.Printf("SKIP %s/%s/%s: %v", mode, alg.Name, sc.Name, err)
					continue
				}
				results = append(results, res)
			}
		}
	}

	return results, nil
}

func (r *Runner) runNyaQueue(ctx context.Context, sc benchmarks.Scenario, alg AlgorithmConfig, mode Mode, dur time.Duration) (ExperimentResult, error) {
	dataDir := fmt.Sprintf("/tmp/nyaqueue-exp-%s-%s-%s", sc.Name, alg.Name, mode)
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "create data dir")
	}

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         mode,
		BrokerConfig: broker.DefaultConfig(),
		DataDir:      dataDir,
		Algorithm:    alg,
	})
	if err != nil {
		return ExperimentResult{}, err
	}
	defer h.Close()

	topicCfg := broker.DefaultTopicConfig()
	topicCfg.NumPartitions = sc.Producers
	if topicCfg.NumPartitions < 4 {
		topicCfg.NumPartitions = 4
	}

	brk := h.Broker()
	if brk != nil {
		if err := brk.CreateTopic("bench", topicCfg); err != nil {
			return ExperimentResult{}, oops.Wrapf(err, "create topic")
		}
		brk.SetScheduler("bench", alg.NewScheduler())
	}

	collector := NewMetricsCollector()
	collector.Start()

	deadline := time.After(dur)
	msgSize := sc.MsgSize
	if msgSize == 0 {
		msgSize = 256
	}

	log.Printf("  running %s / %s / %s for %v ...", sc.Name, alg.Name, mode, dur)

loop:
	for {
		select {
		case <-deadline:
			break loop
		case <-ctx.Done():
			break loop
		default:
			key := []byte("k")
			value := benchmarks.GenerateMessage(msgSize)
			priority := sc.SamplePriority()

			if err := h.Publish(ctx, "bench", key, value, priority); err == nil {
				collector.RecordProduce()
				collector.RecordConsume(0)
			}
		}
	}

	collector.Stop()

	if brk != nil {
		m := brk.Metrics()
		collector.RecordPartitionLoads(m.PartitionLoads)
	}

	return collector.Snapshot(sc.Name, alg.Name, "nyaqueue", mode.String()), nil
}

func (r *Runner) runKafka(ctx context.Context, sc benchmarks.Scenario, dur time.Duration) (ExperimentResult, error) {
	if len(r.KafkaBrokers) == 0 {
		return ExperimentResult{}, oops.Errorf("no kafka brokers configured")
	}

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:         ModeKafka,
		KafkaBrokers: r.KafkaBrokers,
	})
	if err != nil {
		return ExperimentResult{}, err
	}
	defer h.Close()

	collector := NewMetricsCollector()
	collector.Start()

	deadline := time.After(dur)
	msgSize := sc.MsgSize
	if msgSize == 0 {
		msgSize = 256
	}

	log.Printf("  running %s / kafka for %v ...", sc.Name, dur)

loop:
	for {
		select {
		case <-deadline:
			break loop
		case <-ctx.Done():
			break loop
		default:
			key := []byte("k")
			value := benchmarks.GenerateMessage(msgSize)

			if err := h.Publish(ctx, "bench", key, value, 0); err == nil {
				collector.RecordProduce()
				collector.RecordConsume(0)
			}
		}
	}

	collector.Stop()
	return collector.Snapshot(sc.Name, "Kafka", "kafka", "kafka"), nil
}
