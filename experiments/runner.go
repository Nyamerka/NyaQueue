package experiments

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/alitto/pond/v2"
	"github.com/samber/oops"
)

const (
	timestampPrefixBytes = 8
	loadSampleInterval   = 100 * time.Millisecond
	consumeIdleBackoff   = 500 * time.Microsecond
	drainGracePeriod     = 500 * time.Millisecond
	defaultMsgSize       = 256
	expGroup             = "exp-group"
	expTopic             = "bench"
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

	topicCfg := topicConfigFor(sc)
	topicCfg.ScheduleMode = alg.Mode

	if err := h.CreateTopic(ctx, expTopic, topicCfg); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "create topic")
	}
	if brk := h.Broker(); brk != nil {
		brk.SetScheduler(expTopic, alg.NewScheduler())
	}

	log.Printf("  running %s / %s / %s for %v ...", sc.Name, alg.Name, mode, dur)
	return runScenario(ctx, h, sc, alg.Name, "nyaqueue", mode, topicCfg.NumPartitions, dur), nil
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

	topicCfg := topicConfigFor(sc)
	if err := h.CreateTopic(ctx, expTopic, topicCfg); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "create kafka topic")
	}

	log.Printf("  running %s / kafka for %v ...", sc.Name, dur)
	return runScenario(ctx, h, sc, "Kafka", "kafka", ModeKafka, topicCfg.NumPartitions, dur), nil
}

func runScenario(
	ctx context.Context,
	h *Harness,
	sc benchmarks.Scenario,
	algName, system string,
	mode Mode,
	numPartitions int,
	dur time.Duration,
) ExperimentResult {
	collector := NewMetricsCollector()
	collector.Start()

	produceCtx, stopProducers := context.WithCancel(ctx)
	consumeCtx, stopConsumers := context.WithCancel(ctx)
	defer stopConsumers()

	var samplerWG sync.WaitGroup
	if brk := h.Broker(); brk != nil {
		samplerWG.Add(1)
		go func() {
			defer samplerWG.Done()
			sampleLoads(consumeCtx, brk, collector, loadSampleInterval)
		}()
	}

	msgSize := sc.MsgSize
	if msgSize == 0 {
		msgSize = defaultMsgSize
	}
	if msgSize < timestampPrefixBytes {
		msgSize = timestampPrefixBytes
	}

	numProducers := sc.Producers
	if numProducers < 1 {
		numProducers = 1
	}

	producerPool := pond.NewPool(numProducers)
	for i := 0; i < numProducers; i++ {
		producerPool.Submit(func() {
			runProducer(produceCtx, h, sc, msgSize, collector)
		})
	}

	consumerPool := pond.NewPool(numPartitions)
	for p := 0; p < numPartitions; p++ {
		partition := p
		consumerPool.Submit(func() {
			runConsumer(consumeCtx, h, partition, collector)
		})
	}

	select {
	case <-time.After(dur):
	case <-ctx.Done():
	}

	stopProducers()
	producerPool.StopAndWait()

	select {
	case <-time.After(drainGracePeriod):
	case <-ctx.Done():
	}

	stopConsumers()
	consumerPool.StopAndWait()
	samplerWG.Wait()

	collector.Stop()
	return collector.Snapshot(sc.Name, algName, system, mode.String())
}

func runProducer(ctx context.Context, h *Harness, sc benchmarks.Scenario, msgSize int, c *MetricsCollector) {
	key := []byte("k")
	for {
		if ctx.Err() != nil {
			return
		}

		value := encodeValue(msgSize)
		priority := sc.SamplePriority()

		if err := h.Publish(ctx, expTopic, key, value, priority); err != nil {
			if ctx.Err() != nil {
				return
			}
			c.RecordPublishError()
			continue
		}
		c.RecordProduce()
	}
}

func runConsumer(ctx context.Context, h *Harness, partition int, c *MetricsCollector) {
	for {
		if ctx.Err() != nil {
			return
		}

		msg, err := h.Consume(ctx, expTopic, expGroup, partition)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			if errors.Is(err, ErrNoMessage) {
				select {
				case <-ctx.Done():
					return
				case <-time.After(consumeIdleBackoff):
				}
				continue
			}
			c.RecordConsumeError()
			continue
		}

		latency, ok := decodeLatency(msg.Value)
		if !ok {
			c.RecordConsumeError()
			continue
		}
		c.RecordConsume(latency)
	}
}

func sampleLoads(ctx context.Context, brk *broker.Broker, c *MetricsCollector, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.RecordLoadSample(brk.Metrics().PartitionLoads)
		}
	}
}

func encodeValue(size int) []byte {
	buf := benchmarks.GenerateMessage(size)
	binary.BigEndian.PutUint64(buf[:timestampPrefixBytes], uint64(time.Now().UnixNano()))
	return buf
}

func decodeLatency(value []byte) (time.Duration, bool) {
	if len(value) < timestampPrefixBytes {
		return 0, false
	}
	ts := int64(binary.BigEndian.Uint64(value[:timestampPrefixBytes]))
	latency := time.Since(time.Unix(0, ts))
	if latency < 0 {
		return 0, false
	}
	return latency, true
}

func topicConfigFor(sc benchmarks.Scenario) broker.TopicConfig {
	cfg := broker.DefaultTopicConfig()
	cfg.NumPartitions = sc.Producers
	if cfg.NumPartitions < 4 {
		cfg.NumPartitions = 4
	}
	return cfg
}