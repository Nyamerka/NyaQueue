package experiments

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	mrand "math/rand"
	"math/rand/v2"
	"net"
	"os"
	"sync"
	"syscall"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
	"github.com/alitto/pond/v2"
	"github.com/cenkalti/backoff/v4"
	"github.com/samber/oops"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	timestampPrefixBytes = 8
	loadSampleInterval   = 100 * time.Millisecond
	consumeIdleBackoff   = 500 * time.Microsecond
	drainGracePeriod     = 500 * time.Millisecond
	defaultMsgSize       = 256
	expGroup             = "exp-group"
)

func expTopicName(scenario, alg string, mode Mode) string {
	return fmt.Sprintf("exp-%s-%s-%s", scenario, alg, mode)
}

// Runner orchestrates experiment runs across scenarios and algorithms.
type Runner struct {
	Scenarios      []benchmarks.Scenario
	Algorithms     []AlgorithmConfig
	Modes          []Mode
	KafkaBrokers   []string
	Duration       time.Duration // override scenario duration if > 0
	BrokerAddr     string        // gRPC broker address for external mode
	HTTPBrokerAddr string        // HTTP broker address for external mode (default: BrokerAddr)

	sharedGRPC *transport.Client
	sharedHTTP *transport.HTTPClient
}

// RunAll executes every (scenario, algorithm, mode) combination and returns results.
func (r *Runner) RunAll(ctx context.Context) ([]ExperimentResult, error) {
	if r.BrokerAddr != "" && r.sharedGRPC == nil {
		client, err := transport.NewClient(r.BrokerAddr)
		if err == nil {
			r.sharedGRPC = client
			defer func() { r.sharedGRPC.Close(); r.sharedGRPC = nil }()
		}
	}
	if addr := r.HTTPBrokerAddr; addr != "" && r.sharedHTTP == nil {
		r.sharedHTTP = transport.NewHTTPClient(addr)
		defer func() { r.sharedHTTP.Close(); r.sharedHTTP = nil }()
	}

	var results []ExperimentResult

	disabledModes := make(map[Mode]bool)
	const failFastThreshold = 3

	if len(r.Scenarios) > 0 {
		r.runWarmup(ctx)
	}

	for _, sc := range r.Scenarios {
		consecutiveFails := make(map[Mode]int)

		dur := sc.Duration
		if r.Duration > 0 {
			dur = r.Duration
		}

		for _, mode := range r.Modes {
			if disabledModes[mode] {
				log.Printf("SKIP %s/%s: mode disabled (fail-fast)", mode, sc.Name)
				continue
			}

			if mode == ModeKafka {
				res, err := r.runKafka(ctx, sc, dur)
				if err != nil {
					log.Printf("SKIP kafka/%s: %v", sc.Name, err)
					consecutiveFails[mode]++
					if consecutiveFails[mode] >= failFastThreshold {
						log.Printf("DISABLE mode %s: %d consecutive failures", mode, consecutiveFails[mode])
						disabledModes[mode] = true
					}
					continue
				}
				consecutiveFails[mode] = 0
				results = append(results, res)
				continue
			}

			for _, alg := range r.Algorithms {
				if disabledModes[mode] {
					break
				}
				res, err := r.runNyaQueue(ctx, sc, alg, mode, dur)
				if err != nil {
					log.Printf("SKIP %s/%s/%s: %v", mode, alg.Name, sc.Name, err)
					consecutiveFails[mode]++
					if consecutiveFails[mode] >= failFastThreshold {
						log.Printf("DISABLE mode %s: %d consecutive failures", mode, consecutiveFails[mode])
						disabledModes[mode] = true
					}
					continue
				}
				consecutiveFails[mode] = 0
				results = append(results, res)
			}
		}
	}

	return results, nil
}

const warmupDuration = 10 * time.Second

func (r *Runner) runWarmup(ctx context.Context) {
	log.Printf("[warmup] running %v warmup run (results discarded)...", warmupDuration)
	warmupSc := benchmarks.Scenario{
		Name:          "warmup",
		Duration:      warmupDuration,
		Producers:     2,
		NumPartitions: 4,
		MsgSize:       defaultMsgSize,
	}

	for _, mode := range r.Modes {
		dataDir := "/tmp/nyaqueue-warmup"
		_ = os.RemoveAll(dataDir)
		_ = os.MkdirAll(dataDir, 0o755)

		topicCfg := topicConfigFor(warmupSc)
		h, err := NewHarness(ctx, HarnessConfig{
			Mode:           mode,
			BrokerConfig:   broker.DefaultConfig(),
			DataDir:        dataDir,
			Algorithm:      r.Algorithms[0],
			NumPartitions:  topicCfg.NumPartitions,
			KafkaBrokers:   r.KafkaBrokers,
			BrokerAddr:     r.BrokerAddr,
			HTTPBrokerAddr: r.HTTPBrokerAddr,
		})
		if err != nil {
			log.Printf("[warmup] skip mode %s: %v", mode, err)
			continue
		}

		_ = h.CreateTopic(ctx, "warmup-topic", topicCfg)
		if brk := h.Broker(); brk != nil {
			brk.SetScheduler("warmup-topic", r.Algorithms[0].NewScheduler())
		}

		runScenario(ctx, h, warmupSc, "warmup", "warmup", mode, topicCfg.NumPartitions, warmupDuration, "warmup-topic")
		h.Close()
		_ = os.RemoveAll(dataDir)
		log.Printf("[warmup] mode %s done", mode)
	}
}

func (r *Runner) runNyaQueue(ctx context.Context, sc benchmarks.Scenario, alg AlgorithmConfig, mode Mode, dur time.Duration) (ExperimentResult, error) {
	dataDir := fmt.Sprintf("/tmp/nyaqueue-exp-%s-%s-%s", sc.Name, alg.Name, mode)
	if err := os.RemoveAll(dataDir); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "remove data dir")
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "create data dir")
	}

	// Determine partition count before creating the harness so that
	// partition-aware balancers (PSA, DQN) are initialised with the correct K.
	topicCfg := topicConfigFor(sc)
	topicCfg.ScheduleMode = alg.Mode

	h, err := NewHarness(ctx, HarnessConfig{
		Mode:           mode,
		BrokerConfig:   broker.DefaultConfig(),
		DataDir:        dataDir,
		Algorithm:      alg,
		NumPartitions:  topicCfg.NumPartitions,
		BrokerAddr:     r.BrokerAddr,
		HTTPBrokerAddr: r.HTTPBrokerAddr,
	})
	if err != nil {
		return ExperimentResult{}, err
	}
	defer h.Close()

	topic := expTopicName(sc.Name, alg.Name, mode)

	if h.IsExternal() {
		bo := newExpBackoff()
		err := backoff.Retry(func() error {
			delCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			delErr := h.DeleteTopic(delCtx, topic)
			if delErr != nil && !errors.Is(delErr, broker.ErrTopicNotFound) {
				if isRetryableError(delErr) {
					return delErr
				}
				return backoff.Permanent(delErr)
			}
			return nil
		}, bo)
		if err != nil {
			return ExperimentResult{}, oops.Wrapf(err, "delete stale topic")
		}
	}

	createErr := backoff.Retry(func() error {
		err := h.CreateTopic(ctx, topic, topicCfg)
		if err == nil {
			return nil
		}
		if errors.Is(err, broker.ErrTopicAlreadyExists) {
			delCtx, delCancel := context.WithTimeout(ctx, 5*time.Second)
			_ = h.DeleteTopic(delCtx, topic)
			delCancel()
			return err
		}
		if isRetryableError(err) {
			return err
		}
		return backoff.Permanent(err)
	}, newExpBackoff())
	if createErr != nil {
		return ExperimentResult{}, oops.Wrapf(createErr, "create topic")
	}

	if brk := h.Broker(); brk != nil {
		brk.SetScheduler(topic, alg.NewScheduler())
	}

	log.Printf("  running %s / %s / %s for %v (seed=%d) ...", sc.Name, alg.Name, mode, dur, sc.Seed)
	result := runScenario(ctx, h, sc, alg.Name, "nyaqueue", mode, topicCfg.NumPartitions, dur, topic)

	if mode != ModeInProcess {
		time.Sleep(5 * time.Second)
	}
	return result, nil
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

	topic := expTopicName(sc.Name, "Kafka", ModeKafka)

	cleanCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	if delErr := h.DeleteTopic(cleanCtx, topic); delErr != nil && !errors.Is(delErr, broker.ErrTopicNotFound) {
		cancel()
		return ExperimentResult{}, oops.Wrapf(delErr, "delete stale kafka topic")
	}
	cancel()

	topicCfg := topicConfigFor(sc)
	if err := h.CreateTopic(ctx, topic, topicCfg); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "create kafka topic")
	}

	log.Printf("  running %s / kafka for %v ...", sc.Name, dur)
	return runScenario(ctx, h, sc, "Kafka", "kafka", ModeKafka, topicCfg.NumPartitions, dur, topic), nil
}

func runScenario(
	ctx context.Context,
	h *Harness,
	sc benchmarks.Scenario,
	algName, system string,
	mode Mode,
	numPartitions int,
	dur time.Duration,
	topicOverride ...string,
) ExperimentResult {
	topic := expTopicName(sc.Name, algName, mode)
	if len(topicOverride) > 0 && topicOverride[0] != "" {
		topic = topicOverride[0]
	}
	collector := NewMetricsCollector()
	collector.Start()

	produceCtx, stopProducers := context.WithCancel(ctx)
	consumeCtx, stopConsumers := context.WithCancel(ctx)
	defer stopConsumers()

	var samplerWG sync.WaitGroup
	samplerWG.Add(1)
	go func() {
		defer samplerWG.Done()
		sampleLoadsFromHarness(consumeCtx, h, collector, loadSampleInterval)
	}()

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
			runProducer(produceCtx, h, sc, msgSize, collector, topic)
		})
	}

	consumerPool := pond.NewPool(numPartitions)
	for p := 0; p < numPartitions; p++ {
		partition := p
		consumerPool.Submit(func() {
			runConsumer(consumeCtx, h, partition, collector, topic)
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

const (
	produceBatchSize  = 64
	defaultBatchBytes = 16 * 1024 // 16KB, matching Kafka's batch.size default
)

func runProducer(ctx context.Context, h *Harness, sc benchmarks.Scenario, msgSize int, c *MetricsCollector, topic string) {
	if sc.RatePerSec > 0 {
		runRateLimitedProducer(ctx, h, sc, msgSize, c, topic)
	} else {
		runUnlimitedProducer(ctx, h, sc, msgSize, c, topic)
	}
}

const produceLingerInterval = 5 * time.Millisecond

func runRateLimitedProducer(ctx context.Context, h *Harness, sc benchmarks.Scenario, msgSize int, c *MetricsCollector, topic string) {
	perProducer := sc.RatePerSec / sc.Producers
	if perProducer < 1 {
		perProducer = 1
	}
	interval := time.Second / time.Duration(perProducer)

	seed := sc.Seed
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed)^0xCAFE))
	legacyRng := mrand.New(mrand.NewSource(int64(rng.Uint64())))

	targetBytes := sc.BatchBytes
	if targetBytes <= 0 {
		targetBytes = defaultBatchBytes
	}

	batch := make([]BatchItem, 0, produceBatchSize)
	accumulatedBytes := 0
	linger := time.NewTimer(produceLingerInterval)
	linger.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		n, err := h.PublishBatch(ctx, topic, batch)
		for i := 0; i < n; i++ {
			c.RecordProduce()
		}
		if err != nil && n < len(batch) {
			failed := len(batch) - n
			if isThrottleError(err) {
				for range failed {
					c.RecordPublishThrottled()
				}
			} else {
				for range failed {
					c.RecordPublishError()
				}
			}
		}
		batch = batch[:0]
		accumulatedBytes = 0
		linger.Stop()
	}

	keyBuf := make([]byte, 8)
	for {
		if ctx.Err() != nil {
			flush()
			return
		}

		item := BatchItem{
			Key:      generateKeySeeded(keyBuf, sc.SkewRatio, rng),
			Value:    encodeValue(msgSize),
			Priority: sc.SamplePrioritySeeded(legacyRng),
		}
		batch = append(batch, item)
		accumulatedBytes += len(item.Key) + len(item.Value)

		if accumulatedBytes >= targetBytes {
			flush()
		} else if len(batch) == 1 {
			linger.Reset(produceLingerInterval)
		}

		select {
		case <-ctx.Done():
			flush()
			return
		case <-linger.C:
			flush()
		case <-time.After(interval):
		}
	}
}

// runUnlimitedProducer batches messages by bytes for maximum throughput.
// Flushes when accumulated bytes reach the target (default 16KB, matching
// Kafka's batch.size). This gives realistic WAL and compression ratios.
func runUnlimitedProducer(ctx context.Context, h *Harness, sc benchmarks.Scenario, msgSize int, c *MetricsCollector, topic string) {
	seed := sc.Seed
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed)^0xBEEF))
	legacyRng := mrand.New(mrand.NewSource(int64(rng.Uint64())))

	targetBytes := sc.BatchBytes
	if targetBytes <= 0 {
		targetBytes = defaultBatchBytes
	}

	batch := make([]BatchItem, 0, produceBatchSize)
	accumulatedBytes := 0

	flush := func() {
		if len(batch) == 0 {
			return
		}
		n, err := h.PublishBatch(ctx, topic, batch)
		for i := 0; i < n; i++ {
			c.RecordProduce()
		}
		if err != nil && n < len(batch) {
			failed := len(batch) - n
			if isThrottleError(err) {
				for range failed {
					c.RecordPublishThrottled()
				}
			} else {
				for range failed {
					c.RecordPublishError()
				}
			}
		}
		batch = batch[:0]
		accumulatedBytes = 0
	}

	keyBuf := make([]byte, 8)
	for {
		if ctx.Err() != nil {
			flush()
			return
		}

		item := BatchItem{
			Key:      generateKeySeeded(keyBuf, sc.SkewRatio, rng),
			Value:    encodeValue(msgSize),
			Priority: sc.SamplePrioritySeeded(legacyRng),
		}
		batch = append(batch, item)
		accumulatedBytes += len(item.Key) + len(item.Value)

		if accumulatedBytes >= targetBytes {
			flush()
		}
	}
}

func runConsumer(ctx context.Context, h *Harness, partition int, c *MetricsCollector, topic string) {
	for {
		if ctx.Err() != nil {
			return
		}

		msgs, err := h.ConsumeBatch(ctx, topic, expGroup, partition)
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

		for _, msg := range msgs {
			latency, enqueueTime, ok := decodeLatency(msg.Value)
			if !ok {
				c.RecordConsumeError()
				continue
			}
			c.RecordConsumeWithPriority(msg.Priority, latency)
			if msg.ProduceTime > 0 {
				c.RecordConsumeMultiStage(enqueueTime, msg.ProduceTime, msg.AppendTime)
			}
		}
	}
}

func sampleLoadsFromHarness(ctx context.Context, h *Harness, c *MetricsCollector, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			snap, err := h.GetMetricsSnapshot(ctx)
			if err != nil {
				continue
			}
			if snap.HasStdDev {
				c.RecordLoadStdDev(snap.LoadStdDev)
			} else if len(snap.PartitionLoads) > 0 {
				c.RecordLoadSample(snap.PartitionLoads)
			}
		}
	}
}

func encodeValue(size int) []byte {
	buf := benchmarks.GenerateMessage(size)
	binary.BigEndian.PutUint64(buf[:timestampPrefixBytes], uint64(time.Now().UnixNano()))
	return buf
}

func decodeLatency(value []byte) (time.Duration, int64, bool) {
	if len(value) < timestampPrefixBytes {
		return 0, 0, false
	}
	ts := int64(binary.BigEndian.Uint64(value[:timestampPrefixBytes]))
	latency := time.Since(time.Unix(0, ts))
	if latency < 0 {
		return 0, 0, false
	}
	return latency, ts, true
}

func topicConfigFor(sc benchmarks.Scenario) broker.TopicConfig {
	cfg := broker.DefaultTopicConfig()
	if sc.NumPartitions > 0 {
		cfg.NumPartitions = sc.NumPartitions
	} else {
		cfg.NumPartitions = sc.Producers
		if cfg.NumPartitions < 1 {
			cfg.NumPartitions = 1
		}
	}
	return cfg
}

// generateKey returns a message key respecting the scenario's SkewRatio.
// With SkewRatio > 0, that fraction of messages get a fixed hot key (all-zeros)
// which consistent-hashing balancers (PSA) will route to the same partition,
// creating a realistic key-skew workload.
func generateKey(buf []byte, skewRatio float64) []byte {
	key := make([]byte, 8)
	if skewRatio > 0 && rand.Float64() < skewRatio {
		return key
	}
	binary.BigEndian.PutUint64(buf, rand.Uint64())
	copy(key, buf)
	return key
}

func generateKeySeeded(buf []byte, skewRatio float64, rng *rand.Rand) []byte {
	key := make([]byte, 8)
	if skewRatio > 0 && rng.Float64() < skewRatio {
		return key
	}
	binary.BigEndian.PutUint64(buf, rng.Uint64())
	copy(key, buf)
	return key
}

func newExpBackoff() *backoff.ExponentialBackOff {
	bo := backoff.NewExponentialBackOff()
	bo.InitialInterval = 200 * time.Millisecond
	bo.MaxElapsedTime = 10 * time.Second
	return bo
}

func isThrottleError(err error) bool {
	if errors.Is(err, broker.ErrThrottled) {
		return true
	}
	if st, ok := status.FromError(err); ok && st.Code() == codes.ResourceExhausted {
		return true
	}
	return false
}

func isRetryableError(err error) bool {
	if errors.Is(err, syscall.EADDRNOTAVAIL) {
		return true
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	return false
}
