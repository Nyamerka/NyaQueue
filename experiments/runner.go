package experiments

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"net"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/balancer"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/Nyamerka/NyaQueue/pkg/nn"
	"github.com/Nyamerka/NyaQueue/pkg/optimizer"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
	"github.com/alitto/pond/v2"
	"github.com/cenkalti/backoff/v4"
	"github.com/puzpuzpuz/xsync/v3"
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

type Runner struct {
	Scenarios      []benchmarks.Scenario
	Algorithms     []AlgorithmConfig
	Modes          []Mode
	KafkaBrokers   []string
	Duration       time.Duration // override scenario duration if > 0
	BrokerAddr     string        // gRPC broker address for external mode
	HTTPBrokerAddr string        // HTTP broker address for external mode (default: BrokerAddr)
	SkipWarmup     bool          // skip initial 10s discard warmup (CI smoke uses true)

	OnScenarioComplete func(results []ExperimentResult)

	sharedGRPC *transport.Client
	sharedHTTP *transport.HTTPClient

	dqnWarmupCache *xsync.MapOf[string, []nn.Transition]
}

func (r *Runner) RunAll(ctx context.Context) ([]ExperimentResult, error) {
	if r.dqnWarmupCache == nil {
		r.dqnWarmupCache = xsync.NewMapOf[string, []nn.Transition]()
	}
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

	if len(r.Scenarios) > 0 && !r.SkipWarmup {
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

			sortedAlgs := groupAlgorithmsByMatch(r.Algorithms, "DQN")
			for _, alg := range sortedAlgs {
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

		if r.OnScenarioComplete != nil {
			r.OnScenarioComplete(results)
		}
	}

	return results, nil
}

const (
	warmupDuration        = 10 * time.Second
	dqnWarmupMaxDuration  = 5 * time.Minute
	dqnWarmupTargetExp    = 2000
	dqnWarmupPollInterval = 100 * time.Millisecond
)

func dqnWarmupKey(mode Mode, scenario string) string {
	return fmt.Sprintf("%s|%s", mode, scenario)
}

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
		h.SetManagedTopic("warmup-topic")
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

	// For short experiments, increase warmup and hysteresis so TD3 collects
	// enough data before first action while still allowing online learning.
	if dur <= 60*time.Second && alg.UseOptimizer && alg.OptimizerOpts != nil {
		origOpts := alg.OptimizerOpts
		alg.OptimizerOpts = func() ([]optimizer.TunableParam, optimizer.OptimizerConfig) {
			params, cfg := origOpts()
			cfg.WarmupTicks = 200
			cfg.Hysteresis = 0.15
			return params, cfg
		}
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
		return ExperimentResult{}, oops.Wrapf(err, "create harness for %s/%s", sc.Name, alg.Name)
	}
	defer func() {
		h.Close()
		runtime.GC()
		debug.FreeOSMemory()
		_ = os.RemoveAll(dataDir)
	}()

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
			return oops.Wrapf(err, "topic already exists, retrying after delete")
		}
		if isRetryableError(err) {
			return oops.Wrapf(err, "retryable create topic error")
		}
		return backoff.Permanent(err)
	}, newExpBackoff())
	if createErr != nil {
		return ExperimentResult{}, oops.Wrapf(createErr, "create topic")
	}
	h.SetManagedTopic(topic)

	if brk := h.Broker(); brk != nil {
		newBal := alg.NewBalancer(topicCfg.NumPartitions)
		brk.ReplaceBalancerSameAlgorithm(newBal)
	} else {
		if err := h.SwitchBalancer(alg, topicCfg.NumPartitions); err != nil {
			return ExperimentResult{}, oops.Wrapf(err, "switch balancer for %s/%s", sc.Name, alg.Name)
		}
	}
	if brk := h.Broker(); brk != nil {
		brk.SetScheduler(topic, alg.NewScheduler())
	}

	if strings.Contains(alg.Name, "DQN") {
		if brk := h.Broker(); brk != nil {
			readyCtx, readyCancel := context.WithTimeout(ctx, 30*time.Second)
			if err := brk.WaitBalancerReady(readyCtx); err != nil {
				readyCancel()
				return ExperimentResult{}, oops.Wrapf(err, "wait DQN XLA ready")
			}
			readyCancel()
			log.Printf("  [dqn-setup] XLA ready for %s/%s/%s", sc.Name, alg.Name, mode)
		} else if h.IsExternal() {
			readyCtx, readyCancel := context.WithTimeout(ctx, 30*time.Second)
			if err := h.WaitExternalBalancerReady(readyCtx); err != nil {
				readyCancel()
				return ExperimentResult{}, oops.Wrapf(err, "external DQN XLA init failed for %s/%s", sc.Name, alg.Name)
			}
			readyCancel()
			log.Printf("  [dqn-setup] external XLA ready for %s/%s/%s", sc.Name, alg.Name, mode)
		}

		const miniWarmupDuration = 5 * time.Second
		warmupKey := dqnWarmupKey(mode, sc.Name)

		// Try loading from warmup cache.
		cachedBuffer, hasCache := r.dqnWarmupCache.Load(warmupKey)

		if hasCache {
			if brk := h.Broker(); brk != nil {
				if bal, ok := brk.Balancer().(*balancer.DQNBalancer); ok {
					bal.ReplayBuffer().LoadFromBuffer(cachedBuffer)
					log.Printf("  [dqn-warmup] loaded %d cached experiences for %s",
						len(cachedBuffer), warmupKey)
				}
			}
			goto postWarmup
		}

		seen, _ := h.BalancerExperiencesSeen(ctx)
		if seen >= dqnWarmupTargetExp {
			log.Printf("  [dqn-warmup] weights preserved (seen=%d), mini-warmup %v for %s/%s",
				seen, miniWarmupDuration, sc.Name, alg.Name)
			warmupSc := sc
			warmupSc.Duration = miniWarmupDuration
			if warmupSc.RatePerSec == 0 {
				warmupSc.RatePerSec = 5000 * min(warmupSc.Producers, topicCfg.NumPartitions)
			}
			_ = runScenario(ctx, h, warmupSc, alg.Name, "nyaqueue", mode, topicCfg.NumPartitions, miniWarmupDuration, topic)
			goto postWarmup
		}

		maxWarmup := dqnWarmupMaxDuration

		warmupSc := sc
		warmupSc.Duration = maxWarmup
		if warmupSc.RatePerSec == 0 {
			baseRate := 5000 * min(warmupSc.Producers, topicCfg.NumPartitions)
			if mode != ModeInProcess {
				warmupSc.RatePerSec = baseRate * 3
			} else {
				warmupSc.RatePerSec = baseRate * 2
			}
		}

		log.Printf("  [dqn-warmup] running warmup for %s/%s (k=%d, rate=%d, target=%d experiences, max=%v) ...",
			sc.Name, alg.Name, topicCfg.NumPartitions, warmupSc.RatePerSec, dqnWarmupTargetExp, maxWarmup)

		warmupCtx, cancelWarmup := context.WithCancel(ctx)
		warmupDone := make(chan struct{})
		go func() {
			defer close(warmupDone)
			_ = runScenario(warmupCtx, h, warmupSc, alg.Name, "nyaqueue", mode, topicCfg.NumPartitions, maxWarmup, topic)
		}()

		deadline := time.After(maxWarmup)
		ticker := time.NewTicker(dqnWarmupPollInterval)
		defer ticker.Stop()

	warmupLoop:
		for {
			select {
			case <-ticker.C:
				seen, err := h.BalancerExperiencesSeen(ctx)
				if err == nil && seen >= dqnWarmupTargetExp {
					log.Printf("  [dqn-warmup] reached %d experiences, finishing early", seen)
					break warmupLoop
				}
			case <-deadline:
				seen, _ := h.BalancerExperiencesSeen(ctx)
				log.Printf("  [dqn-warmup] max warmup reached, experiences=%d", seen)
				break warmupLoop
			case <-warmupDone:
				break warmupLoop
			case <-ctx.Done():
				break warmupLoop
			}
		}
		cancelWarmup()
		<-warmupDone

		// Save warmup buffer to cache for subsequent experiments.
		if brk := h.Broker(); brk != nil {
			if bal, ok := brk.Balancer().(*balancer.DQNBalancer); ok {
				buf := bal.ExportReplayBuffer(ctx)
				if buf != nil {
					r.dqnWarmupCache.Store(warmupKey, buf)
				}
				log.Printf("  [dqn-warmup] saved %d experiences for %s to cache",
					len(buf), warmupKey)
			}
		}
	}
postWarmup:
	if brk := h.Broker(); brk != nil {
		brk.ResetBalancerForScenario()
	}

	log.Printf("  running %s / %s / %s for %v (seed=%d) ...", sc.Name, alg.Name, mode, dur, sc.Seed)
	result := runScenario(ctx, h, sc, alg.Name, "nyaqueue", mode, topicCfg.NumPartitions, dur, topic)

	drainQueues(ctx, h, 5*time.Second)
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
		return ExperimentResult{}, oops.Wrapf(err, "create kafka harness for %s", sc.Name)
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

	if err := kafkaSmokeCheck(ctx, h, topic); err != nil {
		return ExperimentResult{}, oops.Wrapf(err, "kafka smoke check failed for %s", sc.Name)
	}

	log.Printf("  running %s / kafka for %v ...", sc.Name, dur)
	return runScenario(ctx, h, sc, "Kafka", "kafka", ModeKafka, topicCfg.NumPartitions, dur, topic), nil
}

func kafkaSmokeCheck(ctx context.Context, h *Harness, topic string) error {
	smokeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	batch := []BatchItem{{
		Key:   []byte("smoke-check"),
		Value: encodeValue(defaultMsgSize),
	}}
	n, err := h.PublishBatch(smokeCtx, topic, batch)
	if err != nil {
		return oops.Wrapf(err, "kafka smoke: produce failed")
	}
	if n == 0 {
		return oops.Errorf("kafka smoke: 0 messages produced")
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-smokeCtx.Done():
			return oops.Errorf("kafka smoke: timed out waiting for consumed message on topic %s", topic)
		case <-ticker.C:
			msgs, err := h.ConsumeBatch(smokeCtx, topic, "smoke-check-group", 0)
			if err != nil {
				continue
			}
			if len(msgs) > 0 {
				log.Printf("[kafka-smoke] OK: produce+consume verified on topic %s", topic)
				return nil
			}
		}
	}
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
	h.SetOnReconnect(collector.Reset)
	collector.Start()

	produceCtx, stopProducers := context.WithCancel(ctx)
	consumeCtx, stopConsumers := context.WithCancel(ctx)
	defer stopConsumers()

	samplerPool := pond.NewPool(1)
	samplerPool.Submit(func() {
		sampleLoadsFromHarness(consumeCtx, h, collector, loadSampleInterval)
	})

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

	maxLatency := 2 * dur
	consumerPool := pond.NewPool(numPartitions)
	for p := 0; p < numPartitions; p++ {
		partition := p
		var processingDelay time.Duration
		if sc.ConsumerSkewRate > 0 && p == 0 {
			processingDelay = time.Duration(float64(time.Millisecond) / sc.ConsumerSkewRate)
		}
		consumerPool.Submit(func() {
			runConsumer(consumeCtx, h, partition, collector, topic, maxLatency, processingDelay)
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
	samplerPool.StopAndWait()

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

func lingerInterval(sc benchmarks.Scenario) time.Duration {
	ms := sc.LingerMs
	if ms <= 0 {
		ms = 1
	}
	return time.Duration(ms) * time.Millisecond
}

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

	targetBytes := sc.BatchBytes
	if targetBytes <= 0 {
		targetBytes = defaultBatchBytes
	}

	produceLingerInterval := lingerInterval(sc)
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
			Priority: sc.SamplePrioritySeeded(rng),
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

func runUnlimitedProducer(ctx context.Context, h *Harness, sc benchmarks.Scenario, msgSize int, c *MetricsCollector, topic string) {
	seed := sc.Seed
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed)^0xBEEF))

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
			Priority: sc.SamplePrioritySeeded(rng),
		}
		batch = append(batch, item)
		accumulatedBytes += len(item.Key) + len(item.Value)

		if accumulatedBytes >= targetBytes {
			flush()
		}
	}
}

func runConsumer(ctx context.Context, h *Harness, partition int, c *MetricsCollector, topic string, maxLatency time.Duration, processingDelay ...time.Duration) {
	var delay time.Duration
	if len(processingDelay) > 0 {
		delay = processingDelay[0]
	}
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
			latency, enqueueTime, ok := decodeLatency(msg.Value, maxLatency)
			if !ok {
				c.RecordConsumeError()
				continue
			}
			c.RecordConsumeWithPriority(msg.Priority, latency)
			if msg.ProduceTime > 0 {
				c.RecordConsumeMultiStage(enqueueTime, msg.ProduceTime, msg.AppendTime)
			}
		}

		if delay > 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(delay):
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

func decodeLatency(value []byte, maxLatency time.Duration) (time.Duration, int64, bool) {
	if len(value) < timestampPrefixBytes {
		return 0, 0, false
	}
	ts := int64(binary.BigEndian.Uint64(value[:timestampPrefixBytes]))
	latency := time.Since(time.Unix(0, ts))
	if latency < 0 || latency > maxLatency {
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

func groupAlgorithmsByMatch(algs []AlgorithmConfig, match string) []AlgorithmConfig {
	sorted := make([]AlgorithmConfig, 0, len(algs))
	var matched []AlgorithmConfig
	for _, alg := range algs {
		if strings.Contains(alg.Name, match) {
			matched = append(matched, alg)
		} else {
			sorted = append(sorted, alg)
		}
	}
	return append(sorted, matched...)
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

func drainQueues(ctx context.Context, h *Harness, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		snap, err := h.GetMetricsSnapshot(ctx)
		if err != nil || snap == nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		totalDepth := 0
		for _, d := range snap.QueueDepth {
			totalDepth += d
		}
		if totalDepth == 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	log.Printf("  [drain] timeout after %v, some queues may not be empty", timeout)
}
