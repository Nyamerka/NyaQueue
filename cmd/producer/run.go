package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/alitto/pond/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/samber/oops"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/Nyamerka/NyaQueue/benchmarks"
	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

const timestampPrefixBytes = 8

type producerMetrics struct {
	up        prometheus.Gauge
	published *prometheus.CounterVec
	errors    *prometheus.CounterVec
	latency   *prometheus.HistogramVec
	throttled *prometheus.CounterVec
}

func registerMetrics(reg prometheus.Registerer) *producerMetrics {
	f := promauto.With(reg)
	return &producerMetrics{
		up: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_producer",
			Name:      "up",
			Help:      "1 when producer workers are running.",
		}),
		published: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_producer",
			Name:      "messages_published_total",
			Help:      "Number of messages published successfully.",
		}, []string{"topic", "priority"}),
		errors: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_producer",
			Name:      "publish_errors_total",
			Help:      "Number of publish errors grouped by gRPC code.",
		}, []string{"topic", "code"}),
		latency: f.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "nyaqueue_producer",
			Name:      "publish_latency_seconds",
			Help:      "Produce RPC round-trip latency.",
			Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 16),
		}, []string{"topic"}),
		throttled: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_producer",
			Name:      "throttled_total",
			Help:      "Number of publishes rejected by backpressure.",
		}, []string{"topic"}),
	}
}

func Run(ctx context.Context, cfg Config, client *transport.Client, m *producerMetrics) error {
	ok, sc := benchmarks.AllScenarios().FindByName(cfg.Scenario)
	if !ok {
		return oops.Errorf("unknown scenario: %s", cfg.Scenario)
	}

	if err := client.CreateTopic(ctx, cfg.Topic, int32(cfg.Partitions), pb.ScheduleMode_FIFO); err != nil &&
		!errors.Is(err, broker.ErrTopicAlreadyExists) {
		return oops.Wrapf(err, "create topic %q", cfg.Topic)
	}

	if cfg.Duration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Duration)
		defer cancel()
	}

	pool := pond.NewPool(cfg.Producers)
	defer pool.StopAndWait()
	m.up.Set(1)
	defer m.up.Set(0)

	var seq uint64
	for i := 0; i < cfg.Producers; i++ {
		workerID := i
		pool.Submit(func() {
			runWorker(ctx, cfg, &sc, client, m, workerID, &seq)
		})
	}

	<-ctx.Done()
	return nil
}

func runWorker(
	ctx context.Context,
	cfg Config,
	sc *benchmarks.Scenario,
	client *transport.Client,
	m *producerMetrics,
	workerID int,
	seq *uint64,
) {
	var interval time.Duration
	if sc.RatePerSec > 0 && cfg.Producers > 0 {
		perWorker := sc.RatePerSec / cfg.Producers
		if perWorker < 1 {
			perWorker = 1
		}
		interval = time.Second / time.Duration(perWorker)
	}

	for {
		if ctx.Err() != nil {
			return
		}

		var (
			priority = sc.SamplePriority()
			key      = fmt.Sprintf("w%d-%d", workerID, atomic.AddUint64(seq, 1))
			value    = encodeValue(sc.MsgSize)
		)

		start := time.Now()
		_, _, err := client.Produce(ctx, cfg.Topic, []byte(key), value, uint32(priority))
		elapsed := time.Since(start)

		if err != nil {
			if ctx.Err() != nil {
				return
			}
			code := status.Code(err)
			m.errors.WithLabelValues(cfg.Topic, code.String()).Inc()
			if code == codes.ResourceExhausted {
				m.throttled.WithLabelValues(cfg.Topic).Inc()
			}
		} else {
			m.published.WithLabelValues(cfg.Topic, strconv.Itoa(int(priority))).Inc()
			m.latency.WithLabelValues(cfg.Topic).Observe(elapsed.Seconds())
		}

		if interval > 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(interval):
			}
		}
	}
}

func encodeValue(size int) []byte {
	if size < timestampPrefixBytes {
		size = timestampPrefixBytes
	}
	buf := benchmarks.GenerateMessage(size)
	binary.BigEndian.PutUint64(buf[:timestampPrefixBytes], uint64(time.Now().UnixNano()))
	return buf
}
