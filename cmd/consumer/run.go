package main

import (
	"context"
	"encoding/binary"
	"errors"
	"strconv"
	"time"

	"github.com/alitto/pond/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/samber/oops"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

const (
	timestampPrefixBytes = 8
	fetchMaxBytes        = 1 << 20
	idleSleep            = 50 * time.Millisecond
	resolveRetryDelay    = time.Second
)

type consumerMetrics struct {
	consumed      *prometheus.CounterVec
	errors        *prometheus.CounterVec
	e2eLatency    *prometheus.HistogramVec
	commitLatency *prometheus.HistogramVec
	parseErrors   *prometheus.CounterVec
}

func registerMetrics(reg prometheus.Registerer) *consumerMetrics {
	f := promauto.With(reg)
	latencyBuckets := prometheus.ExponentialBuckets(0.0001, 2, 18)
	return &consumerMetrics{
		consumed: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_consumer",
			Name:      "messages_consumed_total",
			Help:      "Number of messages consumed successfully.",
		}, []string{"topic", "partition", "priority"}),
		errors: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_consumer",
			Name:      "consume_errors_total",
			Help:      "Number of consume errors grouped by gRPC code.",
		}, []string{"topic", "code"}),
		e2eLatency: f.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "nyaqueue_consumer",
			Name:      "e2e_latency_seconds",
			Help:      "End-to-end latency from publish timestamp to consume time.",
			Buckets:   latencyBuckets,
		}, []string{"topic", "partition"}),
		commitLatency: f.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "nyaqueue_consumer",
			Name:      "commit_latency_seconds",
			Help:      "Commit RPC round-trip latency.",
			Buckets:   latencyBuckets,
		}, []string{"topic"}),
		parseErrors: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nyaqueue_consumer",
			Name:      "parse_errors_total",
			Help:      "Number of messages with malformed timestamp prefix.",
		}, []string{"topic"}),
	}
}

func Run(ctx context.Context, cfg Config, client *transport.Client, m *consumerMetrics) error {
	partitions, err := resolvePartitions(ctx, client, cfg)
	if err != nil {
		return oops.Wrapf(err, "resolve partitions")
	}
	if len(partitions) == 0 {
		return oops.Errorf("no partitions to consume on topic %q", cfg.Topic)
	}

	pool := pond.NewPool(cfg.Workers)
	defer pool.StopAndWait()

	for _, p := range partitions {
		partition := p
		pool.Submit(func() {
			runWorker(ctx, cfg, client, m, int32(partition))
		})
	}

	<-ctx.Done()
	return nil
}

func resolvePartitions(ctx context.Context, client *transport.Client, cfg Config) ([]int, error) {
	if len(cfg.Partitions) > 0 {
		return cfg.Partitions, nil
	}
	for {
		topics, err := client.ListTopics(ctx)
		if err != nil {
			return nil, err
		}
		for _, t := range topics {
			if t.Topic != cfg.Topic {
				continue
			}
			out := make([]int, t.NumPartitions)
			for i := range out {
				out[i] = i
			}
			return out, nil
		}
		select {
		case <-ctx.Done():
			return nil, oops.Wrapf(ctx.Err(), "waiting for topic %q", cfg.Topic)
		case <-time.After(resolveRetryDelay):
		}
	}
}

func runWorker(
	ctx context.Context,
	cfg Config,
	client *transport.Client,
	m *consumerMetrics,
	partition int32,
) {
	partLabel := strconv.Itoa(int(partition))

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		msgs, err := client.Consume(ctx, cfg.Topic, cfg.Group, partition, fetchMaxBytes)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			m.errors.WithLabelValues(cfg.Topic, status.Code(err).String()).Inc()
			time.Sleep(idleSleep)
			continue
		}

		if len(msgs) == 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(idleSleep):
			}
			continue
		}

		now := time.Now()
		for _, env := range msgs {
			observeE2E(m, cfg.Topic, partLabel, env, now)
			m.consumed.WithLabelValues(cfg.Topic, partLabel, strconv.FormatUint(uint64(env.Priority), 10)).Inc()

			commitStart := time.Now()
			if err := client.Commit(ctx, cfg.Topic, cfg.Group, partition, env.Offset+1); err != nil {
				if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
					return
				}
				m.errors.WithLabelValues(cfg.Topic, status.Code(err).String()).Inc()
				continue
			}
			m.commitLatency.WithLabelValues(cfg.Topic).Observe(time.Since(commitStart).Seconds())
		}
	}
}

func observeE2E(m *consumerMetrics, topic, partLabel string, env *pb.MessageEnvelope, now time.Time) {
	if len(env.Value) < timestampPrefixBytes {
		m.parseErrors.WithLabelValues(topic).Inc()
		return
	}
	ts := int64(binary.BigEndian.Uint64(env.Value[:timestampPrefixBytes]))
	latency := now.Sub(time.Unix(0, ts)).Seconds()
	if latency < 0 {
		m.parseErrors.WithLabelValues(topic).Inc()
		return
	}
	m.e2eLatency.WithLabelValues(topic, partLabel).Observe(latency)
}
