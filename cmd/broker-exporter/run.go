package main

import (
	"context"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"gonum.org/v1/gonum/stat"

	"github.com/Nyamerka/NyaQueue/pkg/transport"
)

type exporterMetrics struct {
	throughput         prometheus.Gauge
	avgLatency         prometheus.Gauge
	successRate        prometheus.Gauge
	partitionLoad      *prometheus.GaugeVec
	queueDepth         *prometheus.GaugeVec
	partitionLoadStdev prometheus.Gauge
	scrapeErrors       prometheus.Counter
	lastScrapeSuccess  prometheus.Gauge
}

func registerMetrics(reg prometheus.Registerer) *exporterMetrics {
	f := promauto.With(reg)
	return &exporterMetrics{
		throughput: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "throughput_messages_per_sec",
			Help:      "Messages per second aggregated across all partitions.",
		}),
		avgLatency: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "avg_latency_seconds",
			Help:      "Average internal broker latency.",
		}),
		successRate: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "success_rate",
			Help:      "Fraction of deliveries that succeeded.",
		}),
		partitionLoad: f.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "partition_load",
			Help:      "Normalized load per partition in [0, 1].",
		}, []string{"partition"}),
		queueDepth: f.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "queue_depth",
			Help:      "Number of pending messages per partition.",
		}, []string{"partition"}),
		partitionLoadStdev: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker",
			Name:      "partition_load_stddev",
			Help:      "Standard deviation of partition load — queue balance indicator.",
		}),
		scrapeErrors: f.NewCounter(prometheus.CounterOpts{
			Namespace: "nyaqueue_broker_exporter",
			Name:      "scrape_errors_total",
			Help:      "Number of failed scrapes against the broker.",
		}),
		lastScrapeSuccess: f.NewGauge(prometheus.GaugeOpts{
			Namespace: "nyaqueue_broker_exporter",
			Name:      "last_scrape_success",
			Help:      "1 if the last scrape succeeded, 0 otherwise.",
		}),
	}
}

func Run(ctx context.Context, cfg Config, client *transport.Client, m *exporterMetrics) error {
	if err := scrape(ctx, client, m); err != nil {
		m.scrapeErrors.Inc()
		m.lastScrapeSuccess.Set(0)
	}

	ticker := time.NewTicker(cfg.ScrapeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := scrape(ctx, client, m); err != nil {
				m.scrapeErrors.Inc()
				m.lastScrapeSuccess.Set(0)
			}
		}
	}
}

func scrape(ctx context.Context, client *transport.Client, m *exporterMetrics) error {
	resp, err := client.GetMetrics(ctx)
	if err != nil {
		return err
	}

	m.throughput.Set(resp.Throughput)
	m.avgLatency.Set(resp.AvgLatency)
	m.successRate.Set(resp.SuccessRate)

	for i, load := range resp.PartitionLoads {
		m.partitionLoad.WithLabelValues(strconv.Itoa(i)).Set(load)
	}
	for i, depth := range resp.QueueDepth {
		m.queueDepth.WithLabelValues(strconv.Itoa(i)).Set(float64(depth))
	}

	if len(resp.PartitionLoads) > 1 {
		m.partitionLoadStdev.Set(stat.StdDev(resp.PartitionLoads, nil))
	} else {
		m.partitionLoadStdev.Set(0)
	}

	m.lastScrapeSuccess.Set(1)
	return nil
}
