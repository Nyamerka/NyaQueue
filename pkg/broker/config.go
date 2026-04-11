package broker

import (
	"time"

	"github.com/nobl9/govy/pkg/govy"
	"github.com/nobl9/govy/pkg/rules"
)

type ScheduleMode int

const (
	ModeFIFO           ScheduleMode = iota // strict FIFO, priority ignored
	ModeStrictPriority                     // per-priority FIFO (highest first)
	ModeDQNAdaptive                        // DQN balances priority vs fairness
)

type Config struct {
	// Storage (5)
	SegmentMaxBytes   int           `koanf:"segment_max_bytes"`
	SegmentMaxCount   int           `koanf:"segment_max_count"`
	RetentionMaxBytes int64         `koanf:"retention_max_bytes"`
	RetentionMaxAge   time.Duration `koanf:"retention_max_age"`
	FlushIntervalMs   int           `koanf:"flush_interval_ms"`

	// Compression (1)
	CompressionType int `koanf:"compression_type"` // 0=none, 1=snappy, 2=gzip, 3=lz4

	// Network (7)
	MaxConnections       int `koanf:"max_connections"`
	RecvBufferBytes      int `koanf:"recv_buffer_bytes"`
	SendBufferBytes      int `koanf:"send_buffer_bytes"`
	MaxMessageBytes      int `koanf:"max_message_bytes"`
	ReadTimeoutMs        int `koanf:"read_timeout_ms"`
	WriteTimeoutMs       int `koanf:"write_timeout_ms"`
	NumNetworkGoroutines int `koanf:"num_network_goroutines"`

	// IO (2)
	NumIOGoroutines   int `koanf:"num_io_goroutines"`
	MaxQueuedRequests int `koanf:"max_queued_requests"`

	// Producer batching (3)
	BatchSize        int `koanf:"batch_size"`
	BatchMemoryBytes int `koanf:"batch_memory_bytes"`
	LingerMs         int `koanf:"linger_ms"`

	// Consumer (4)
	FetchMinBytes            int `koanf:"fetch_min_bytes"`
	FetchMaxWaitMs           int `koanf:"fetch_max_wait_ms"`
	MaxFetchBytes            int `koanf:"max_fetch_bytes"`
	ConsumerSessionTimeoutMs int `koanf:"consumer_session_timeout_ms"`
}

var configValidator = govy.New(
	govy.For(func(c Config) int { return c.SegmentMaxBytes }).
		WithName("segment_max_bytes").
		Rules(rules.GTE(1<<20), rules.LTE(64<<20)),
	govy.For(func(c Config) int { return c.SegmentMaxCount }).
		WithName("segment_max_count").
		Rules(rules.GTE(1)),
	govy.For(func(c Config) int64 { return c.RetentionMaxBytes }).
		WithName("retention_max_bytes").
		Rules(rules.GTE[int64](1<<20)),
	govy.For(func(c Config) int { return c.FlushIntervalMs }).
		WithName("flush_interval_ms").
		Rules(rules.GTE(1), rules.LTE(60000)),
	govy.For(func(c Config) int { return c.CompressionType }).
		WithName("compression_type").
		Rules(rules.GTE(0), rules.LTE(3)),
	govy.For(func(c Config) int { return c.MaxConnections }).
		WithName("max_connections").
		Rules(rules.GTE(1), rules.LTE(65535)),
	govy.For(func(c Config) int { return c.RecvBufferBytes }).
		WithName("recv_buffer_bytes").
		Rules(rules.GTE(1024)),
	govy.For(func(c Config) int { return c.SendBufferBytes }).
		WithName("send_buffer_bytes").
		Rules(rules.GTE(1024)),
	govy.For(func(c Config) int { return c.MaxMessageBytes }).
		WithName("max_message_bytes").
		Rules(rules.GTE(1024)),
	govy.For(func(c Config) int { return c.NumNetworkGoroutines }).
		WithName("num_network_goroutines").
		Rules(rules.GTE(1), rules.LTE(256)),
	govy.For(func(c Config) int { return c.NumIOGoroutines }).
		WithName("num_io_goroutines").
		Rules(rules.GTE(1), rules.LTE(256)),
	govy.For(func(c Config) int { return c.MaxQueuedRequests }).
		WithName("max_queued_requests").
		Rules(rules.GTE(1)),
	govy.For(func(c Config) int { return c.BatchSize }).
		WithName("batch_size").
		Rules(rules.GTE(1)),
	govy.For(func(c Config) int { return c.FetchMinBytes }).
		WithName("fetch_min_bytes").
		Rules(rules.GTE(1)),
	govy.For(func(c Config) int { return c.FetchMaxWaitMs }).
		WithName("fetch_max_wait_ms").
		Rules(rules.GTE(1)),
	govy.For(func(c Config) int { return c.ConsumerSessionTimeoutMs }).
		WithName("consumer_session_timeout_ms").
		Rules(rules.GTE(1000)),
)

// Validate checks Config fields against defined constraints.
func (c Config) Validate() error {
	return configValidator.WithName("broker.Config").Validate(c)
}

type TopicConfig struct {
	ScheduleMode      ScheduleMode
	NumPartitions     int
	PriorityLevels    int           // 1-10, default 1 = FIFO
	AntiStarvationTTL time.Duration // promote stale messages after this duration
	DQNThrottleOnLoad float64       // DQN falls back to FIFO at this load level
}

func DefaultConfig() Config {
	return Config{
		SegmentMaxBytes:   20 * 1024 * 1024, // 20MB
		SegmentMaxCount:   100,
		RetentionMaxBytes: 1 << 30, // 1GB
		RetentionMaxAge:   168 * time.Hour,
		FlushIntervalMs:   1000,

		CompressionType: 0,

		MaxConnections:       1024,
		RecvBufferBytes:      65536,
		SendBufferBytes:      65536,
		MaxMessageBytes:      1 << 20, // 1MB
		ReadTimeoutMs:        30000,
		WriteTimeoutMs:       30000,
		NumNetworkGoroutines: 4,

		NumIOGoroutines:   4,
		MaxQueuedRequests: 500,

		BatchSize:        100,
		BatchMemoryBytes: 65536,
		LingerMs:         5,

		FetchMinBytes:            1,
		FetchMaxWaitMs:           500,
		MaxFetchBytes:            1 << 20, // 1MB
		ConsumerSessionTimeoutMs: 30000,
	}
}

func DefaultTopicConfig() TopicConfig {
	return TopicConfig{
		ScheduleMode:      ModeFIFO,
		NumPartitions:     4,
		PriorityLevels:    1,
		AntiStarvationTTL: 10 * time.Second,
		DQNThrottleOnLoad: 0.9,
	}
}

const NumTunableParams = 22

// ParamRange describes bounds for one tunable parameter (used by DDPG optimizer).
type ParamRange struct {
	Name string
	Min  float64
	Max  float64
}

func TunableParamRanges() []ParamRange {
	return []ParamRange{
		{"SegmentMaxBytes", 1 << 20, 64 << 20},
		{"SegmentMaxCount", 10, 500},
		{"RetentionMaxBytes", 100 << 20, 10 << 30},
		{"RetentionMaxAge", float64(time.Hour), float64(720 * time.Hour)},
		{"FlushIntervalMs", 100, 10000},
		{"CompressionType", 0, 3},
		{"MaxConnections", 64, 8192},
		{"RecvBufferBytes", 4096, 1 << 20},
		{"SendBufferBytes", 4096, 1 << 20},
		{"MaxMessageBytes", 1024, 10 << 20},
		{"ReadTimeoutMs", 1000, 120000},
		{"WriteTimeoutMs", 1000, 120000},
		{"NumNetworkGoroutines", 1, 32},
		{"NumIOGoroutines", 1, 32},
		{"MaxQueuedRequests", 50, 5000},
		{"BatchSize", 1, 1000},
		{"BatchMemoryBytes", 1024, 1 << 20},
		{"LingerMs", 0, 100},
		{"FetchMinBytes", 1, 1 << 20},
		{"FetchMaxWaitMs", 10, 5000},
		{"MaxFetchBytes", 1024, 10 << 20},
		{"ConsumerSessionTimeoutMs", 5000, 120000},
	}
}
