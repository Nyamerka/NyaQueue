# NyaQueue

Distributed message queue with ML-powered optimization, written in pure Go.

## Features

- **Topic/Partition model** — messages stored in WAL-backed partitions (`tidwall/wal`)
- **Priority scheduling** — per-topic FIFO with DQN-adaptive priority and anti-starvation
- **Pluggable balancers** — Round Robin, Weighted Round Robin, PSA, DQN
- **Pluggable schedulers** — FIFO, Strict Priority, DQN-adaptive
- **Predictive backpressure** — LSTM predicts overload, throttles producers proactively
- **Auto-configuration** — DDPG online optimizer tunes broker parameters
- **Persistent offsets** — consumer group offsets stored in `bbolt`
- **gRPC transport** — Produce, Consume, Admin APIs

## Quick Start

```bash
# Run via Docker
docker compose -f deploy/docker-compose.yml up -d nyaqueue

# Build and run locally
make build
./bin/broker
```

## Stack

| Layer | Implementation |
|-------|---------------|
| Storage | [`tidwall/wal`](https://github.com/tidwall/wal) — WAL segments per partition |
| Metadata | [`bbolt`](https://github.com/etcd-io/bbolt) — consumer group offsets |
| Math / ML | [`gonum`](https://gonum.org) + [`pehringer/simd`](https://github.com/pehringer/simd) |
| Config | [`koanf`](https://github.com/knadh/koanf) — YAML + env |
| Transport | `grpc` + `protobuf` |

## Configuration

Configuration is loaded from `config.yaml` and can be overridden via environment variables.

```yaml
broker:
  partitions: 4
  balancer: dqn        # rr | wrr | psa | dqn
  scheduler: dqn       # fifo | priority | dqn
  backpressure:
    enabled: true
    threshold: 0.85
  optimizer:
    enabled: true
    interval: 5s
```

## Development

```bash
make test        # run all tests
make bench       # run benchmarks
make lint        # golangci-lint
make experiment  # run in-process experiment across all algorithms
```

## License

MIT
