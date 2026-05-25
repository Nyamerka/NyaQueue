ARG GO_VERSION=1.26
FROM golang:${GO_VERSION}-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Broker and experiment — CGO_ENABLED=1 for XLA PJRT
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /bin/broker     ./cmd/broker
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /bin/experiment  ./cmd/experiment

# Others — no CGO, static binaries
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /bin/producer        ./cmd/producer
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /bin/consumer        ./cmd/consumer
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /bin/broker-exporter ./cmd/broker-exporter

# Broker/experiment image — needs glibc for XLA .so and PJRT plugin
FROM debian:bookworm-slim AS ml-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g 1000 nyaqueue && \
    useradd -u 1000 -g nyaqueue nyaqueue

FROM ml-base AS broker
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY deploy/configs/broker.yaml /etc/nyaqueue/config.yaml
RUN mkdir -p /data && chown -R nyaqueue:nyaqueue /data
USER nyaqueue
EXPOSE 9090 8080
ENTRYPOINT ["broker", "-config", "/etc/nyaqueue/config.yaml"]

FROM ml-base AS experiment
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY --from=builder /bin/experiment /usr/local/bin/experiment
COPY deploy/configs/broker.yaml /etc/nyaqueue/config.yaml
RUN mkdir -p /results
ENTRYPOINT ["experiment"]

# Others — alpine (no CGO dependencies)
FROM alpine:3.20 AS base-alpine
RUN apk add --no-cache ca-certificates && \
    addgroup -S -g 1000 nyaqueue && \
    adduser  -S -u 1000 -G nyaqueue nyaqueue

FROM base-alpine AS producer
COPY --from=builder /bin/producer /usr/local/bin/producer
COPY deploy/configs/producer.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8081
ENTRYPOINT ["producer", "-config", "/etc/nyaqueue/config.yaml"]

FROM base-alpine AS consumer
COPY --from=builder /bin/consumer /usr/local/bin/consumer
COPY deploy/configs/consumer.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8082
ENTRYPOINT ["consumer", "-config", "/etc/nyaqueue/config.yaml"]

FROM base-alpine AS broker-exporter
COPY --from=builder /bin/broker-exporter /usr/local/bin/broker-exporter
COPY deploy/configs/broker-exporter.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8083
ENTRYPOINT ["broker-exporter", "-config", "/etc/nyaqueue/config.yaml"]
