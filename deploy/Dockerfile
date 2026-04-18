FROM golang:1.25-alpine AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /bin/broker          ./cmd/broker
RUN CGO_ENABLED=0 go build -o /bin/producer        ./cmd/producer
RUN CGO_ENABLED=0 go build -o /bin/consumer        ./cmd/consumer
RUN CGO_ENABLED=0 go build -o /bin/broker-exporter ./cmd/broker-exporter
RUN CGO_ENABLED=0 go build -o /bin/experiment      ./cmd/experiment

FROM alpine:3.20 AS base
RUN apk add --no-cache ca-certificates && \
    addgroup -S -g 1000 nyaqueue && \
    adduser  -S -u 1000 -G nyaqueue nyaqueue

FROM base AS broker
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY deploy/configs/broker.yaml /etc/nyaqueue/config.yaml
RUN mkdir -p /data && chown -R nyaqueue:nyaqueue /data
USER nyaqueue
EXPOSE 9090
ENTRYPOINT ["broker", "-config", "/etc/nyaqueue/config.yaml"]

FROM base AS producer
COPY --from=builder /bin/producer /usr/local/bin/producer
COPY deploy/configs/producer.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8081
ENTRYPOINT ["producer", "-config", "/etc/nyaqueue/config.yaml"]

FROM base AS consumer
COPY --from=builder /bin/consumer /usr/local/bin/consumer
COPY deploy/configs/consumer.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8082
ENTRYPOINT ["consumer", "-config", "/etc/nyaqueue/config.yaml"]

FROM base AS broker-exporter
COPY --from=builder /bin/broker-exporter /usr/local/bin/broker-exporter
COPY deploy/configs/broker-exporter.yaml /etc/nyaqueue/config.yaml
USER nyaqueue
EXPOSE 8083
ENTRYPOINT ["broker-exporter", "-config", "/etc/nyaqueue/config.yaml"]

FROM base AS experiment
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY --from=builder /bin/experiment /usr/local/bin/experiment
COPY deploy/configs/broker.yaml /etc/nyaqueue/config.yaml
RUN mkdir -p /results && chown -R nyaqueue:nyaqueue /results
USER nyaqueue
ENTRYPOINT ["experiment"]
