FROM golang:1.23-alpine AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /bin/broker     ./cmd/broker
RUN CGO_ENABLED=0 go build -o /bin/loadgen    ./cmd/loadgen
RUN CGO_ENABLED=0 go build -o /bin/experiment ./cmd/experiment

# --- broker ---
FROM alpine:3.20 AS broker
RUN apk add --no-cache ca-certificates
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY deploy/configs/nyaqueue.yaml /etc/nyaqueue/config.yaml
EXPOSE 9090
ENTRYPOINT ["broker", "-config", "/etc/nyaqueue/config.yaml"]

# --- loadgen ---
FROM alpine:3.20 AS loadgen
RUN apk add --no-cache ca-certificates
COPY --from=builder /bin/loadgen /usr/local/bin/loadgen
ENTRYPOINT ["loadgen"]

# --- experiment ---
FROM alpine:3.20 AS experiment
RUN apk add --no-cache ca-certificates
COPY --from=builder /bin/broker /usr/local/bin/broker
COPY --from=builder /bin/experiment /usr/local/bin/experiment
COPY deploy/configs/nyaqueue.yaml /etc/nyaqueue/config.yaml
ENTRYPOINT ["experiment"]
