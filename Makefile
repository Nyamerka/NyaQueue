.PHONY: build broker producer consumer broker-exporter experiment proto clean test bench lint \
        compose-up compose-down compose-build compose-experiment compose-kafka

BINARY_DIR := bin
PROTO_DIR  := pkg/proto

build: broker producer consumer broker-exporter experiment

broker:
	go build -o $(BINARY_DIR)/broker          ./cmd/broker

producer:
	go build -o $(BINARY_DIR)/producer        ./cmd/producer

consumer:
	go build -o $(BINARY_DIR)/consumer        ./cmd/consumer

broker-exporter:
	go build -o $(BINARY_DIR)/broker-exporter ./cmd/broker-exporter

experiment:
	go build -o $(BINARY_DIR)/experiment      ./cmd/experiment

proto:
	protoc \
		--go_out=$(PROTO_DIR) --go_opt=paths=source_relative \
		--go-grpc_out=$(PROTO_DIR) --go-grpc_opt=paths=source_relative \
		-I $(PROTO_DIR) \
		$(PROTO_DIR)/model.proto $(PROTO_DIR)/request.proto \
		$(PROTO_DIR)/response.proto $(PROTO_DIR)/queue.proto

clean:
	rm -rf $(BINARY_DIR) data/

test:
	go test ./... -count=1

bench:
	go test ./benchmarks/ -bench=. -benchmem -count=3

lint:
	golangci-lint run ./...

run-experiment:
	go run ./cmd/experiment --mode=inprocess --scenarios=all --algorithms=all --duration=10s

compose-build:
	docker compose -f deploy/docker-compose.yml build

compose-up:
	docker compose -f deploy/docker-compose.yml up -d

compose-down:
	docker compose -f deploy/docker-compose.yml down -v

compose-kafka:
	docker compose -f deploy/docker-compose.yml --profile kafka up -d

compose-experiment:
	docker compose -f deploy/docker-compose.yml --profile experiment run --build --rm experiment
