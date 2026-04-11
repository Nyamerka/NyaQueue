.PHONY: build proto clean test bench lint \
       experiment docker-build docker-up docker-down docker-experiment

BINARY_DIR := bin
PROTO_DIR  := proto
GEN_DIR    := pkg/transport/gen

build:
	go build -o $(BINARY_DIR)/broker     ./cmd/broker
	go build -o $(BINARY_DIR)/loadgen    ./cmd/loadgen
	go build -o $(BINARY_DIR)/experiment ./cmd/experiment

proto:
	protoc \
		--go_out=$(GEN_DIR) --go_opt=paths=source_relative \
		--go-grpc_out=$(GEN_DIR) --go-grpc_opt=paths=source_relative \
		-I $(PROTO_DIR) \
		$(PROTO_DIR)/nyaqueue.proto

clean:
	rm -rf $(BINARY_DIR) data/

test:
	go test ./... -v -count=1

bench:
	go test ./benchmarks/ -bench=. -benchmem -count=3

lint:
	golangci-lint run ./...

experiment:
	go run ./cmd/experiment --mode=inprocess --scenarios=all --algorithms=all --duration=10s

docker-build:
	docker compose -f deploy/docker-compose.yml build

docker-up:
	docker compose -f deploy/docker-compose.yml up -d nyaqueue kafka

docker-down:
	docker compose -f deploy/docker-compose.yml down -v

docker-experiment:
	docker compose -f deploy/docker-compose.yml run --rm experiment
