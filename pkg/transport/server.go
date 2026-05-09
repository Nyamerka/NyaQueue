package transport

import (
	"context"
	"errors"
	"log"
	"net"
	"time"

	"github.com/samber/oops"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

var ErrServerOverloaded = status.Error(codes.ResourceExhausted, "server overloaded: max queued requests reached")

type Server struct {
	pb.UnimplementedNyaQueueServer

	broker   *broker.Broker
	grpc     *grpc.Server
	listener net.Listener
	reqSem   chan struct{}
}

func NewServer(b *broker.Broker) *Server {
	cfg := b.Config()

	var sem chan struct{}
	if cfg.MaxQueuedRequests > 0 {
		sem = make(chan struct{}, cfg.MaxQueuedRequests)
	}

	semaphoreInterceptor := func(
		ctx context.Context,
		req any,
		_ *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (any, error) {
		if sem != nil {
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			default:
				return nil, ErrServerOverloaded
			}
		}
		return handler(ctx, req)
	}

	opts := []grpc.ServerOption{
		grpc.MaxConcurrentStreams(uint32(cfg.MaxConnections)),
		grpc.NumStreamWorkers(uint32(cfg.NumNetworkGoroutines)),
		grpc.MaxRecvMsgSize(cfg.MaxMessageBytes + 1024),
		grpc.MaxSendMsgSize(cfg.MaxFetchBytes + 1024),
		grpc.ReadBufferSize(cfg.RecvBufferBytes),
		grpc.WriteBufferSize(cfg.SendBufferBytes),
		grpc.UnaryInterceptor(semaphoreInterceptor),
	}

	opts = append(opts, grpc.KeepaliveParams(keepalive.ServerParameters{
		MaxConnectionAge:      4 * time.Minute,
		MaxConnectionAgeGrace: 30 * time.Second,
		MaxConnectionIdle:     5 * time.Minute,
		Time:                  10 * time.Second,
		Timeout:               1 * time.Second,
	}))

	return &Server{
		broker: b,
		grpc:   grpc.NewServer(opts...),
		reqSem: sem,
	}
}

func (s *Server) Start(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return oops.Wrapf(err, "listen %s", addr)
	}
	s.listener = lis

	pb.RegisterNyaQueueServer(s.grpc, s)

	go s.grpc.Serve(lis)
	return nil
}

func (s *Server) Stop() {
	s.grpc.GracefulStop()
}

func (s *Server) Addr() string {
	if s.listener != nil {
		return s.listener.Addr().String()
	}
	return ""
}

func (s *Server) Produce(_ context.Context, req *pb.ProduceRequest) (*pb.ProduceResponse, error) {
	if len(req.Messages) > 0 {
		return s.produceBatch(req)
	}
	return s.produceSingle(req)
}

func (s *Server) produceSingle(req *pb.ProduceRequest) (*pb.ProduceResponse, error) {
	msg := &broker.Message{
		Header: broker.MessageHeader{
			Priority:  uint8(req.Priority),
			Timestamp: time.Now().UnixNano(),
		},
		Key:   req.Key,
		Value: req.Value,
	}

	partition, offset, err := s.broker.Publish(req.Topic, msg)
	if err != nil {
		return nil, err
	}

	return &pb.ProduceResponse{
		Partition: int32(partition),
		Offset:    int64(offset),
	}, nil
}

func (s *Server) produceBatch(req *pb.ProduceRequest) (*pb.ProduceResponse, error) {
	now := time.Now().UnixNano()
	msgs := make([]*broker.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = &broker.Message{
			Header: broker.MessageHeader{
				Priority:  uint8(m.Priority),
				Timestamp: now,
			},
			Key:   m.Key,
			Value: m.Value,
		}
	}

	batchResults := s.broker.PublishBatch(req.Topic, msgs)

	results := make([]*pb.ProduceResult, len(batchResults))
	var firstErr error
	for i, r := range batchResults {
		if r.Err != nil && firstErr == nil {
			firstErr = r.Err
		}
		results[i] = &pb.ProduceResult{
			Partition: int32(r.Partition),
			Offset:    int64(r.Offset),
		}
	}

	if firstErr != nil && allFailed(batchResults) {
		return nil, firstErr
	}

	return &pb.ProduceResponse{Results: results}, nil
}

func allFailed(results []broker.PublishResult) bool {
	for _, r := range results {
		if r.Err == nil {
			return false
		}
	}
	return true
}

// Consume fetches messages up to maxBytes, tracking the offset locally within
// the batch. A single Commit at the end avoids per-message offset store writes.
func (s *Server) Consume(_ context.Context, req *pb.ConsumeRequest) (*pb.ConsumeResponse, error) {
	cfg := s.broker.Config()

	maxBytes := int(req.MaxBytes)
	if maxBytes <= 0 {
		maxBytes = cfg.MaxFetchBytes
	}
	if cfg.MaxFetchBytes > 0 && maxBytes > cfg.MaxFetchBytes {
		maxBytes = cfg.MaxFetchBytes
	}

	fetchMinBytes := cfg.FetchMinBytes
	fetchMaxWait := time.Duration(cfg.FetchMaxWaitMs) * time.Millisecond
	deadline := time.Now().Add(fetchMaxWait)

	// Load offset once; advance locally in the loop.
	currentOffset, err := s.broker.LoadOffset(req.Group, req.Topic, int(req.Partition))
	if err != nil {
		currentOffset = 1
	}

	var (
		envelopes  []*pb.MessageEnvelope
		totalBytes int
		lastOffset int64 = -1
	)

	for totalBytes < maxBytes {
		msg, nextOffset, err := s.broker.ConsumeFrom(req.Topic, req.Group, int(req.Partition), currentOffset)
		if err != nil {
			if errors.Is(err, broker.ErrNoMessages) {
				if totalBytes < fetchMinBytes && time.Now().Before(deadline) {
					time.Sleep(time.Millisecond)
					continue
				}
				break
			}
			if len(envelopes) > 0 {
				break
			}
			return nil, err
		}

		env := &pb.MessageEnvelope{
			Offset:    int64(nextOffset) - 1,
			Key:       msg.Key,
			Value:     msg.Value,
			Priority:  uint32(msg.Header.Priority),
			Timestamp: msg.Header.Timestamp,
		}
		envelopes = append(envelopes, env)
		totalBytes += len(msg.Key) + len(msg.Value)
		lastOffset = int64(nextOffset)
		currentOffset = nextOffset
	}

	if lastOffset >= 0 {
		if err := s.broker.Commit(req.Group, req.Topic, int(req.Partition), lastOffset); err != nil {
			log.Printf("auto-commit failed (topic=%s group=%s partition=%d offset=%d): %v",
				req.Topic, req.Group, req.Partition, lastOffset, err)
		}
	}

	return &pb.ConsumeResponse{Messages: envelopes}, nil
}

func (s *Server) Commit(_ context.Context, req *pb.CommitRequest) (*pb.CommitResponse, error) {
	err := s.broker.Commit(req.Group, req.Topic, int(req.Partition), req.Offset)
	if err != nil {
		return nil, err
	}
	return &pb.CommitResponse{}, nil
}

func (s *Server) CreateTopic(_ context.Context, req *pb.CreateTopicRequest) (*pb.CreateTopicResponse, error) {
	cfg := broker.DefaultTopicConfig()
	cfg.NumPartitions = int(req.NumPartitions)

	if req.ScheduleConfig != nil {
		switch req.ScheduleConfig.Mode {
		case pb.ScheduleMode_STRICT_PRIORITY:
			cfg.ScheduleMode = broker.ModeStrictPriority
		case pb.ScheduleMode_DQN_ADAPTIVE:
			cfg.ScheduleMode = broker.ModeDQNAdaptive
		default:
			cfg.ScheduleMode = broker.ModeFIFO
		}
		if req.ScheduleConfig.PriorityLevels > 0 {
			cfg.PriorityLevels = int(req.ScheduleConfig.PriorityLevels)
		}
		if req.ScheduleConfig.AntiStarvationMs > 0 {
			cfg.AntiStarvationTTL = time.Duration(req.ScheduleConfig.AntiStarvationMs) * time.Millisecond
		}
		if req.ScheduleConfig.DqnThrottleOnLoad > 0 {
			cfg.DQNThrottleOnLoad = req.ScheduleConfig.DqnThrottleOnLoad
		}
	}

	if err := s.broker.CreateTopic(req.Topic, cfg); err != nil {
		return nil, mapBrokerError(err)
	}
	return &pb.CreateTopicResponse{}, nil
}

func (s *Server) DeleteTopic(_ context.Context, req *pb.DeleteTopicRequest) (*pb.DeleteTopicResponse, error) {
	if err := s.broker.DeleteTopic(req.Topic); err != nil {
		return nil, mapBrokerError(err)
	}
	return &pb.DeleteTopicResponse{}, nil
}

func (s *Server) ListTopics(_ context.Context, _ *pb.ListTopicsRequest) (*pb.ListTopicsResponse, error) {
	topics := s.broker.ListTopics()
	infos := make([]*pb.TopicInfo, len(topics))
	for i, t := range topics {
		mode := pb.ScheduleMode_FIFO
		switch t.Config().ScheduleMode {
		case broker.ModeStrictPriority:
			mode = pb.ScheduleMode_STRICT_PRIORITY
		case broker.ModeDQNAdaptive:
			mode = pb.ScheduleMode_DQN_ADAPTIVE
		}
		infos[i] = &pb.TopicInfo{
			Topic:         t.Name(),
			NumPartitions: int32(t.NumPartitions()),
			Mode:          mode,
		}
	}
	return &pb.ListTopicsResponse{Topics: infos}, nil
}

func (s *Server) GetMetrics(_ context.Context, _ *pb.MetricsRequest) (*pb.MetricsResponse, error) {
	m := s.broker.Metrics()

	depths := make([]int64, len(m.QueueDepth))
	for i, d := range m.QueueDepth {
		depths[i] = int64(d)
	}

	return &pb.MetricsResponse{
		Throughput:     m.Throughput,
		AvgLatency:     m.AvgLatency,
		PartitionLoads: m.PartitionLoads,
		SuccessRate:    m.SuccessRate,
		QueueDepth:     depths,
		LoadStddev:     m.LoadStdDev,
		MsgRate:        m.MsgRate,
		AvgMsgSize:     m.AvgMsgSize,
		PredictedLoads: m.PredictedLoads,
	}, nil
}
