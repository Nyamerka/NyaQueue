package transport

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	pb "github.com/Nyamerka/NyaQueue/pkg/transport/gen"
	"google.golang.org/grpc"
)

type Server struct {
	pb.UnimplementedNyaQueueServer

	broker   *broker.Broker
	grpc     *grpc.Server
	listener net.Listener
}

func NewServer(b *broker.Broker) *Server {
	return &Server{
		broker: b,
		grpc:   grpc.NewServer(),
	}
}

func (s *Server) Start(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
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

// --- gRPC method implementations ---

func (s *Server) Produce(_ context.Context, req *pb.ProduceRequest) (*pb.ProduceResponse, error) {
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

func (s *Server) Consume(_ context.Context, req *pb.ConsumeRequest) (*pb.ConsumeResponse, error) {
	msg, offset, err := s.broker.Consume(req.Topic, req.Group, int(req.Partition))
	if err != nil {
		return nil, err
	}

	env := &pb.MessageEnvelope{
		Offset:    int64(offset),
		Key:       msg.Key,
		Value:     msg.Value,
		Priority:  uint32(msg.Header.Priority),
		Timestamp: msg.Header.Timestamp,
	}

	return &pb.ConsumeResponse{
		Messages: []*pb.MessageEnvelope{env},
	}, nil
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

	err := s.broker.CreateTopic(req.Topic, cfg)
	if err != nil {
		return nil, err
	}
	return &pb.CreateTopicResponse{}, nil
}

func (s *Server) DeleteTopic(_ context.Context, req *pb.DeleteTopicRequest) (*pb.DeleteTopicResponse, error) {
	err := s.broker.DeleteTopic(req.Topic)
	if err != nil {
		return nil, err
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
	}, nil
}
