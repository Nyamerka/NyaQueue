package transport

import (
	"context"
	"fmt"
	"time"

	pb "github.com/Nyamerka/NyaQueue/pkg/transport/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is a producer/consumer SDK wrapping gRPC calls.
type Client struct {
	conn   *grpc.ClientConn
	client pb.NyaQueueClient
}

func NewClient(addr string) (*Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to %s: %w", addr, err)
	}

	return &Client{
		conn:   conn,
		client: pb.NewNyaQueueClient(conn),
	}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

// Produce sends a message to the broker.
func (c *Client) Produce(ctx context.Context, topic string, key, value []byte, priority uint32) (int32, int64, error) {
	resp, err := c.client.Produce(ctx, &pb.ProduceRequest{
		Topic:    topic,
		Key:      key,
		Value:    value,
		Priority: priority,
	})
	if err != nil {
		return 0, 0, err
	}
	return resp.Partition, resp.Offset, nil
}

// Consume fetches the next message from a partition.
func (c *Client) Consume(ctx context.Context, topic, group string, partition int32, maxBytes int32) ([]*pb.MessageEnvelope, error) {
	resp, err := c.client.Consume(ctx, &pb.ConsumeRequest{
		Topic:     topic,
		Group:     group,
		Partition: partition,
		MaxBytes:  maxBytes,
	})
	if err != nil {
		return nil, err
	}
	return resp.Messages, nil
}

// Commit acknowledges a consumer offset.
func (c *Client) Commit(ctx context.Context, topic, group string, partition int32, offset int64) error {
	_, err := c.client.Commit(ctx, &pb.CommitRequest{
		Topic:     topic,
		Group:     group,
		Partition: partition,
		Offset:    offset,
	})
	return err
}

// CreateTopic creates a new topic on the broker.
func (c *Client) CreateTopic(ctx context.Context, topic string, numPartitions int32, mode pb.ScheduleMode) error {
	_, err := c.client.CreateTopic(ctx, &pb.CreateTopicRequest{
		Topic:         topic,
		NumPartitions: numPartitions,
		ScheduleConfig: &pb.TopicScheduleConfig{
			Mode: mode,
		},
	})
	return err
}

// DeleteTopic removes a topic from the broker.
func (c *Client) DeleteTopic(ctx context.Context, topic string) error {
	_, err := c.client.DeleteTopic(ctx, &pb.DeleteTopicRequest{Topic: topic})
	return err
}

// ListTopics lists all topics on the broker.
func (c *Client) ListTopics(ctx context.Context) ([]*pb.TopicInfo, error) {
	resp, err := c.client.ListTopics(ctx, &pb.ListTopicsRequest{})
	if err != nil {
		return nil, err
	}
	return resp.Topics, nil
}

// GetMetrics retrieves current broker metrics.
func (c *Client) GetMetrics(ctx context.Context) (*pb.MetricsResponse, error) {
	return c.client.GetMetrics(ctx, &pb.MetricsRequest{})
}
