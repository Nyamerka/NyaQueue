package transport

import (
	"context"
	"sync"
	"time"

	"github.com/samber/oops"

	pb "github.com/Nyamerka/NyaQueue/pkg/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

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
		return nil, oops.Wrapf(err, "connect to %s", addr)
	}

	return &Client{
		conn:   conn,
		client: pb.NewNyaQueueClient(conn),
	}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

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

// ProduceBatch sends a batch of messages in a single RPC.
func (c *Client) ProduceBatch(ctx context.Context, topic string, msgs []*pb.ProduceMessage) ([]*pb.ProduceResult, error) {
	resp, err := c.client.Produce(ctx, &pb.ProduceRequest{
		Topic:    topic,
		Messages: msgs,
	})
	if err != nil {
		return nil, err
	}
	return resp.Results, nil
}

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

func (c *Client) Commit(ctx context.Context, topic, group string, partition int32, offset int64) error {
	_, err := c.client.Commit(ctx, &pb.CommitRequest{
		Topic:     topic,
		Group:     group,
		Partition: partition,
		Offset:    offset,
	})
	return err
}

func (c *Client) CreateTopic(ctx context.Context, topic string, numPartitions int32, mode pb.ScheduleMode) error {
	_, err := c.client.CreateTopic(ctx, &pb.CreateTopicRequest{
		Topic:         topic,
		NumPartitions: numPartitions,
		ScheduleConfig: &pb.TopicScheduleConfig{
			Mode: mode,
		},
	})
	return mapClientError(err)
}

func (c *Client) DeleteTopic(ctx context.Context, topic string) error {
	_, err := c.client.DeleteTopic(ctx, &pb.DeleteTopicRequest{Topic: topic})
	return mapClientError(err)
}

func (c *Client) ListTopics(ctx context.Context) ([]*pb.TopicInfo, error) {
	resp, err := c.client.ListTopics(ctx, &pb.ListTopicsRequest{})
	if err != nil {
		return nil, err
	}
	return resp.Topics, nil
}

func (c *Client) GetMetrics(ctx context.Context) (*pb.MetricsResponse, error) {
	return c.client.GetMetrics(ctx, &pb.MetricsRequest{})
}

// ---------- BufferedProducer ----------

// BufferedProducer accumulates messages and sends them in batches via a single
// Produce RPC, amortising the per-RPC cost across many messages.
type BufferedProducer struct {
	client    *Client
	topic     string
	batchSize int
	linger    time.Duration

	bgCtx    context.Context
	bgCancel context.CancelFunc

	mu     sync.Mutex
	buf    []*pb.ProduceMessage
	timer  *time.Timer
	asyncErr error
	closed bool
}

// NewBufferedProducer creates a producer that flushes when either batchSize
// messages are accumulated or linger time elapses, whichever comes first.
func NewBufferedProducer(c *Client, topic string, batchSize int, linger time.Duration) *BufferedProducer {
	bgCtx, cancel := context.WithCancel(context.Background())
	return &BufferedProducer{
		client:    c,
		topic:     topic,
		batchSize: batchSize,
		linger:    linger,
		bgCtx:     bgCtx,
		bgCancel:  cancel,
		buf:       make([]*pb.ProduceMessage, 0, batchSize),
	}
}


func (p *BufferedProducer) Send(ctx context.Context, key, value []byte, priority uint32) error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return oops.Errorf("buffered producer: closed")
	}

	if err := p.asyncErr; err != nil {
		p.asyncErr = nil
		p.mu.Unlock()
		return err
	}

	p.buf = append(p.buf, &pb.ProduceMessage{
		Key:      key,
		Value:    value,
		Priority: priority,
	})

	if len(p.buf) >= p.batchSize {
		batch := p.buf
		p.buf = make([]*pb.ProduceMessage, 0, p.batchSize)
		p.stopTimerLocked()
		p.mu.Unlock()
		return p.send(ctx, batch)
	}

	if p.timer == nil && p.linger > 0 {
		p.timer = time.AfterFunc(p.linger, p.flushFromTimer)
	}
	p.mu.Unlock()
	return nil
}

// Flush sends any buffered messages immediately.
func (p *BufferedProducer) Flush(ctx context.Context) error {
	p.mu.Lock()
	if len(p.buf) == 0 {
		p.mu.Unlock()
		return nil
	}
	batch := p.buf
	p.buf = make([]*pb.ProduceMessage, 0, p.batchSize)
	p.stopTimerLocked()
	p.mu.Unlock()
	return p.send(ctx, batch)
}

// Close drains any buffered messages, stops the linger timer, and prevents
// further Send calls. Returns the error from the final flush (or nil).
func (p *BufferedProducer) Close(ctx context.Context) error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	batch := p.buf
	p.buf = nil
	p.stopTimerLocked()
	p.mu.Unlock()

	p.bgCancel()

	if len(batch) == 0 {
		return nil
	}
	return p.send(ctx, batch)
}

func (p *BufferedProducer) flushFromTimer() {
	if err := p.Flush(p.bgCtx); err != nil {
		p.mu.Lock()
		if p.asyncErr == nil {
			p.asyncErr = err
		}
		p.mu.Unlock()
	}
}

func (p *BufferedProducer) send(ctx context.Context, batch []*pb.ProduceMessage) error {
	_, err := p.client.ProduceBatch(ctx, p.topic, batch)
	return err
}

func (p *BufferedProducer) stopTimerLocked() {
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}
}
