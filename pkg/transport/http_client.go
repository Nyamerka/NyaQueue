package transport

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/samber/oops"
)

type HTTPClient struct {
	base   string
	client *http.Client
}

func NewHTTPClient(addr string) *HTTPClient {
	return &HTTPClient{
		base: "http://" + addr,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *HTTPClient) Close() error {
	c.client.CloseIdleConnections()
	return nil
}

type HTTPMetricsResponse struct {
	Throughput     float64   `json:"Throughput"`
	PartitionLoads []float64 `json:"PartitionLoads"`
	PredictedLoads []float64 `json:"PredictedLoads"`
	SuccessRate    float64   `json:"SuccessRate"`
}

func (c *HTTPClient) GetMetrics(ctx context.Context) (*HTTPMetricsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/metrics", nil)
	if err != nil {
		return nil, oops.Wrapf(err, "new request")
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, readHTTPError(resp)
	}

	var m HTTPMetricsResponse
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, oops.Wrapf(err, "decode metrics")
	}
	return &m, nil
}

func (c *HTTPClient) Produce(ctx context.Context, topic string, key, value []byte, priority uint32) (int, int64, error) {
	results, err := c.ProduceBatch(ctx, topic, []HTTPProduceRecord{{Key: key, Value: value, Priority: priority}})
	if err != nil {
		return 0, 0, err
	}
	if len(results) == 0 {
		return 0, 0, oops.Errorf("empty produce response")
	}
	return results[0].Partition, results[0].Offset, nil
}

func (c *HTTPClient) ProduceBatch(ctx context.Context, topic string, records []HTTPProduceRecord) ([]HTTPProduceResult, error) {
	body, err := json.Marshal(HTTPProduceRequest{Records: records})
	if err != nil {
		return nil, oops.Wrapf(err, "marshal")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.base+"/topics/"+url.PathEscape(topic)+"/messages",
		bytes.NewReader(body))
	if err != nil {
		return nil, oops.Wrapf(err, "new request")
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, readHTTPError(resp)
	}

	var result HTTPProduceResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, oops.Wrapf(err, "decode response")
	}
	return result.Results, nil
}

func (c *HTTPClient) Consume(ctx context.Context, topic, group string, partition int32, maxBytes int32) ([]HTTPMessageEnvelope, error) {
	u := fmt.Sprintf("%s/topics/%s/messages?group=%s&partition=%d&maxBytes=%d",
		c.base, url.PathEscape(topic), url.QueryEscape(group), partition, maxBytes)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, oops.Wrapf(err, "new request")
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, readHTTPError(resp)
	}

	var result HTTPConsumeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, oops.Wrapf(err, "decode response")
	}
	return result.Messages, nil
}

func (c *HTTPClient) Commit(ctx context.Context, topic, group string, partition int32, offset int64) error {
	body, err := json.Marshal(HTTPCommitRequest{
		Group:     group,
		Partition: int(partition),
		Offset:    offset,
	})
	if err != nil {
		return oops.Wrapf(err, "marshal")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.base+"/topics/"+url.PathEscape(topic)+"/offsets",
		bytes.NewReader(body))
	if err != nil {
		return oops.Wrapf(err, "new request")
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		return readHTTPError(resp)
	}
	return nil
}

func (c *HTTPClient) CreateTopic(ctx context.Context, topic string, numPartitions int32, mode string) error {
	body, err := json.Marshal(HTTPCreateTopicRequest{
		Topic:         topic,
		NumPartitions: numPartitions,
		Mode:          mode,
	})
	if err != nil {
		return oops.Wrapf(err, "marshal")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/topics", bytes.NewReader(body))
	if err != nil {
		return oops.Wrapf(err, "new request")
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return readHTTPError(resp)
	}
	return nil
}

func (c *HTTPClient) DeleteTopic(ctx context.Context, topic string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		c.base+"/topics/"+url.PathEscape(topic), nil)
	if err != nil {
		return oops.Wrapf(err, "new request")
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		return readHTTPError(resp)
	}
	return nil
}

func (c *HTTPClient) ListTopics(ctx context.Context) ([]HTTPTopicInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/topics", nil)
	if err != nil {
		return nil, oops.Wrapf(err, "new request")
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, oops.Wrapf(err, "do request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, readHTTPError(resp)
	}

	var infos []HTTPTopicInfo
	if err := json.NewDecoder(resp.Body).Decode(&infos); err != nil {
		return nil, oops.Wrapf(err, "decode")
	}
	return infos, nil
}

func readHTTPError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	var errResp HTTPErrorResponse
	if json.Unmarshal(body, &errResp) == nil && errResp.Error != "" {
		return oops.Errorf("http %d: %s", resp.StatusCode, errResp.Error)
	}
	return oops.Errorf("http %s: %s", strconv.Itoa(resp.StatusCode), string(body))
}
