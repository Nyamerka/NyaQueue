package transport

import (
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/samber/oops"
)

type HTTPServer struct {
	broker   *broker.Broker
	server   *http.Server
	listener net.Listener
}

func NewHTTPServer(b *broker.Broker) *HTTPServer {
	s := &HTTPServer{broker: b}
	mux := http.NewServeMux()

	mux.HandleFunc("POST /topics", s.handleCreateTopic)
	mux.HandleFunc("GET /topics", s.handleListTopics)
	mux.HandleFunc("DELETE /topics/{topic}", s.handleDeleteTopic)
	mux.HandleFunc("POST /topics/{topic}/messages", s.handleProduce)
	mux.HandleFunc("GET /topics/{topic}/messages", s.handleConsume)
	mux.HandleFunc("POST /topics/{topic}/offsets", s.handleCommit)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	s.server = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	return s
}

func (s *HTTPServer) Start(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return oops.Wrapf(err, "http listen %s", addr)
	}
	s.listener = ln
	go s.server.Serve(ln)
	return nil
}

func (s *HTTPServer) Stop() {
	if s.server != nil {
		s.server.Close()
	}
}

func (s *HTTPServer) Addr() string {
	if s.listener != nil {
		return s.listener.Addr().String()
	}
	return ""
}

type HTTPProduceRecord struct {
	Key      []byte `json:"key"`
	Value    []byte `json:"value"`
	Priority uint32 `json:"priority"`
}

type HTTPProduceRequest struct {
	Records []HTTPProduceRecord `json:"records"`
}

type HTTPProduceResult struct {
	Partition int   `json:"partition"`
	Offset    int64 `json:"offset"`
}

type HTTPProduceResponse struct {
	Results []HTTPProduceResult `json:"results"`
}

type HTTPMessageEnvelope struct {
	Offset    int64  `json:"offset"`
	Key       []byte `json:"key"`
	Value     []byte `json:"value"`
	Priority  uint32 `json:"priority"`
	Timestamp int64  `json:"timestamp"`
}

type HTTPConsumeResponse struct {
	Messages []HTTPMessageEnvelope `json:"messages"`
}

type HTTPCommitRequest struct {
	Group     string `json:"group"`
	Partition int    `json:"partition"`
	Offset    int64  `json:"offset"`
}

type HTTPCreateTopicRequest struct {
	Topic         string `json:"topic"`
	NumPartitions int32  `json:"num_partitions"`
	Mode          string `json:"mode"`
}

type HTTPTopicInfo struct {
	Topic         string `json:"topic"`
	NumPartitions int32  `json:"num_partitions"`
	Mode          string `json:"mode"`
}

type HTTPErrorResponse struct {
	Error string `json:"error"`
}

func (s *HTTPServer) handleProduce(w http.ResponseWriter, r *http.Request) {
	topic := r.PathValue("topic")

	var req HTTPProduceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err)
		return
	}

	msgs := make([]*broker.Message, len(req.Records))
	now := time.Now().UnixNano()
	for i, rec := range req.Records {
		msgs[i] = &broker.Message{
			Header: broker.MessageHeader{
				Priority:  uint8(rec.Priority),
				Timestamp: now,
			},
			Key:   rec.Key,
			Value: rec.Value,
		}
	}

	batchResults := s.broker.PublishBatch(topic, msgs)
	results := make([]HTTPProduceResult, len(batchResults))
	for i, br := range batchResults {
		if br.Err != nil {
			writeHTTPError(w, http.StatusInternalServerError, br.Err)
			return
		}
		results[i] = HTTPProduceResult{
			Partition: br.Partition,
			Offset:    int64(br.Offset),
		}
	}

	writeJSON(w, http.StatusOK, HTTPProduceResponse{Results: results})
}

func (s *HTTPServer) handleConsume(w http.ResponseWriter, r *http.Request) {
	topic := r.PathValue("topic")
	group := r.URL.Query().Get("group")
	partitionStr := r.URL.Query().Get("partition")
	maxBytesStr := r.URL.Query().Get("maxBytes")

	partition, _ := strconv.Atoi(partitionStr)
	maxBytes := 1 << 20
	if v, err := strconv.Atoi(maxBytesStr); err == nil && v > 0 {
		maxBytes = v
	}

	var (
		envelopes  []HTTPMessageEnvelope
		totalBytes int
	)

	for totalBytes < maxBytes {
		msg, nextOffset, err := s.broker.Consume(topic, group, partition)
		if err != nil {
			if errors.Is(err, broker.ErrNoMessages) {
				break
			}
			if len(envelopes) > 0 {
				break
			}
			writeHTTPError(w, http.StatusInternalServerError, err)
			return
		}

		envelopes = append(envelopes, HTTPMessageEnvelope{
			Offset:    int64(nextOffset) - 1,
			Key:       msg.Key,
			Value:     msg.Value,
			Priority:  uint32(msg.Header.Priority),
			Timestamp: msg.Header.Timestamp,
		})
		totalBytes += len(msg.Key) + len(msg.Value)

		_ = s.broker.Commit(group, topic, partition, int64(nextOffset))
	}

	writeJSON(w, http.StatusOK, HTTPConsumeResponse{Messages: envelopes})
}

func (s *HTTPServer) handleCommit(w http.ResponseWriter, r *http.Request) {
	topic := r.PathValue("topic")

	var req HTTPCommitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err)
		return
	}

	if err := s.broker.Commit(req.Group, topic, req.Partition, req.Offset); err != nil {
		writeHTTPError(w, http.StatusInternalServerError, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *HTTPServer) handleCreateTopic(w http.ResponseWriter, r *http.Request) {
	var req HTTPCreateTopicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err)
		return
	}

	cfg := broker.DefaultTopicConfig()
	cfg.NumPartitions = int(req.NumPartitions)
	switch req.Mode {
	case "strict_priority":
		cfg.ScheduleMode = broker.ModeStrictPriority
	case "dqn_adaptive":
		cfg.ScheduleMode = broker.ModeDQNAdaptive
	default:
		cfg.ScheduleMode = broker.ModeFIFO
	}

	if err := s.broker.CreateTopic(req.Topic, cfg); err != nil {
		if errors.Is(err, broker.ErrTopicAlreadyExists) {
			writeHTTPError(w, http.StatusConflict, err)
			return
		}
		writeHTTPError(w, http.StatusInternalServerError, err)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

func (s *HTTPServer) handleDeleteTopic(w http.ResponseWriter, r *http.Request) {
	topic := r.PathValue("topic")

	if err := s.broker.DeleteTopic(topic); err != nil {
		if errors.Is(err, broker.ErrTopicNotFound) {
			writeHTTPError(w, http.StatusNotFound, err)
			return
		}
		writeHTTPError(w, http.StatusInternalServerError, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *HTTPServer) handleListTopics(w http.ResponseWriter, _ *http.Request) {
	topics := s.broker.ListTopics()
	infos := make([]HTTPTopicInfo, len(topics))
	for i, t := range topics {
		mode := "fifo"
		switch t.Config().ScheduleMode {
		case broker.ModeStrictPriority:
			mode = "strict_priority"
		case broker.ModeDQNAdaptive:
			mode = "dqn_adaptive"
		}
		infos[i] = HTTPTopicInfo{
			Topic:         t.Name(),
			NumPartitions: int32(t.NumPartitions()),
			Mode:          mode,
		}
	}

	writeJSON(w, http.StatusOK, infos)
}

func (s *HTTPServer) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	m := s.broker.Metrics()
	writeJSON(w, http.StatusOK, m)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeHTTPError(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(HTTPErrorResponse{Error: err.Error()})
}
