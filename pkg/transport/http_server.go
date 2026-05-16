package transport

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"strconv"
	"time"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
	"github.com/samber/oops"
	"golang.org/x/net/http2"
	"golang.org/x/net/netutil"
)

type HTTPServer struct {
	broker   *broker.Broker
	server   *http.Server
	listener net.Listener
	reqSem   chan struct{}
}

func NewHTTPServer(b *broker.Broker) *HTTPServer {
	cfg := b.Config()

	var sem chan struct{}
	if cfg.MaxQueuedRequests > 0 {
		sem = make(chan struct{}, cfg.MaxQueuedRequests)
	}

	s := &HTTPServer{broker: b, reqSem: sem}
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
	mux.HandleFunc("GET /readyz", s.handleReadyz)
	mux.Handle("GET /debug/pprof/", http.DefaultServeMux)
	mux.Handle("GET /debug/pprof/{cmd}", http.DefaultServeMux)

	readTimeout := time.Duration(cfg.ReadTimeoutMs) * time.Millisecond
	writeTimeout := time.Duration(cfg.WriteTimeoutMs) * time.Millisecond
	if readTimeout == 0 {
		readTimeout = 30 * time.Second
	}
	if writeTimeout == 0 {
		writeTimeout = 30 * time.Second
	}

	var handler http.Handler = mux
	if sem != nil {
		handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
				mux.ServeHTTP(w, r)
			default:
				http.Error(w, `{"error":"server overloaded"}`, http.StatusServiceUnavailable)
			}
		})
	}

	s.server = &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	http2.ConfigureServer(s.server, &http2.Server{})
	return s
}

func (s *HTTPServer) MaxConnections() int {
	return s.broker.Config().MaxConnections
}

func (s *HTTPServer) Start(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return oops.Wrapf(err, "http listen %s", addr)
	}

	maxConn := s.broker.Config().MaxConnections
	if maxConn > 0 {
		ln = netutil.LimitListener(ln, maxConn)
	}

	s.listener = ln
	go s.server.Serve(ln)
	return nil
}

func (s *HTTPServer) Stop() {
	if s.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = s.server.Shutdown(ctx)
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
	Offset      int64  `json:"offset"`
	Key         []byte `json:"key"`
	Value       []byte `json:"value"`
	Priority    uint32 `json:"priority"`
	Timestamp   int64  `json:"timestamp"`
	ProduceTime int64  `json:"produce_time,omitempty"`
	AppendTime  int64  `json:"append_time,omitempty"`
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

func (s *HTTPServer) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	topics := s.broker.ListTopics()
	if len(topics) == 0 {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("no topics loaded"))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ready"))
}

func (s *HTTPServer) handleProduce(w http.ResponseWriter, r *http.Request) {
	topic := r.PathValue("topic")
	cfg := s.broker.Config()
	r.Body = http.MaxBytesReader(w, r.Body, int64(cfg.MaxMessageBytes))

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

	batchResults := s.broker.PublishBatch(r.Context(), topic, msgs)
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

	cfg := s.broker.Config()

	partition, _ := strconv.Atoi(partitionStr)
	maxBytes := cfg.MaxFetchBytes
	if v, err := strconv.Atoi(maxBytesStr); err == nil && v > 0 {
		maxBytes = v
	}
	if cfg.MaxFetchBytes > 0 && maxBytes > cfg.MaxFetchBytes {
		maxBytes = cfg.MaxFetchBytes
	}

	fetchMinBytes := cfg.FetchMinBytes
	fetchMaxWait := time.Duration(cfg.FetchMaxWaitMs) * time.Millisecond
	deadline := time.Now().Add(fetchMaxWait)

	// Load offset once; advance locally in the loop.
	currentOffset, err := s.broker.LoadOffset(group, topic, partition)
	if err != nil {
		currentOffset = 1
	}

	var (
		envelopes  []HTTPMessageEnvelope
		totalBytes int
		lastOffset int64 = -1
	)

	for totalBytes < maxBytes {
		msg, nextOffset, err := s.broker.ConsumeFrom(r.Context(), topic, group, partition, currentOffset)
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
			writeHTTPError(w, http.StatusInternalServerError, err)
			return
		}

		envelopes = append(envelopes, HTTPMessageEnvelope{
			Offset:      int64(nextOffset) - 1,
			Key:         msg.Key,
			Value:       msg.Value,
			Priority:    uint32(msg.Header.Priority),
			Timestamp:   msg.Header.Timestamp,
			ProduceTime: msg.Header.ProduceTime,
			AppendTime:  msg.Header.AppendTime,
		})
		totalBytes += len(msg.Key) + len(msg.Value)
		lastOffset = int64(nextOffset)
		currentOffset = nextOffset
	}

	if lastOffset >= 0 {
		if err := s.broker.Commit(group, topic, partition, lastOffset); err != nil {
			log.Printf("auto-commit failed (topic=%s group=%s partition=%d offset=%d): %v",
				topic, group, partition, lastOffset, err)
		}
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
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeHTTPError(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(HTTPErrorResponse{Error: err.Error()})
}
