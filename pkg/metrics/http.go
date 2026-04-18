package metrics

import (
	"context"
	"errors"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/alitto/pond/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/samber/oops"
)

type Server struct {
	http *http.Server
	ln   net.Listener
	pool pond.Pool
}

func Serve(addr string, reg *prometheus.Registry) (*Server, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, oops.Wrapf(err, "listen %s", addr)
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{Registry: reg}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	s := &Server{
		http: &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second},
		ln:   ln,
		pool: pond.NewPool(1),
	}

	s.pool.Submit(func() {
		if err := s.http.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("metrics http server: %v", err)
		}
	})

	return s, nil
}

func (s *Server) Addr() string {
	return s.ln.Addr().String()
}

func (s *Server) Shutdown(ctx context.Context) error {
	if err := s.http.Shutdown(ctx); err != nil {
		return oops.Wrapf(err, "metrics shutdown")
	}
	s.pool.StopAndWait()
	return nil
}
