package metrics

import (
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type HTTPSuite struct {
	suite.Suite
}

func TestHTTPSuite(t *testing.T) { suite.Run(t, new(HTTPSuite)) }

func (s *HTTPSuite) TestHealthzAndMetrics() {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector())

	srv, err := Serve(":0", reg)
	require.NoError(s.T(), err)
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		require.NoError(s.T(), srv.Shutdown(ctx))
	}()

	base := "http://" + srv.Addr()

	tests := []struct {
		path     string
		contains string
	}{
		{"/healthz", "ok"},
		{"/metrics", "go_goroutines"},
	}

	for _, tc := range tests {
		s.Run(tc.path, func() {
			resp, err := http.Get(base + tc.path)
			require.NoError(s.T(), err)
			defer resp.Body.Close()
			require.Equal(s.T(), http.StatusOK, resp.StatusCode)

			body, err := io.ReadAll(resp.Body)
			require.NoError(s.T(), err)
			require.Contains(s.T(), string(body), tc.contains)
		})
	}
}
