package experiments

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// ExportCSV writes results as a flat CSV table.
func ExportCSV(results []ExperimentResult, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	path := filepath.Join(dir, fmt.Sprintf("results-%s.csv", time.Now().Format("20060102-150405")))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"scenario", "algorithm", "system", "mode",
		"throughput_msg_sec", "latency_p50_us", "latency_p95_us", "latency_p99_us",
		"load_stddev", "produced", "consumed", "duration_sec",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		row := []string{
			r.Scenario,
			r.Algorithm,
			r.System,
			r.Mode,
			fmt.Sprintf("%.2f", r.Throughput),
			fmt.Sprintf("%.2f", float64(r.LatencyP50)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyP95)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyP99)/float64(time.Microsecond)),
			fmt.Sprintf("%.6f", r.LoadStdDev),
			fmt.Sprintf("%d", r.Produced),
			fmt.Sprintf("%d", r.Consumed),
			fmt.Sprintf("%.3f", r.Duration.Seconds()),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// ExportJSON writes results as a structured JSON file.
func ExportJSON(results []ExperimentResult, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	path := filepath.Join(dir, fmt.Sprintf("results-%s.json", time.Now().Format("20060102-150405")))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(results)
}
