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
		"enqueue_to_flush_p50_us", "flush_to_append_p50_us", "append_to_consume_p50_us",
		"load_stddev", "produced", "consumed",
		"publish_errors", "consume_errors", "duration_sec",
		"high_prio_p50_us", "high_prio_p99_us",
		"low_prio_p50_us", "low_prio_p99_us",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		highP50, highP99 := prioGroupStats(r.LatencyByPriority[:], 0, 3)
		lowP50, lowP99 := prioGroupStats(r.LatencyByPriority[:], 7, 10)

		row := []string{
			r.Scenario,
			r.Algorithm,
			r.System,
			r.Mode,
			fmt.Sprintf("%.2f", r.Throughput),
			fmt.Sprintf("%.2f", float64(r.LatencyP50)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyP95)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyP99)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyEnqueueToFlushP50)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyFlushToAppendP50)/float64(time.Microsecond)),
			fmt.Sprintf("%.2f", float64(r.LatencyAppendToConsumeP50)/float64(time.Microsecond)),
			fmt.Sprintf("%.6f", r.LoadStdDev),
			fmt.Sprintf("%d", r.Produced),
			fmt.Sprintf("%d", r.Consumed),
			fmt.Sprintf("%d", r.PublishErrors),
			fmt.Sprintf("%d", r.ConsumeErrors),
			fmt.Sprintf("%.3f", r.Duration.Seconds()),
			fmt.Sprintf("%.2f", highP50),
			fmt.Sprintf("%.2f", highP99),
			fmt.Sprintf("%.2f", lowP50),
			fmt.Sprintf("%.2f", lowP99),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func prioGroupStats(stats []PriorityStats, from, to int) (p50us, p99us float64) {
	var sumP50, sumP99 float64
	count := 0
	for i := from; i < to && i < len(stats); i++ {
		if stats[i].Count == 0 {
			continue
		}
		sumP50 += float64(stats[i].P50) / float64(time.Microsecond)
		sumP99 += float64(stats[i].P99) / float64(time.Microsecond)
		count++
	}
	if count == 0 {
		return 0, 0
	}
	return sumP50 / float64(count), sumP99 / float64(count)
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
