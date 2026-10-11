#!/usr/bin/env bash
# scripts/run_predictive_routing_experiments.sh
#
# Experiment driver for the 5 predictive load-routing algorithms:
#   1. LODESTAR      — Online-learning MLP reward predictor (Lim et al., 2026)
#   2. PAST_FUTURE   — Histogram-based memory prediction (ASPLOS 2025)
#   3. BF_IO         — Balance-Future Integer Optimization (2026)
#   4. BALANCEROUTE_H — Termination classifier + mean regressor (2026)
#   5. QUARTZ_PRED   — Quantile-aware routing (QUARTZ, 2026)
#
# These are compared against all default LLMServingSim baselines:
#   RR, LOAD, RAND
#
# Usage:
#   ./scripts/run_predictive_routing_experiments.sh [--detach|--status|--logs|--stop]
#
# Configuration via environment variables (see defaults below).
#
# Output structure:
#   outputs/predictive_routing/
#     <scenario>_policy_<POLICY>.csv        — per-request results
#     summary.csv                           — aggregated comparison table
#     predictive_routing.log                — run log
#     predictive_routing_status.txt         — run status

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTAINER="${CONTAINER:-servingsim_docker}"
# Default to high-saturation cluster config to expose queueing bottlenecks
CONFIG="${CONFIG:-configs/cluster/high_saturation_policy_study.json}"
CACHE_MODE="${CACHE_MODE:-xpu}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/predictive_routing}"
WORKLOAD_DIR="${WORKLOAD_DIR:-workloads/fairroute_hypothesis}"
NUM_REQS="${NUM_REQS:-1000}"
INCLUDE_REAL_WORKLOAD="${INCLUDE_REAL_WORKLOAD:-1}"
REAL_WORKLOAD="${REAL_WORKLOAD:-workloads/workload-llama8b-full-sps10-100clients.jsonl}"
REAL_WORKLOAD_LIMIT="${REAL_WORKLOAD_LIMIT:-1000}"

STATUS_FILE="${STATUS_FILE:-$OUTPUT_DIR/predictive_routing_status.txt}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/predictive_routing.log}"
PID_FILE="${PID_FILE:-$OUTPUT_DIR/predictive_routing.pid}"

# The 5 predictive routing policies + 3 baselines for comparison
PREDICTIVE_POLICIES=(
  LODESTAR
  PAST_FUTURE
  BF_IO
  BALANCEROUTE_H
  QUARTZ_PRED
)

BASELINE_POLICIES=(
  RR
  LOAD
  RAND
)

ALL_POLICIES=("${BASELINE_POLICIES[@]}" "${PREDICTIVE_POLICIES[@]}")

# ---------------------------------------------------------------------------
# Control-plane helpers (--detach / --status / --logs / --stop)
# ---------------------------------------------------------------------------
ACTION="${1:-}"

case "$ACTION" in
  --detach)
    mkdir -p "$(dirname "$LOG_FILE")"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Experiment is already running with PID $(cat "$PID_FILE")."
      exit 0
    fi
    nohup "$0" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "Started predictive routing experiments with PID $!."
    echo "Status: $STATUS_FILE"
    echo "Logs:   $LOG_FILE"
    exit 0
    ;;
  --status)
    [[ -f "$STATUS_FILE" ]] && cat "$STATUS_FILE" || echo "No status file found: $STATUS_FILE"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "host_process=RUNNING pid=$(cat "$PID_FILE")"
    else
      echo "host_process=STOPPED"
    fi
    exit 0
    ;;
  --logs)
    if [[ ! -f "$LOG_FILE" ]]; then
      echo "No log file found: $LOG_FILE" >&2
      exit 1
    fi
    tail -n "${TAIL_LINES:-40}" -f "$LOG_FILE"
    exit 0
    ;;
  --stop)
    if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
      docker exec -i "$CONTAINER" bash -s <<'STOP_SCRIPT'
  set +e
  for pid in $(pgrep -f 'python3 -m serving' 2>/dev/null); do
    command_line=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    if [[ "$command_line" == *"predictive_routing"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
STOP_SCRIPT
    fi
    if [[ -f "$PID_FILE" ]]; then
      host_pid=$(cat "$PID_FILE")
      kill -TERM "$host_pid" 2>/dev/null || true
      rm -f "$PID_FILE"
    fi
    printf '%s status=STOPPED\n' "$(date -Is)" >"$STATUS_FILE"
    echo "Stopped predictive routing experiments."
    exit 0
    ;;
  "")
    ;;  # fall through to run
  *)
    echo "Usage: $0 [--detach|--status|--logs|--stop]" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Docker container check
# ---------------------------------------------------------------------------
if ! docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "Container '$CONTAINER' is not running. Starting container..."
  docker start "$CONTAINER"
fi

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
printf '%s status=RUNNING container=%s\n' "$(date -Is)" "$CONTAINER" >"$STATUS_FILE"

docker exec \
  -i \
  -e "CONFIG=$CONFIG" \
  -e "WORKLOAD_DIR=$WORKLOAD_DIR" \
  -e "OUTPUT_DIR=$OUTPUT_DIR" \
  -e "NUM_REQS=$NUM_REQS" \
  -e "SKIP_EXISTING=$SKIP_EXISTING" \
  -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
  -e "INCLUDE_REAL_WORKLOAD=$INCLUDE_REAL_WORKLOAD" \
  -e "REAL_WORKLOAD=$REAL_WORKLOAD" \
  -e "REAL_WORKLOAD_LIMIT=$REAL_WORKLOAD_LIMIT" \
  -e "CACHE_MODE=$CACHE_MODE" \
  -e "STATUS_FILE=$STATUS_FILE" \
  "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
cd /app/LLMServingSim
mkdir -p "$OUTPUT_DIR"
printf '%s status=RUNNING container=%s\n' "$(date -Is)" "$HOSTNAME" >"$STATUS_FILE"

case "$CACHE_MODE" in
  xpu) CACHE_ARGS=() ;;
  cpu) CACHE_ARGS=(--enable-prefix-sharing --prefix-storage CPU) ;;
  cxl) CACHE_ARGS=(--enable-prefix-sharing --prefix-storage CXL) ;;
  *) echo "CACHE_MODE must be xpu, cpu, or cxl: $CACHE_MODE" >&2; exit 2 ;;
esac

if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be a positive integer: $PARALLEL_JOBS" >&2
  exit 2
fi

# Optional real workload
if [[ "$INCLUDE_REAL_WORKLOAD" == "1" ]]; then
  bounded_workload="$WORKLOAD_DIR/real_workload_${REAL_WORKLOAD_LIMIT}.jsonl"
  if [[ ! -s "$bounded_workload" ]]; then
    if [[ ! -s "$REAL_WORKLOAD" ]]; then
      echo "Real workload not found: $REAL_WORKLOAD" >&2
      exit 3
    fi
    head -n "$REAL_WORKLOAD_LIMIT" "$REAL_WORKLOAD" >"$bounded_workload"
  fi
fi

# 5 predictive + 3 baseline policies
POLICIES=(
  RR LOAD RAND
  LODESTAR PAST_FUTURE BF_IO BALANCEROUTE_H QUARTZ_PRED
)

shopt -s nullglob
datasets=("$WORKLOAD_DIR"/*.jsonl)
if (( ${#datasets[@]} == 0 )); then
  echo "No JSONL workloads found in $WORKLOAD_DIR." >&2
  exit 3
fi

STATE_DIR="$OUTPUT_DIR/.predictive_state"
mkdir -p "$STATE_DIR"
if [[ "$SKIP_EXISTING" != "1" ]]; then
  rm -f "$STATE_DIR"/*
fi
total_jobs=$(( ${#datasets[@]} * ${#POLICIES[@]} ))

write_status() {
  local status="$1"
  local current="${2:-}"
  local done_count active_count skipped_count failed_count remaining_count
  done_count=$(find "$STATE_DIR" -maxdepth 1 -name '*.done' -type f | wc -l)
  active_count=$(find "$STATE_DIR" -maxdepth 1 -name '*.running' -type f | wc -l)
  skipped_count=$(find "$STATE_DIR" -maxdepth 1 -name '*.skipped' -type f | wc -l)
  failed_count=$(find "$STATE_DIR" -maxdepth 1 -name '*.failed' -type f | wc -l)
  remaining_count=$(( total_jobs - done_count - skipped_count - failed_count - active_count ))
  (( remaining_count < 0 )) && remaining_count=0
  local tmp_status
  tmp_status=$(mktemp "${STATUS_FILE}.tmp.XXXXXX")
  printf '%s status=%s cache_mode=%s total=%s completed=%s skipped=%s active=%s failed=%s remaining=%s current=%s\n' \
    "$(date -Is)" "$status" "$CACHE_MODE" "$total_jobs" "$done_count" \
    "$skipped_count" "$active_count" "$failed_count" "$remaining_count" "$current" >"$tmp_status"
  mv -f "$tmp_status" "$STATUS_FILE"
}

run_policy() {
  local dataset="$1"
  local scenario="$2"
  local policy="$3"
  local output="$OUTPUT_DIR/${scenario}_policy_${policy}.csv"
  local state_key="${scenario}__${policy}"
  local state_file="$STATE_DIR/$state_key.running"
  local expected_rows actual_rows

  expected_rows=$(wc -l <"$dataset")

  : >"$state_file"
  write_status RUNNING "$scenario/$policy"
  actual_rows=0
  if [[ -s "$output" ]]; then
    actual_rows=$(wc -l <"$output")
  fi
  if [[ "$SKIP_EXISTING" == "1" && "$actual_rows" -ge $((expected_rows + 1)) ]]; then
    echo "=== SKIP $scenario / $policy ==="
    mv -f "$state_file" "$STATE_DIR/$state_key.skipped"
    write_status RUNNING "$scenario/$policy"
    return 0
  fi

  echo "=== RUN $scenario / $policy ==="
  if python3 -m serving \
      --cluster-config "$CONFIG" \
      --request-routing-policy "$policy" \
      --dtype bfloat16 --block-size 16 \
      "${CACHE_ARGS[@]}" \
      --dataset "$dataset" --num-reqs "$NUM_REQS" \
      --output "$output" \
      --log-level WARNING; then
    mv -f "$state_file" "$STATE_DIR/$state_key.done"
    write_status RUNNING "$scenario/$policy"
  else
    mv -f "$state_file" "$STATE_DIR/$state_key.failed"
    write_status FAILED "$scenario/$policy"
    return 1
  fi
}

echo "=========================================================================="
echo "  Predictive Routing Experiment"
echo "  Policies: ${POLICIES[*]}"
echo "  Datasets: ${#datasets[@]} workloads"
echo "  Total jobs: $total_jobs"
echo "=========================================================================="

running_jobs=0
failed_jobs=0
for dataset in "${datasets[@]}"; do
  scenario="${dataset##*/}"
  scenario="${scenario%.jsonl}"
  for policy in "${POLICIES[@]}"; do
    run_policy "$dataset" "$scenario" "$policy" &
    running_jobs=$((running_jobs + 1))
    if (( running_jobs >= PARALLEL_JOBS )); then
      wait -n || failed_jobs=$((failed_jobs + 1))
      running_jobs=$((running_jobs - 1))
    fi
  done
done

while (( running_jobs > 0 )); do
  wait -n || failed_jobs=$((failed_jobs + 1))
  running_jobs=$((running_jobs - 1))
done

if (( failed_jobs > 0 )); then
  write_status FAILED "failed_jobs=$failed_jobs"
  echo "Some jobs failed ($failed_jobs failures)." >&2
  exit 4
fi

# ---------------------------------------------------------------------------
# Generate comparison summary
# ---------------------------------------------------------------------------
echo "=========================================================================="
echo "  Generating comparison summary ..."
echo "=========================================================================="

python3 - "$OUTPUT_DIR" <<'SUMMARY_SCRIPT'
import csv
import glob
import os
import sys
from collections import defaultdict

output_dir = sys.argv[1]
summary_path = os.path.join(output_dir, "summary.csv")

# Collect all result CSVs
pattern = os.path.join(output_dir, "*_policy_*.csv")
files = sorted(glob.glob(pattern))

if not files:
    print("No result files found.")
    sys.exit(0)

rows = []
for fpath in files:
    fname = os.path.basename(fpath)
    # Parse: <scenario>_policy_<POLICY>.csv
    parts = fname.replace(".csv", "").split("_policy_")
    if len(parts) != 2:
        continue
    scenario = parts[0]
    policy = parts[1]

    # Read CSV
    latencies = []
    ttfts = []
    tpots = []
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize column names to lowercase
            lower_row = {k.strip().lower(): v for k, v in row.items() if k}
            try:
                if 'latency' in lower_row and lower_row['latency']:
                    latencies.append(float(lower_row['latency']))
                if 'ttft' in lower_row and lower_row['ttft']:
                    ttfts.append(float(lower_row['ttft']))
                if 'tpot' in lower_row and lower_row['tpot']:
                    tpots.append(float(lower_row['tpot']))
            except (ValueError, KeyError):
                continue

    if not latencies:
        continue

    n = len(latencies)
    latencies.sort()
    ttfts.sort()
    tpots.sort()

    def percentile(arr, p):
        if not arr:
            return 0
        idx = int(p / 100.0 * (len(arr) - 1))
        return arr[idx]

    def mean(arr):
        return sum(arr) / len(arr) if arr else 0

    rows.append({
        "scenario": scenario,
        "policy": policy,
        "num_requests": n,
        "latency_mean": f"{mean(latencies):.0f}",
        "latency_p50": f"{percentile(latencies, 50):.0f}",
        "latency_p95": f"{percentile(latencies, 95):.0f}",
        "latency_p99": f"{percentile(latencies, 99):.0f}",
        "ttft_mean": f"{mean(ttfts):.0f}",
        "ttft_p50": f"{percentile(ttfts, 50):.0f}",
        "ttft_p95": f"{percentile(ttfts, 95):.0f}",
        "ttft_p99": f"{percentile(ttfts, 99):.0f}",
        "tpot_mean": f"{mean(tpots):.0f}",
        "tpot_p50": f"{percentile(tpots, 50):.0f}",
        "tpot_p95": f"{percentile(tpots, 95):.0f}",
        "tpot_p99": f"{percentile(tpots, 99):.0f}",
    })

if not rows:
    print("No valid results to summarize.")
    sys.exit(0)

# Write summary CSV
fieldnames = list(rows[0].keys())
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary written to {summary_path} ({len(rows)} entries)")

# Print comparison table to stdout
print("\n" + "=" * 120)
print(f"{'Scenario':<30} {'Policy':<18} {'Reqs':>5} {'Lat Mean':>10} {'Lat P99':>10} {'TTFT Mean':>10} {'TTFT P99':>10} {'TPOT Mean':>10}")
print("=" * 120)
for r in sorted(rows, key=lambda x: (x["scenario"], x["policy"])):
    print(f"{r['scenario']:<30} {r['policy']:<18} {r['num_requests']:>5} {r['latency_mean']:>10} {r['latency_p99']:>10} {r['ttft_mean']:>10} {r['ttft_p99']:>10} {r['tpot_mean']:>10}")
print("=" * 120)
SUMMARY_SCRIPT

write_status COMPLETE
echo "Predictive routing experiments complete!"
echo "Results: $OUTPUT_DIR/"
echo "Summary: $OUTPUT_DIR/summary.csv"
CONTAINER_SCRIPT

printf '%s status=COMPLETE\n' "$(date -Is)" >"$STATUS_FILE"
echo "Done."
