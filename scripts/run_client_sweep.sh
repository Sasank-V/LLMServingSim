#!/usr/bin/env bash
# Run the Client Scalability Sweep Experiment (Number of Clients vs. Routing Policies)
#
# Sweeps number of concurrent synthetic clients: N in [5, 10, 20, 50, 100, 200]
# Evaluates policies: FAIRROUTE_V2, LOAD, LOCALITY, FAIRNESS, PREBLE, H0, H1, H4
# Outputs CSVs to: outputs/client_sweep/
# Generates publication plots to: outputs/client_sweep/plots/

set -euo pipefail

CONTAINER="${CONTAINER:-servingsim_docker}"
CONFIG="${CONFIG:-configs/cluster/dual_node_multi_instance_policy_study.json}"
WORKLOAD_DIR="${WORKLOAD_DIR:-workloads/client_sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/client_sweep}"
PLOTS_DIR="${PLOTS_DIR:-$OUTPUT_DIR/plots}"
NUM_REQS="${NUM_REQS:-500}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
CACHE_MODE="${CACHE_MODE:-xpu}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_DIR/sweep_status.txt}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/sweep.log}"
PID_FILE="${PID_FILE:-$OUTPUT_DIR/sweep.pid}"

CLIENT_COUNTS=(5 10 20 50 100 200)
POLICIES=(FAIRROUTE_V2 LOAD LOCALITY FAIRNESS PREBLE H0 H1 H4)
SCENARIOS=(full_adversarial locality_conflict fairness_conflict)

case "${1:-}" in
  --detach)
    mkdir -p "$(dirname "$LOG_FILE")"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Client sweep is already running with PID $(cat "$PID_FILE")."
      exit 0
    fi
    nohup "$0" "${@:2}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "Started client sweep with PID $!."
    echo "Status: $STATUS_FILE"
    echo "Logs:   $LOG_FILE"
    exit 0
    ;;
  --status)
    if [[ -f "$STATUS_FILE" ]]; then
      cat "$STATUS_FILE"
    else
      echo "No status file found: $STATUS_FILE"
    fi
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
    ;;
  --stop)
    if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
      docker exec -i "$CONTAINER" bash -s <<'STOP_SCRIPT'
  set +e
  for pid in $(pgrep -f 'python3 -m serving' 2>/dev/null); do
    command_line=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    if [[ "$command_line" == *"client_sweep"* ]]; then
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
    echo "Stopped client sweep workers."
    exit 0
    ;;
esac

docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true || {
  echo "Container '$CONTAINER' is not running. Starting container..."
  docker start "$CONTAINER"
}

docker exec \
  -i \
  -e "CONFIG=$CONFIG" \
  -e "WORKLOAD_DIR=$WORKLOAD_DIR" \
  -e "OUTPUT_DIR=$OUTPUT_DIR" \
  -e "PLOTS_DIR=$PLOTS_DIR" \
  -e "NUM_REQS=$NUM_REQS" \
  -e "SKIP_EXISTING=$SKIP_EXISTING" \
  -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
  -e "CACHE_MODE=$CACHE_MODE" \
  -e "STATUS_FILE=$STATUS_FILE" \
  "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
cd /app/LLMServingSim
mkdir -p "$OUTPUT_DIR" "$WORKLOAD_DIR" "$PLOTS_DIR"
printf '%s status=RUNNING container=%s\n' "$(date -Is)" "$HOSTNAME" >"$STATUS_FILE"

CLIENT_COUNTS=(5 10 20 50 100 200)
POLICIES=(FAIRROUTE_V2 LOAD LOCALITY FAIRNESS PREBLE H0 H1 H4)

case "$CACHE_MODE" in
  xpu) CACHE_ARGS=() ;;
  cpu) CACHE_ARGS=(--enable-prefix-sharing --prefix-storage CPU) ;;
  cxl) CACHE_ARGS=(--enable-prefix-sharing --prefix-storage CXL) ;;
  *) echo "CACHE_MODE must be xpu, cpu, or cxl: $CACHE_MODE" >&2; exit 2 ;;
esac

echo "=========================================================================="
echo "  Starting Client Scalability Sweep: Clients = ${CLIENT_COUNTS[*]}"
echo "=========================================================================="

# Step 1: Generate client workloads
for n_clients in "${CLIENT_COUNTS[@]}"; do
  client_wl_dir="$WORKLOAD_DIR/clients_${n_clients}"
  mkdir -p "$client_wl_dir"
  echo "Generating workloads for num_clients = $n_clients ..."
  python3 -m workloads.generators.generate_hypothesis_workloads \
    --output-dir "$client_wl_dir" \
    --num-reqs "$NUM_REQS" \
    --num-clients "$n_clients" \
    --seed 42 >/dev/null
done

# Step 2: Run simulation jobs
STATE_DIR="$OUTPUT_DIR/.sweep_state"
mkdir -p "$STATE_DIR"
if [[ "$SKIP_EXISTING" != "1" ]]; then
  rm -f "$STATE_DIR"/*
fi

total_jobs=0
for n_clients in "${CLIENT_COUNTS[@]}"; do
  shopt -s nullglob
  datasets=("$WORKLOAD_DIR/clients_${n_clients}"/*.jsonl)
  total_jobs=$(( total_jobs + ${#datasets[@]} * ${#POLICIES[@]} ))
done

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
  local n_clients="$1"
  local dataset="$2"
  local scenario="$3"
  local policy="$4"
  local output="$OUTPUT_DIR/clients_${n_clients}_${scenario}_policy_${policy}.csv"
  local state_key="c${n_clients}_${scenario}_${policy}"
  local state_file="$STATE_DIR/$state_key.running"

  : >"$state_file"
  write_status RUNNING "c${n_clients}/${scenario}/${policy}"

  if [[ "$SKIP_EXISTING" == "1" && -s "$output" ]]; then
    mv -f "$state_file" "$STATE_DIR/$state_key.skipped"
    write_status RUNNING "c${n_clients}/${scenario}/${policy}"
    return 0
  fi

  echo "=== RUN clients=$n_clients / scenario=$scenario / policy=$policy ==="
  if python3 -m serving \
      --cluster-config "$CONFIG" \
      --request-routing-policy "$policy" \
      --dtype bfloat16 --block-size 16 \
      "${CACHE_ARGS[@]}" \
      --dataset "$dataset" --num-reqs "$NUM_REQS" \
      --output "$output" \
      --log-level WARNING; then
    mv -f "$state_file" "$STATE_DIR/$state_key.done"
    write_status RUNNING "c${n_clients}/${scenario}/${policy}"
  else
    mv -f "$state_file" "$STATE_DIR/$state_key.failed"
    write_status FAILED "c${n_clients}/${scenario}/${policy}"
    return 1
  fi
}

running_jobs=0
failed_jobs=0
for n_clients in "${CLIENT_COUNTS[@]}"; do
  shopt -s nullglob
  datasets=("$WORKLOAD_DIR/clients_${n_clients}"/*.jsonl)
  for dataset in "${datasets[@]}"; do
    scenario="${dataset##*/}"
    scenario="${scenario%.jsonl}"
    for policy in "${POLICIES[@]}"; do
      run_policy "$n_clients" "$dataset" "$scenario" "$policy" &
      running_jobs=$((running_jobs + 1))
      if (( running_jobs >= PARALLEL_JOBS )); then
        wait -n || failed_jobs=$((failed_jobs + 1))
        running_jobs=$((running_jobs - 1))
      fi
    done
  done
done

while (( running_jobs > 0 )); do
  wait -n || failed_jobs=$((failed_jobs + 1))
  running_jobs=$((running_jobs - 1))
done

if (( failed_jobs > 0 )); then
  write_status FAILED "failed_jobs=$failed_jobs"
  echo "Some client sweep jobs failed ($failed_jobs failures)." >&2
  exit 4
fi

# Step 3: Generate Plot Figures
echo "=========================================================================="
echo "  Generating Plot Figures via scripts/plot_client_sweep.py ..."
echo "=========================================================================="
python3 scripts/plot_client_sweep.py --input-dir "$OUTPUT_DIR" --output-dir "$PLOTS_DIR"

write_status COMPLETE
echo "Client Scalability Sweep Complete! Plots saved in: $PLOTS_DIR"

CONTAINER_SCRIPT
