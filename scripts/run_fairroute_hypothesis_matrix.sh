#!/usr/bin/env bash
# Run the FairRoute hypothesis matrix inside the simulator container.
set -euo pipefail

CONTAINER="${CONTAINER:-servingsim_docker}"
CONFIG="${CONFIG:-configs/cluster/dual_node_multi_instance_policy_study.json}"
WORKLOAD_DIR="${WORKLOAD_DIR:-workloads/fairroute_hypothesis}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fairroute_hypothesis}"
NUM_REQS="${NUM_REQS:-1000}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
INCLUDE_REAL_WORKLOAD="${INCLUDE_REAL_WORKLOAD:-1}"
REAL_WORKLOAD="${REAL_WORKLOAD:-workloads/workload-llama8b-full-sps10-100clients.jsonl}"
REAL_WORKLOAD_LIMIT="${REAL_WORKLOAD_LIMIT:-1000}"
CACHE_MODE="${CACHE_MODE:-xpu}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_DIR/matrix_status.txt}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/matrix.log}"
PID_FILE="${PID_FILE:-$OUTPUT_DIR/matrix.pid}"

case "${1:-}" in
  --detach)
    mkdir -p "$(dirname "$LOG_FILE")"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Matrix is already running with PID $(cat "$PID_FILE")."
      exit 0
    fi
    nohup "$0" "${@:2}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "Started matrix with PID $!."
    echo "Status: $STATUS_FILE"
    echo "Logs:   $LOG_FILE"
    exit 0
    ;;
  --status)
    if [[ -f "$STATUS_FILE" ]]; then
      cat "$STATUS_FILE"
    else
      echo "No matrix status file found: $STATUS_FILE"
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
      echo "No matrix log found: $LOG_FILE" >&2
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
    if [[ "$command_line" == *"fairroute_hypothesis"* ]]; then
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
    echo "Stopped FairRoute matrix workers. Docker container remains running."
    exit 0
    ;;
esac

POLICIES=(
  RR LOAD RAND
  FAIRNESS LOCALITY PREDICTION F_L L_P F_P F_L_P
  PREBLE LBGR DUALMAP CACHE_ROUTE VTC EQUINOX QUARTZ ISJL
  NEXUSSCHED BALANCEROUTE PILLM ONLINE_LP
  H0 H1 H2 H3 H4 FAIRROUTE
)

docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true || {
  echo "Container '$CONTAINER' is not running." >&2
  exit 2
}

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

POLICIES=(
  RR LOAD RAND
  FAIRNESS LOCALITY PREDICTION F_L L_P F_P F_L_P
  PREBLE LBGR DUALMAP CACHE_ROUTE VTC EQUINOX QUARTZ ISJL
  NEXUSSCHED BALANCEROUTE PILLM ONLINE_LP
  H0 H1 H2 H3 H4 FAIRROUTE
)

shopt -s nullglob
datasets=("$WORKLOAD_DIR"/*.jsonl)
if (( ${#datasets[@]} == 0 )); then
  echo "No JSONL workloads found in $WORKLOAD_DIR." >&2
  exit 3
fi

STATE_DIR="$OUTPUT_DIR/.matrix_state"
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
  exit 4
fi

write_status COMPLETE

CONTAINER_SCRIPT
