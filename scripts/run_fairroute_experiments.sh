#!/usr/bin/env bash
# scripts/run_fairroute_experiments.sh
#
# Unified entry point for the two FairRoute experiment drivers that used to be
# separate scripts:
#   - scripts/run_fairroute_hypothesis_matrix.sh  (the 29-policy x 6-scenario matrix)
#   - scripts/run_client_sweep.sh                 (client-count scalability sweep)
#
# Usage:
#   ./scripts/run_fairroute_experiments.sh <matrix|sweep|all> [--detach|--status|--logs|--stop]
#
# Examples:
#   ./scripts/run_fairroute_experiments.sh matrix              # run the policy matrix in the foreground
#   ./scripts/run_fairroute_experiments.sh sweep --detach       # run the client sweep in the background
#   ./scripts/run_fairroute_experiments.sh all --detach         # run matrix, then sweep, in the background
#   ./scripts/run_fairroute_experiments.sh all --status         # status of both stages
#   ./scripts/run_fairroute_experiments.sh all --logs           # tail the combined log
#   ./scripts/run_fairroute_experiments.sh all --stop           # stop whichever stage is running
#
# All environment variables from both original scripts are preserved (with the
# same defaults), plus a MATRIX_/SWEEP_ prefix where a name would otherwise
# collide between the two stages (e.g. MATRIX_OUTPUT_DIR vs SWEEP_OUTPUT_DIR).
# See the "Configuration" section below for the full list.

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
MODE="${1:-}"
case "$MODE" in
  matrix|sweep|all) ;;
  *)
    echo "Usage: $0 <matrix|sweep|all> [--detach|--status|--logs|--stop]" >&2
    exit 2
    ;;
esac
shift
ACTION="${1:-}"

# ---------------------------------------------------------------------------
# Configuration (shared)
# ---------------------------------------------------------------------------
CONTAINER="${CONTAINER:-servingsim_docker}"
CONFIG="${CONFIG:-configs/cluster/dual_node_multi_instance_policy_study.json}"
CACHE_MODE="${CACHE_MODE:-xpu}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# --- Matrix-specific (was scripts/run_fairroute_hypothesis_matrix.sh) ------
MATRIX_WORKLOAD_DIR="${MATRIX_WORKLOAD_DIR:-workloads/fairroute_hypothesis}"
MATRIX_OUTPUT_DIR="${MATRIX_OUTPUT_DIR:-outputs/fairroute_hypothesis}"
MATRIX_NUM_REQS="${MATRIX_NUM_REQS:-1000}"
INCLUDE_REAL_WORKLOAD="${INCLUDE_REAL_WORKLOAD:-1}"
REAL_WORKLOAD="${REAL_WORKLOAD:-workloads/workload-llama8b-full-sps10-100clients.jsonl}"
REAL_WORKLOAD_LIMIT="${REAL_WORKLOAD_LIMIT:-1000}"
MATRIX_STATUS_FILE="${MATRIX_STATUS_FILE:-$MATRIX_OUTPUT_DIR/matrix_status.txt}"
MATRIX_LOG_FILE="${MATRIX_LOG_FILE:-$MATRIX_OUTPUT_DIR/matrix.log}"
MATRIX_POLICIES=(
  RR LOAD RAND
  FAIRNESS LOCALITY PREDICTION F_L L_P F_P F_L_P
  PREBLE LBGR DUALMAP CACHE_ROUTE VTC EQUINOX QUARTZ ISJL
  NEXUSSCHED BALANCEROUTE PILLM ONLINE_LP
  H0 H1 H2 H3 H4 H5 FAIRROUTE FAIRROUTE_V2
)

# --- Sweep-specific (was scripts/run_client_sweep.sh) -----------------------
SWEEP_WORKLOAD_DIR="${SWEEP_WORKLOAD_DIR:-workloads/client_sweep}"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-outputs/client_sweep}"
SWEEP_PLOTS_DIR="${SWEEP_PLOTS_DIR:-$SWEEP_OUTPUT_DIR/plots}"
SWEEP_NUM_REQS="${SWEEP_NUM_REQS:-500}"
SWEEP_STATUS_FILE="${SWEEP_STATUS_FILE:-$SWEEP_OUTPUT_DIR/sweep_status.txt}"
SWEEP_LOG_FILE="${SWEEP_LOG_FILE:-$SWEEP_OUTPUT_DIR/sweep.log}"
CLIENT_COUNTS=(5 10 20 50 100 200)
SWEEP_POLICIES=(FAIRROUTE_V2 LOAD LOCALITY FAIRNESS PREBLE H0 H1 H4)

# --- Combined-run control files (used only for MODE=all) -------------------
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/run_all}"
STATUS_FILE="${STATUS_FILE:-$RUN_ROOT_DIR/run_status.txt}"
LOG_FILE="${LOG_FILE:-$RUN_ROOT_DIR/run.log}"
PID_FILE="${PID_FILE:-$RUN_ROOT_DIR/run.pid}"

# ---------------------------------------------------------------------------
# Control-plane helpers (--detach / --status / --logs / --stop)
# ---------------------------------------------------------------------------
# For MODE=matrix or MODE=sweep, control acts on that stage's own status/log/pid
# files (identical behavior to the original two scripts). For MODE=all, control
# acts on the combined run_status.txt/run.log/run.pid, which in turn reports on
# whichever underlying stage is currently active.

case "$MODE" in
  matrix) _STATUS_FILE="$MATRIX_STATUS_FILE"; _LOG_FILE="$MATRIX_LOG_FILE"; _PID_FILE="$MATRIX_OUTPUT_DIR/matrix.pid" ;;
  sweep)  _STATUS_FILE="$SWEEP_STATUS_FILE";  _LOG_FILE="$SWEEP_LOG_FILE";  _PID_FILE="$SWEEP_OUTPUT_DIR/sweep.pid" ;;
  all)    _STATUS_FILE="$STATUS_FILE";        _LOG_FILE="$LOG_FILE";       _PID_FILE="$PID_FILE" ;;
esac

case "$ACTION" in
  --detach)
    mkdir -p "$(dirname "$_LOG_FILE")"
    if [[ -f "$_PID_FILE" ]] && kill -0 "$(cat "$_PID_FILE")" 2>/dev/null; then
      echo "'$MODE' is already running with PID $(cat "$_PID_FILE")."
      exit 0
    fi
    nohup "$0" "$MODE" >"$_LOG_FILE" 2>&1 &
    echo $! >"$_PID_FILE"
    echo "Started '$MODE' with PID $!."
    echo "Status: $_STATUS_FILE"
    echo "Logs:   $_LOG_FILE"
    exit 0
    ;;
  --status)
    if [[ "$MODE" == "all" ]]; then
      echo "--- matrix ---"
      [[ -f "$MATRIX_STATUS_FILE" ]] && cat "$MATRIX_STATUS_FILE" || echo "No matrix status file found: $MATRIX_STATUS_FILE"
      echo "--- sweep ---"
      [[ -f "$SWEEP_STATUS_FILE" ]] && cat "$SWEEP_STATUS_FILE" || echo "No sweep status file found: $SWEEP_STATUS_FILE"
      echo "--- combined ---"
      [[ -f "$STATUS_FILE" ]] && cat "$STATUS_FILE" || echo "No combined status file found: $STATUS_FILE"
    else
      [[ -f "$_STATUS_FILE" ]] && cat "$_STATUS_FILE" || echo "No status file found: $_STATUS_FILE"
    fi
    if [[ -f "$_PID_FILE" ]] && kill -0 "$(cat "$_PID_FILE")" 2>/dev/null; then
      echo "host_process=RUNNING pid=$(cat "$_PID_FILE")"
    else
      echo "host_process=STOPPED"
    fi
    exit 0
    ;;
  --logs)
    if [[ ! -f "$_LOG_FILE" ]]; then
      echo "No log file found: $_LOG_FILE" >&2
      exit 1
    fi
    tail -n "${TAIL_LINES:-40}" -f "$_LOG_FILE"
    exit 0
    ;;
  --stop)
    if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
      docker exec -i "$CONTAINER" bash -s <<'STOP_SCRIPT'
  set +e
  for pid in $(pgrep -f 'python3 -m serving' 2>/dev/null); do
    command_line=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    if [[ "$command_line" == *"fairroute_hypothesis"* || "$command_line" == *"client_sweep"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
STOP_SCRIPT
    fi
    for f in "$_PID_FILE" "$MATRIX_OUTPUT_DIR/matrix.pid" "$SWEEP_OUTPUT_DIR/sweep.pid"; do
      if [[ -f "$f" ]]; then
        host_pid=$(cat "$f")
        kill -TERM "$host_pid" 2>/dev/null || true
        rm -f "$f"
      fi
    done
    printf '%s status=STOPPED\n' "$(date -Is)" >"$_STATUS_FILE"
    echo "Stopped FairRoute '$MODE' workers. Docker container remains running."
    exit 0
    ;;
  "")
    ;;  # fall through to actually running the stage(s) in the foreground
  *)
    echo "Unknown action: $ACTION (expected --detach|--status|--logs|--stop)" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Docker container check (shared)
# ---------------------------------------------------------------------------
if ! docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "Container '$CONTAINER' is not running. Starting container..."
  docker start "$CONTAINER"
fi

# ---------------------------------------------------------------------------
# Stage 1: policy matrix (was run_fairroute_hypothesis_matrix.sh)
# ---------------------------------------------------------------------------
run_matrix_stage() {
  mkdir -p "$MATRIX_OUTPUT_DIR"
  printf '%s status=RUNNING stage=matrix container=%s\n' "$(date -Is)" "$CONTAINER" >"$MATRIX_STATUS_FILE"

  docker exec \
    -i \
    -e "CONFIG=$CONFIG" \
    -e "WORKLOAD_DIR=$MATRIX_WORKLOAD_DIR" \
    -e "OUTPUT_DIR=$MATRIX_OUTPUT_DIR" \
    -e "NUM_REQS=$MATRIX_NUM_REQS" \
    -e "SKIP_EXISTING=$SKIP_EXISTING" \
    -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
    -e "INCLUDE_REAL_WORKLOAD=$INCLUDE_REAL_WORKLOAD" \
    -e "REAL_WORKLOAD=$REAL_WORKLOAD" \
    -e "REAL_WORKLOAD_LIMIT=$REAL_WORKLOAD_LIMIT" \
    -e "CACHE_MODE=$CACHE_MODE" \
    -e "STATUS_FILE=$MATRIX_STATUS_FILE" \
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
  H0 H1 H2 H3 H4 H5 FAIRROUTE FAIRROUTE_V2
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

  echo "Matrix stage complete."
}

# ---------------------------------------------------------------------------
# Stage 2: client sweep (was run_client_sweep.sh)
# ---------------------------------------------------------------------------
run_sweep_stage() {
  mkdir -p "$SWEEP_OUTPUT_DIR" "$SWEEP_WORKLOAD_DIR" "$SWEEP_PLOTS_DIR"
  printf '%s status=RUNNING stage=sweep container=%s\n' "$(date -Is)" "$CONTAINER" >"$SWEEP_STATUS_FILE"

  docker exec \
    -i \
    -e "CONFIG=$CONFIG" \
    -e "WORKLOAD_DIR=$SWEEP_WORKLOAD_DIR" \
    -e "OUTPUT_DIR=$SWEEP_OUTPUT_DIR" \
    -e "PLOTS_DIR=$SWEEP_PLOTS_DIR" \
    -e "NUM_REQS=$SWEEP_NUM_REQS" \
    -e "SKIP_EXISTING=$SKIP_EXISTING" \
    -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
    -e "CACHE_MODE=$CACHE_MODE" \
    -e "STATUS_FILE=$SWEEP_STATUS_FILE" \
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

  echo "Sweep stage complete."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
mkdir -p "$RUN_ROOT_DIR"
printf '%s status=RUNNING mode=%s\n' "$(date -Is)" "$MODE" >"$STATUS_FILE"

case "$MODE" in
  matrix)
    run_matrix_stage
    ;;
  sweep)
    run_sweep_stage
    ;;
  all)
    echo "=========================================================================="
    echo "  Stage 1/2: policy matrix"
    echo "=========================================================================="
    run_matrix_stage
    printf '%s status=RUNNING mode=all stage=sweep\n' "$(date -Is)" >"$STATUS_FILE"
    echo "=========================================================================="
    echo "  Stage 2/2: client sweep"
    echo "=========================================================================="
    run_sweep_stage
    ;;
esac

printf '%s status=COMPLETE mode=%s\n' "$(date -Is)" "$MODE" >"$STATUS_FILE"
echo "Done: $MODE"
