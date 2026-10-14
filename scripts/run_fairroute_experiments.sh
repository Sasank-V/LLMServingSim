#!/usr/bin/env bash
# scripts/run_fairroute_experiments.sh
#
# Unified entry point for the policy matrix and client-count experiment drivers that used to be
# separate scripts:
#   - the request-routing policy matrix
#   - scripts/run_client_sweep.sh                 (client-count scalability sweep)
#
# Usage:
#   ./scripts/run_fairroute_experiments.sh <matrix|sweep|all> [--detach|--status|--logs|--stop]
#
# Examples:
#   ./scripts/run_fairroute_experiments.sh matrix              # run every workload under workloads/fairroute
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
DOCKER_USER="${DOCKER_USER:-0:0}"
CONFIG="${CONFIG:-configs/cluster/dual_node_multi_instance_policy_study.json}"
CACHE_MODE="${CACHE_MODE:-xpu}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Per-simulation wall-clock timeout in seconds (0 = no timeout).
# Protects the run from a single hung simulation stalling the entire matrix.
SIM_TIMEOUT="${SIM_TIMEOUT:-1800}"

# --- Matrix-specific (was scripts/run_fairroute_hypothesis_matrix.sh) ------
MATRIX_WORKLOAD_DIR="${MATRIX_WORKLOAD_DIR:-workloads/fairroute/main}"
MATRIX_WORKLOAD_PATTERN="${MATRIX_WORKLOAD_PATTERN:-trace.jsonl}"
MATRIX_OUTPUT_DIR="${MATRIX_OUTPUT_DIR:-outputs/fairroute/main}"
MATRIX_NUM_REQS="${MATRIX_NUM_REQS:-1000}"
MATRIX_NUM_CLIENTS="${MATRIX_NUM_CLIENTS:-8}"
MATRIX_SEED="${MATRIX_SEED:-42}"
MATRIX_WORKLOAD_SCENARIOS="${MATRIX_WORKLOAD_SCENARIOS:-c0_calibration,balanced_control,locality_conflict,fairness_conflict,prediction_burst,full_adversarial,u5_vtc_as_published,w2_heavy_light,w4_flooding,u2_multiturn,u3_loogle_proxy}"
GENERATE_WORKLOADS="${GENERATE_WORKLOADS:-0}"
INCLUDE_REAL_WORKLOAD="${INCLUDE_REAL_WORKLOAD:-0}"
REAL_WORKLOAD="${REAL_WORKLOAD:-workloads/workload-llama8b-full-sps10-100clients.jsonl}"
REAL_WORKLOAD_LIMIT="${REAL_WORKLOAD_LIMIT:-1000}"
MATRIX_STATUS_FILE="${MATRIX_STATUS_FILE:-$MATRIX_OUTPUT_DIR/matrix_status.txt}"
MATRIX_LOG_FILE="${MATRIX_LOG_FILE:-$MATRIX_OUTPUT_DIR/matrix.log}"
# Router policies. Use MATRIX_LOCAL_POLICIES="FCFS VTC DLPM" for the factorial
# routing × local-scheduling comparison; MATRIX_LOCAL_POLICY remains a
# backwards-compatible single-policy override.
# Keep these as space-separated strings: environment variables are strings,
# and the container expands them into Bash arrays immediately before dispatch.
# Initializing them as arrays here would overwrite caller-provided values.
MATRIX_POLICIES="${MATRIX_POLICIES:-RR LOAD RAND PRISM D2LPM FAIRROUTE FAIRROUTE_SLACK FAIRROUTE_GATED}"
MATRIX_LOCAL_POLICY="${MATRIX_LOCAL_POLICY:-FCFS}"
MATRIX_LOCAL_POLICIES="${MATRIX_LOCAL_POLICIES:-$MATRIX_LOCAL_POLICY}"

# --- Sweep-specific (was scripts/run_client_sweep.sh) -----------------------
SWEEP_WORKLOAD_DIR="${SWEEP_WORKLOAD_DIR:-workloads/client_sweep}"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-outputs/client_sweep}"
SWEEP_PLOTS_DIR="${SWEEP_PLOTS_DIR:-$SWEEP_OUTPUT_DIR/plots}"
SWEEP_NUM_REQS="${SWEEP_NUM_REQS:-500}"
SWEEP_STATUS_FILE="${SWEEP_STATUS_FILE:-$SWEEP_OUTPUT_DIR/sweep_status.txt}"
SWEEP_LOG_FILE="${SWEEP_LOG_FILE:-$SWEEP_OUTPUT_DIR/sweep.log}"
CLIENT_COUNTS=(5 10 20 50 100 200)

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

check_writable_dir() {
  local path="$1"
  mkdir -p "$path" 2>/dev/null || {
    echo "Cannot create experiment directory: $path" >&2
    return 1
  }
  local probe
  probe=$(mktemp "$path/.write_probe.XXXXXX" 2>/dev/null) || {
    local fs_type
    fs_type=$(stat -f -c '%T' "$path" 2>/dev/null || printf 'unknown')
    cat >&2 <<EOF
Cannot create files in '$path' (filesystem: $fs_type).
The filesystem is reporting ENOSPC despite having free bytes. This is a
problem with the NTFS/FUSE mount, not Docker memory or simulator concurrency.
Remount the volume, or move this repository to a native Linux filesystem
(for example under /home), then rerun the experiment.
EOF
    return 1
  }
  rm -f "$probe"
}

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
    echo "Forcefully stopping all FairRoute '$MODE' processes (host & container) and cleaning stale data..."

    # 1. Kill host-side script processes and docker exec commands
    for f in "$_PID_FILE" "$MATRIX_OUTPUT_DIR/matrix.pid" "$SWEEP_OUTPUT_DIR/sweep.pid"; do
      if [[ -f "$f" ]]; then
        host_pid=$(cat "$f")
        if kill -0 "$host_pid" 2>/dev/null; then
          kill -9 "$host_pid" 2>/dev/null || true
        fi
        rm -f "$f"
      fi
    done
    pkill -9 -f "run_fairroute_experiments.sh" 2>/dev/null || true
    pkill -9 -f "docker exec.*$CONTAINER" 2>/dev/null || true

    # 2. Kill processes inside container (python frontend, C++ ASTRA-Sim backend, bash loops)
    if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
      docker exec -i --user 0 "$CONTAINER" bash -c 'for p in $(ps -eo pid=); do if [ "$p" -gt 1 ] && [ "$p" -ne "$$" ]; then kill -9 "$p" 2>/dev/null || true; fi; done' 2>/dev/null || true
    fi

    # 3. Clean stale data & orphaned state
    rm -rf astra-sim/inputs/runs/* 2>/dev/null || true

    for state_dir in "$MATRIX_OUTPUT_DIR/.matrix_state" "$SWEEP_OUTPUT_DIR/.sweep_state"; do
      if [[ -d "$state_dir" ]]; then
        rm -rf "$state_dir" 2>/dev/null || true
      fi
    done
    find "$MATRIX_OUTPUT_DIR" "$SWEEP_OUTPUT_DIR" "$RUN_ROOT_DIR" -name '*.tmp.*' -delete 2>/dev/null || true

    # 4. Update status files to STOPPED
    for sf in "$_STATUS_FILE" "$MATRIX_STATUS_FILE" "$SWEEP_STATUS_FILE" "$STATUS_FILE"; do
      if [[ -n "$sf" ]]; then
        mkdir -p "$(dirname "$sf")" 2>/dev/null || true
        printf '%s status=STOPPED\n' "$(date -Is)" >"$sf" 2>/dev/null || true
      fi
    done

    echo "Forcefully stopped all FairRoute '$MODE' processes and cleaned stale trace/state data."
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
case "$MODE" in
  matrix) check_writable_dir "$MATRIX_OUTPUT_DIR" ;;
  sweep) check_writable_dir "$SWEEP_OUTPUT_DIR" ;;
  all)
    check_writable_dir "$RUN_ROOT_DIR"
    check_writable_dir "$MATRIX_OUTPUT_DIR"
    check_writable_dir "$SWEEP_OUTPUT_DIR"
    ;;
esac

if ! docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "Container '$CONTAINER' is not running. Starting container..."
  docker start "$CONTAINER"
fi

# ---------------------------------------------------------------------------
# Stage 1: policy matrix (was run_fairroute_hypothesis_matrix.sh)
# ---------------------------------------------------------------------------
run_matrix_stage() {
  mkdir -p "$MATRIX_OUTPUT_DIR"
  matrix_fs_type=$(stat -f -c '%T' "$MATRIX_OUTPUT_DIR" 2>/dev/null || printf 'unknown')
  echo "Output filesystem type: $matrix_fs_type"
  printf '%s status=RUNNING stage=matrix container=%s\n' "$(date -Is)" "$CONTAINER" >"$MATRIX_STATUS_FILE"

  docker exec \
    -i \
    --user "$DOCKER_USER" \
    -e "CONFIG=$CONFIG" \
    -e "WORKLOAD_DIR=$MATRIX_WORKLOAD_DIR" \
    -e "OUTPUT_DIR=$MATRIX_OUTPUT_DIR" \
    -e "NUM_REQS=$MATRIX_NUM_REQS" \
    -e "NUM_CLIENTS=$MATRIX_NUM_CLIENTS" \
    -e "GENERATOR_SEED=$MATRIX_SEED" \
    -e "WORKLOAD_SCENARIOS=$MATRIX_WORKLOAD_SCENARIOS" \
    -e "GENERATE_WORKLOADS=$GENERATE_WORKLOADS" \
    -e "SKIP_EXISTING=$SKIP_EXISTING" \
    -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
    -e "MAX_NUM_SEQS=$MAX_NUM_SEQS" \
    -e "WORKLOAD_PATTERN=$MATRIX_WORKLOAD_PATTERN" \
    -e "INCLUDE_REAL_WORKLOAD=$INCLUDE_REAL_WORKLOAD" \
    -e "REAL_WORKLOAD=$REAL_WORKLOAD" \
    -e "REAL_WORKLOAD_LIMIT=$REAL_WORKLOAD_LIMIT" \
    -e "MATRIX_NUM_REQS=$MATRIX_NUM_REQS" \
    -e "MATRIX_POLICIES=${MATRIX_POLICIES[*]}" \
    -e "MATRIX_LOCAL_POLICY=$MATRIX_LOCAL_POLICY" \
    -e "MATRIX_LOCAL_POLICIES=$MATRIX_LOCAL_POLICIES" \
    -e "CACHE_MODE=$CACHE_MODE" \
    -e "STATUS_FILE=$MATRIX_STATUS_FILE" \
    "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
cd /app/LLMServingSim
mkdir -p "$OUTPUT_DIR" log astra-sim/inputs/runs
chmod 777 log astra-sim/inputs/runs 2>/dev/null || true
printf '%s status=RUNNING container=%s\n' "$(date -Is)" "$HOSTNAME" >"$STATUS_FILE"

if [[ "$GENERATE_WORKLOADS" == "1" ]]; then
  echo "Generating FairRoute workloads: scenarios=$WORKLOAD_SCENARIOS seed=$GENERATOR_SEED clients=$NUM_CLIENTS"
  python3 -m workloads.generators.generate_hypothesis_workloads \
    --output-dir "$WORKLOAD_DIR" \
    --num-reqs "$NUM_REQS" \
    --num-clients "$NUM_CLIENTS" \
    --seed "$GENERATOR_SEED" \
    --scenarios "$WORKLOAD_SCENARIOS"
fi

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

if [[ -n "${MATRIX_POLICIES:-}" ]]; then
  read -r -a POLICIES <<< "$MATRIX_POLICIES"
else
  POLICIES=(
    RR LOAD RAND
    PRISM D2LPM
    FAIRROUTE FAIRROUTE_SLACK FAIRROUTE_GATED
  )
fi
read -r -a LOCAL_POLICIES <<< "${MATRIX_LOCAL_POLICIES:-${MATRIX_LOCAL_POLICY:-FCFS}}"

mapfile -d '' datasets < <(
  find "$WORKLOAD_DIR" -type f -name "${WORKLOAD_PATTERN:-trace.jsonl}" -print0 | sort -z
)
if (( ${#datasets[@]} == 0 )); then
  echo "No JSONL workloads found recursively under $WORKLOAD_DIR." >&2
  exit 3
fi

STATE_DIR="$OUTPUT_DIR/.matrix_state"
mkdir -p "$STATE_DIR"
if [[ "$SKIP_EXISTING" != "1" ]]; then
  rm -f "$STATE_DIR"/*
fi
total_jobs=$(( ${#datasets[@]} * ${#POLICIES[@]} * ${#LOCAL_POLICIES[@]} ))

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
  chmod 666 "$tmp_status" 2>/dev/null || true
  mv -f "$tmp_status" "$STATUS_FILE"
  chmod 666 "$STATUS_FILE" 2>/dev/null || true
}

run_policy() {
  local dataset="$1"
  local scenario="$2"
  local policy="$3"
  local local_policy="$4"
  local local_suffix=""
  if [[ "$local_policy" != "FCFS" ]]; then
    local_suffix="_local_${local_policy}"
  fi
  local output="$OUTPUT_DIR/${scenario}_policy_${policy}${local_suffix}.csv"
  local state_key="${scenario//\//__}__${policy}${local_suffix}"
  local run_key="${scenario//\//__}"
  local state_file="$STATE_DIR/$state_key.running"
  local expected_rows actual_rows

  expected_rows=$(wc -l <"$dataset")
  if [[ "$NUM_REQS" -gt 0 && "$expected_rows" -gt "$NUM_REQS" ]]; then
    expected_rows="$NUM_REQS"
  fi

  : >"$state_file"
  write_status RUNNING "$scenario/$policy/$local_policy"
  actual_rows=0
  if [[ -s "$output" ]]; then
    actual_rows=$(wc -l <"$output")
  fi

  # Remove partial/incomplete outputs from interrupted runs
  if [[ -s "$output" && "$actual_rows" -lt $((expected_rows + 1)) ]]; then
    echo "=== CLEANUP partial output ($actual_rows/$((expected_rows + 1)) rows): $scenario / $policy ==="
    rm -f "$output" "${output%.csv}_timeseries.csv" "${output}.routing.json" "${output}.sidecar.jsonl" "${output%.csv}_sidecar.jsonl" "${output}.manifest.json"
    actual_rows=0
  fi

  if [[ "$SKIP_EXISTING" == "1" && "$actual_rows" -ge $((expected_rows + 1)) ]]; then
    echo "=== SKIP (complete) $scenario / $policy ==="
    mv -f "$state_file" "$STATE_DIR/$state_key.skipped"
    write_status RUNNING "$scenario/$policy/$local_policy"
    return 0
  fi

  echo "=== RUN $scenario / $policy ==="
  mkdir -p "$(dirname "$output")"
  local timeout_prefix=()
  if [[ "${SIM_TIMEOUT:-0}" -gt 0 ]]; then
    timeout_prefix=(timeout --kill-after=10s "$SIM_TIMEOUT")
  fi
  local seq_args=()
  if [[ -n "${MAX_NUM_SEQS:-}" && "${MAX_NUM_SEQS}" -gt 0 ]]; then
    seq_args=(--max-num-seqs "$MAX_NUM_SEQS")
  fi
  if "${timeout_prefix[@]}" python3 -m serving \
      --cluster-config "$CONFIG" \
      --request-routing-policy "$policy" \
      --local-scheduling-policy "$local_policy" \
      --dtype bfloat16 --block-size 16 \
      "${seq_args[@]}" \
      "${CACHE_ARGS[@]}" \
      --dataset "$dataset" --num-reqs "$NUM_REQS" \
      --run-id "fairroute_${run_key}_${policy}_${local_policy}_${BASHPID}" \
      --output "$output" \
      --log-level WARNING; then
    mv -f "$state_file" "$STATE_DIR/$state_key.done"
    write_status RUNNING "$scenario/$policy/$local_policy"
  else
    local exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
      echo "=== TIMEOUT $scenario / $policy (${SIM_TIMEOUT}s) ==" >&2
    fi
    mv -f "$state_file" "$STATE_DIR/$state_key.failed"
    write_status FAILED "$scenario/$policy/$local_policy"
    return 1
  fi
}

running_jobs=0
failed_jobs=0
for dataset in "${datasets[@]}"; do
  scenario="${dataset#"$WORKLOAD_DIR"/}"
  scenario="${scenario%.jsonl}"
  for policy in "${POLICIES[@]}"; do
    for local_policy in "${LOCAL_POLICIES[@]}"; do
      run_policy "$dataset" "$scenario" "$policy" "$local_policy" &
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
  sweep_fs_type=$(stat -f -c '%T' "$SWEEP_OUTPUT_DIR" 2>/dev/null || printf 'unknown')
  echo "Output filesystem type: $sweep_fs_type"
  printf '%s status=RUNNING stage=sweep container=%s\n' "$(date -Is)" "$CONTAINER" >"$SWEEP_STATUS_FILE"

  docker exec \
    -i \
    --user "$DOCKER_USER" \
    -e "CONFIG=$CONFIG" \
    -e "WORKLOAD_DIR=$SWEEP_WORKLOAD_DIR" \
    -e "OUTPUT_DIR=$SWEEP_OUTPUT_DIR" \
    -e "PLOTS_DIR=$SWEEP_PLOTS_DIR" \
    -e "NUM_REQS=$SWEEP_NUM_REQS" \
    -e "SKIP_EXISTING=$SKIP_EXISTING" \
    -e "PARALLEL_JOBS=$PARALLEL_JOBS" \
    -e "MAX_NUM_SEQS=$MAX_NUM_SEQS" \
    -e "CACHE_MODE=$CACHE_MODE" \
    -e "STATUS_FILE=$SWEEP_STATUS_FILE" \
    "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
cd /app/LLMServingSim
mkdir -p "$OUTPUT_DIR" "$WORKLOAD_DIR" "$PLOTS_DIR" log astra-sim/inputs/runs
chmod 777 log astra-sim/inputs/runs 2>/dev/null || true
printf '%s status=RUNNING container=%s\n' "$(date -Is)" "$HOSTNAME" >"$STATUS_FILE"

CLIENT_COUNTS=(5 10 20 50 100 200)
POLICIES=(LOAD RR RAND PRISM D2LPM FAIRROUTE FAIRROUTE_SLACK FAIRROUTE_GATED)

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
  chmod 666 "$tmp_status" 2>/dev/null || true
  mv -f "$tmp_status" "$STATUS_FILE"
  chmod 666 "$STATUS_FILE" 2>/dev/null || true
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

  local expected_rows="$NUM_REQS"
  local actual_rows=0
  if [[ -s "$output" ]]; then
    actual_rows=$(wc -l <"$output")
  fi

  # Remove partial/incomplete outputs from interrupted runs
  if [[ -s "$output" && "$expected_rows" -gt 0 && "$actual_rows" -lt $((expected_rows + 1)) ]]; then
    echo "=== CLEANUP partial output ($actual_rows/$((expected_rows + 1)) rows): c${n_clients}/${scenario}/${policy} ==="
    rm -f "$output" "${output%.csv}_timeseries.csv" "${output}.manifest.json"
    actual_rows=0
  fi

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    if [[ "$expected_rows" -gt 0 && "$actual_rows" -ge $((expected_rows + 1)) ]] || \
       [[ "$expected_rows" -eq 0 && -s "$output" ]]; then
      echo "=== SKIP (complete) c${n_clients}/${scenario}/${policy} ==="
      mv -f "$state_file" "$STATE_DIR/$state_key.skipped"
      write_status RUNNING "c${n_clients}/${scenario}/${policy}"
      return 0
    fi
  fi

  echo "=== RUN clients=$n_clients / scenario=$scenario / policy=$policy ==="
  local timeout_prefix=()
  if [[ "${SIM_TIMEOUT:-0}" -gt 0 ]]; then
    timeout_prefix=(timeout --kill-after=10s "$SIM_TIMEOUT")
  fi
  local seq_args=()
  if [[ -n "${MAX_NUM_SEQS:-}" && "${MAX_NUM_SEQS}" -gt 0 ]]; then
    seq_args=(--max-num-seqs "$MAX_NUM_SEQS")
  fi
  if "${timeout_prefix[@]}" python3 -m serving \
      --cluster-config "$CONFIG" \
      --request-routing-policy "$policy" \
      --dtype bfloat16 --block-size 16 \
      "${seq_args[@]}" \
      "${CACHE_ARGS[@]}" \
      --dataset "$dataset" --num-reqs "$NUM_REQS" \
      --run-id "fairroute_sweep_c${n_clients}_${scenario}_${policy}_${BASHPID}" \
      --output "$output" \
      --log-level WARNING; then
    mv -f "$state_file" "$STATE_DIR/$state_key.done"
    write_status RUNNING "c${n_clients}/${scenario}/${policy}"
  else
    local exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
      echo "=== TIMEOUT clients=$n_clients/$scenario/$policy (${SIM_TIMEOUT}s) ==" >&2
    fi
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
