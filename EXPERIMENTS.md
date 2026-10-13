The experiment runner now includes all registered request-routing policies, including `PRISM` and `D2LPM`.

Use the following sequence from the repository root.

```bash
cd "/media/sasank-v/NewVolume/Studies/College/VIT/Sem 7/Project I /Code/LLMServingSim"
```

## 1. Verify the virtual environment

```bash
.venv/bin/python --version
.venv/bin/python -c "import tokenizers; print('tokenizers OK')"
```

If your environment is `.venv-fairroute` instead:

```bash
.venv-fairroute/bin/python --version
.venv-fairroute/bin/python -c "import tokenizers; print('tokenizers OK')"
```

The scripts automatically prefer `.venv-fairroute`, then `.venv`.

## 2. Generate pilot workloads

```bash
PY="$PWD/.venv/bin/python" \
RAW="$PWD/workloads/datasets/raw" \
./scripts/generate_fairroute_workloads.sh pilot
```

If using `.venv-fairroute`:

```bash
PY="$PWD/.venv-fairroute/bin/python" \
RAW="$PWD/workloads/datasets/raw" \
./scripts/generate_fairroute_workloads.sh pilot
```

Check the generated pilot traces:

```bash
find workloads/fairroute/pilot -type f -name '*.jsonl' | sort
```

## 3. Set the saturation rates

The main and scaling workload generators require one measured saturation rate per workload. Replace these example values with the rates determined from the RR pilot runs:

```bash
export LAM_W1=2
export LAM_W2=2
export LAM_W3=2
export LAM_W4=0.5
export LAM_W5=4
```

These are the current RR saturation-rate estimates (requests/s, except W4
session starts/s), read from the observed RR knee in the checked-in analysis
outputs. The pilot traces exist, but there are no RR pilot result files under
`outputs/fairroute/pilot/` yet. Run the pilot with RR and point the notebook at
those results to replace these estimates with direct pilot measurements before
regenerating the workload suite. The notebook documents the knee rule and
prints the suggested `LAM_*` assignments.

Analysis notebook: [`notebooks/fairroute_rr_breaking_point.ipynb`](notebooks/fairroute_rr_breaking_point.ipynb).

You can first inspect the commands without generating files:

```bash
PY="$PWD/.venv/bin/python" \
DRY=1 \
WL="w1 w2 w3 w4 w5" \
./scripts/generate_fairroute_workloads.sh main
```

## 4. Generate the complete main workload suite

```bash
PY="$PWD/.venv/bin/python" \
RAW="$PWD/workloads/datasets/raw" \
SEEDS=100 \
WL="w1 w2 w3 w4 w5" \
./scripts/generate_fairroute_workloads.sh main
```

For the tuning workloads:

```bash
PY="$PWD/.venv/bin/python" \
RAW="$PWD/workloads/datasets/raw" \
SEEDS=7 \
WL="w1 w3" \
LEVELS="0.8 1.0" \
./scripts/generate_fairroute_workloads.sh main
```

## 5. Generate the scaling workloads

```bash
PY="$PWD/.venv/bin/python" \
RAW="$PWD/workloads/datasets/raw" \
SEEDS=100 \
./scripts/generate_fairroute_workloads.sh scaling
```

The generated workloads should be under:

```text
workloads/fairroute/
├── pilot/
├── main/
└── scaling/
```

Verify the expected counts:

```bash
./scripts/generate_fairroute_workloads.sh check
```

## 6. Ensure the simulator container is running

Check for the expected container:

```bash
docker ps -a --filter name=servingsim_docker
```

Start it if it already exists but is stopped:

```bash
docker start servingsim_docker
```

If it has not been created yet, use the repository’s simulator setup script according to your existing Docker setup:

```bash
./scripts/docker-sim.sh
```

## 7. Run every workload with every policy

The matrix runner recursively discovers all `.jsonl` files under `workloads/fairroute`.

Run everything in the foreground:

```bash
MATRIX_WORKLOAD_DIR=workloads/fairroute \
MATRIX_OUTPUT_DIR=outputs/fairroute \
CONFIG=configs/cluster/dual_node_multi_instance_policy_study.json \
CACHE_MODE=xpu \
PARALLEL_JOBS=1 \
SKIP_EXISTING=0 \
./scripts/run_fairroute_experiments.sh matrix
```

The complete policy set is:

```text
RR
LOAD
RAND
FAIRROUTE
FAIRROUTE_V2
FAIRROUTE_LINEAR
FAIRROUTE_MULTIPLICATIVE
FAIRROUTE_GATED
FAIRROUTE_LEXICOGRAPHIC
FAIRROUTE_SLACK
FAIRROUTE_LEARNED
PRISM
D2LPM
```

With `PARALLEL_JOBS=4` (the default), four policy runs execute concurrently on
ext4 — safe and significantly faster than serial.

A per-run wall-clock timeout of `SIM_TIMEOUT=1800` (30 min) is active by
default. Any simulation that hangs (e.g. ASTRA-Sim crash, disk issue) will be
killed after that deadline and marked failed rather than stalling the whole
matrix. Increase or disable with `SIM_TIMEOUT=3600` / `SIM_TIMEOUT=0`.

## 8. Run in the background

```bash
MATRIX_WORKLOAD_DIR=workloads/fairroute \
MATRIX_OUTPUT_DIR=outputs/fairroute \
CONFIG=configs/cluster/dual_node_multi_instance_policy_study.json \
CACHE_MODE=xpu \
PARALLEL_JOBS=4 \
SKIP_EXISTING=0 \
./scripts/run_fairroute_experiments.sh matrix --detach
```

Check status:

```bash
./scripts/run_fairroute_experiments.sh matrix --status
```

Follow the log:

```bash
TAIL_LINES=100 \
./scripts/run_fairroute_experiments.sh matrix --logs
```

Stop the run if needed:

```bash
./scripts/run_fairroute_experiments.sh matrix --stop
```

## 9. Prefix-storage variants

Run with CPU prefix storage:

```bash
MATRIX_WORKLOAD_DIR=workloads/fairroute \
MATRIX_OUTPUT_DIR=outputs/fairroute_cpu \
CONFIG=configs/cluster/dual_node_multi_instance_policy_study.json \
CACHE_MODE=cpu \
PARALLEL_JOBS=1 \
./scripts/run_fairroute_experiments.sh matrix
```

Run with CXL prefix storage:

```bash
MATRIX_WORKLOAD_DIR=workloads/fairroute \
MATRIX_OUTPUT_DIR=outputs/fairroute_cxl \
CONFIG=configs/cluster/dual_node_multi_instance_policy_study.json \
CACHE_MODE=cxl \
PARALLEL_JOBS=1 \
./scripts/run_fairroute_experiments.sh matrix
```

## 10. Inspect results

Each result is written as a policy-specific CSV:

```bash
find outputs/fairroute -type f -name '*.csv' | sort
```

For example:

```text
outputs/fairroute/main/w1/..._policy_RR.csv
outputs/fairroute/main/w1/..._policy_LOAD.csv
outputs/fairroute/main/w1/..._policy_PRISM.csv
outputs/fairroute/main/w1/..._policy_D2LPM.csv
outputs/fairroute/main/w1/..._policy_FAIRROUTE_V2.csv
```

Count generated result files:

```bash
find outputs/fairroute -type f -name '*.csv' | wc -l
```

The runner was updated in `run_fairroute_experiments.sh` to include `PRISM` and `D2LPM`.
