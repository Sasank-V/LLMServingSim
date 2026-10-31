<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/img/llmservingsim_full_primary_dark_transparent.png">
    <img alt="LLMServingSim" src="docs/static/img/llmservingsim_full_primary_transparent.png" width="70%">
  </picture>
</p>

<h3 align="center">A cycle-level simulator for heterogeneous and disaggregated LLM serving</h3>

<p align="center">
  <a href="https://llmservingsim.ai">Website</a> ·
  <a href="https://llmservingsim.ai/docs/getting-started/overview">Documentation</a> ·
  <a href="https://llmservingsim.ai/docs/contributor/welcome">Contributing</a>
</p>

LLMServingSim combines a Python model of a vLLM-style serving system with
ASTRA-Sim's analytical network backend. It runs a request trace against a
configurable cluster and reports request-level latency, queueing, memory, and
routing behavior. Layer-level latency tables acquired from real vLLM execution
drive the simulation.

The simulator supports dense and MoE models, sparse and linear attention,
continuous batching, KV-cache and prefix sharing, CPU/CXL/PIM memory tiers,
speculative decoding, and TP/PP/EP/DP configurations.

## Simulation flow

```text
JSONL workload
  → Router selects an instance; local scheduler admits a batch
  → Trace generator looks up profiled layer latency and collectives
  → Chakra graph generator emits an execution trace (.et)
  → ASTRA-Sim returns cycle-level execution/network time
  → Scheduler updates request state, KV cache, queues, and output CSVs
```

| Path | Responsibility |
| --- | --- |
| `serving/` | Simulator CLI, batching, routing, memory/KV model, trace creation, and ASTRA-Sim IPC |
| `profiler/` | vLLM layerwise profiling and per-hardware latency-table generation |
| `bench/` | Real vLLM replay, recording, and simulator-versus-vLLM validation |
| `configs/` | Model, cluster/topology, and PIM configurations |
| `workloads/` | JSONL traces, source-data conversion, and workload builders |
| `scripts/` | Container/build entry points and FairRoute experiment drivers |
| `astra-sim/` | ASTRA-Sim backend submodule plus Chakra graph converter |

`python -m serving` reads a cluster JSON and workload JSONL. Its result CSV
has one row per completed request, including arrival, queueing delay, TTFT,
TPOT, latency, routing decision, and cache fields. Agentic workload rows may
contain ordered `sub_requests`, released only after prior turns and modeled
tool delay complete.

## Quick start

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
./scripts/docker-sim.sh
./scripts/compile.sh
./serving/run.sh
```

```bash
python -m serving \
  --cluster-config configs/cluster/dual_node_multi_instance_policy_study.json \
  --dataset workloads/example_trace.jsonl \
  --request-routing-policy RR \
  --output outputs/example.csv
```

Profiling new hardware requires the vLLM environment. Simulation of committed
profile bundles and validation against recorded examples are CPU-only. See the
[documentation](https://llmservingsim.ai/docs/getting-started/overview) for
installation, profiler, benchmark, and CLI details.

## Routing and scheduling algorithms

Routing and local admission are separate decisions:

1. The **global router** chooses an instance for an arriving request.
2. The selected instance's **local scheduler** chooses its next admission.

Implementation is in `serving/core/router.py` and
`serving/core/algorithms/`. The formulas below describe the code in this
repository; a policy name does not imply full reproduction of every detail of
the similarly named external paper.

### Shared candidate model

For replica `i`, the router computes prefix reuse `H_i`, uncached work
`n_ext = max(0, n_in - H_i)`, queued token work `W_i`, active request count
`B_i`, decode-like count `D_i`, free KV capacity, and a per-client output
estimate. Prefix locality is:

```text
h_i = clamp(H_i / (n_in + 1e-8), 0, 1)
```

The live LOAD pressure is:

```text
load_i = (4 × waiting_i + running_i) / max_num_seqs_i
```

Queued requests count four times as much as running requests. A deterministic
MD5-derived request/candidate jitter breaks score ties without permanent
low-index bias.

Request-aware policies use the analytical iteration model:

```text
P(B,S)  = p_max (1 - exp(-k_B B)) (1 - exp(-k_S S))
t_iter  = tau_0 + (w_0 + w_S S) / P(B,S) + tau_B B + tau_S S

TTFT_i  = ceil((W_i + n_ext) / max(1, S_max - D_i)) × t_iter(B_i + 1, S_max)
decode_i = output_estimate × t_iter(D_i + 1, D_i + 1)
T_i     = TTFT_i + decode_i
SLO     = k × ceil(n_in / S_max) × t_iter(1, S_max)
```

`T_i` is a fast candidate-ranking predictor, not the final cycle-level result
from ASTRA-Sim. Candidates also carry their required KV capacity
`n_in + output_estimate`.

### Global routing policies

| Policy | Implemented behavior |
| --- | --- |
| `RR` | Cycles through instances independently for prefill and decode. |
| `RAND` | Seeded uniform random instance choice. |
| `LOAD` | Minimizes normalized `(4 × waiting + running) / capacity`; rotating scan resolves equal scores. |
| `PRISM` | Simulator scoring approximation, not a full-paper reproduction. Maximizes `s_lat × s_load × s_cap × affinity`, where `s_lat=1/(1+(T_i/SLO)^p)`, `s_load=1/(1+((W_i/S_max)/SLO)^p)`, `s_cap=clamp((freeKV_i-KV_i)/KV_i,0,1)`, and affinity is `beta` only for a sufficiently large prefix hit. |
| `D2LPM` | Maintains a deficit per client/replica. It refills an exhausted client's deficits, prefers replicas with the longest prefix match, then minimizes predicted latency, queue length, and deterministic jitter. It charges input/extension work and output work. |
| `FAIRROUTE` / `FAIRROUTE_V2` | Feasibility gate, urgency-dependent latency slack, cache-locality credit, client fairness, and replica-service deficit. |
| `FAIRROUTE_GATED` | Same FairRoute candidate model, but selects without the locality slack set. |
| `FAIRROUTE_SLACK` | Same FairRoute candidate model with slack-set/lexicographic selection. |
| `FAIRROUTE_P` | Separately calibrated token- and slot-aware FairRoute. Fit `fp_prefill_tps` and `fp_tpot` from low-load RR measurements before use. |

`PrebleE2`, `PerfModel`, and `lens_search` are implemented components, but
not selectable full routing policies in this experiment driver.

### FairRoute formulas and selection

FairRoute records delivered client service. For total service `S_total`,
client service `S_c`, and `N` known clients:

```text
target service A = S_total / N
fairness debt d_c = max(0, A - S_c) / (A + 1e-8)
urgency u_c = 1 - exp(-2 d_c)
```

Higher urgency steers an underserved client toward less queued replicas.
Candidate safety combines queue and KV conditions. Its queue projection is
`max(0, Q + (arrival_rate - service_rate) × horizon)`; the default FairRoute
call uses zero rates, so the queue portion is current-queue persistence. The
TTFT/decode predictor is the primary forward-looking component.

The FairRoute gate keeps candidates with enough free KV and
`TTFT_i ≤ gamma × SLO`; when none pass it falls back to all candidates. The
latency band is:

```text
epsilon = min(eps_max, rho × min_i(T_i)) × (1 - u_c)
slack = { i | T_i ≤ min_j(T_j) + epsilon }
```

The primary `FAIRROUTE` policy selects lexicographically within the band:

1. a hit of at least `fairroute_h_min` tokens with positive locality credit;
2. client fairness toward less queued work;
3. replica service deficit;
4. lower predicted latency;
5. deterministic jitter.

Selection spends a per-client/per-replica credit by uncached prompt work.
Once every replica's credit for that client is depleted, they are refilled.
This bounds unlimited cache-locality preference. FairRoute variants share
candidate construction, avoiding a predictor change as a confound.

### Local admission policies

| Local policy | Implemented behavior |
| --- | --- |
| `FCFS` | Ordinary arrival-order admission. |
| `VTC` | Virtual-time fairness: choose the lowest client counter; first admission charges `w_p × prompt_tokens`, each output step charges `w_q × count`. |
| `DLPM` | Deficit local policy: when no queued client has positive deficit, issue a quantum; charge uncached prompt work at admission and output work during generation. Prefer larger prefix hits, then earlier arrival. |

## FairRoute experiment structure

The real-data FairRoute suite and the synthetic hypothesis generator are
separate experiment families:

```text
raw public data → conversion / proxy construction → pilot traces
                → RR pilots → knee-rate selection → main/scaling traces
                → policy × local-policy matrix

synthetic hypothesis generator → controlled diagnostic scenarios only
```

### Workloads

`scripts/build_fairroute_suite.sh` is the generator;
`scripts/generate_fairroute_workloads.sh` is its compatibility wrapper.

| Workload | Stressor and source |
| --- | --- |
| W1 `FAIRMIX` | Two long arXiv-heavy clients carry 12.5% of load; `K-2` ShareGPT light clients carry the rest. |
| W2 `LENMIX` | 90% short ShareGPT, 5% long-prefill arXiv, and 5% long-decode Code-Reasoning requests. |
| W3 `FLOOD` | Controlled prefix/cache pressure: one flood client has 35% of requests and a long shared prefix; normal clients have separate prefixes. The checked-in matrix traces use `flood_proxy.py`; the current builder's RAG path is `rag_workloads.py`. |
| W4 `ARENA` | LMSYS Arena sessions. Selected model name becomes the client; multi-turn sessions retain think/tool delays. |
| W5 `BURST` | Azure conversation-trace replay, rate-rescaled while retaining burst shape. Synthetic token IDs are used because the source has counts, not text. |

`workloads/datasets/download_datasets.py` obtains raw data into
`workloads/datasets/raw/`. The converter tokenizes text using the local
Llama-3.1 tokenizer, applies caps/truncation, creates reproducible client IDs
and arrivals, and records exact arguments and hashes in manifests. Synthetic
Azure IDs are simulator identities, not reconstructed prompts.

Recommended sequence:

```bash
# 1. Download sources; some Hugging Face sources are gated.
python workloads/datasets/download_datasets.py

# 2. Build seed-7 pilots, run them under RR, and inspect the knee notebook.
RAW="$PWD/workloads/datasets/raw" ./scripts/generate_fairroute_workloads.sh pilot
# notebooks/fairroute_rr_breaking_point.ipynb

# 3. Supply RR knee rates and generate held-out main traces.
export LAM_W1=<rate> LAM_W2=<rate> LAM_W3=<rate> LAM_W4=<session-rate> LAM_W5=<rate>
SEEDS="101 102 103 104 105" ./scripts/generate_fairroute_workloads.sh main

# 4. Optionally generate W1/W3 client-count scaling traces.
SEEDS="101 102 103 104 105" ./scripts/generate_fairroute_workloads.sh scaling
```

The knee notebook selects the last RR rate before p99 queue delay doubles
versus the previous rate or exceeds one second. W4 is session starts/s; the
other workloads use requests/s. Seed 7 is tuning-only.

### Matrix experiment

The matrix runner defaults to `workloads/fairroute/main`, does not generate
workloads (`GENERATE_WORKLOADS=0`), discovers only `trace.jsonl`, and limits
each simulation to `MATRIX_NUM_REQS=1000` requests by default. It writes
policy-specific request CSVs, time series, routing sidecars, and manifests to
`outputs/fairroute/main/`.

```bash
./scripts/run_fairroute_experiments.sh matrix
```

Default global policies:

```text
RR RAND LOAD FAIRROUTE FAIRROUTE_GATED FAIRROUTE_SLACK FAIRROUTE_P D2LPM PRISM
```

Default local policies are `FCFS VTC DLPM`; a complete matrix for `N`
workloads therefore has `N × 9 × 3` simulations. `VALIDATE_MATRIX=1` checks
matrix completeness before analysis. `--detach`, `--status`, `--logs`, and
`--stop` control a matrix run. The client sweep is separate:

```bash
./scripts/run_fairroute_experiments.sh sweep
```

The existing main suite in this checkout has 150 traces: five workloads ×
five load multipliers × six present seeds. Record the workload root used:
older `EXPERIMENTS.md` commands set `MATRIX_WORKLOAD_DIR=workloads/fairroute`,
while the current runner default is the narrower `workloads/fairroute/main`.

## Validation and reproducibility

```bash
./serving/validate.sh
python -m bench validate ...
python scripts/validate_fairroute_matrix.py outputs/fairroute/main
pytest tests/test_routing_formulas.py tests/test_fairroute_p.py
```

The simulator is deterministic for the same configuration, workload, seed,
and flags. Results still depend on the hardware profile bundle, model,
topology, cache tier, and policy parameters. A passing matrix or benchmark is
evidence for that specific deployment/workload combination, not a universal
accuracy guarantee.

## Publications

**ISPASS 2026**

*LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure*

Jaehong Cho*, Hyunmin Choi*, Guseul Heo, Jongse Park (KAIST)
[[Paper]](https://doi.org/10.1109/ISPASS69572.2026.00012)

**CAL 2025**

*LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure*

Jaehong Cho, Hyunmin Choi, Jongse Park (KAIST)
[[Paper]](https://doi.org/10.1109/LCA.2025.3628325)

**IISWC 2024**

*LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale*

Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)
[[Paper]](https://doi.org/10.1109/IISWC63097.2024.00012)

## Citation

```bibtex
@INPROCEEDINGS{11527300,
  author={Cho, Jaehong and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
  booktitle={2026 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  title={{LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure}},
  year={2026}, pages={1-14}, doi={10.1109/ISPASS69572.2026.00012}
}

@ARTICLE{11224567,
  author={Cho, Jaehong and Choi, Hyunmin and Park, Jongse},
  journal={IEEE Computer Architecture Letters},
  title={{LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure}},
  year={2025}, volume={24}, number={02}, pages={361-364},
  doi={10.1109/LCA.2025.3628325}
}

@INPROCEEDINGS{10763697,
  author={Cho, Jaehong and Kim, Minsu and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
  booktitle={2024 IEEE International Symposium on Workload Characterization (IISWC)},
  title={{LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale}},
  year={2024}, pages={1-9}, doi={10.1109/IISWC63097.2024.00012}
}
```
