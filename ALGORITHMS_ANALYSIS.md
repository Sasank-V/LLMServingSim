# Comprehensive Deep Analysis of Routing and Fairness Algorithms

This document presents a rigorous, deep mathematical and algorithmic analysis of all files and functions inside [`serving/core/algorithms/`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/) on the `v2` branch of **LLMServingSim**.

---

## Table of Contents

1. [Architectural Overview & System Context](#1-architectural-overview--system-context)
2. [File-by-File Detailed Analysis](#2-file-by-file-detailed-analysis)
   - 2.1 [`serving/core/algorithms/fairness.py`](#21-servingcorealgorithmsfairnesspy)
     - `FairnessTracker` Class
       - `__init__`
       - `record_service`
       - `get_service`
       - `record_latency`
       - `get_mean_latency`
       - `jains_index_latency`
     - `fairness_target`
     - `fairness_debt`
     - `fairness_urgency`
   - 2.2 [`serving/core/algorithms/locality.py`](#22-servingcorealgorithmslocalitypy)
     - `locality_score`
     - Historical & Extended Locality Formulations (`recompute_tokens`, `recompute_cost`, `locality_cost`, `adaptive_locality_weight`)
   - 2.3 [`serving/core/algorithms/prediction.py`](#23-servingcorealgorithmspredictionpy)
     - `predict_queue`
     - `replica_is_safe`
     - Historical & Extended State Predictions (`ewma_prediction`, `predict_kv_usage`, `predicted_kv_utilization`, `prediction_risk`, `prediction_benefit`)
   - 2.4 [`serving/core/algorithms/utils.py`](#24-servingcorealgorithmsutilspy)
     - `reactive_load`
     - `kv_utilization`
     - Extended Utilities (`normalize`, `normalize_inverse`, `load_benefit`, `kv_headroom`)
   - 2.5 [`serving/core/algorithms/fairroute.py`](#25-servingcorealgorithmsfairroutepy)
     - `iteration_time`
     - `predict_latency_seconds`
     - `fairroute_gate`
     - `fairroute_slack_set`
     - `choose_fairroute_candidate`
     - Deficit-Driven Adaptive Scoring (`score_fairroute_v2` / $H_5$)
     - Multiplicative, AFLC, and Gated Variant Scoring
3. [End-to-End Routing Integration in `router.py`](#3-end-to-end-routing-integration-in-routerpy)
   - Predictor Feature Extraction & Queued Work Calculation
   - Two-Phase TTFT and Decode Latency Estimation
   - Empirical Bayes Output Estimation ($n_0$)
   - Credit Lifecycle & Deficit Balance
   - Deterministic Hash Jitter Tie-Breaking
   - Comparison with Literature Policies (`PRISM`, `D2LPM`, `LOAD`, `RR`)
4. [Master Mathematical Reference Table](#4-master-mathematical-reference-table)

---

## 1. Architectural Overview & System Context

In multi-tenant distributed LLM serving systems, requests arrive continuously from multiple clients with heterogeneous prompt lengths, generation lengths, and Service Level Objectives (SLOs). Replicas maintain independent Key-Value (KV) caches, local queues, and compute engines.

The algorithms inside [`serving/core/algorithms/`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/) provide the decision-making mathematical primitives that drive **FairRoute**, a multi-objective request routing architecture balancing:
1. **Multi-tenant Fairness**: Mitigating starvation, tail latency inflation, and service deficits between competing clients.
2. **Prefix Locality**: Maximizing KV cache reuse by routing prompts to replicas that already store identical prefix blocks.
3. **Execution Predictability & Queue Load**: Projecting queuing delays, TTFT (Time-To-First-Token), and decode latency to avoid replica hotspots and prevent SLO violations.

```
Incoming Request (client_id, input_tokens, prefix_hashes)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   fairness.py                    │
        │   • Compute Target: T            │
        │   • Compute Debt: D_c            │
        │   • Compute Urgency: U_c         │
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   For each replica candidate i:  │
        │   • locality.py: hit ratio h_i   │
        │   • utils.py: load L_i, KV util  │
        │   • prediction.py: safety check  │
        │   • fairroute.py: predicted TTFT │
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   fairroute.py::fairroute_gate   │ ── Feasibility Filter (SLO & Safety)
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ fairroute.py::fairroute_slack_set│ ── Urgency-Adaptive Slack Filter
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ choose_fairroute_candidate       │ ── Lexicographic Selection
        └────────────────┬─────────────────┘
                         │
                         ▼
               Assigned Replica Index
```

---

## 2. File-by-File Detailed Analysis

---

### 2.1 `serving/core/algorithms/fairness.py`

This module tracks delivered client service, records end-to-end request latencies, and computes egalitarian fair-share debt and non-linear control urgency.

#### Class: `FairnessTracker`

`FairnessTracker` maintains running totals for tokens delivered and request turnaround latency per client across the simulation timeline.

##### 1. `__init__(self)`
- **Purpose**: Initializes internal hash maps for service accumulation, cumulative latency, and request completion counts.
- **State Variables**:
  - `self.service`: $\mathcal{S}: \text{client\_id} \to \mathbb{R}_{\ge 0}$ (accumulated token volume).
  - `self.latency`: $\mathcal{L}_{\text{total}}: \text{client\_id} \to \mathbb{R}_{\ge 0}$ (accumulated latency in nanoseconds).
  - `self.latency_count`: $\mathcal{N}: \text{client\_id} \to \mathbb{Z}_{\ge 0}$ (number of completed requests).

##### 2. `record_service(self, client_id, amount: float)`
- **Purpose**: Updates cumulative service delivered to a specified client.
- **Inputs**:
  - `client_id`: Hashable client identifier.
  - `amount`: Token count delivered ($S_{\text{req}} = N_{\text{output}} - N_{\text{input}}$ or total generation tokens).
- **Mathematical Formula**:
  $$S_c^{(t)} = S_c^{(t-1)} + \Delta S$$
  where $S_c$ represents total cumulative service to client $c$.

##### 3. `get_service(self, client_id) -> float`
- **Purpose**: Queries the current cumulative service for client $c$.
- **Mathematical Formula**:
  $$S_c = \begin{cases} \mathcal{S}[c] & \text{if } c \in \text{dom}(\mathcal{S}) \\ 0.0 & \text{otherwise} \end{cases}$$

##### 4. `record_latency(self, client_id, latency_ns)`
- **Purpose**: Records the turnaround latency of a completed request and increments client request count.
- **Inputs**:
  - `client_id`: Hashable client identifier.
  - `latency_ns`: Request completion latency $\tau_{\text{complete}} - \tau_{\text{arrival}}$ in nanoseconds.
- **Mathematical Formula**:
  $$L_c^{\text{total}} \leftarrow L_c^{\text{total}} + \ell_{\text{req}}$$
  $$N_c \leftarrow N_c + 1$$

##### 5. `get_mean_latency(self, client_id) -> float`
- **Purpose**: Computes the sample mean request turnaround latency for client $c$.
- **Inputs**: `client_id`.
- **Output**: Mean latency $\bar{L}_c \in [0, \infty)$ in nanoseconds.
- **Mathematical Formula**:
  $$\bar{L}_c = \begin{cases} 0.0 & \text{if } N_c \le 0 \\ \frac{L_c^{\text{total}}}{N_c} & \text{if } N_c > 0 \end{cases}$$

##### 6. `jains_index_latency(self) -> float`
- **Purpose**: Evaluates Jain's Fairness Index over the set of active clients using their mean request turnaround latencies.
- **Mathematical Formula**:
  Let $\mathcal{C}_{\text{active}} = \{ c \mid N_c > 0 \}$ be the set of active clients, and $n = |\mathcal{C}_{\text{active}}|$.
  If $n < 2$, returns $1.0$.
  Let $x_i = \bar{L}_{c_i}$ for each client $c_i \in \mathcal{C}_{\text{active}}$.
  $$J(x) = \frac{\left( \sum_{i=1}^n x_i \right)^2}{n \cdot \sum_{i=1}^n x_i^2}$$
- **Bounds & Edge Cases**:
  - Theoretical bounds: $J(x) \in \left[\frac{1}{n}, 1.0\right]$.
  - If $\sum_{i=1}^n x_i^2 = 0.0$ (all latencies zero), returns $1.0$.
  - Perfect fairness ($J(x) = 1.0$) occurs if and only if $\bar{L}_1 = \bar{L}_2 = \dots = \bar{L}_n$.
- **Design Rationale**: Prior versions computed Jain's index on request output token counts. Because output lengths are fixed by the workload traces, token-based Jain's index was invariant to routing. Computing Jain's index over *mean client latency* makes it sensitive to routing-induced queuing and head-of-line blocking.

---

#### Function: `fairness_target`

```python
def fairness_target(total_service: float, num_clients: int) -> float:
```

- **Purpose**: Computes the egalitarian fair-share service allocation per client.
- **Inputs**:
  - `total_service`: Aggregate service across all clients, $S_{\text{total}} = \sum_{c} S_c$.
  - `num_clients`: Number of registered clients, $N_{\text{clients}} = |\mathcal{C}|$.
- **Mathematical Formula**:
  $$T = \begin{cases} 0.0 & \text{if } N_{\text{clients}} \le 0 \\ \frac{S_{\text{total}}}{N_{\text{clients}}} & \text{if } N_{\text{clients}} > 0 \end{cases}$$
- **Domain & Range**: $T \in [0, \infty)$.

---

#### Function: `fairness_debt`

```python
def fairness_debt(client_service: float, target_service: float, eps: float = 1e-8) -> float:
```

- **Purpose**: Computes the normalized service deficit (debt) of a client relative to the egalitarian target.
- **Inputs**:
  - `client_service`: Actual cumulative service received by client $c$, $S_c$.
  - `target_service`: Egalitarian fair-share target $T$.
  - `eps`: Numerical regularization factor $\epsilon = 10^{-8}$.
- **Mathematical Formula**:
  $$\text{deficit}_c = \max\left(0.0, \; T - S_c\right)$$
  $$D_c = \frac{\text{deficit}_c}{T + \epsilon} = \frac{\max\left(0.0, \; T - S_c\right)}{T + \epsilon}$$
- **Properties & Sensitivity**:
  - If $S_c \ge T$ (the client has received at least their fair share or is over-served), $D_c = 0.0$.
  - If $S_c = 0$ (the client has received zero service), $D_c = \frac{T}{T + \epsilon} \approx 1.0$.
  - Normalization ensures $D_c \in [0.0, 1.0]$ regardless of simulation run duration.

---

#### Function: `fairness_urgency`

```python
def fairness_urgency(debt: float, lambda_d: float = 2.0) -> float:
```

- **Purpose**: Maps linear service debt into an exponential saturation control signal $U_c \in [0.0, 1.0)$.
- **Inputs**:
  - `debt`: Normalized client debt $D_c \ge 0$.
  - `lambda_d`: Sensitivity decay constant $\lambda_d > 0$ (default: $2.0$).
- **Mathematical Formula**:
  $$U_c(D_c) = 1.0 - \exp\left(-\lambda_d \cdot \max(0.0, \; D_c)\right)$$
- **Derivation & Behavior**:
  - At zero debt ($D_c = 0$): $U_c = 1 - e^0 = 0.0$.
  - At $D_c = 0.5$: $U_c = 1 - e^{-1.0} \approx 0.6321$.
  - At $D_c = 1.0$: $U_c = 1 - e^{-2.0} \approx 0.8647$.
  - As $D_c \to \infty$: $U_c \to 1.0$.
  - First derivative:
    $$\frac{dU_c}{dD_c} = \lambda_d \cdot e^{-\lambda_d D_c} \quad (\text{for } D_c > 0)$$
    The marginal urgency gain is steepest near $D_c = 0$ ($\left.\frac{dU_c}{dD_c}\right|_{0^+} = \lambda_d = 2.0$), allowing rapid response to early service skew, and smoothly saturates as debt grows to avoid unbounded priority monopolization.

---

### 2.2 `serving/core/algorithms/locality.py`

This module quantifies prefix-cache hit reuse for prompt tokens.

#### Function: `locality_score`

```python
def locality_score(prefix_hit_tokens: int, input_tokens: int, eps: float = 1e-8) -> float:
```

- **Purpose**: Computes the normalized fraction of prompt tokens whose KV cache state is already present in a replica's prefix cache.
- **Inputs**:
  - `prefix_hit_tokens`: Number of prompt tokens matching existing Radix tree blocks, $H \ge 0$.
  - `input_tokens`: Total prompt length, $N_{\text{in}}$.
  - `eps`: Numerical regularization factor $\epsilon = 10^{-8}$.
- **Mathematical Formula**:
  $$\text{locality\_score}(H, N_{\text{in}}) = \begin{cases} 0.0 & \text{if } N_{\text{in}} \le 0 \\ \min\left(1.0, \; \max\left(0.0, \; \frac{H}{N_{\text{in}} + \epsilon}\right)\right) & \text{if } N_{\text{in}} > 0 \end{cases}$$
- **Interpretation**:
  - $h = 1.0$: Complete prompt cache hit $\implies$ zero prefill compute required.
  - $h = 0.0$: Complete prompt cache miss $\implies$ full prefill recomputation required.
  - $0 < h < 1.0$: Partial prefix hit; uncached prompt tokens $n_{\text{ext}} = N_{\text{in}} - H$ must be computed during prefill.

#### Extended Locality Formulations (Repository Reference & Companion Functions)

In previous iterations and companion configurations within the repository, additional locality metrics were formulated:

1. **Recompute Tokens**:
   $$n_{\text{ext}} = \max(0, \; N_{\text{in}} - H)$$
2. **Recompute Cost**:
   $$C_{\text{recompute}} = 1.0 - \text{locality\_score}(H, N_{\text{in}})$$
3. **Tier-Aware Locality Cost** (Hierarchical NPU $\to$ CPU $\to$ CXL/Storage):
   $$C_{\text{tier}} = \frac{H_{\text{npu}} \cdot c_{\text{npu}} + H_{\text{cpu}} \cdot c_{\text{cpu}} + H_{\text{storage}} \cdot c_{\text{storage}} + n_{\text{ext}} \cdot c_{\text{recompute}}}{N_{\text{in}} + \epsilon}$$
   Default cost coefficients: $c_{\text{npu}} = 0.0$, $c_{\text{cpu}} = 0.25$, $c_{\text{storage}} = 0.60$, $c_{\text{recompute}} = 1.00$.
4. **Adaptive Locality Weight**:
   $$\beta(D_c) = \beta_0 \cdot \exp\left(-\lambda_d \cdot \max(0.0, D_c)\right)$$
   As a client's debt increases, the priority assigned to locality decays exponentially, forcing the router to prioritize load balance and fairness over cache affinity.

---

### 2.3 `serving/core/algorithms/prediction.py`

This module evaluates replica safety constraints and models future queue lengths under continuous batching.

#### Function: `predict_queue`

```python
def predict_queue(current_queue: float, arrival_rate: float,
                  service_rate: float, horizon: float) -> float:
```

- **Purpose**: Computes a fluid-flow first-order projection of replica queue length over a lookahead time horizon $H$.
- **Inputs**:
  - `current_queue`: Instantaneous waiting queue size, $Q(t)$.
  - `arrival_rate`: Estimated request arrival rate, $\lambda$.
  - `service_rate`: Estimated request service rate, $\mu$.
  - `horizon`: Lookahead duration, $H$.
- **Mathematical Formula**:
  $$\hat{Q}(t + H) = \max\left(0.0, \; Q(t) + \lambda \cdot H - \mu \cdot H\right) = \max\left(0.0, \; Q(t) + (\lambda - \mu) \cdot H\right)$$
- **Properties**:
  - If arrival rate matches service rate ($\lambda = \mu$), queue is stationary: $\hat{Q}(t+H) = Q(t)$.
  - If the replica drains faster than arrivals ($\mu > \lambda$), queue decays to zero: $\hat{Q}(t+H) = \max(0, Q(t) - (\mu-\lambda)H)$.
  - In `router.py`, default lookahead uses $\lambda=0, \mu=0, H=1$, evaluating instantaneous persistence: $\hat{Q} = Q(t)$.

---

#### Function: `replica_is_safe`

```python
def replica_is_safe(predicted_queue: float, max_queue: float,
                    predicted_kv_utilization: float,
                    max_kv_utilization: float = 0.95) -> bool:
```

- **Purpose**: Verifies that a replica has sufficient queue and KV cache memory headroom before allowing candidate selection.
- **Inputs**:
  - `predicted_queue`: Projected queue size, $\hat{Q}$.
  - `max_queue`: Maximum allowable queue capacity, $Q_{\max} = 4 \times \text{capacity}$.
  - `predicted_kv_utilization`: Projected memory utilization, $\hat{U}_{\text{KV}} \in [0, 1]$.
  - `max_kv_utilization`: Safety ceiling threshold, $U_{\text{KV}}^{\max}$ (default: $0.95 = 95\%$).
- **Mathematical Formula**:
  $$\text{replica\_is\_safe}(\hat{Q}, Q_{\max}, \hat{U}_{\text{KV}}, U_{\text{KV}}^{\max}) = \left(\hat{Q} < Q_{\max}\right) \land \left(\hat{U}_{\text{KV}} < U_{\text{KV}}^{\max}\right)$$
- **System Implication**: Prevents catastrophic out-of-memory (OOM) eviction cascades and head-of-line blocking by disqualifying replicas within 5% of KV exhaustion.

---

#### Extended Prediction Formulations (Repository Reference)

1. **EWMA Prediction**:
   $$\hat{X}^{(t)} = \alpha \cdot X^{(t)} + (1 - \alpha) \cdot \hat{X}^{(t-1)}$$
   used for smoothing arrival rates, output token counts, and scheduling intervals ($\alpha \in \{0.05, 0.1, 0.3\}$).
2. **KV Usage Projection**:
   $$\hat{K}(t + H) = \max\left(0.0, \; K(t) + \lambda_{\text{kv}} \cdot H - \mu_{\text{kv}} \cdot H\right)$$
3. **Prediction Risk & Temperature Benefit**:
   $$\text{Risk} = \hat{W} + \kappa_{\sigma} \cdot \sigma_{\text{pred}}$$
   $$\text{Benefit}(\text{Risk}) = \exp\left(-\frac{\text{Risk}}{\tau}\right)$$
   where $\tau$ is a temperature parameter controlling risk sensitivity.

---

### 2.4 `serving/core/algorithms/utils.py`

This module provides normalized live-state pressure and memory utilization metrics.

#### Function: `reactive_load`

```python
def reactive_load(waiting_requests: int, running_requests: int,
                  capacity: int) -> float:
```

- **Purpose**: Calculates the replica's normalized scheduling pressure under continuous batching.
- **Inputs**:
  - `waiting_requests`: Number of queued requests awaiting prefill, $N_{\text{wait}}$.
  - `running_requests`: Number of actively executing sequences in decode/prefill, $N_{\text{run}}$.
  - `capacity`: Maximum sequence capacity limit of the engine, $C$ (`max_num_seqs`).
- **Mathematical Formula**:
  $$\mathcal{L} = \begin{cases} \infty & \text{if } C \le 0 \\ \frac{4 \cdot N_{\text{wait}} + N_{\text{run}}}{C} & \text{if } C > 0 \end{cases}$$
- **Weighting Rationale**:
  - Waiting requests are assigned a $4\times$ weight penalty relative to running requests ($4 \cdot N_{\text{wait}}$).
  - Stalled waiting requests represent full uncached prompts requiring compute-intensive chunked prefill and blocking subsequent sequence allocations. Running requests are already in steady-state decoding, consuming only 1 token generation step per iteration.

---

#### Function: `kv_utilization`

```python
def kv_utilization(kv_used: float, kv_capacity: float, eps: float = 1e-8) -> float:
```

- **Purpose**: Evaluates bounded physical KV cache memory pool occupancy.
- **Inputs**:
  - `kv_used`: Number of allocated memory blocks, $M_{\text{used}}$.
  - `kv_capacity`: Total pool memory blocks, $M_{\text{total}}$.
  - `eps`: Regularization factor $\epsilon = 10^{-8}$.
- **Mathematical Formula**:
  $$U_{\text{KV}} = \begin{cases} 1.0 & \text{if } M_{\text{total}} \le 0 \\ \min\left(1.0, \; \max\left(0.0, \; \frac{M_{\text{used}}}{M_{\text{total}} + \epsilon}\right)\right) & \text{if } M_{\text{total}} > 0 \end{cases}$$
- **Output**: Bounded utilization $U_{\text{KV}} \in [0.0, 1.0]$.
- **Companion Headroom Metric**:
  $$\text{KV}_{\text{headroom}} = \max\left(0.0, \; 1.0 - U_{\text{KV}}\right)$$

---

### 2.5 `serving/core/algorithms/fairroute.py`

This module defines kernel iteration models, latency predictors, safety gating, slack sets, and candidate scoring functions.

#### Function: `iteration_time`

```python
def iteration_time(
    batch_size: float,
    sequence_count: float,
    *,
    tau0: float = 0.001,
    w0: float = 0.001,
    ws: float = 0.00001,
    p_max: float = 2048.0,
    k_batch: float = 0.05,
    k_sequence: float = 0.05,
    tau_batch: float = 0.00001,
    tau_sequence: float = 0.00001,
) -> float:
```

- **Purpose**: Evaluates an analytical 2D non-linear execution time model for a single simulation step/iteration under continuous batching. Derived from the NexusSched iteration model.
- **Inputs**:
  - `batch_size`: Number of tokens processed in the batch iteration, $B \ge 0$.
  - `sequence_count`: Number of concurrent sequences active in the iteration, $S \ge 0$.
- **Model Parameters (Defaults)**:
  - $\tau_0 = 10^{-3}$ s (1 ms): Base iteration launch and synchronization overhead.
  - $w_0 = 10^{-3}$ s, $w_s = 10^{-5}$ s: Sequence work coefficients.
  - $p_{\max} = 2048.0$: Asymptotic throughput saturation scale (tokens/sec).
  - $k_{\text{batch}} = 0.05$, $k_{\text{sequence}} = 0.05$: Exponential curvature rates.
  - $\tau_{\text{batch}} = 10^{-5}$ s, $\tau_{\text{sequence}} = 10^{-5}$ s: Linear per-token and per-sequence memory traversal overheads.
- **Mathematical Formula**:
  $$B = \max(0.0, \text{batch\_size}), \quad S = \max(0.0, \text{sequence\_count})$$
  $$P(B, S) = p_{\max} \cdot \left(1.0 - \exp\left(-k_{\text{batch}} \cdot B\right)\right) \cdot \left(1.0 - \exp\left(-k_{\text{sequence}} \cdot S\right)\right)$$
  $$T_{\text{iter}}(B, S) = \begin{cases} \tau_0 + \tau_{\text{batch}} \cdot B + \tau_{\text{sequence}} \cdot S & \text{if } P(B, S) \le 0.0 \\ \tau_0 + \frac{w_0 + w_s \cdot S}{P(B, S)} + \tau_{\text{batch}} \cdot B + \tau_{\text{sequence}} \cdot S & \text{if } P(B, S) > 0.0 \end{cases}$$
- **Asymptotic & Physical Properties**:
  - As $B \to \infty$ and $S \to \infty$, $P(B, S) \to p_{\max}$.
  - The middle term $\frac{w_0 + w_s S}{P(B,S)}$ captures GPU Tensor Core compute saturation where incremental tokens achieve higher throughput until memory bandwidth or compute peak is hit.
  - The trailing linear terms $\tau_{\text{batch}} B + \tau_{\text{sequence}} S$ represent KV cache memory bus loads and Attention context scans.

---

#### Function: `predict_latency_seconds`

```python
def predict_latency_seconds(load, extend_tokens, decode_tokens):
```

- **Purpose**: Fast analytical estimation of total request processing latency from replica load and token workloads.
- **Inputs**:
  - `load`: Normalized replica load $\mathcal{L}$.
  - `extend_tokens`: Uncached prompt tokens requiring prefill computation, $n_{\text{ext}} = N_{\text{in}} - H$.
  - `decode_tokens`: Expected number of generated tokens, $\hat{Y}$.
- **Mathematical Formula**:
  $$\tilde{\mathcal{L}} = \min\left(1.0, \; \max(0.0, \; \text{load})\right)$$
  $$W_{\text{ext}} = \frac{\max(0.0, \; \text{extend\_tokens})}{2048.0}$$
  $$W_{\text{dec}} = \frac{\max(0.0, \; \text{decode\_tokens})}{512.0}$$
  $$\Theta(\tilde{\mathcal{L}}) = \max\left(0.05, \; 1.0 - 0.65 \cdot \tilde{\mathcal{L}}\right)$$
  $$\hat{T}_{\text{lat}}(\tilde{\mathcal{L}}, W_{\text{ext}}, W_{\text{dec}}) = \frac{0.01 + W_{\text{ext}} + W_{\text{dec}}}{\Theta(\tilde{\mathcal{L}})}$$
- **Analysis**:
  - Normalization scales 2048 prefill tokens and 512 decode tokens to unit work ($1.0$).
  - Effective throughput $\Theta(\tilde{\mathcal{L}})$ drops linearly from $1.0$ at zero load to $0.35$ at full load ($\tilde{\mathcal{L}} = 1.0$), reflecting interference and scheduling contention.
  - At zero load ($\tilde{\mathcal{L}} = 0$): $\hat{T}_{\text{lat}} = 0.01 + W_{\text{ext}} + W_{\text{dec}}$.
  - The lower bound $\Theta \ge 0.05$ caps the maximum penalty factor at $20\times$.

---

#### Function: `fairroute_gate`

```python
def fairroute_gate(candidates, ttft_slo, gamma=1.0):
```

- **Purpose**: Applies a hard feasibility filter based on memory safety and predicted TTFT SLO compliance.
- **Inputs**:
  - `candidates`: List of candidate replica dictionaries, each with fields `"safe"` (bool) and `"predicted_latency"` ($\hat{T}_i$).
  - `ttft_slo`: Time-To-First-Token SLO threshold in seconds.
  - `gamma`: SLO headroom multiplier $\gamma > 0$ (default: $1.0$).
- **Mathematical Formula**:
  Let $\mathcal{C}$ be the set of all candidate replicas.
  $$\mathcal{C}_{\text{feasible}} = \left\{ c \in \mathcal{C} \;\middle|\; c[\text{"safe"}] = \text{True} \;\land\; c[\text{"predicted\_latency"}] \le \gamma \cdot \text{ttft\_slo} \right\}$$
  $$\mathcal{C}_{\text{safe}} = \left\{ c \in \mathcal{C} \;\middle|\; c[\text{"safe"}] = \text{True} \right\}$$
  $$\text{fairroute\_gate}(\mathcal{C}, \text{SLO}, \gamma) = \begin{cases} (\mathcal{C}_{\text{feasible}}, \; \text{slo\_risk}=\text{False}) & \text{if } \mathcal{C}_{\text{feasible}} \neq \emptyset \\ (\mathcal{C}_{\text{safe}}, \; \text{slo\_risk}=\text{True}) & \text{if } \mathcal{C}_{\text{feasible}} = \emptyset \land \mathcal{C}_{\text{safe}} \neq \emptyset \\ (\mathcal{C}, \; \text{slo\_risk}=\text{True}) & \text{if } \mathcal{C}_{\text{feasible}} = \emptyset \land \mathcal{C}_{\text{safe}} = \emptyset \end{cases}$$
- **Fallback Philosophy**: If no replica can satisfy the SLO target, the gate relaxes the latency requirement and falls back to memory-safe candidates, flagging `slo_risk = True` so downstream scoring can apply penalty offsets.

---

#### Function: `fairroute_slack_set`

```python
def fairroute_slack_set(candidates, rho=0.25, eps_max=0.05, urgency=0.0):
```

- **Purpose**: Identifies the subset of candidates whose predicted latency is within an urgency-modulated margin of the fastest replica.
- **Inputs**:
  - `candidates`: Feasible candidate set $\mathcal{C}_{\text{feasible}}$.
  - `rho`: Fractional slack coefficient $\rho \in (0, 1)$ (default: $0.25 = 25\%$).
  - `eps_max`: Maximum absolute slack ceiling $\epsilon_{\max}$ (default: $0.05$ s = 50 ms).
  - `urgency`: Client fairness urgency $U_c \in [0.0, 1.0]$.
- **Mathematical Formula**:
  If $\mathcal{C} = \emptyset$, returns $(\emptyset, 0.0)$.
  Let $T^* = \min_{c \in \mathcal{C}} c[\text{"predicted\_latency"}]$.
  $$u = \min\left(1.0, \; \max(0.0, \; \text{urgency})\right)$$
  $$\epsilon = \min\left(\epsilon_{\max}, \; \rho \cdot T^*\right) \cdot (1.0 - u)$$
  $$\mathcal{C}_{\text{slack}} = \left\{ c \in \mathcal{C} \;\middle|\; c[\text{"predicted\_latency"}] \le T^* + \epsilon \right\}$$
- **Control Theory Interpretation**:
  - **Case $U_c \to 1.0$ (Starved/High-Debt Client)**:
    $$\epsilon \to 0.0 \implies \mathcal{C}_{\text{slack}} = \left\{ c \in \mathcal{C} \mid c[\text{"predicted\_latency"}] = T^* \right\}$$
    The slack collapses completely. The system eliminates all routing flexibility and strictly routes the request to the fastest replica to immediately minimize client delay.
  - **Case $U_c \to 0.0$ (Well-Served Client)**:
    $$\epsilon = \min(\epsilon_{\max}, \; \rho \cdot T^*)$$
    Slack expands up to $25\%$ of the optimal latency (capped at 50 ms). Any replica within this margin is admitted into $\mathcal{C}_{\text{slack}}$, allowing prefix locality and deficit equalization to guide routing without penalizing the client beyond the bounded tolerance $\epsilon$.

---

#### Function: `choose_fairroute_candidate`

```python
def choose_fairroute_candidate(candidates, mode="SLACK_LEXICOGRAPHIC",
                               urgency=0.0, rho=0.25, eps_max=0.05):
```

- **Purpose**: Final candidate selection dispatch supporting multiple mathematical decision policies.
- **Modes & Formulations**:

##### 1. Mode: `"GATED"`
Greedy latency minimization over the gated set:
$$c^* = \arg\min_{c \in \mathcal{C}} c[\text{"predicted\_latency"}], \quad \epsilon = 0.0$$

##### 2. Mode: `"MULTIPLICATIVE"`
Multiplicative balance between latency, prefix locality, and replica deficit:
Let $T^* = \min_{c \in \mathcal{C}} c[\text{"predicted\_latency"}]$.
$$c^* = \arg\min_{c \in \mathcal{C}} \left[ \frac{c[\text{"predicted\_latency"}]}{\max(10^{-8}, T^*)} \cdot \frac{1.0 + c[\text{"locality"}]}{\max(10^{-8}, \; 1.0 + c[\text{"replica\_deficit"}])} \right], \quad \epsilon = 0.0$$

##### 3. Mode: `"LINEAR"`
Additive weighted scoring:
$$c^* = \arg\max_{c \in \mathcal{C}} \left[ -c[\text{"predicted\_latency"}] + 0.02 \cdot c[\text{"locality"}] + 0.01 \cdot c[\text{"replica\_deficit"}] \right], \quad \epsilon = 0.0$$

##### 4. Mode: `"LEARNED"`
Linear combination incorporating client token credit:
$$c^* = \arg\max_{c \in \mathcal{C}} \left[ -c[\text{"predicted\_latency"}] + 0.01 \cdot c[\text{"locality"}] + 0.01 \cdot c[\text{"replica\_deficit"}] + 0.01 \cdot c[\text{"fairness\_credit"}] \right], \quad \epsilon = 0.0$$

##### 5. Mode: `"SLACK_LEXICOGRAPHIC"` / `"V2"` (Default FairRoute Policy)
Combines slack filtering with hierarchical lexicographic tie-breaking:
1. Filter candidates by urgency-adaptive slack:
   $$(\mathcal{C}_{\text{slack}}, \epsilon) = \text{fairroute\_slack\_set}(\mathcal{C}, \rho, \epsilon_{\max}, u)$$
2. Filter for locality eligibility:
   $$\mathcal{C}_{\text{eligible}} = \left\{ c \in \mathcal{C}_{\text{slack}} \;\middle|\; c[\text{"hit"}] > 0 \;\land\; c[\text{"fairness\_credit"}] > 0 \right\}$$
3. Determine pool:
   $$\text{Pool} = \begin{cases} \mathcal{C}_{\text{eligible}} & \text{if } \mathcal{C}_{\text{eligible}} \neq \emptyset \\ \mathcal{C}_{\text{slack}} & \text{if } \mathcal{C}_{\text{eligible}} = \emptyset \end{cases}$$
4. Evaluate multi-dimensional lexicographic tuple (highest priority first):
   $$c^* = \arg\max_{c \in \text{Pool}} \Big( c[\text{"locality"}], \; c[\text{"future\_locality"}], \; c[\text{"replica\_deficit"}], \; -c[\text{"predicted\_latency"}] \Big)$$
   where comparisons proceed left-to-right:
   - Primary: Maximum prefix cache hit ratio $h$.
   - Secondary: Maximum future locality score.
   - Tertiary: Maximum replica deficit (routing to least-utilized replica to balance load).
   - Quaternary: Minimum predicted latency (negated for maximization).

---

#### Deficit-Driven Adaptive Scoring: FairRoute-v2 ($H_5$)

Documented in [`README.md`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/README.md) and earlier commit versions:

```python
def score_fairroute_v2(
    debt: float,
    urgency: float,
    locality: float,
    prediction: float,
    normalized_load_benefit: float,
    w_f_base: float = 1.0,
    w_l: float = 1.0,
    w_p: float = 1.0,
) -> float:
```

- **Mathematical Formula**:
  $$w_f(D_c) = w_{f, \text{base}} + 2.0 \cdot \max(0.0, \; D_c)$$
  $$\text{FairnessBenefit}_i = U_c \cdot \text{normalized\_load\_benefit}_i$$
  $$\text{Score}_i = w_f(D_c) \cdot \text{FairnessBenefit}_i + w_l \cdot \text{locality}_i + w_p \cdot \text{prediction}_i$$
- **Candidate-Varying Mechanism**:
  $$\text{FairnessBenefit}_i = U_c \cdot (1.0 - \mathcal{L}_i)$$
  Because load headroom $(1 - \mathcal{L}_i)$ varies across candidate replicas, the fairness signal actively directs underserved clients (high $U_c$) toward the least-loaded replicas, preventing candidate-invariant score collapse.

---

#### Adaptive Fairness-Locality Control (AFLC / $H_3$)

Documented in earlier iterations of `fairroute.py`:
$$\beta(D_c) = \beta_0 \cdot \exp\left(-\lambda_d \cdot \max(0.0, D_c)\right)$$
$$\text{Score}_{\text{AFLC}} = \left(1.0 + w_f \cdot U_c\right) \cdot (\epsilon + L)^{\beta(D_c)} \cdot (\epsilon + P)^\gamma$$

---

## 3. End-to-End Routing Integration in `router.py`

To understand how the mathematical functions in `serving/core/algorithms/` execute in practice, this section documents their invocation in [`serving/core/router.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/router.py).

### 3.1 Predictor Terms & Work Modeling (`_predictor_terms`)

For each candidate replica $i$, `Router._predictor_terms` gathers system states and calculates:

1. **Queued Work ($W_i$)**:
   $$W_i = \sum_{r \in \text{waiting}} \text{tokens}(r) + \sum_{r \in \text{running}} \max\left(0, \; \text{tokens}(r) - \text{computed\_tokens}(r)\right)$$
2. **Uncached Extension Tokens ($n_{\text{ext}}$)**:
   $$n_{\text{ext}} = \max\left(0.0, \; N_{\text{in}} - H_i\right)$$
3. **Active Sequences ($B_i$) and Active Decoders ($D_i$)**:
   $$B_i = |\text{running}_i| + |\text{waiting}_i|$$
   $$D_i = \sum_{r \in \text{running}_i} \mathbb{I}\left(\text{tokens}(r) \le \text{computed\_tokens}(r) + 1\right)$$
4. **Engine Capacity Headroom**:
   $$S_{\max} = \text{max\_num\_batched\_tokens}$$
   $$\text{Capacity}_i = \max\left(1, \; S_{\max} - D_i\right)$$
5. **Time-To-First-Token (TTFT) Prediction**:
   $$T_{\text{base}} = \text{iteration\_time}(B_i + 1, S_{\max}, p_{\max}=S_{\max})$$
   $$\text{TTFT}_i = \left\lceil \frac{W_i + n_{\text{ext}}}{\text{Capacity}_i} \right\rceil \cdot T_{\text{base}}$$
6. **Decode Latency Prediction**:
   $$\text{Decode}_i = \hat{Y}_c \cdot \text{iteration\_time}(D_i + 1, D_i + 1, p_{\max}=S_{\max})$$
7. **Total Projected Latency**:
   $$\hat{T}_{\text{lat}, i} = \text{TTFT}_i + \text{Decode}_i$$
8. **Target TTFT SLO**:
   $$\text{SLO} = k_{\text{SLO}} \cdot \left\lceil \frac{N_{\text{in}}}{S_{\max}} \right\rceil \cdot \text{iteration\_time}(1, S_{\max}, p_{\max}=S_{\max})$$
   where $k_{\text{SLO}} = 10.0$ (`fairroute_k`).

---

### 3.2 Empirical Bayes Output Token Estimation ($\hat{Y}_c$)

Request generation lengths are unknown at arrival. `Router` estimates client generation length using an Empirical Bayes shrinkage estimator with prior $n_0$:

1. Per-client online EWMA:
   $$\bar{Y}_c^{(t)} = 0.1 \cdot Y_{c, \text{observed}} + 0.9 \cdot \bar{Y}_c^{(t-1)}$$
2. Global median output across all clients:
   $$\bar{Y}_{\text{global}} = \text{median}(\{ Y_{\text{samples}} \})$$
3. Bayes shrinkage estimate:
   $$\hat{Y}_c = \begin{cases} \bar{Y}_c & \text{if } n_0 = 0 \\ \frac{\bar{Y}_c + n_0 \cdot \bar{Y}_{\text{global}}}{1.0 + n_0} & \text{if } n_0 > 0 \end{cases}$$
   For clients with sparse history, $\hat{Y}_c$ shrinks toward the population median $\bar{Y}_{\text{global}}$, preventing outlier predictions.

---

### 3.3 Client Delay EWMA & Median Threshold Urgency

When enabled via `fairroute_n0 > 0`:
1. Client turnaround delay EWMA:
   $$\bar{\delta}_c^{(t)} = 0.05 \cdot \delta_c + 0.95 \cdot \bar{\delta}_c^{(t-1)}$$
2. Median delay across registered clients:
   $$\delta_{\text{median}} = \text{median}\left(\left\{ \bar{\delta}_{c'} \mid c' \in \mathcal{C}_{\text{known}} \right\}\right)$$
3. Relative delay inflation urgency:
   $$U_c = \begin{cases} \min\left(1.0, \; \frac{\bar{\delta}_c}{\delta_{\text{median}}} - 1.0\right) & \text{if } \delta_{\text{median}} > 0 \land \frac{\bar{\delta}_c}{\delta_{\text{median}}} > 1.0 \\ 0.0 & \text{otherwise} \end{cases}$$

---

### 3.4 Multi-Tenant Token Credit Quantum & Updates

To prevent free-riding, clients consume token credits when routing to replicas:

1. **Credit Quantum**:
   $$Q_{\text{credit}} = \kappa \cdot \left( w_e \cdot M_{\text{in}}^{\max} + w_q \cdot M_{\text{out}}^{\max} \right)$$
   where $M_{\text{in}}^{\max} = 2048$, $M_{\text{out}}^{\max} = 1024$, $w_e = 1.0$, $w_q = 2.0$, $\kappa = 1.0$ ($Q_{\text{credit}} = 4096$).
2. **Debit on Route**:
   $$\text{Credit}_{(c, i)} \leftarrow \text{Credit}_{(c, i)} - w_e \cdot n_{\text{ext}}$$
3. **Credit Refill**:
   When all replicas for client $c$ reach $\text{Credit} \le 0$:
   $$\forall i: \quad \text{Credit}_{(c, i)} \leftarrow \text{Credit}_{(c, i)} + Q_{\text{credit}}$$

---

### 3.5 Deterministic Hash Jitter Tie-Breaking (`_tiebreak_jitter`)

When candidate scores tie, deterministic hash jitter breaks ties without systematic replica bias:
$$h = \text{MD5}(r_{\text{id}} : i_{\text{candidate}})$$
$$\text{Jitter}(r_{\text{id}}, i_{\text{candidate}}) = \frac{\text{int32\_little\_endian}(h[0:4])}{2^{32} - 1} \in [0.0, 1.0]$$
This replaces legacy `-candidate["index"]` which previously hot-spotted 95%+ of tied decisions onto Instance 0.

---

### 3.6 Comparison with Literature Policies in `Router`

#### PRISM (`_select_prism`)
Computes composite geometric score across latency, load, KV capacity, and prefix affinity:
$$S_{\text{lat}} = \frac{1.0}{1.0 + \left( \frac{\hat{T}_{\text{lat}, i}}{\text{SLO}} \right)^p}$$
$$S_{\text{load}} = \frac{1.0}{1.0 + \left( \frac{\rho_i}{\text{SLO}} \right)^p}, \quad \text{where } \rho_i = \frac{W_i}{\max(1, S_{\max})}$$
$$S_{\text{cap}} = \min\left(1.0, \; \max\left(0.0, \; \frac{\text{FreeKV}_i - \text{KV}_{\text{req}}}{\max(\text{KV}_{\text{req}}, 10^{-9})}\right)\right)$$
$$\text{Affinity}_i = \begin{cases} \beta & \text{if } H_i \ge h_{\min} \\ 1.0 & \text{otherwise} \end{cases}$$
$$\text{Score}_{\text{PRISM}} = S_{\text{lat}} \cdot S_{\text{load}} \cdot S_{\text{cap}} \cdot \text{Affinity}_i$$
Default parameters: $p = 4.0$, $\beta = 2.0$, $h_{\min} = 16$.

#### D2LPM (`_select_d2lpm`)
Deficit-driven locality and predictive matching:
1. Identify max hit group:
   $$\mathcal{C}_{\text{max\_hit}} = \left\{ c \;\middle|\; c[\text{"hit"}] = \max_j c_j[\text{"hit"}] \ge h_{\min} \right\}$$
2. Filter by credit availability:
   $$\mathcal{C}_{\text{active}} = \{ c \mid c[\text{"credit"}] > 0 \}$$
   $$\text{Pool} = (\mathcal{C}_{\text{max\_hit}} \cap \mathcal{C}_{\text{active}}) \text{ or } \mathcal{C}_{\text{active}} \text{ or } \mathcal{C}$$
3. Select instance minimizing backlog:
   $$c^* = \arg\min_{c \in \text{Pool}} \Big( c[\text{"requests"}], \; c[\text{"index"}] \Big)$$

#### Reactive Load (`_least_load_select`)
$$c^* = \arg\min_{i} \frac{4 \cdot N_{\text{wait}, i} + N_{\text{run}, i}}{C_i}$$

---

## 4. Master Mathematical Reference Table

| Module | Function / Class | Core Mathematical Formula | Domain / Bounds |
| :--- | :--- | :--- | :--- |
| **`fairness.py`** | `FairnessTracker.record_service` | $S_c \leftarrow S_c + \Delta S$ | $S_c \ge 0$ |
| | `FairnessTracker.record_latency` | $L_c \leftarrow L_c + \ell, \quad N_c \leftarrow N_c + 1$ | $L_c \ge 0, N_c \in \mathbb{N}$ |
| | `FairnessTracker.get_mean_latency` | $\bar{L}_c = L_c / N_c$ | $\bar{L}_c \ge 0$ |
| | `FairnessTracker.jains_index_latency` | $J(x) = \frac{(\sum x_i)^2}{n \sum x_i^2}$ | $J \in [1/n, 1.0]$ |
| | `fairness_target` | $T = S_{\text{total}} / N_{\text{clients}}$ | $T \ge 0$ |
| | `fairness_debt` | $D_c = \frac{\max(0, T - S_c)}{T + \epsilon}$ | $D_c \in [0.0, 1.0]$ |
| | `fairness_urgency` | $U_c = 1 - e^{-\lambda_d \cdot \max(0, D_c)}$ | $U_c \in [0.0, 1.0)$ |
| **`locality.py`** | `locality_score` | $h = \text{clip}\left(\frac{H}{N_{\text{in}} + \epsilon}, 0, 1\right)$ | $h \in [0.0, 1.0]$ |
| **`prediction.py`** | `predict_queue` | $\hat{Q} = \max(0, Q + (\lambda - \mu)H)$ | $\hat{Q} \ge 0$ |
| | `replica_is_safe` | $(\hat{Q} < Q_{\max}) \land (\hat{U}_{\text{KV}} < U_{\text{KV}}^{\max})$ | Boolean |
| **`utils.py`** | `reactive_load` | $\mathcal{L} = \frac{4 N_{\text{wait}} + N_{\text{run}}}{C}$ | $\mathcal{L} \ge 0$ |
| | `kv_utilization` | $U_{\text{KV}} = \text{clip}\left(\frac{M_{\text{used}}}{M_{\text{total}} + \epsilon}, 0, 1\right)$ | $U_{\text{KV}} \in [0.0, 1.0]$ |
| **`fairroute.py`** | `iteration_time` | $T_{\text{iter}} = \tau_0 + \frac{w_0 + w_s S}{p_{\max}(1 - e^{-k_b B})(1 - e^{-k_s S})} + \tau_b B + \tau_s S$ | $T_{\text{iter}} > 0$ (seconds) |
| | `predict_latency_seconds` | $\hat{T}_{\text{lat}} = \frac{0.01 + n_{\text{ext}}/2048 + \hat{Y}/512}{\max(0.05, 1.0 - 0.65 \mathcal{L})}$ | $\hat{T}_{\text{lat}} > 0$ (seconds) |
| | `fairroute_gate` | $\mathcal{C}_{\text{feasible}} = \{ c \mid c.\text{safe} \land \hat{T}_c \le \gamma \cdot \text{SLO} \}$ | Candidate subset |
| | `fairroute_slack_set` | $\mathcal{C}_{\text{slack}} = \{ c \mid \hat{T}_c \le T^* + \min(\epsilon_{\max}, \rho T^*)(1 - U_c) \}$ | Candidate subset |
| | `choose_fairroute_candidate` | $c^* = \arg\max_{c \in \text{Pool}} (h_c, h_{c, \text{future}}, \text{deficit}_c, -\hat{T}_c)$ | Selected replica index |
| | `score_fairroute_v2` | $\text{Score} = (w_{f, \text{base}} + 2 D_c) \cdot U_c (1 - \mathcal{L}_i) + w_l h_i + w_p P_i$ | Real scalar score |

---
*Generated after deep analysis of the `v2` branch of LLMServingSim.*
