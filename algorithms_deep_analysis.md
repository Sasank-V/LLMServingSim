# Deep Mathematical Analysis: `serving/core/algorithms/` (v2 Branch)

This document provides a complete, mathematically rigorous breakdown of every function and class inside [`serving/core/algorithms/`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/) on the `v2` branch of **LLMServingSim**, along with its system integration in [`serving/core/router.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/router.py).

The generated repository file is also saved at [`serving/core/algorithms/ALGORITHMS_ANALYSIS.md`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/ALGORITHMS_ANALYSIS.md).

---

## 1. Summary of Files

| File | Purpose | Key Functions / Classes |
| :--- | :--- | :--- |
| [`fairness.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/fairness.py) | Client service tracking, turnaround latency profiling, egalitarian target & debt calculation, urgency function, and latency-based Jain's Index | `FairnessTracker`, `fairness_target`, `fairness_debt`, `fairness_urgency` |
| [`locality.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/locality.py) | Prompt prefix-cache hit ratio and KV reuse quantification | `locality_score` *(companion: `recompute_cost`, `locality_cost`, `adaptive_locality_weight`)* |
| [`prediction.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/prediction.py) | Fluid-flow queue length projection and dual-constraint replica safety | `predict_queue`, `replica_is_safe` *(companion: `ewma_prediction`, `predict_kv_usage`, `prediction_risk`)* |
| [`utils.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/utils.py) | Continuous-batching reactive pressure scoring and bounded physical KV cache pool utilization | `reactive_load`, `kv_utilization` *(companion: `normalize`, `load_benefit`, `kv_headroom`)* |
| [`fairroute.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/fairroute.py) | Analytical GPU iteration model (NexusSched), latency predictor, feasibility gate, adaptive slack set filter, and candidate scoring dispatch | `iteration_time`, `predict_latency_seconds`, `fairroute_gate`, `fairroute_slack_set`, `choose_fairroute_candidate`, `score_fairroute_v2` |

---

## 2. File-by-File Functions & Exact Mathematical Formulas

### 2.1 `fairness.py`

#### 1. `FairnessTracker.record_service(client_id, amount: float)`
Accumulates total service tokens delivered to client $c$:
$$S_c \leftarrow S_c + \Delta S$$

#### 2. `FairnessTracker.get_service(client_id) -> float`
Retrieves cumulative service $S_c$ (defaults to $0.0$ if unseen).

#### 3. `FairnessTracker.record_latency(client_id, latency_ns)`
Accumulates total turnaround latency and request completions:
$$L_c^{\text{total}} \leftarrow L_c^{\text{total}} + \ell, \quad N_c \leftarrow N_c + 1$$

#### 4. `FairnessTracker.get_mean_latency(client_id) -> float`
Computes sample mean turnaround latency in nanoseconds:
$$\bar{L}_c = \begin{cases} 0.0 & \text{if } N_c \le 0 \\ \frac{L_c^{\text{total}}}{N_c} & \text{if } N_c > 0 \end{cases}$$

#### 5. `FairnessTracker.jains_index_latency() -> float`
Computes Jain's Fairness Index across all active clients ($N_c > 0$) using mean client latencies:
Let $\mathcal{C} = \{ c \mid N_c > 0 \}$ and $n = |\mathcal{C}|$. If $n < 2$, returns $1.0$.
$$J(\vec{x}) = \frac{\left( \sum_{i=1}^n x_i \right)^2}{n \cdot \sum_{i=1}^n x_i^2}, \quad \text{where } x_i = \bar{L}_{c_i}$$
Bounds: $J \in [1/n, 1.0]$. If denominator is $0.0$, returns $1.0$.

#### 6. `fairness_target(total_service: float, num_clients: int) -> float`
Computes egalitarian fair-share target service:
$$T = \begin{cases} 0.0 & \text{if } N_{\text{clients}} \le 0 \\ \frac{S_{\text{total}}}{N_{\text{clients}}} & \text{if } N_{\text{clients}} > 0 \end{cases}$$

#### 7. `fairness_debt(client_service: float, target_service: float, eps: float = 1e-8) -> float`
Normalized service deficit relative to fair-share target:
$$\Delta_c = \max(0.0, \; T - S_c)$$
$$D_c = \frac{\Delta_c}{T + \epsilon} = \frac{\max(0.0, \; T - S_c)}{T + \epsilon}$$
Bounds: $D_c \in [0.0, 1.0]$. When $S_c \ge T$, $D_c = 0.0$.

#### 8. `fairness_urgency(debt: float, lambda_d: float = 2.0) -> float`
Maps linear debt into exponential saturation control signal $U_c \in [0.0, 1.0)$:
$$U_c(D_c) = 1.0 - \exp\left(-\lambda_d \cdot \max(0.0, \; D_c)\right)$$
Default: $\lambda_d = 2.0$. First derivative $\frac{dU_c}{dD_c} = \lambda_d e^{-\lambda_d D_c}$.

---

### 2.2 `locality.py`

#### `locality_score(prefix_hit_tokens: int, input_tokens: int, eps: float = 1e-8) -> float`
Evaluates normalized prefix-cache reuse ratio:
$$h(H, N_{\text{in}}) = \begin{cases} 0.0 & \text{if } N_{\text{in}} \le 0 \\ \min\left(1.0, \; \max\left(0.0, \; \frac{H}{N_{\text{in}} + \epsilon}\right)\right) & \text{if } N_{\text{in}} > 0 \end{cases}$$
Uncached extension prompt tokens required during prefill:
$$n_{\text{ext}} = \max(0, \; N_{\text{in}} - H)$$

---

### 2.3 `prediction.py`

#### 1. `predict_queue(current_queue: float, arrival_rate: float, service_rate: float, horizon: float) -> float`
Fluid-flow queue projection over lookahead horizon $H$:
$$\hat{Q}(t + H) = \max\left(0.0, \; Q(t) + (\lambda - \mu) \cdot H\right)$$

#### 2. `replica_is_safe(predicted_queue: float, max_queue: float, predicted_kv_utilization: float, max_kv_utilization: float = 0.95) -> bool`
Dual-constraint safety predicate:
$$\text{replica\_is\_safe} = \left(\hat{Q} < Q_{\max}\right) \land \left(\hat{U}_{\text{KV}} < U_{\text{KV}}^{\max}\right)$$
Default: $U_{\text{KV}}^{\max} = 0.95$ ($95\%$ memory occupancy ceiling).

---

### 2.4 `utils.py`

#### 1. `reactive_load(waiting_requests: int, running_requests: int, capacity: int) -> float`
Continuous-batching pressure metric:
$$\mathcal{L} = \begin{cases} \infty & \text{if } C \le 0 \\ \frac{4 \cdot N_{\text{wait}} + N_{\text{run}}}{C} & \text{if } C > 0 \end{cases}$$
$4\times$ weight penalizes waiting requests due to uncached prefill compute & memory allocation pressure.

#### 2. `kv_utilization(kv_used: float, kv_capacity: float, eps: float = 1e-8) -> float`
Bounded physical KV cache pool utilization:
$$U_{\text{KV}} = \begin{cases} 1.0 & \text{if } M_{\text{total}} \le 0 \\ \min\left(1.0, \; \max\left(0.0, \; \frac{M_{\text{used}}}{M_{\text{total}} + \epsilon}\right)\right) & \text{if } M_{\text{total}} > 0 \end{cases}$$

---

### 2.5 `fairroute.py`

#### 1. `iteration_time(batch_size, sequence_count, ...)`
Evaluates the NexusSched saturating 2D execution time model for one simulation step:
$$B = \max(0.0, \text{batch\_size}), \quad S = \max(0.0, \text{sequence\_count})$$
$$P(B, S) = p_{\max} \cdot \left(1.0 - e^{-k_{\text{batch}} \cdot B}\right) \cdot \left(1.0 - e^{-k_{\text{sequence}} \cdot S}\right)$$
$$T_{\text{iter}}(B, S) = \begin{cases} \tau_0 + \tau_{\text{batch}} B + \tau_{\text{sequence}} S & \text{if } P(B, S) \le 0.0 \\ \tau_0 + \frac{w_0 + w_s S}{P(B, S)} + \tau_{\text{batch}} B + \tau_{\text{sequence}} S & \text{if } P(B, S) > 0.0 \end{cases}$$
Defaults: $\tau_0 = 10^{-3}$, $w_0 = 10^{-3}$, $w_s = 10^{-5}$, $p_{\max} = 2048.0$, $k_{\text{batch}} = 0.05$, $k_{\text{sequence}} = 0.05$, $\tau_{\text{batch}} = 10^{-5}$, $\tau_{\text{sequence}} = 10^{-5}$.

#### 2. `predict_latency_seconds(load, extend_tokens, decode_tokens)`
Fast analytical request execution latency prediction:
$$\tilde{\mathcal{L}} = \text{clip}(\text{load}, 0.0, 1.0), \quad W_{\text{ext}} = \frac{n_{\text{ext}}}{2048.0}, \quad W_{\text{dec}} = \frac{\hat{Y}}{512.0}$$
$$\Theta(\tilde{\mathcal{L}}) = \max(0.05, \; 1.0 - 0.65 \cdot \tilde{\mathcal{L}})$$
$$\hat{T}_{\text{lat}} = \frac{0.01 + W_{\text{ext}} + W_{\text{dec}}}{\Theta(\tilde{\mathcal{L}})}$$

#### 3. `fairroute_gate(candidates, ttft_slo, gamma=1.0)`
Feasibility filter enforcing safety and TTFT SLO:
$$\mathcal{C}_{\text{feasible}} = \left\{ c \in \mathcal{C} \;\middle|\; c[\text{"safe"}] \land \left(c[\text{"predicted\_latency"}] \le \gamma \cdot \text{ttft\_slo}\right) \right\}$$
$$\text{Output} = \begin{cases} (\mathcal{C}_{\text{feasible}}, \text{slo\_risk}=\text{False}) & \text{if } \mathcal{C}_{\text{feasible}} \neq \emptyset \\ (\mathcal{C}_{\text{safe}}, \text{slo\_risk}=\text{True}) & \text{if } \mathcal{C}_{\text{feasible}} = \emptyset \land \mathcal{C}_{\text{safe}} \neq \emptyset \\ (\mathcal{C}, \text{slo\_risk}=\text{True}) & \text{otherwise} \end{cases}$$

#### 4. `fairroute_slack_set(candidates, rho=0.25, eps_max=0.05, urgency=0.0)`
Load-adaptive latency slack set:
$$T^* = \min_{c \in \mathcal{C}} c[\text{"predicted\_latency"}], \quad u = \text{clip}(U_c, 0.0, 1.0)$$
$$\epsilon = \min\left(\epsilon_{\max}, \; \rho \cdot T^*\right) \cdot (1.0 - u)$$
$$\mathcal{C}_{\text{slack}} = \left\{ c \in \mathcal{C} \;\middle|\; c[\text{"predicted\_latency"}] \le T^* + \epsilon \right\}$$
When $U_c \to 1.0$, $\epsilon \to 0$ (greedy minimization for starved clients). When $U_c \to 0$, slack expands up to $\min(\epsilon_{\max}, \rho T^*)$ allowing cache reuse to guide placement.

#### 5. `choose_fairroute_candidate(candidates, mode, urgency, rho, eps_max)`
- **`GATED`**: $\arg\min \hat{T}_c$
- **`MULTIPLICATIVE`**: $\arg\min \left( \frac{\hat{T}_c}{\max(1e-8, T^*)} \cdot \frac{1 + h_c}{\max(1e-8, 1 + \text{deficit}_c)} \right)$
- **`LINEAR`**: $\arg\max \left( -\hat{T}_c + 0.02 \cdot h_c + 0.01 \cdot \text{deficit}_c \right)$
- **`LEARNED`**: $\arg\max \left( -\hat{T}_c + 0.01 \cdot h_c + 0.01 \cdot \text{deficit}_c + 0.01 \cdot \text{credit}_c \right)$
- **`SLACK_LEXICOGRAPHIC` / `V2` (Default)**:
  Filter $\mathcal{C}_{\text{slack}}$.
  If any candidate has $H_c > 0$ and $\text{credit}_c > 0$, narrow pool to locality-eligible candidates.
  Maximized hierarchical lexicographic tuple:
  $$c^* = \arg\max_{c \in \text{Pool}} \Big( h_c, \; h_{c, \text{future}}, \; \text{deficit}_c, \; -\hat{T}_c \Big)$$

#### 6. Deficit-Driven Adaptive Scoring (`score_fairroute_v2` / $H_5$)
$$w_f(D_c) = w_{f, \text{base}} + 2.0 \cdot \max(0.0, D_c)$$
$$\text{FairnessBenefit}_i = U_c \cdot (1.0 - \mathcal{L}_i)$$
$$\text{Score}_i = w_f(D_c) \cdot \text{FairnessBenefit}_i + w_l \cdot h_i + w_p \cdot P_i$$

---

## 3. End-to-End Predictor & Router Formulas (`router.py`)

1. **Queued Work**:
   $$W_i = \sum_{r \in \text{waiting}} \text{tokens}(r) + \sum_{r \in \text{running}} \max(0, \text{tokens}(r) - \text{computed}(r))$$
2. **TTFT Latency**:
   $$\text{TTFT}_i = \left\lceil \frac{W_i + n_{\text{ext}}}{\max(1, S_{\max} - D_i)} \right\rceil \cdot \text{iteration\_time}(B_i + 1, S_{\max}, p_{\max}=S_{\max})$$
3. **Decode Latency**:
   $$\text{Decode}_i = \hat{Y}_c \cdot \text{iteration\_time}(D_i + 1, D_i + 1, p_{\max}=S_{\max})$$
4. **Target TTFT SLO**:
   $$\text{SLO} = k_{\text{SLO}} \cdot \left\lceil \frac{N_{\text{in}}}{S_{\max}} \right\rceil \cdot \text{iteration\_time}(1, S_{\max}, p_{\max}=S_{\max})$$
5. **Empirical Bayes Output Prediction**:
   $$\hat{Y}_c = \frac{\bar{Y}_c + n_0 \cdot \bar{Y}_{\text{global}}}{1.0 + n_0}$$
6. **PRISM Composite Score**:
   $$\text{Score}_{\text{PRISM}} = \frac{1}{1 + (\hat{T}_i / \text{SLO})^p} \cdot \frac{1}{1 + (\rho_i / \text{SLO})^p} \cdot \text{clip}\left(\frac{\text{FreeKV} - \text{KV}}{\text{KV}}, 0, 1\right) \cdot (\beta \text{ if } H_i \ge h_{\min} \text{ else } 1)$$
7. **Deterministic Tie-Breaking Jitter**:
   $$\text{Jitter}(r, i) = \frac{\text{int32\_le}(\text{MD5}(r : i)[0:4])}{2^{32} - 1}$$
