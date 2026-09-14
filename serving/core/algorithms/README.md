# Routing Algorithms & FairRoute Implementation Guide

This directory (`serving/core/algorithms/`) contains the modular scoring functions and fairness tracking logic for request routing in **LLMServingSim**.

---

## 1. Executive Summary & Problem Diagnosis

Previous evaluations observed that composite routing policies (`FAIRNESS`, `LOCALITY`, `F_L`, `H0`, `H1`, `H4`/`FAIRROUTE`) behaved identically and dumped 95%+ of requests onto Instance 0.

### Root Causes Identified

1. **Dual-Pass Overwrite Bug**: In `router.py`, `_select_fairroute_candidate` calculated un-normalized candidate scores in a first pass (lines 228–254), but then re-calculated normalized candidate scores in a second pass (lines 270–334) and **overwrote** `candidate["score"]`.
2. **Scalar Candidate-Invariant Fairness**: In the second pass, `fairness_value` was computed as a scalar `0.5 + 0.5 * min(1, debt)` derived solely from the requesting client's debt. Because this value was identical across all candidate instances for a given request, any policy evaluating fairness produced identical scores for all candidates.
3. **Hotspotting Tie-Breaker**: When candidate scores tied (e.g., all 0.5 or 0.0), candidates were sorted by `-candidate["index"]`, which strictly prioritized candidate 0. Consequently, all single-component and composite policies collapsed to choosing Instance 0.
4. **Workload-Invariant Fairness Metric**: Jain's Fairness Index was previously computed over total request `output_toks`. Because token counts are fixed by the workload dataset and independent of routing decisions, Jain's Index showed 0 variation across all policies.

---

## 2. Applied Fixes & Architectural Enhancements

### A. Candidate-Varying Fairness Signal (`router.py`)
Fairness is now computed per candidate instance relative to instance load headroom:
$$\text{fairness\_values}[i] = \text{urgency} \times \text{load\_headroom}[i]$$
Where:
- $\text{urgency} = 1 - e^{-\lambda_d \cdot \max(0, \text{debt})}$
- $\text{load\_headroom}[i] = \text{normalize\_candidates}(\text{load}, \text{higher\_is\_better}=\text{False})$

This ensures that an underserved client (high debt/urgency) is steered toward less-loaded instances, while a well-served client (low urgency) lets prefix locality and prediction dominate.

### B. Deterministic Hash Jitter Tie-Breaking (`router.py`)
Replaced `-candidate["index"]` with MD5 hash-based jitter:
```python
def _tiebreak_jitter(request_index, candidate_index):
    h = hashlib.md5(f"{request_index}:{candidate_index}".encode()).digest()
    return int.from_bytes(h[:4], 'little') / 0xFFFFFFFF
```
This distributes ties uniformly across replicas without introducing non-deterministic simulation variance.

### C. Routing-Sensitive Per-Client Latency Fairness (`fairness.py` & `router.py`)
- Extended `FairnessTracker` with per-client latency tracking: `record_latency(client_id, latency_ns)` and `get_mean_latency(client_id)`.
- Added `jains_index_latency()` to compute Jain's Fairness Index over mean client latencies:
$$J(x) = \frac{\left(\sum_{i=1}^n x_i\right)^2}{n \sum_{i=1}^n x_i^2}$$
where $x_i$ is client $i$'s mean request latency.

### D. FairRoute-v2 / H5 (`fairroute.py` & `router.py`)
Introduced `score_fairroute_v2` implementing deficit-driven adaptive weighted scoring:
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
    w_f = w_f_base + 2.0 * max(0.0, debt)
    fairness_benefit = urgency * normalized_load_benefit
    return (
        w_f * fairness_benefit
        + w_l * locality
        + w_p * prediction
    )
```

---

## 3. Supported Routing Policies

| Policy | Category | Description |
|--------|----------|-------------|
| `LOAD` | Baseline | vLLM-style weighted least-loaded replica selection |
| `RR` | Baseline | Round-robin across active replicas |
| `RAND` | Baseline | Uniform random instance selection |
| `FAIRNESS` | Single-component | Routes based purely on per-candidate fairness benefit |
| `LOCALITY` | Single-component | Routes based purely on prefix-cache hit ratio |
| `PREDICTION` | Single-component | Routes based purely on predicted queue depth / load |
| `F_L`, `L_P`, `F_P`, `F_L_P` | Combinations | Linear combinations of normalized component benefits |
| `H0` | Hypothesis 0 | Equal weighted linear sum ($F + L + P$) |
| `H1` | Hypothesis 1 | Multiplicative component scoring ($(1+F) \cdot L \cdot P$) |
| `H2` | Hypothesis 2 | Fairness-gated locality scoring |
| `H3` | Hypothesis 3 | Adaptive Fairness-Locality Control (AFLC) |
| `H4` / `FAIRROUTE` | Hypothesis 4 | FairRoute v1 temperature-scaled score |
| `H5` / `FAIRROUTE_V2` | Hypothesis 5 | **FairRoute-v2**: Deficit-driven adaptive weighted scoring |
| `PREBLE`, `LBGR`, `DUALMAP`, `CACHE_ROUTE`, `VTC`, `EQUINOX`, `QUARTZ`, `ISJL`, `NEXUSSCHED`, `BALANCEROUTE`, `PILLM`, `ONLINE_LP` | Literature | Implementations of published multi-tenant LLM routing algorithms |

---

## 4. File Structure

- [`fairness.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/fairness.py): Client debt tracking (`FairnessTracker`), target calculation, urgency functions, and latency-based Jain's Index.
- [`fairroute.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/fairroute.py): Scoring functions for `H0`–`H5`, `FAIRROUTE`, AFLC, and gated routing.
- [`locality.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/locality.py): Prefix-cache hit ratio and locality scoring functions.
- [`prediction.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/prediction.py): Queue prediction, load benefit, and risk estimation.
- [`utils.py`](file:///home/sriram/sem7/p1/LLMServingSim/serving/core/algorithms/utils.py): Feature normalization (`normalize_candidates`) and utility helpers.
