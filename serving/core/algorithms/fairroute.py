"""Predictors and decision primitives for the routing policies."""

import math
from .utils import tiebreak_jitter


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
    """Evaluate NexusSched's iteration-time model.

    The defaults are conservative [D] fallbacks.  A caller can provide fitted
    constants without changing the predictor's shape or units (seconds).
    """
    batch = max(0.0, float(batch_size))
    sequences = max(0.0, float(sequence_count))
    throughput = p_max * (1.0 - math.exp(-k_batch * batch)) * (
        1.0 - math.exp(-k_sequence * sequences)
    )
    if throughput <= 0.0:
        return tau0 + tau_batch * batch + tau_sequence * sequences
    return (
        tau0
        + (w0 + ws * sequences) / throughput
        + tau_batch * batch
        + tau_sequence * sequences
    )


def predict_latency_seconds(load, extend_tokens, decode_tokens):
    """Estimate request latency from load and uncached work."""
    load = max(0.0, min(1.0, float(load)))
    work = max(0.0, float(extend_tokens)) / 2048.0
    decode = max(0.0, float(decode_tokens)) / 512.0
    throughput = max(0.05, 1.0 - 0.65 * load)
    return (0.01 + work + decode) / throughput


def fairroute_gate(candidates, ttft_slo, gamma=1.0):
    """Apply KV safety and predicted-TTFT feasibility with fallback."""
    feasible = [
        candidate for candidate in candidates
        if candidate["safe"]
        and candidate["predicted_latency"] <= gamma * ttft_slo
    ]
    if feasible:
        return feasible, False
    safe = [candidate for candidate in candidates if candidate["safe"]]
    return (safe or candidates), True


def fairroute_slack_set(candidates, rho=0.25, eps_max=0.05, urgency=0.0):
    """Keep candidates within load-adaptive latency slack of the fastest one."""
    if not candidates:
        return [], 0.0
    best = min(candidate["predicted_latency"] for candidate in candidates)
    urgency = max(0.0, min(1.0, float(urgency)))
    epsilon = min(float(eps_max), float(rho) * best) * (1.0 - urgency)
    return [
        candidate for candidate in candidates
        if candidate["predicted_latency"] <= best + epsilon
    ], epsilon


def _candidate_load_headrooms(candidates):
    """Return normalised load headroom in [0, 1] for each candidate.

    Headroom 1.0 = least-loaded replica; 0.0 = most-loaded.  When all
    candidates share the same load, every headroom is 0.5.  The ``load``
    key is read from each candidate dict and defaults to 0.0 when absent,
    so the function is safe to call on candidates from any source.
    """
    loads = [c.get("load", 0.0) for c in candidates]
    l_min, l_max = min(loads), max(loads)
    span = l_max - l_min
    if span < 1e-8:
        return [0.5] * len(candidates)
    return [(l_max - l) / span for l in loads]


def choose_fairroute_candidate(candidates, mode="SLACK_LEXICOGRAPHIC",
                               urgency=0.0, rho=0.25, eps_max=0.05,
                               request_index=0):
    """Choose a candidate using one FairRoute combination form.

    Every scoring mode incorporates **both** replica-level and client-level
    fairness signals:

    * **Replica-level fairness** (``replica_deficit``): balances accumulated
      service across replica instances so no single instance is over-burdened.
    * **Client-level fairness** (``urgency * load_headroom``): steers
      underserved clients (high ``urgency``) toward less-loaded replicas,
      ensuring no client is starved regardless of which replica is chosen.

    The client fairness term is candidate-varying: the least-loaded replica
    gets headroom=1, the most-loaded gets 0, so the signal genuinely
    differentiates replicas.  When ``urgency`` is 0.0 (well-served client)
    the client-level contribution is zero and the formula reduces to the
    replica-only form.  When ``urgency`` is 1.0 (starved client) the pull
    toward low-load replicas is maximal.

    Candidates must carry a ``load`` key (normalised replica load from
    ``reactive_load``); when absent a headroom of 0.5 is assumed.
    """
    if not candidates:
        raise ValueError("FairRoute requires at least one candidate")

    mode = mode.upper()
    urgency = max(0.0, min(1.0, float(urgency)))

    # Build a per-candidate client-level fairness signal.
    # Normalising across the candidate set makes the signal candidate-varying
    # even though ``urgency`` is a single per-client scalar.
    # An identity-keyed dict avoids mutating the caller's dicts while remaining
    # correct after list-comprehension filtering (Python preserves object
    # identity in list comprehensions, so id(item) still matches).
    headrooms = _candidate_load_headrooms(candidates)
    cf = {id(c): urgency * h for c, h in zip(candidates, headrooms)}

    if mode == "GATED":
        pool = candidates
        epsilon = 0.0
        # Prefer low latency; break ties by favouring the less-loaded replica
        # when the client is underserved (urgency > 0).
        selected = max(
            pool,
            key=lambda item: (
                cf[id(item)] - item["predicted_latency"]
            ),
        )
    elif mode == "MULTIPLICATIVE":
        pool = candidates
        epsilon = 0.0
        best_latency = min(item["predicted_latency"] for item in pool)
        selected = min(
            pool,
            key=lambda item: (
                item["predicted_latency"] / max(1e-8, best_latency)
                * (1.0 + item["locality"])
                / max(1e-8, 1.0 + item["replica_deficit"])  # replica-level
                / max(1e-8, 1.0 + cf[id(item)])             # client-level
            ),
        )
    elif mode == "LINEAR":
        pool = candidates
        epsilon = 0.0
        selected = max(
            pool,
            key=lambda item: (
                -item["predicted_latency"]
                + 0.02 * item["locality"]
                + 0.01 * item["replica_deficit"]  # replica-level fairness
                + 0.05 * cf[id(item)]             # client-level fairness
            ),
        )
    elif mode == "LEARNED":
        pool = candidates
        epsilon = 0.0
        selected = max(
            pool,
            key=lambda item: (
                -item["predicted_latency"]
                + 0.01 * item["locality"]
                + 0.01 * item["replica_deficit"]   # replica-level fairness
                + 0.01 * item["fairness_credit"]
                + 0.02 * cf[id(item)]              # client-level fairness
            ),
        )
    else:
        pool, epsilon = fairroute_slack_set(
            candidates, rho=rho, eps_max=eps_max, urgency=urgency
        )
        locality_eligible = [
            item for item in pool
            if item["hit"] > 0 and item["fairness_credit"] > 0
        ]
        if locality_eligible and mode in {"SLACK_LEXICOGRAPHIC", "V2"}:
            pool = locality_eligible
        selected = max(
            pool,
            key=lambda item: (
                item["locality"],
                item["future_locality"],
                cf[id(item)],              # client-level fairness
                item["replica_deficit"],   # replica-level fairness
                -item["predicted_latency"],
                -tiebreak_jitter(request_index, item["index"]),
            ),
        )
    return selected, epsilon
