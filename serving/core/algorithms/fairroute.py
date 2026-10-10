"""Predictors and decision primitives for the routing policies."""

import math


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


def choose_fairroute_candidate(candidates, mode="SLACK_LEXICOGRAPHIC",
                               urgency=0.0, rho=0.25, eps_max=0.05):
    """Choose a candidate using one FairRoute combination form."""
    if not candidates:
        raise ValueError("FairRoute requires at least one candidate")

    mode = mode.upper()
    if mode == "GATED":
        pool = candidates
        epsilon = 0.0
        selected = min(pool, key=lambda item: item["predicted_latency"])
    elif mode == "MULTIPLICATIVE":
        pool = candidates
        epsilon = 0.0
        best_latency = min(item["predicted_latency"] for item in pool)
        selected = min(
            pool,
            key=lambda item: (
                item["predicted_latency"] / max(1e-8, best_latency)
                * (1.0 + item["locality"])
                / max(1e-8, 1.0 + item["replica_deficit"])
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
                + 0.01 * item["replica_deficit"]
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
                + 0.01 * item["replica_deficit"]
                + 0.01 * item["fairness_credit"]
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
                item["replica_deficit"],
                -item["predicted_latency"],
            ),
        )
    return selected, epsilon
