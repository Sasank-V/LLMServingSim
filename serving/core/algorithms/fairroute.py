from serving.core.algorithms.prediction import prediction_benefit
from serving.core.algorithms.locality import adaptive_locality_weight
import math


def bounded_benefit(value, scale=1.0):
    """Map a non-negative cost/pressure to a smooth, informative [0, 1] benefit."""
    scale = max(float(scale), 1e-8)
    return 0.5 + 0.5 * math.tanh((float(value) - scale) / scale)


def normalize_feature(value, low=0.0, high=1.0, neutral=0.5):
    """Normalize a feature without collapsing constant candidate values to zero."""
    span = float(high) - float(low)
    if span <= 1e-8:
        return float(neutral)
    return min(1.0, max(0.0, (float(value) - low) / span))


def normalize_candidates(values, higher_is_better=True):
    """Candidate-relative normalization used by every composite router."""
    values = [float(value) for value in values]
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-8:
        return [0.5] * len(values)
    normalized = [(value - low) / (high - low) for value in values]
    return normalized if higher_is_better else [1.0 - value for value in normalized]


def score_combination(fairness, locality, prediction, components="FLP",
                      weights=None):
    """Publication-friendly additive score for all F/L/P combinations.

    All inputs are benefits in [0, 1]. Additive scoring avoids the near-zero
    product collapse of multiplicative formulas while preserving the selected
    feature trade-off.
    """
    weights = weights or {"F": 1.0, "L": 1.0, "P": 1.0}
    active = [key for key in "FLP" if key in components]
    if not active:
        return 0.0
    values = {"F": fairness, "L": locality, "P": prediction}
    total_weight = sum(weights[key] for key in active)
    return sum(weights[key] * values[key] for key in active) / total_weight


def score_preble(current_load, eviction_cost, request_prefill):
    """Preble: predictive load + eviction cost + candidate prefill cost."""
    return -(current_load + eviction_cost + request_prefill)


def score_lbgr(cached_tokens, prompt_tokens, output_tokens,
               cached_cost=0.25, miss_cost=1.0):
    """LBGR cache-hit/miss/output service-time cost (returned as a benefit)."""
    cost = (cached_cost * cached_tokens
            + miss_cost * max(0, prompt_tokens - cached_tokens)
            + output_tokens)
    return 1.0 / (1.0 + cost)


def score_dualmap(queue_delay, compute_cost):
    """DualMap chooses the candidate with minimum queue plus compute cost."""
    return -(queue_delay + compute_cost)


def score_cache_route(prefix_rate, capacity, assigned_load):
    """CacheRoute LPT placement benefit for a prefix key."""
    replication = max(1, math.ceil(prefix_rate / max(capacity, 1e-8)))
    return -(assigned_load + prefix_rate / replication)


def score_vtc(client_counter, request_cost):
    """VTC deficit-style priority: lower virtual service counter wins."""
    return -(client_counter + request_cost)


def score_equinox(user_counter, resource_counter, alpha=0.7, beta=0.3):
    """Equinox holistic fairness counter."""
    return -(alpha * user_counter + beta * resource_counter)


def score_quartz(predicted_quantile, arrival_age, service_counter,
                 aging_weight=0.1, fairness_weight=0.1):
    """QUARTZ quantile-aware admission key, expressed as a benefit."""
    return -(predicted_quantile + aging_weight * arrival_age
             + fairness_weight * service_counter)


def score_isjl(progress, alpha=1.0):
    """ISJL short-job insertion benefit under a progress disparity limit."""
    return -(alpha * max(0.0, progress))


def score_nexussched(batch_size, sequence_tokens, coefficients=None):
    """NexusSched structural iteration-latency model, returned as a benefit."""
    coefficients = coefficients or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    tau0, tau_b, tau_s, w0, ws, pmax = coefficients
    denom = max(pmax * (1.0 - math.exp(-batch_size))
                * (1.0 - math.exp(-sequence_tokens / 1024.0)), 1e-8)
    latency = tau0 + (w0 + ws * sequence_tokens) / denom
    return -(latency + tau_b * batch_size + tau_s * sequence_tokens)


def score_pillm(prefill_tokens, active_tokens, predicted_decode,
                alpha=1e-4, beta=1.0, gamma=0.0, lambd=1.0, mu=0.0):
    """PiLLM polynomial prefill/decode demand model, returned as a benefit."""
    demand = (alpha * prefill_tokens ** 2 + beta * prefill_tokens + gamma
              + lambd * active_tokens * predicted_decode + mu)
    return 1.0 / (1.0 + demand)


def score_balance_route(margin, overflow, horizon=1.0, discount=0.95,
                        alpha=1.0, beta=1.0):
    """BalanceRoute horizon-discounted margin minus overflow penalty."""
    horizon_weight = sum(discount ** step for step in range(max(1, int(horizon))))
    return alpha * horizon_weight * margin - beta * overflow


def score_online_lp(reward, resource_demand, shadow_price):
    """Online-LP reduced reward: utility minus capacity shadow price."""
    return reward - resource_demand * shadow_price

def score_linear(
    fairness: float,
    locality: float,
    prediction: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:

    return (
        alpha * fairness
        + beta * locality
        + gamma * prediction
    )

def score_multiplicative(
    fairness: float,
    locality: float,
    prediction: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:

    return (
        (1.0 + alpha * fairness)
        * (1.0 + beta * locality)
        * (1.0 + gamma * prediction)
    )

def score_fairness_gated(
    fairness_urgency_value: float,
    fairness_threshold: float,
    locality: float,
    prediction: float,
    beta: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> float:

    if fairness_urgency_value < fairness_threshold:
        return (
            (eps + locality) ** beta
            * (eps + prediction) ** gamma
        )

    # Once fairness becomes urgent,
    # fairness gets stronger influence.
    return (
        (1.0 + fairness_urgency_value)
        * (eps + locality) ** beta
        * (eps + prediction) ** gamma
    )

""" 
Not Implemented
features = [
    fairness_debt,
    locality,
    prediction,
    waiting,
    running,
    kv_utilization,
]
"""
def learned_linear_score(
    features,
    weights,
    bias=0.0,
) -> float:

    return (
        sum(
            w * x
            for w, x in zip(weights, features)
        )
        + bias
    )

def score_aflc(
    fairness_urgency_value: float,
    locality: float,
    prediction: float,
    fairness_weight: float,
    base_beta: float,
    gamma: float,
    fairness_debt_value: float,
    lambda_d: float = 1.0,
    eps: float = 1e-8,
) -> float:

    beta = adaptive_locality_weight(
        base_beta=base_beta,
        fairness_debt_value=fairness_debt_value,
        lambda_d=lambda_d,
    )

    return (
        (1.0 + fairness_weight * fairness_urgency_value)
        * (eps + locality) ** beta
        * (eps + prediction) ** gamma
    )

def fairroute_score(
    fairness_debt_value: float,
    fairness_urgency_value: float,
    locality: float,
    predicted_risk: float,

    fairness_weight: float = 1.0,
    base_beta: float = 1.0,
    prediction_gamma: float = 1.0,
    lambda_d: float = 1.0,

    prediction_temperature: float = 1.0,
) -> float:

    prediction = prediction_benefit(
        risk=predicted_risk,
        temperature=prediction_temperature,
    )

    return score_aflc(
        fairness_urgency_value=fairness_urgency_value,
        locality=locality,
        prediction=prediction,
        fairness_weight=fairness_weight,
        base_beta=base_beta,
        gamma=prediction_gamma,
        fairness_debt_value=fairness_debt_value,
        lambda_d=lambda_d,
    )