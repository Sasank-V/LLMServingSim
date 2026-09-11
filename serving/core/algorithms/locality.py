import math


def locality_score(
    prefix_hit_tokens: int,
    input_tokens: int,
    eps: float = 1e-8,
) -> float:
    """
    Fraction of the prompt whose KV state can be reused.

    Returns:
        [0, 1]
    """
    if input_tokens <= 0:
        return 0.0

    return min(
        1.0,
        max(
            0.0,
            prefix_hit_tokens / (input_tokens + eps)
        )
    )

def recompute_tokens(
    input_tokens: int,
    prefix_hit_tokens: int,
) -> int:
    """
    Number of input tokens that must be recomputed.
    """
    return max(
        0,
        input_tokens - prefix_hit_tokens
    )

def recompute_cost(
    prefix_hit_tokens: int,
    input_tokens: int,
) -> float:
    """
    Normalized recomputation cost in [0, 1].
    
    100% prefix hit → recompute cost = 0
    50% prefix hit  → recompute cost = 0.5
    0% prefix hit   → recompute cost = 1
    """
    return 1.0 - locality_score(
        prefix_hit_tokens,
        input_tokens
    )

"""
Tier Aware Locality Cost

NPU cache
   ↓ miss
CPU cache
   ↓ miss
CXL/storage
   ↓ miss
recompute

"""
def locality_cost(
    npu_hit_tokens: int,
    cpu_hit_tokens: int,
    storage_hit_tokens: int,
    input_tokens: int,
    npu_cost: float = 0.0,
    cpu_cost: float = 0.25,
    storage_cost: float = 0.6,
    recompute_cost_value: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Estimates normalized locality cost.

    Lower is better.
    """

    cached_tokens = (
        npu_hit_tokens
        + cpu_hit_tokens
        + storage_hit_tokens
    )

    recompute = max(
        0,
        input_tokens - cached_tokens
    )

    cost = (
        npu_hit_tokens * npu_cost
        + cpu_hit_tokens * cpu_cost
        + storage_hit_tokens * storage_cost
        + recompute * recompute_cost_value
    )

    return cost / (input_tokens + eps)

def adaptive_locality_weight(
    base_beta: float,
    fairness_debt_value: float,
    lambda_d: float = 1.0,
) -> float:

    return (
        base_beta
        * math.exp(
            -lambda_d
            * max(0.0, fairness_debt_value)
        )
    )