def normalize(value, min_value, max_value, eps=1e-8):
    """
    Min-max normalization into [0, 1].
    """
    if max_value - min_value < eps:
        return 0.0

    return (value - min_value) / (max_value - min_value)

def normalize_inverse(value, min_value, max_value, eps=1e-8):
    """
    Converts a cost where lower is better into a benefit where higher is better.
    """
    if max_value - min_value < eps:
        return 1.0

    return 1.0 - (value - min_value) / (max_value - min_value)

def load_headroom(load: float) -> float:
    """
    Convert load into a non-negative headroom benefit [0, 1].
    """
    return max(0.0, 1.0 - load)

# Load and locality scores

def reactive_load(
    waiting_requests: int,
    running_requests: int,
    capacity: int,
) -> float:
    """
    Equivalent conceptual form of the simulator's LOAD score.

    Lower is better.
    """
    if capacity <= 0:
        return float("inf")

    return (
        4.0 * waiting_requests
        + running_requests
    ) / capacity

def load_benefit(load: float) -> float:
    """
    Convert load into a bounded benefit.
    """
    return 1.0 / (1.0 + load)

# KV cache 

def kv_utilization(
    kv_used: float,
    kv_capacity: float,
    eps: float = 1e-8,
) -> float:
    if kv_capacity <= 0:
        return 1.0

    return min(
        1.0,
        max(
            0.0,
            kv_used / (kv_capacity + eps)
        )
    )

def kv_headroom(
    kv_used: float,
    kv_capacity: float,
    eps: float = 1e-8,
) -> float:
    return max(
        0.0,
        1.0 - kv_utilization(
            kv_used,
            kv_capacity,
            eps
        )
    )