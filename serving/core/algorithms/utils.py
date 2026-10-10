"""Small live-state utilities used by Router."""


def reactive_load(waiting_requests: int, running_requests: int,
                  capacity: int) -> float:
    """Return the simulator's normalized LOAD-policy pressure."""
    if capacity <= 0:
        return float("inf")
    return (4.0 * waiting_requests + running_requests) / capacity


def kv_utilization(kv_used: float, kv_capacity: float, eps: float = 1e-8) -> float:
    """Return bounded KV-cache utilization."""
    if kv_capacity <= 0:
        return 1.0
    return min(1.0, max(0.0, kv_used / (kv_capacity + eps)))
