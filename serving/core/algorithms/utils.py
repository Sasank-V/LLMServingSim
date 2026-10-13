"""Small live-state utilities used by Router."""
import hashlib


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


def tiebreak_jitter(request_index: int, candidate_index: int) -> float:
    """Deterministic hash-based jitter to break ties without systematic bias.

    Using the replica index directly as a tiebreaker always selects the lowest
    index, causing pathological hotspotting when scores are equal.  This hashes
    the (request, candidate) pair to a stable but uniformly scattered float in
    [0, 1), so ties distribute evenly across replicas across requests.
    """
    h = hashlib.md5(f"{request_index}:{candidate_index}".encode()).digest()
    return int.from_bytes(h[:4], "little") / 0xFFFFFFFF
