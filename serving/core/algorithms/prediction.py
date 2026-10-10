"""Replica safety and queue prediction used by the FairRoute router."""


def predict_queue(current_queue: float, arrival_rate: float,
                  service_rate: float, horizon: float) -> float:
    """Project queue size over a short prediction horizon."""
    return max(
        0.0,
        current_queue + arrival_rate * horizon - service_rate * horizon,
    )


def replica_is_safe(predicted_queue: float, max_queue: float,
                    predicted_kv_utilization: float,
                    max_kv_utilization: float = 0.95) -> bool:
    """Check queue and KV headroom before a candidate enters the gate."""
    return (
        predicted_queue < max_queue
        and predicted_kv_utilization < max_kv_utilization
    )
