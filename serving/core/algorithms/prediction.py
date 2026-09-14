import math
from dataclasses import dataclass


@dataclass
class ReplicaState:
    waiting: int
    running: int

    kv_used: float
    kv_capacity: float

    arrival_rate: float
    service_rate: float

    batch_size: int
    prefill_tokens: int
    decode_tokens: int

def persistence_prediction(current_value: float) -> float:
    """
    Predict future value using the most recent observation.
    """
    return current_value

def predict_replica_state_persistence(
    state: ReplicaState,
) -> ReplicaState:

    return ReplicaState(
        waiting=state.waiting,
        running=state.running,

        kv_used=state.kv_used,
        kv_capacity=state.kv_capacity,

        arrival_rate=state.arrival_rate,
        service_rate=state.service_rate,

        batch_size=state.batch_size,
        prefill_tokens=state.prefill_tokens,
        decode_tokens=state.decode_tokens,
    )


# Better Prediction 

"""
Experiment with alpha values
alpha = 0.1
alpha = 0.3
alpha = 0.5
alpha = 0.7
alpha = 0.9
"""
def ewma_prediction(
    current_value: float,
    previous_prediction: float,
    alpha: float = 0.3,
) -> float:

    return (
        alpha * current_value
        + (1.0 - alpha) * previous_prediction
    )


# Queue Prediction

def predict_queue(
    current_queue: float,
    arrival_rate: float,
    service_rate: float,
    horizon: float,
) -> float:

    predicted = (
        current_queue
        + arrival_rate * horizon
        - service_rate * horizon
    )

    return max(0.0, predicted)


# KV Prediction

def predict_kv_usage(
    current_kv: float,
    kv_arrival_rate: float,
    kv_release_rate: float,
    horizon: float,
) -> float:

    predicted = (
        current_kv
        + kv_arrival_rate * horizon
        - kv_release_rate * horizon
    )

    return max(0.0, predicted)

def predicted_kv_utilization(
    predicted_kv: float,
    kv_capacity: float,
) -> float:

    if kv_capacity <= 0:
        return 1.0

    return min(
        1.0,
        predicted_kv / kv_capacity
    )

# Prediction Risk

def prediction_risk(
    predicted_work: float,
    prediction_std: float,
    uncertainty_weight: float = 1.0,
) -> float:

    return (
        predicted_work
        + uncertainty_weight * prediction_std
    )

"""
Higher predicted risk → smaller Prediction Benefit
Lower predicted risk → larger Prediction Benefit
"""
def prediction_benefit(
    risk: float,
    temperature: float = 1.0,
) -> float:

    if temperature <= 0:
        temperature = 1.0

    return math.exp(
        -risk / temperature
    )

def replica_is_safe(
    predicted_queue: float,
    max_queue: float,
    predicted_kv_utilization: float,
    max_kv_utilization: float = 0.95,
) -> bool:

    return (
        predicted_queue < max_queue
        and
        predicted_kv_utilization < max_kv_utilization
    )