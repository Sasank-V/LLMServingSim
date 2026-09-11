from serving.core.algorithms.prediction import prediction_benefit
from serving.core.algorithms.locality import adaptive_locality_weight

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
) -> float:

    if fairness_urgency_value < fairness_threshold:
        return (
            locality ** beta
            * prediction ** gamma
        )

    # Once fairness becomes urgent,
    # fairness gets stronger influence.
    return (
        (1.0 + fairness_urgency_value)
        * locality ** beta
        * prediction ** gamma
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