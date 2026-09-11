import math


class FairnessTracker:

    def __init__(self):
        self.service = {}

    def record_service(
        self,
        client_id,
        amount: float,
    ):
        self.service[client_id] = (
            self.service.get(client_id, 0.0)
            + amount
        )

    def get_service(
        self,
        client_id,
    ) -> float:

        return self.service.get(
            client_id,
            0.0
        )

def fairness_target(
    total_service: float,
    num_clients: int,
) -> float:

    if num_clients <= 0:
        return 0.0

    return total_service / num_clients

def fairness_debt(
    client_service: float,
    target_service: float,
    eps: float = 1e-8,
) -> float:

    deficit = max(
        0.0,
        target_service - client_service
    )

    return deficit / (
        target_service + eps
    )

def fairness_urgency(
    debt: float,
    lambda_d: float = 2.0,
) -> float:

    return 1.0 - math.exp(
        -lambda_d * max(0.0, debt)
    )