import math


class FairnessTracker:

    def __init__(self):
        self.service = {}
        self.latency = {}
        self.latency_count = {}

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

    def record_latency(
        self,
        client_id,
        latency_ns,
    ):
        self.latency[client_id] = (
            self.latency.get(client_id, 0.0)
            + latency_ns
        )
        self.latency_count[client_id] = (
            self.latency_count.get(client_id, 0)
            + 1
        )

    def get_mean_latency(
        self,
        client_id,
    ) -> float:

        count = self.latency_count.get(client_id, 0)
        if count <= 0:
            return 0.0

        return self.latency.get(
            client_id,
            0.0
        ) / count

    def jains_index_latency(self) -> float:
        clients = [
            client_id
            for client_id, count in self.latency_count.items()
            if count > 0
        ]
        n = len(clients)
        if n < 2:
            return 1.0

        means = [
            self.get_mean_latency(client_id)
            for client_id in clients
        ]
        sum_x = sum(means)
        sum_sq = sum(x * x for x in means)

        denom = n * sum_sq
        if denom == 0.0:
            return 1.0

        return (sum_x ** 2) / denom

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