#!/usr/bin/env python3
"""Generate controlled FairRoute hypothesis workloads.

The scenarios keep the request count and token ranges fixed while varying one
routing signal: tenant skew, prefix locality, burstiness, or signal conflict.
The output is JSONL compatible with LLMServingSim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCENARIOS = (
    "balanced_control",
    "locality_conflict",
    "fairness_conflict",
    "prediction_burst",
    "full_adversarial",
)


def token_ids(prefix: int, length: int, request_index: int, shared: bool) -> list[int]:
    if shared:
        return [prefix] * max(1, length - 8) + list(range(request_index * 8, request_index * 8 + 8))
    return list(range(request_index * max(1, length), (request_index + 1) * max(1, length)))


def make_records(scenario: str, num_reqs: int, num_clients: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    base_interval_ns = 100_000_000  # 10 requests/s outside bursts

    for index in range(num_reqs):
        client_id = index % num_clients
        shared_prefix = False
        arrival_ns = index * base_interval_ns
        output_toks = int(rng.integers(64, 192))
        input_toks = int(rng.integers(256, 768))

        if scenario == "locality_conflict":
            # Tenant 0 owns a warm prefix; other tenants have cold prompts.
            client_id = index % num_clients
            shared_prefix = client_id == 0
        elif scenario == "fairness_conflict":
            # One tenant repeatedly sends warm requests while all tenants remain
            # present. Locality-only routing should favor that tenant's replica.
            client_id = 0 if index % 2 == 0 else (index % (num_clients - 1)) + 1
            shared_prefix = client_id == 0
        elif scenario == "prediction_burst":
            # Bursts create queue pressure while tenant shares remain balanced.
            client_id = index % num_clients
            burst = (index // 20) % 2 == 1
            arrival_ns = (index // 20) * 2_000_000_000 + (index % 20) * (10_000_000 if burst else base_interval_ns)
        elif scenario == "full_adversarial":
            # Hot tenant, warm prefix, and synchronized bursts conflict with
            # fairness and prediction simultaneously.
            client_id = 0 if index % 3 != 2 else (index % (num_clients - 1)) + 1
            shared_prefix = client_id == 0
            arrival_ns = (index // 25) * 2_000_000_000 + (index % 25) * 5_000_000
            output_toks = int(rng.integers(256, 768))
            input_toks = int(rng.integers(512, 1536))

        prefix_id = 10_000 + (client_id if shared_prefix else index)
        records.append({
            "input_toks": input_toks,
            "output_toks": output_toks,
            "arrival_time_ns": arrival_ns,
            "input_tok_ids": token_ids(prefix_id, input_toks, index, shared_prefix),
            "output_tok_ids": token_ids(20_000 + client_id, output_toks, index, False),
            "client_id": client_id,
            "scenario": scenario,
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="LLMServingSim/workloads/fairroute_hypothesis")
    parser.add_argument("--num-reqs", type=int, default=1000)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_reqs <= 0 or args.num_clients <= 1:
        raise SystemExit("--num-reqs must be positive and --num-clients must exceed 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        path = output_dir / f"{scenario}.jsonl"
        with path.open("w", encoding="utf-8") as workload:
            for record in make_records(scenario, args.num_reqs, args.num_clients, args.seed):
                workload.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(path)


if __name__ == "__main__":
    main()
