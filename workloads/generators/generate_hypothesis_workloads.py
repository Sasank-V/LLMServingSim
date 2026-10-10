#!/usr/bin/env python3
"""Generate controlled FairRoute hypothesis workloads.

The scenarios keep the request count and token ranges fixed while varying one
routing signal: tenant skew, prefix locality, burstiness, or signal conflict.
The output is JSONL compatible with LLMServingSim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


SCENARIOS = (
    "c0_calibration",
    "balanced_control",
    "locality_conflict",
    "fairness_conflict",
    "prediction_burst",
    "full_adversarial",
    "u5_vtc_as_published",
    "w2_heavy_light",
    "w4_flooding",
    "u2_multiturn",
    "u3_loogle_proxy",
)

VOCAB_SIZE = 128_256
BLOCK_SIZE = 16


def token_ids(prefix: int, length: int, request_index: int, shared: bool) -> list[int]:
    """Create deterministic, in-range IDs without accidental prefix sharing."""
    length = max(1, int(length))
    if shared:
        shared_len = max(BLOCK_SIZE, length - 8)
        shared_len -= shared_len % BLOCK_SIZE
        seed = (prefix * 1_000_003 + request_index * 97) % VOCAB_SIZE
        suffix = [((seed + offset) % (VOCAB_SIZE - 1)) + 1 for offset in range(length - shared_len)]
        return [prefix % VOCAB_SIZE] * shared_len + suffix
    seed = (prefix * 1_000_003 + request_index * 97) % (VOCAB_SIZE - 1) + 1
    return [((seed + offset) % (VOCAB_SIZE - 1)) + 1 for offset in range(length)]


def _rng_for(seed: int, scenario: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{scenario}:{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "little"))


def _record(index, scenario, client_id, input_toks, output_toks, arrival_ns,
            shared_prefix=False, prefix_group=None):
    group = client_id if prefix_group is None else prefix_group
    prefix = 1_000 + (group * BLOCK_SIZE)
    return {
        "input_toks": int(input_toks),
        "output_toks": int(output_toks),
        "arrival_time_ns": int(arrival_ns),
        "input_tok_ids": token_ids(prefix, input_toks, index, shared_prefix),
        "output_tok_ids": token_ids(50_000 + client_id, output_toks, index, False),
        "client_id": int(client_id),
        "scenario": scenario,
        "prefix_group_id": f"{scenario}:{group}" if shared_prefix else f"{scenario}:unique:{index}",
    }


def make_records(scenario: str, num_reqs: int, num_clients: int, seed: int) -> list[dict]:
    records = []
    base_interval_ns = 100_000_000

    for index in range(num_reqs):
        local_rng = _rng_for(seed, scenario, index)
        client_id = index % num_clients
        shared_prefix = False
        arrival_ns = index * base_interval_ns
        output_toks = local_rng.randrange(64, 192)
        input_toks = local_rng.randrange(256, 768)

        if scenario == "c0_calibration":
            client_id = index % 8
            input_toks = local_rng.randrange(128, 512)
            output_toks = local_rng.randrange(32, 128)
            shared_prefix = True
            arrival_ns = int(local_rng.expovariate(1 / 100_000_000) + (0 if index == 0 else records[-1]["arrival_time_ns"]))
        elif scenario == "locality_conflict":
            # Tenant 0 owns a warm prefix; other tenants have cold prompts.
            client_id = index % num_clients
            shared_prefix = client_id == 0
        elif scenario == "fairness_conflict":
            # One tenant repeatedly sends warm requests while all tenants remain
            # present. Locality-only routing should favor that tenant's replica.
            client_id = 0 if index % 2 == 0 else ((index // 2) % (num_clients - 1)) + 1
            shared_prefix = client_id == 0
        elif scenario == "prediction_burst":
            # Bursts create queue pressure while tenant shares remain balanced.
            client_id = index % num_clients
            burst = (index // 20) % 2 == 1
            arrival_ns = (index // 20) * 2_000_000_000 + (index % 20) * (10_000_000 if burst else base_interval_ns)
        elif scenario == "full_adversarial":
            # Hot tenant, warm prefix, and synchronized bursts conflict with
            # fairness and prediction simultaneously.
            client_id = 0 if index % 3 != 2 else ((index // 3) % (num_clients - 1)) + 1
            shared_prefix = client_id == 0
            arrival_ns = (index // 25) * 2_000_000_000 + (index % 25) * 5_000_000
            output_toks = local_rng.randrange(256, 768)
            input_toks = local_rng.randrange(512, 1536)
        elif scenario == "u5_vtc_as_published":
            client_id = 0 if index % 2 == 0 else 1
            input_toks, output_toks = ((64, 64) if client_id == 0 else (256, 256))
            arrival_ns = index * (125_000_000 if client_id == 0 else 666_666_666)
        elif scenario == "w2_heavy_light":
            client_id = index % num_clients
            if client_id < 2:
                input_toks = local_rng.randrange(4096, 8193)
                output_toks = local_rng.randrange(256, 768)
            else:
                input_toks = local_rng.randrange(128, 512)
                output_toks = local_rng.randrange(32, 128)
        elif scenario == "w4_flooding":
            client_id = 0 if index % 2 == 0 else ((index // 2) % (num_clients - 1)) + 1
            shared_prefix = client_id == 0
            input_toks = 4096 if shared_prefix else local_rng.randrange(256, 768)
            output_toks = local_rng.randrange(64, 192)
        elif scenario == "u3_loogle_proxy":
            client_id = index % num_clients
            input_toks = local_rng.randrange(4096, 12289)
            output_toks = local_rng.randrange(12, 20)
            shared_prefix = True

        if scenario == "u2_multiturn":
            # Session dependencies are materialized in make_agentic_records.
            continue

        records.append(_record(index, scenario, client_id, input_toks, output_toks,
                              arrival_ns, shared_prefix))

    return records


def make_agentic_records(num_reqs: int, num_clients: int, seed: int) -> list[dict]:
    records = []
    for session_index in range(num_reqs):
        client_id = session_index % num_clients
        sub_requests = []
        history = []
        for turn in range(3):
            rng = _rng_for(seed, "u2_multiturn", session_index * 3 + turn)
            user_len = rng.randrange(64, 160)
            output_len = rng.randrange(64, 192)
            input_len = 256 + turn * (user_len + output_len)
            input_ids = token_ids(8_000 + client_id * BLOCK_SIZE, input_len, session_index * 3 + turn, True)
            output_ids = token_ids(60_000 + client_id, output_len, session_index * 3 + turn, False)
            sub_requests.append({
                "input_toks": input_len,
                "output_toks": output_len,
                "input_tok_ids": input_ids,
                "output_tok_ids": output_ids,
                "tool_duration_ns": 100_000_000,
            })
            history.extend(input_ids)
            history.extend(output_ids)
        records.append({
            "session_id": f"u2-session-{session_index}",
            "client_id": client_id,
            "arrival_time_ns": session_index * 250_000_000,
            "sub_requests": sub_requests,
        })
    return records


def validate_records(records: list[dict], agentic: bool = False) -> dict:
    arrivals = [int(row["arrival_time_ns"]) for row in records]
    assert arrivals == sorted(arrivals), "arrival_time_ns must be non-decreasing"
    flat = [sub for row in records for sub in row["sub_requests"]] if agentic else records
    for row in flat:
        assert len(row["input_tok_ids"]) == int(row["input_toks"])
        assert len(row["output_tok_ids"]) == int(row["output_toks"])
        assert all(0 <= token < VOCAB_SIZE for token in row["input_tok_ids"] + row["output_tok_ids"])
    clients = sorted({int(row["client_id"]) for row in records})
    assert clients == list(range(len(clients))), f"client IDs are not dense: {clients}"
    return {
        "records": len(records),
        "requests": len(flat),
        "clients": len(clients),
        "duration_s": (arrivals[-1] / 1e9 if arrivals else 0.0),
        "input_mean": statistics.fmean([r["input_toks"] for r in flat]) if flat else 0.0,
        "output_mean": statistics.fmean([r["output_toks"] for r in flat]) if flat else 0.0,
        "status": "DELIVERED" if not agentic else "APPROX",
    }


def write_card(output_dir: Path, scenario: str, records: list[dict], agentic: bool, seed: int) -> None:
    card_dir = output_dir / "cards"
    card_dir.mkdir(exist_ok=True)
    card = validate_records(records, agentic)
    card.update({"workload": scenario, "seed": seed, "block_size": BLOCK_SIZE,
                 "vocabulary_limit": VOCAB_SIZE,
                 "limitation": "Synthetic stress workload; no paper dataset is claimed."})
    (card_dir / f"{scenario}.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, paths: list[Path], seed: int) -> None:
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workload", "path", "sha256", "seed", "status"])
        writer.writeheader()
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow({"workload": path.stem, "path": str(path.relative_to(output_dir)), "sha256": digest,
                             "seed": seed, "status": "DELIVERED"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="LLMServingSim/workloads/fairroute_hypothesis")
    parser.add_argument("--num-reqs", type=int, default=1000)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS),
                        help="comma-separated scenarios to generate")
    args = parser.parse_args()

    if args.num_reqs <= 0 or args.num_clients <= 1:
        raise SystemExit("--num-reqs must be positive and --num-clients must exceed 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")
    for scenario in scenarios:
        path = output_dir / f"{scenario}.jsonl"
        agentic = scenario == "u2_multiturn"
        records = (make_agentic_records(args.num_reqs, args.num_clients, args.seed)
                   if agentic else make_records(scenario, args.num_reqs, args.num_clients, args.seed))
        with path.open("w", encoding="utf-8") as workload:
            for record in records:
                workload.write(json.dumps(record, separators=(",", ":")) + "\n")
        validate_records(records, agentic)
        write_card(output_dir, scenario, records, agentic, args.seed)
        paths.append(path)
        print(path)
    write_manifest(output_dir, paths, args.seed)


if __name__ == "__main__":
    main()
