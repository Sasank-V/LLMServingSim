#!/usr/bin/env python3
"""Produce seed-paired, routing-sensitive FairRoute summaries from CSV output."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from validate_fairroute_matrix import _identity, _rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def bootstrap_mean(values: list[float], rng: random.Random, draws: int = 10000) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return mean, samples[int(draws * .025)], samples[int(draws * .975)]


def load(root: Path) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    runs = {}
    for path in root.rglob("*.csv"):
        try:
            workload, seed, policy, local = _identity(path, root)
        except ValueError:
            continue
        runs[workload, seed, policy, local] = _rows(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--isolated-results", type=Path, required=True,
                        help="paired one-request/zero-queue baselines; never substitute RR here")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    runs = load(args.results.resolve())
    isolated = load(args.isolated_results.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = {}
    for (workload, seed, _, _), rows in isolated.items():
        for row in rows:
            reference[workload, seed, row["request id"]] = max(1.0, float(row["latency"]))

    per_run, per_client = [], []
    for (workload, seed, policy, local), rows in sorted(runs.items()):
        slowdowns = []
        clients = defaultdict(list)
        ttfts = []
        for row in rows:
            baseline = reference.get((workload, seed, row["request id"]))
            if baseline is None:
                raise SystemExit(f"missing isolated baseline for {workload}/seed={seed}/request={row['request id']}")
            latency_ms = float(row["latency"]) / 1e6
            ttft_ms = float(row["TTFT"]) / 1e6
            slowdown = float(row["latency"]) / baseline
            slowdowns.append(slowdown)
            clients[row["client id"]].append(slowdown)
            ttfts.append(ttft_ms)
        client_p99 = {client: percentile(values, .99) for client, values in clients.items()}
        client_means = [sum(values) / len(values) for values in clients.values()]
        mean_client = sum(client_means) / len(client_means)
        cov = (math.sqrt(sum((value - mean_client) ** 2 for value in client_means) / len(client_means)) / mean_client
               if mean_client else 0.0)
        worst_ratio = max(client_p99.values()) / max(1e-12, percentile(list(client_p99.values()), .5))
        for client, values in clients.items():
            per_client.append({"workload": workload, "seed": seed, "policy": policy, "local_policy": local,
                               "client_id": client, "n": len(values), "slowdown_mean": sum(values) / len(values),
                               "slowdown_p99": percentile(values, .99)})
        per_run.append({"workload": workload, "seed": seed, "policy": policy, "local_policy": local,
                        "n": len(rows), "ttft_p50_ms": percentile(ttfts, .50),
                        "ttft_p95_ms": percentile(ttfts, .95), "ttft_p99_ms": percentile(ttfts, .99),
                        "ttft_mean_ms": sum(ttfts) / len(ttfts),
                        "slowdown_p99": percentile(slowdowns, .99), "slowdown_mean": sum(slowdowns) / len(slowdowns),
                        "client_slowdown_cov": cov, "worst_client_p99_ratio": worst_ratio})

    grouped = defaultdict(list)
    for row in per_run:
        grouped[row["workload"], row["policy"], row["local_policy"]].append(row)
    rng = random.Random(args.bootstrap_seed)
    aggregate = []
    for key, rows in sorted(grouped.items()):
        for metric in ("ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "ttft_mean_ms",
                       "slowdown_p99", "client_slowdown_cov", "worst_client_p99_ratio"):
            mean, lo, hi = bootstrap_mean([row[metric] for row in rows], rng)
            aggregate.append({"workload": key[0], "policy": key[1], "local_policy": key[2], "metric": metric,
                              "seed_count": len(rows), "mean": mean, "ci95_low": lo, "ci95_high": hi})

    for name, rows in (("per_run.csv", per_run), ("per_client.csv", per_client), ("seed_summary.csv", aggregate)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
