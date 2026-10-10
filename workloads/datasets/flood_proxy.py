"""Deterministic synthetic flood/normal workload proxy.

The proxy intentionally contains no conversation text.  It is used for
scaling experiments where a shared prefix is the independent variable.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


TOKEN_MIN = 1000
TOKEN_MAX = 120000
BLOCK_SIZE = 16


def _rng(seed: int, purpose: str, client: int, index: int) -> random.Random:
    material = f"{seed}:{purpose}:{client}:{index}".encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _prefix(seed: int, client: int, length: int) -> list[int]:
    rng = _rng(seed, "prefix", client, 0)
    return [rng.randrange(TOKEN_MIN, TOKEN_MAX) for _ in range(length)]


def _lognormal_clipped(
    rng: random.Random, median: float, sigma: float, lower: int, upper: int
) -> int:
    value = rng.lognormvariate(math.log(median), sigma)
    return max(lower, min(upper, int(round(value))))


def _scaled_arrivals(
    seed: int, purpose: str, client: int, count: int, rate: float
) -> list[int]:
    if rate <= 0:
        raise ValueError("R must be positive")
    elapsed = 0.0
    arrivals: list[int] = []
    for index in range(count):
        # Exponential gaps preserve the requested Poisson process while the
        # content RNG remains independent of R.
        unit_gap = _rng(seed, f"{purpose}:arrival", client, index).expovariate(1.0)
        elapsed += unit_gap / rate
        arrivals.append(round(elapsed * 1_000_000_000))
    return arrivals


def build_flood(
    R: float,
    seed: int,
    N: int,
    out: str | Path,
    *,
    K: int = 16,
    flood_share: float = 0.35,
    flood_prefix: int = 4096,
    normal_prefix: int = 512,
) -> list[dict[str, Any]]:
    """Build a deterministic FLOOD proxy and write it as JSONL.

    ``R`` changes only arrival times.  For fixed ``seed`` and ``K``, all
    client assignments, lengths, and token ids are therefore identical across
    rates.  Counts are allocated with largest-remainder rounding so exactly
    ``N`` rows are emitted and the flooder share is stable to one request.
    """
    if K < 2:
        raise ValueError("K must be at least 2")
    if N < 1:
        raise ValueError("N must be positive")
    if not 0 < flood_share < 1:
        raise ValueError("flood_share must be between 0 and 1")
    if flood_prefix % BLOCK_SIZE or normal_prefix % BLOCK_SIZE:
        raise ValueError("prefix lengths must be multiples of 16")

    flood_count = int(round(N * flood_share))
    counts = [flood_count] + [
        (N - flood_count) // (K - 1) for _ in range(K - 1)
    ]
    for client in range((N - flood_count) % (K - 1)):
        counts[client + 1] += 1

    prefixes = [_prefix(seed, 0, flood_prefix)] + [
        _prefix(seed, client, normal_prefix) for client in range(1, K)
    ]
    rows: list[dict[str, Any]] = []
    for client, count in enumerate(counts):
        role = "flooder" if client == 0 else "normal"
        rate = R * flood_share if client == 0 else R * (1 - flood_share) / (K - 1)
        arrivals = _scaled_arrivals(seed, "flood", client, count, rate)
        for index, arrival in enumerate(arrivals):
            content_rng = _rng(seed, "content", client, index)
            suffix_len = _lognormal_clipped(content_rng, 100, 0.7, 8, 512)
            output_len = _lognormal_clipped(content_rng, 150, 0.8, 8, 1024)
            suffix = [
                content_rng.randrange(TOKEN_MIN, TOKEN_MAX)
                for _ in range(suffix_len)
            ]
            output = [
                content_rng.randrange(TOKEN_MIN, TOKEN_MAX)
                for _ in range(output_len)
            ]
            input_ids = prefixes[client] + suffix
            rows.append({
                "input_toks": len(input_ids),
                "output_toks": len(output),
                "arrival_time_ns": arrival,
                "input_tok_ids": input_ids,
                "output_tok_ids": output,
                "client_id": client,
                "role": role,
                "prefix_group": f"{role}_{client}",
                "source": "flood_proxy",
                "proxy": "PROXY",
                "source_id": f"{seed}:{client}:{index}",
            })

    rows.sort(key=lambda row: (row["arrival_time_ns"], row["source_id"]))
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return rows
