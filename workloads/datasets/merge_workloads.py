#!/usr/bin/env python3
"""Merge sorted flat workload JSONL files with bounded memory.

Each input is specified as PATH:CLIENT_OFFSET:ROLE. Rows are streamed twice:
the first pass finds the common duration, and the second pass performs a
heap-based merge without materialising all requests.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Iterator


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("parts", nargs="+", help="PATH:CLIENT_OFFSET:ROLE")
    return parser.parse_args()


def _part_spec(raw: str) -> tuple[Path, int, str]:
    path, offset, role = raw.rsplit(":", 2)
    return Path(path), int(offset), role


def _rows(path: Path, offset: int, role: str, end: int) -> Iterator[tuple[int, int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            arrival = int(row["arrival_time_ns"])
            if arrival > end:
                continue
            row["client_id"] += offset
            row["role"] = role
            yield arrival, int(row["client_id"]), row


def main() -> int:
    args = _arguments()
    specs = [_part_spec(raw) for raw in args.parts]
    ends: list[int] = []
    for path, _, _ in specs:
        last = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                last = int(json.loads(line)["arrival_time_ns"])
        if last is None:
            raise ValueError(f"Workload part is empty: {path}")
        ends.append(last)
    end = min(ends)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    clients: set[int] = set()
    with args.output.open("w", encoding="utf-8") as handle:
        streams = [_rows(*spec, end) for spec in specs]
        for _, _, row in heapq.merge(*streams):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
            clients.add(int(row["client_id"]))
    print(f"{args.output} {count} rows, {len(clients)} clients, duration {end / 1e9:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
