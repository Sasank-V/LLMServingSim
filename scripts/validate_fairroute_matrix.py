#!/usr/bin/env python3
"""Fail closed on incomplete or non-comparable FairRoute experiment matrices.

Run this after the simulator matrix and before opening an analysis notebook.
It checks paired factorial completeness, request-count equality, routing-sidecar
coverage, and predictor telemetry.  It intentionally does not compute paper
metrics: an invalid matrix must not be rescued by aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


PREDICTION_POLICIES = {"PRISM", "D2LPM", "FAIRROUTE", "FAIRROUTE_V2", "FAIRROUTE_GATED", "FAIRROUTE_SLACK"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sidecar(path: Path) -> list[dict]:
    candidates = [Path(f"{path}.sidecar.jsonl"), path.with_name(f"{path.stem}_sidecar.jsonl")]
    sidecar = next((candidate for candidate in candidates if candidate.is_file()), None)
    if sidecar is None:
        raise FileNotFoundError("routing sidecar missing")
    with sidecar.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity(path: Path, root: Path) -> tuple[str, str, str, str]:
    """Return workload, seed, router, and local scheduler from an output path."""
    if not path.name.endswith(".csv") or "_policy_" not in path.stem:
        raise ValueError("filename must end in _policy_<POLICY>[_local_<LOCAL>].csv")
    workload_name, policy_part = path.stem.rsplit("_policy_", 1)
    policy, separator, local = policy_part.partition("_local_")
    workload = str(path.relative_to(root).with_name(workload_name))
    seed_match = re.search(r"(?:seed_|_s)(\d+)", workload)
    if not seed_match:
        raise ValueError("workload path must include a seed directory or _s<seed> suffix")
    return workload, seed_match.group(1), policy, local if separator else "FCFS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--local-policies", nargs="+", default=["FCFS"])
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    problems: list[str] = []
    matrix: dict[tuple[str, str], dict[tuple[str, str], Path]] = defaultdict(dict)

    for path in sorted(root.rglob("*.csv")):
        if path.name.startswith(".") or path.parent.name.startswith("."):
            continue
        try:
            workload, seed, policy, local = _identity(path, root)
        except ValueError:
            continue
        key = (workload, seed)
        if (policy, local) in matrix[key]:
            problems.append(f"duplicate cell {key}: {policy}/{local}")
        matrix[key][policy, local] = path

    seen_seeds = {seed for _, seed in matrix}
    if len(seen_seeds) < args.min_seeds:
        problems.append(f"only {len(seen_seeds)} evaluation seed(s); need at least {args.min_seeds}")
    if "7" in seen_seeds:
        problems.append("tuning seed 7 appears in evaluation results")

    expected = {(policy, local) for policy in args.policies for local in args.local_policies}
    for key, cells in sorted(matrix.items()):
        missing = expected - cells.keys()
        extra = cells.keys() - expected
        if missing:
            problems.append(f"{key}: missing {sorted(missing)}")
        if extra:
            problems.append(f"{key}: unexpected {sorted(extra)}")
        request_ids: dict[tuple[str, str], set[str]] = {}
        for policy_local, path in cells.items():
            try:
                rows = _rows(path)
                ids = {row["request id"] for row in rows}
                if not rows or len(ids) != len(rows):
                    problems.append(f"{path}: empty output or duplicate request ids")
                request_ids[policy_local] = ids
                decisions = _sidecar(path)
                decision_ids = {str(row["request_id"]) for row in decisions}
                if ids != decision_ids:
                    problems.append(f"{path}: output and routing sidecar request ids differ")
                if policy_local[0] in PREDICTION_POLICIES:
                    missing_observations = [row["request_id"] for row in decisions if not row.get("prediction_available", False)]
                    if missing_observations:
                        problems.append(f"{path}: {len(missing_observations)} requests lack a valid prediction")
                    for row in decisions:
                        value = row.get("prediction_error_s")
                        if value is not None and not math.isfinite(float(value)):
                            problems.append(f"{path}: non-finite prediction error")
                            break
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"{path}: {exc}")
        if request_ids:
            reference = next(iter(request_ids.values()))
            for policy_local, ids in request_ids.items():
                if ids != reference:
                    problems.append(f"{key}: request set differs for {policy_local}")

    report = {"valid": not problems, "cells": len(matrix), "seeds": sorted(seen_seeds), "problems": problems}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
