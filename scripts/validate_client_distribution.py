#!/usr/bin/env python3
"""Validate client_id balance in an LLMServingSim JSONL workload."""

import argparse
import json
import math
from collections import Counter


def gini(values):
    values = sorted(float(value) for value in values)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2 * weighted) / (len(values) * total) - (len(values) + 1) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", help="JSONL workload to inspect")
    parser.add_argument("--expected-clients", type=int, default=None)
    args = parser.parse_args()

    counts = Counter()
    total = 0
    with open(args.workload, encoding="utf-8") as workload:
        for line_number, line in enumerate(workload, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON at line {line_number}: {error}")
            if "client_id" not in record:
                continue
            counts[str(record["client_id"])] += 1
            total += 1

    print(f"requests with client_id: {total}")
    print(f"distinct clients: {len(counts)}")
    if not counts:
        print("No client_id values found; FairRoute fairness is not measurable.")
        return 1

    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probabilities)
    print(f"Shannon entropy: {entropy:.3f} bits")
    print(f"normalized entropy: {entropy / math.log(len(counts), 2) if len(counts) > 1 else 1.0:.3f}")
    print(f"Gini imbalance: {gini(counts.values()):.3f}")
    print(f"min/max client requests: {min(counts.values())}/{max(counts.values())}")
    for client_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  client {client_id}: {count} ({100 * count / total:.2f}%)")

    if args.expected_clients is not None:
        missing = sorted(set(map(str, range(args.expected_clients))) - set(counts))
        if missing:
            print(f"missing expected client IDs: {', '.join(missing)}")
            return 2

    if len(counts) > 1 and gini(counts.values()) > 0.45:
        print("Recommendation: use a lower Zipf alpha or a balanced/stratified client assignment")
        print("for policy comparisons; retain this skewed trace as a separate stress test.")
    else:
        print("Distribution is suitable for a balanced policy comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
