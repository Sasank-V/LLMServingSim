import hashlib
import json

from workloads.datasets.flood_proxy import build_flood


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_deterministic_and_rate_dilation(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    fast = tmp_path / "fast.jsonl"
    build_flood(2.0, 7, 1000, first, K=16)
    build_flood(2.0, 7, 1000, second, K=16)
    build_flood(4.0, 7, 1000, fast, K=16)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()

    slow_rows = _rows(first)
    fast_rows = _rows(fast)
    for slow, quick in zip(slow_rows, fast_rows):
        assert slow["client_id"] == quick["client_id"]
        assert slow["input_tok_ids"] == quick["input_tok_ids"]
        assert slow["output_tok_ids"] == quick["output_tok_ids"]
        assert abs(quick["arrival_time_ns"] / slow["arrival_time_ns"] - 0.5) < 0.001


def test_aggregate_invariance_and_minimum_per_client(tmp_path):
    paths = []
    for clients in (4, 8, 16, 32, 64):
        path = tmp_path / f"K{clients}.jsonl"
        build_flood(1.0, 100, 10000, path, K=clients)
        paths.append(_rows(path))

    flood_shares = []
    flood_means = []
    for rows in paths:
        flood = [row for row in rows if row["role"] == "flooder"]
        flood_shares.append(len(flood) / len(rows))
        flood_means.append(sum(row["input_toks"] for row in flood) / len(flood))
        counts = {}
        for row in rows:
            counts[row["client_id"]] = counts.get(row["client_id"], 0) + 1
        assert min(counts.values()) >= 100
    assert max(flood_shares) - min(flood_shares) <= 0.01
    assert max(flood_means) / min(flood_means) <= 1.05


def test_ids_and_prefixes(tmp_path):
    path = tmp_path / "ids.jsonl"
    build_flood(1.0, 3, 100, path, K=4)
    rows = _rows(path)
    assert [row["arrival_time_ns"] for row in rows] == sorted(
        row["arrival_time_ns"] for row in rows
    )
    for row in rows:
        assert len(row["input_tok_ids"]) == row["input_toks"]
        assert len(row["output_tok_ids"]) == row["output_toks"]
        assert all(0 <= token < 128256 and token < 128000 for token in row["input_tok_ids"])
        assert all(0 <= token < 128256 and token < 128000 for token in row["output_tok_ids"])
        prefix_length = 4096 if row["client_id"] == 0 else 512
        assert prefix_length % 16 == 0
