#!/usr/bin/env python3
"""
Generate LLMServingSim-compatible workload JSONL from the Nebius
Llama-3.1-8B-Instruct-Infinity-Instruct-0625 HuggingFace dataset.

Output format (one JSON object per line):
  {
    "input_toks":      <int>,
    "output_toks":     <int>,
    "arrival_time_ns": <int>,
    "input_tok_ids":   [<int>, ...],
    "output_tok_ids":  [<int>, ...]
  }

This matches the "flat requests" format documented in:
  https://github.com/casys-kaist/LLMServingSim/blob/main/workloads/README.md

Usage:
  pip install datasets transformers numpy tqdm
  python generate_workload.py \
      --num-reqs 500 \
      --sps 10.0 \
      --output workload-llama8b-500-sps10.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "nebius/Llama-3.1-8B-Instruct-Infinity-Instruct-0625"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
NS_PER_SEC = 1_000_000_000


# ---------------------------------------------------------------------------
# Dataset & Turn Extraction
# ---------------------------------------------------------------------------


def extract_turn_from_row(row: dict) -> tuple[str, str] | None:
    """Extract (input_text, output_text) pair from a single dataset row."""
    conversation = row.get("conversation", [])
    generated = row.get("generated_message", {})
    output_text = generated.get("content", "")
    finish_reason = row.get("finish_reason", "")

    if not output_text or not output_text.strip():
        return None
    if finish_reason != "stop":
        return None

    input_parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            input_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

    if not input_parts:
        return None

    input_text = (
        "<|begin_of_text|>"
        + "".join(input_parts)
        + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return input_text, output_text


def load_dataset(dataset_dir: str, hf_token: str | None = None):
    from datasets import load_dataset as hf_load_dataset, load_from_disk

    cache_path = Path(dataset_dir)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    if cache_path.exists() and (cache_path / "dataset_info.json").exists():
        print(f"Loading local dataset from: {cache_path} ...")
        return load_from_disk(str(cache_path))
    else:
        print(f"Downloading dataset from HuggingFace ({DATASET_ID}) ...")
        ds = hf_load_dataset(DATASET_ID, split="train", **kwargs)
        print(f"Saving dataset locally to: {cache_path} (for future offline runs) ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_path))
        print("  Dataset saved locally.")
        return ds


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def get_tokenizer(tokenizer_dir: str, hf_token: str | None = None):
    from transformers import AutoTokenizer

    tok_path = Path(tokenizer_dir)
    if tok_path.exists() and (tok_path / "tokenizer_config.json").exists():
        print(f"Loading local tokenizer from: {tok_path} ...")
        return AutoTokenizer.from_pretrained(str(tok_path))
    else:
        print(f"Loading tokenizer from HuggingFace ({MODEL_ID}) ...")
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
        print(f"Saving tokenizer locally to: {tok_path} (for future offline runs) ...")
        tok_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_path))
        return tokenizer


def tokenize_turn(
    turn: tuple[str, str],
    tokenizer: Any,
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
    max_kv: int,
) -> tuple[dict[str, Any] | None, str | None]:
    input_text, output_text = turn
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)

    n_in = len(input_ids)
    n_out = len(output_ids)

    if n_in < min_input:
        return None, "short_input"
    if n_in > max_input:
        return None, "long_input"
    if n_out < min_output:
        return None, "short_output"
    if n_out > max_output:
        return None, "long_output"
    if n_in + n_out > max_kv:
        return None, "kv_overflow"

    return {
        "input_tok_ids": input_ids,
        "output_tok_ids": output_ids,
        "input_toks": n_in,
        "output_toks": n_out,
    }, None


# ---------------------------------------------------------------------------
# Arrival time generation (Poisson process)
# ---------------------------------------------------------------------------


def generate_arrival_times(
    num_reqs: int,
    sps: float,
    first_arrival_sec: float,
    rng: np.random.Generator,
) -> list[int]:
    inter_arrivals = rng.exponential(scale=1.0 / sps, size=num_reqs)
    arrival_times_sec = np.cumsum(inter_arrivals) + first_arrival_sec
    arrival_times_ns = (arrival_times_sec * NS_PER_SEC).astype(np.int64)
    return arrival_times_ns.tolist()


# ---------------------------------------------------------------------------
# Client ID assignment (Zipfian distribution)
# ---------------------------------------------------------------------------


def assign_client_ids(
    num_reqs: int,
    num_clients: int,
    zipf_alpha: float,
    rng: np.random.Generator,
    distribution: str = "zipf",
) -> list[int]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if distribution == "balanced":
        client_ids = np.arange(num_reqs, dtype=np.int64) % num_clients
        rng.shuffle(client_ids)
        return client_ids.tolist()
    if distribution != "zipf":
        raise ValueError(f"unknown client distribution: {distribution}")

    ranks = np.arange(1, num_clients + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, zipf_alpha)
    probabilities = weights / weights.sum()
    client_ids = rng.choice(num_clients, size=num_reqs, p=probabilities)
    return client_ids.tolist()


# ---------------------------------------------------------------------------
# Main generation logic (Streamed & O(1) Memory)
# ---------------------------------------------------------------------------


def generate_workload(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    # ── Step 1: Load dataset ──────────────────────────────────────────
    ds = load_dataset(args.dataset_dir, hf_token=args.hf_token)
    total_rows = len(ds)
    limit = min(args.max_sessions, total_rows) if args.max_sessions > 0 else total_rows
    print(f"Processing up to {limit} dataset rows...")

    # ── Step 2: Tokenize & Checkpoint in Streaming Batches ─────────────
    tokenizer = get_tokenizer(args.tokenizer_dir, hf_token=args.hf_token)
    checkpoint_path = Path(
        args.checkpoint_file or f"{args.output}.tokenize.checkpoint.jsonl"
    )

    checkpoint_config = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "min_input": args.min_input_toks,
        "max_input": args.max_input_toks,
        "min_output": args.min_output_toks,
        "max_output": args.max_output_toks,
        "max_kv": args.max_kv_toks,
        "limit": limit,
    }

    start_index = 0
    valid_count = 0
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        print(f"Checking existing tokenization checkpoint: {checkpoint_path} ...")
        with checkpoint_path.open("r", encoding="utf-8") as chk_in:
            _header = chk_in.readline()
            for line in chk_in:
                if not line.strip():
                    continue
                start_index += 1
                try:
                    rec = json.loads(line)
                    if rec.get("result") is not None:
                        valid_count += 1
                except json.JSONDecodeError:
                    break
        print(f"  Resuming from turn index {start_index}/{limit} ({valid_count} valid turns found).")

    if start_index < limit:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
        chk_handle = checkpoint_path.open("a", encoding="utf-8")
        if is_new:
            chk_handle.write(json.dumps({"config": checkpoint_config}) + "\n")
            chk_handle.flush()

        workers = args.num_workers or min(32, (os.cpu_count() or 1) + 4)
        batch_size = max(args.checkpoint_interval, args.checkpoint_interval * workers)
        print(f"Tokenizing remaining {limit - start_index} turns using {workers} workers...")

        pbar = tqdm(total=limit, initial=start_index, desc="Tokenizing turns", unit="turn")

        def process_row_idx(idx: int):
            row = ds[idx]
            turn = extract_turn_from_row(row)
            if turn is None:
                return idx, None, "extract_failed"
            res, reason = tokenize_turn(
                turn,
                tokenizer,
                args.min_input_toks,
                args.max_input_toks,
                args.min_output_toks,
                args.max_output_toks,
                args.max_kv_toks,
            )
            return idx, res, reason

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for b_start in range(start_index, limit, batch_size):
                b_end = min(limit, b_start + batch_size)
                batch_indices = range(b_start, b_end)
                batch_results = list(executor.map(process_row_idx, batch_indices))

                for idx, res, _reason in batch_results:
                    chk_handle.write(json.dumps({"index": idx, "result": res}) + "\n")
                    if res is not None:
                        valid_count += 1
                    pbar.update(1)
                chk_handle.flush()
                del batch_results
        pbar.close()
        chk_handle.close()

    print(f"Tokenization complete: {valid_count} valid turns available.")

    if valid_count == 0:
        print("ERROR: No turns passed length filters.", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Select requests ────────────────────────────────────────
    if args.use_all:
        num_reqs = valid_count
        selected_counts = None
        print(f"  --use-all: emitting all {num_reqs} valid turns.")
    else:
        num_reqs = args.num_reqs
        if num_reqs is None:
            print("ERROR: Specify --num-reqs N or --use-all.", file=sys.stderr)
            sys.exit(1)
        if valid_count >= num_reqs:
            indices = rng.choice(valid_count, size=num_reqs, replace=False)
        else:
            print(f"  WARNING: Only {valid_count} valid turns available, but {num_reqs} requested. Sampling with replacement.")
            indices = rng.choice(valid_count, size=num_reqs, replace=True)
        selected_counts = Counter(indices)

    # ── Step 4: Generate Arrival Times & Client IDs ────────────────────
    arrival_times = generate_arrival_times(
        num_reqs=num_reqs,
        sps=args.sps,
        first_arrival_sec=args.first_arrival_sec,
        rng=rng,
    )

    client_ids = None
    if args.add_client_ids:
        client_ids = assign_client_ids(
            num_reqs=num_reqs,
            num_clients=args.num_clients,
            zipf_alpha=args.zipf_alpha,
            rng=rng,
            distribution=args.client_distribution,
        )

    # ── Step 5: Write Output JSONL (Streamed) ─────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_lens: list[int] = []
    output_lens: list[int] = []

    print(f"Writing {num_reqs} JSONL records to {output_path} ...")
    with checkpoint_path.open("r", encoding="utf-8") as chk_file, open(
        output_path, "w", encoding="utf-8"
    ) as out_file:
        _header = chk_file.readline()
        valid_idx = 0
        req_idx = 0
        pbar = tqdm(total=num_reqs, desc="Writing JSONL output", unit="req")

        for line in chk_file:
            if req_idx >= num_reqs and args.use_all:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            res = rec.get("result")
            if res is None:
                continue

            repeat = 1
            if selected_counts is not None:
                repeat = selected_counts.get(valid_idx, 0)
            valid_idx += 1

            for _ in range(repeat):
                if req_idx >= num_reqs:
                    break
                out_rec = {
                    "input_toks": res["input_toks"],
                    "output_toks": res["output_toks"],
                    "arrival_time_ns": arrival_times[req_idx],
                    "input_tok_ids": res["input_tok_ids"],
                    "output_tok_ids": res["output_tok_ids"],
                }
                if client_ids is not None:
                    out_rec["client_id"] = client_ids[req_idx]

                out_file.write(json.dumps(out_rec, separators=(",", ":")) + "\n")
                input_lens.append(res["input_toks"])
                output_lens.append(res["output_toks"])
                req_idx += 1
                pbar.update(1)

        pbar.close()

    # ── Step 6: Summary Statistics ────────────────────────────────────
    total_duration_sec = arrival_times[-1] / NS_PER_SEC if arrival_times else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Output: {output_path}")
    print(f"  Requests: {len(input_lens)}")
    print(f"  Arrival rate: {args.sps} req/s (Poisson)")
    print(f"  Total duration: {total_duration_sec:.1f} sec")
    if input_lens:
        print(
            f"  Input tokens:  min={min(input_lens)}, "
            f"max={max(input_lens)}, "
            f"mean={np.mean(input_lens):.0f}, "
            f"median={np.median(input_lens):.0f}"
        )
        print(
            f"  Output tokens: min={min(output_lens)}, "
            f"max={max(output_lens)}, "
            f"mean={np.mean(output_lens):.0f}, "
            f"median={np.median(output_lens):.0f}"
        )
    if client_ids is not None:
        counts = Counter(client_ids)
        print(
            f"  Client distribution ({args.client_distribution}, Zipf alpha={args.zipf_alpha}):"
        )
        for cid in sorted(counts.keys()):
            pct = counts[cid] / len(client_ids) * 100
            print(f"    Client {cid}: {counts[cid]} requests ({pct:.1f}%)")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate LLMServingSim workload JSONL from the "
            "Nebius Llama-3.1-8B-Instruct-Infinity-Instruct-0625 "
            "dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--num-reqs",
        type=int,
        default=None,
        help="Number of requests to emit. Not needed with --use-all.",
    )
    p.add_argument(
        "--use-all",
        action="store_true",
        default=False,
        dest="use_all",
        help="Emit all rows that pass filters (ignores --num-reqs).",
    )
    p.add_argument(
        "--sps",
        type=float,
        required=True,
        help="Arrival rate (requests/sec). Poisson-distributed.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default="./nebius_dataset_cache",
        help="Local directory to cache/load downloaded HuggingFace dataset.",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=str,
        default="./llama_tokenizer_cache",
        help="Local directory to cache/load downloaded Llama tokenizer.",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token for gated access (defaults to $HF_TOKEN env var).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=5000,
        help="Cap on dataset rows to load (0 = no cap).",
    )
    p.add_argument(
        "--first-arrival-sec",
        type=float,
        default=0.0,
        help="Offset (seconds) added to first arrival time.",
    )
    p.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Tokenization checkpoint JSONL path (default: <output>.tokenize.checkpoint.jsonl).",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Flush the tokenization checkpoint after this many conversions.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Tokenizer worker threads (0 = choose automatically).",
    )

    p.add_argument(
        "--min-input-toks",
        type=int,
        default=4,
        help="Drop turns with input shorter than this.",
    )
    p.add_argument(
        "--max-input-toks",
        type=int,
        default=8192,
        help="Drop turns with input longer than this.",
    )
    p.add_argument(
        "--min-output-toks",
        type=int,
        default=1,
        help="Drop turns with output shorter than this.",
    )
    p.add_argument(
        "--max-output-toks",
        type=int,
        default=8192,
        help="Drop turns with output longer than this.",
    )
    p.add_argument(
        "--max-kv-toks",
        type=int,
        default=8192,
        help="Drop turns where input+output exceeds this.",
    )

    client_id_group = p.add_mutually_exclusive_group()
    client_id_group.add_argument(
        "--add-client-ids",
        action="store_true",
        dest="add_client_ids",
        help="Add Zipfian-distributed client_id to each request (default).",
    )
    client_id_group.add_argument(
        "--no-client-ids",
        action="store_false",
        dest="add_client_ids",
        help="Do not add client_id to request records.",
    )
    p.set_defaults(add_client_ids=True)
    p.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of synthetic clients (with --add-client-ids).",
    )
    p.add_argument(
        "--zipf-alpha",
        type=float,
        default=1.1,
        help="Zipfian skew parameter (with --add-client-ids).",
    )
    p.add_argument(
        "--client-distribution",
        choices=["zipf", "balanced"],
        default="zipf",
        help="Client assignment mode. Use balanced for policy comparisons and zipf for skew stress tests.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_workload(args)
