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
  pip install datasets transformers numpy
  python generate_workload.py \
      --num-reqs 500 \
      --sps 10.0 \
      --output workload-llama8b-500-sps10.jsonl
Optional flags:
  --seed              RNG seed (default: 42)
  --max-sessions      Cap on HF rows to load (default: 5000)
  --min-input-toks    Minimum input token count (default: 4)
  --max-input-toks    Maximum input token count (default: 8192)
  --min-output-toks   Minimum output token count (default: 1)
  --max-output-toks   Maximum output token count (default: 8192)
  --max-kv-toks       Maximum input+output combined (default: 8192)
  --first-arrival-sec Offset (seconds) for first arrival (default: 0)
  --add-client-ids    Add a "client_id" field with Zipfian skew
  --num-clients       Number of synthetic clients (default: 5)
  --zipf-alpha        Zipfian skew parameter (default: 1.1)
    --client-distribution  Client assignment: zipf or balanced (default: zipf)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_rows(
    max_sessions: int,
    seed: int,
    dataset_dir: str = "./nebius_dataset_cache",
    hf_token: str | None = None,
) -> list[dict]:
    """Load rows from local disk cache if present, else download from HF and save locally."""
    from datasets import load_dataset, load_from_disk

    cache_path = Path(dataset_dir)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    if cache_path.exists() and (cache_path / "dataset_info.json").exists():
        print(f"Loading local dataset from: {cache_path} ...")
        ds = load_from_disk(str(cache_path))
    else:
        print(f"Downloading dataset from HuggingFace ({DATASET_ID}) ...")
        ds = load_dataset(DATASET_ID, split="train", **kwargs)
        print(f"Saving dataset locally to: {cache_path} (for future offline runs) ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_path))
        print("  Dataset saved locally.")

    total_rows = len(ds)
    limit = min(max_sessions, total_rows) if max_sessions > 0 else total_rows

    rows: list[dict] = []
    pbar = tqdm(total=limit, desc="Loading dataset rows", unit="row")
    for i, row in enumerate(ds):
        if max_sessions > 0 and i >= max_sessions:
            break
        rows.append(row)
        pbar.update(1)
    pbar.close()

    print(f"  Loaded {len(rows)} rows.")
    return rows


# ---------------------------------------------------------------------------
# Conversation -> (input_text, output_text) extraction
# ---------------------------------------------------------------------------


def extract_turns(rows: list[dict]) -> list[tuple[str, str]]:
    """
    Extract (input_text, output_text) pairs from the dataset.

    For single-turn rows: input = conversation[0].content,
                          output = generated_message.content.

    For multi-turn rows: input = all prior turns concatenated
                         (preserving shared-prefix structure),
                         output = generated_message.content.
    """
    turns: list[tuple[str, str]] = []

    for row in rows:
        conversation = row.get("conversation", [])
        generated = row.get("generated_message", {})
        output_text = generated.get("content", "")
        finish_reason = row.get("finish_reason", "")

        # Skip rows with empty output or non-stop finishes
        if not output_text or not output_text.strip():
            continue
        if finish_reason != "stop":
            continue

        # Build the full input text from the conversation history.
        # This uses Llama-3.1-Instruct chat template tokens so the
        # tokenizer produces the same IDs as a real serving request.
        input_parts: list[str] = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                input_parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                )

        if not input_parts:
            continue

        # The full prompt includes the conversation + assistant header
        input_text = (
            "<|begin_of_text|>"
            + "".join(input_parts)
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        turns.append((input_text, output_text))

    return turns


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_turns(
    turns: list[tuple[str, str]],
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
    max_kv: int,
    tokenizer_dir: str = "./llama_tokenizer_cache",
    hf_token: str | None = None,
    checkpoint_file: str | None = None,
    checkpoint_interval: int = 10,
    num_workers: int = 0,
    load_results: bool = True,
) -> list[dict[str, Any]]:
    """
    Tokenize (input_text, output_text) pairs using the Llama-3.1-8B
    tokenizer and filter by length constraints.

    Returns a list of dicts with:
      - input_tok_ids: list[int]
      - output_tok_ids: list[int]
      - input_toks: int
      - output_toks: int
    """
    from transformers import AutoTokenizer

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    tok_path = Path(tokenizer_dir)
    if tok_path.exists() and (tok_path / "tokenizer_config.json").exists():
        print(f"Loading local tokenizer from: {tok_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
    else:
        print(f"Loading tokenizer from HuggingFace ({MODEL_ID}) ...")
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
        print(f"Saving tokenizer locally to: {tok_path} (for future offline runs) ...")
        tok_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_path))

    print(f"  Vocab size: {tokenizer.vocab_size}")

    checkpoint_path = Path(checkpoint_file) if checkpoint_file else None
    checkpoint_config = {
        "tokenizer_dir": str(tok_path),
        "min_input": min_input,
        "max_input": max_input,
        "min_output": min_output,
        "max_output": max_output,
        "max_kv": max_kv,
        "turn_count": len(turns),
    }

    start_index = 0
    checkpoint_handle = None
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading tokenization checkpoint from: {checkpoint_path} ...")
        with checkpoint_path.open("rb+") as checkpoint_input:
            header = json.loads(checkpoint_input.readline())
            if header.get("config") != checkpoint_config:
                raise ValueError(
                    "Checkpoint settings do not match this run. "
                    f"Delete {checkpoint_path} to start over."
                )
            last_valid_offset = checkpoint_input.tell()
            while line := checkpoint_input.readline():
                if not line.endswith(b"\n"):
                    checkpoint_input.seek(last_valid_offset)
                    checkpoint_input.truncate()
                    break
                record = json.loads(line)
                if record["index"] != start_index:
                    raise ValueError(
                        f"Invalid tokenization checkpoint: {checkpoint_path}"
                    )
                start_index += 1
                last_valid_offset = checkpoint_input.tell()
        print(f"  Resuming after {start_index}/{len(turns)} conversions.")

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_handle = checkpoint_path.open("a")
        if start_index == 0:
            checkpoint_handle.write(json.dumps({"config": checkpoint_config}) + "\n")
            checkpoint_handle.flush()

    skipped_short_input = 0
    skipped_long_input = 0
    skipped_short_output = 0
    skipped_long_output = 0
    skipped_kv = 0

    def convert_turn(turn: tuple[str, str]) -> tuple[dict[str, Any] | None, str | None]:
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

    workers = num_workers or min(32, (os.cpu_count() or 1) + 4)
    batch_size = max(checkpoint_interval, checkpoint_interval * workers)
    remaining = turns[start_index:]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        progress = tqdm(
            total=len(turns),
            initial=start_index,
            desc="Tokenizing & filtering turns",
            unit="turn",
        )
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]
            batch_results = list(executor.map(convert_turn, batch))
            for offset, (result, skip_reason) in enumerate(batch_results):
                index = start_index + batch_start + offset
                if skip_reason == "short_input":
                    skipped_short_input += 1
                elif skip_reason == "long_input":
                    skipped_long_input += 1
                elif skip_reason == "short_output":
                    skipped_short_output += 1
                elif skip_reason == "long_output":
                    skipped_long_output += 1
                elif skip_reason == "kv_overflow":
                    skipped_kv += 1
                if checkpoint_handle:
                    checkpoint_handle.write(
                        json.dumps({"index": index, "result": result}) + "\n"
                    )
                progress.update(1)
            if checkpoint_handle:
                checkpoint_handle.flush()
            del batch_results
            del batch
        progress.close()
    if checkpoint_handle:
        checkpoint_handle.close()

    if not checkpoint_path:
        raise ValueError("checkpoint_file is required for resumable tokenization")
    if not load_results:
        return []

    valid_results: list[dict[str, Any]] = []
    with checkpoint_path.open() as checkpoint_input:
        next(checkpoint_input)
        for line in checkpoint_input:
            result = json.loads(line)["result"]
            if result is not None:
                valid_results.append(result)
    print(f"  Tokenized {len(valid_results)} valid turns from {len(turns)} total.")
    print(
        f"  Skipped: short_input={skipped_short_input}, "
        f"long_input={skipped_long_input}, "
        f"short_output={skipped_short_output}, "
        f"long_output={skipped_long_output}, "
        f"kv_overflow={skipped_kv}"
    )

    return valid_results


# ---------------------------------------------------------------------------
# Arrival time generation (Poisson process)
# ---------------------------------------------------------------------------


def generate_arrival_times(
    num_reqs: int,
    sps: float,
    first_arrival_sec: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Generate Poisson-distributed arrival times in nanoseconds.

    Args:
        num_reqs: Number of requests to generate arrivals for.
        sps: Arrival rate in requests per second.
        first_arrival_sec: Offset (seconds) for the first arrival.
        rng: NumPy random generator.

    Returns:
        Sorted list of arrival times in nanoseconds.
    """
    # Inter-arrival times are exponentially distributed with rate = sps
    inter_arrivals = rng.exponential(scale=1.0 / sps, size=num_reqs)

    # Convert to cumulative arrival times
    arrival_times_sec = np.cumsum(inter_arrivals)

    # Add the first-arrival offset
    arrival_times_sec += first_arrival_sec

    # Convert to nanoseconds (int64 to match LLMServingSim format)
    arrival_times_ns = (arrival_times_sec * NS_PER_SEC).astype(np.int64)

    return arrival_times_ns.tolist()


# ---------------------------------------------------------------------------
# Client ID assignment (Zipfian distribution for fairness experiments)
# ---------------------------------------------------------------------------


def assign_client_ids(
    num_reqs: int,
    num_clients: int,
    zipf_alpha: float,
    rng: np.random.Generator,
    distribution: str = "zipf",
) -> list[int]:
    """
    Assign client IDs following a Zipfian distribution.
    Client 0 sends the most requests, Client num_clients-1 the fewest.

    Args:
        num_reqs: Number of requests.
        num_clients: Number of synthetic clients.
        zipf_alpha: Zipfian skew parameter (>1 = more skewed).
        rng: NumPy random generator.

    Returns:
        List of client IDs (0-indexed), one per request.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if distribution == "balanced":
        client_ids = np.arange(num_reqs, dtype=np.int64) % num_clients
        rng.shuffle(client_ids)
        return client_ids.tolist()
    if distribution != "zipf":
        raise ValueError(f"unknown client distribution: {distribution}")

    # Compute Zipfian weights: rank-1 gets highest probability
    ranks = np.arange(1, num_clients + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, zipf_alpha)
    probabilities = weights / weights.sum()

    # Sample client IDs
    client_ids = rng.choice(num_clients, size=num_reqs, p=probabilities)
    return client_ids.tolist()


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_workload(args: argparse.Namespace) -> None:
    """Generate the LLMServingSim-compatible workload JSONL file."""
    rng = np.random.default_rng(args.seed)

    # ── Step 1: Load dataset ──────────────────────────────────────────
    rows = load_dataset_rows(
        args.max_sessions,
        args.seed,
        dataset_dir=args.dataset_dir,
        hf_token=args.hf_token,
    )

    # ── Step 2: Extract turns ─────────────────────────────────────────
    turns = extract_turns(rows)
    print(f"  Extracted {len(turns)} conversation turns.")

    if len(turns) == 0:
        print(
            "ERROR: No valid turns extracted. Check dataset format.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 3: Tokenize and filter ───────────────────────────────────
    tokenized = tokenize_turns(
        turns,
        min_input=args.min_input_toks,
        max_input=args.max_input_toks,
        min_output=args.min_output_toks,
        max_output=args.max_output_toks,
        max_kv=args.max_kv_toks,
        tokenizer_dir=args.tokenizer_dir,
        hf_token=args.hf_token,
        checkpoint_file=args.checkpoint_file
        or f"{args.output}.tokenize.checkpoint.jsonl",
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
    )
    del rows
    del turns

    if len(tokenized) == 0:
        print(
            "ERROR: No turns passed the length filters. Relax the constraints.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 4: Select requests ────────────────────────────────────────
    if args.use_all:
        # Use every row that survived filtering — no sampling
        selected = tokenized
        num_reqs = len(selected)
        print(f"  --use-all: emitting all {num_reqs} valid turns.")
    else:
        num_reqs = args.num_reqs
        if num_reqs is None:
            print(
                "ERROR: Specify --num-reqs N or --use-all.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(tokenized) >= num_reqs:
            indices = rng.choice(len(tokenized), size=num_reqs, replace=False)
        else:
            print(
                f"  WARNING: Only {len(tokenized)} valid turns "
                f"available, but {num_reqs} requested. "
                f"Sampling with replacement."
            )
            indices = rng.choice(len(tokenized), size=num_reqs, replace=True)
        selected = [tokenized[i] for i in indices]

    # ── Step 5: Generate Poisson arrival times ────────────────────────
    arrival_times = generate_arrival_times(
        num_reqs=num_reqs,
        sps=args.sps,
        first_arrival_sec=args.first_arrival_sec,
        rng=rng,
    )

    # ── Step 6: Optionally assign client IDs ──────────────────────────
    client_ids = None
    if args.add_client_ids:
        client_ids = assign_client_ids(
            num_reqs=num_reqs,
            num_clients=args.num_clients,
            zipf_alpha=args.zipf_alpha,
            rng=rng,
            distribution=args.client_distribution,
        )
    # ── Step 7: Write JSONL ───────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_lens = [s["input_toks"] for s in selected]
    output_lens = [s["output_toks"] for s in selected]

    with open(output_path, "w") as f:
        for i in range(num_reqs):
            record: dict[str, Any] = {
                "input_toks": selected[i]["input_toks"],
                "output_toks": selected[i]["output_toks"],
                "arrival_time_ns": arrival_times[i],
                "input_tok_ids": selected[i]["input_tok_ids"],
                "output_tok_ids": selected[i]["output_tok_ids"],
            }
            # client_id is NOT part of the core LLMServingSim spec,
            # but is included as metadata for FairRoute's fairness
            # experiments. LLMServingSim ignores unknown fields.
            if client_ids is not None:
                record["client_id"] = client_ids[i]

            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            selected[i] = None

    # ── Summary statistics ────────────────────────────────────────────
    del tokenized
    total_duration_sec = arrival_times[-1] / NS_PER_SEC

    print(f"\n{'=' * 60}")
    print(f"  Output: {output_path}")
    print(f"  Requests: {num_reqs}")
    print(f"  Arrival rate: {args.sps} req/s (Poisson)")
    print(f"  Total duration: {total_duration_sec:.1f} sec")
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
        from collections import Counter

        counts = Counter(client_ids)
        print(
            f"  Client distribution ({args.client_distribution}, Zipf alpha={args.zipf_alpha}):"
        )
        for cid in sorted(counts.keys()):
            pct = counts[cid] / num_reqs * 100
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

    # Required
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

    # Optional
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

    # Length filters
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

    # FairRoute client assignment
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
