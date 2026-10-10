#!/usr/bin/env python3
"""Convert FairRoute raw datasets into LLMServingSim JSONL workloads.

The converter is deliberately explicit about information that is not present in
the source data.  Text datasets are tokenized with the locally downloaded
Llama-3.1-8B tokenizer.  Azure/BurstGPT traces contain token counts but no
prompt text, so they are emitted as count-only requests unless
``--trace-token-ids synthetic`` is selected.

Examples:
    python workloads/datasets/convert_to_llmservingsim.py
    python workloads/datasets/convert_to_llmservingsim.py \
        --dataset sharegpt lmsys_chatbot_arena --emit sessions \
        --client-strategy session --sample-sessions 20000 --max-total-tokens 8192
    python workloads/datasets/convert_to_llmservingsim.py \
        --dataset azure_llm_inference burstgpt \
        --trace-token-ids synthetic --client-strategy weighted
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


DATASET_NAMES = (
    "sharegpt",
    "lmsys_chatbot_arena",
    "loogle",
    "arxiv_summarization",
    "code_reasoning",
    "azure_llm_inference",
    "burstgpt",
)


def _load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "The tokenizers package is required for text datasets. "
            "Install it with: pip install tokenizers"
        ) from exc
    tokenizer_path = path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing Llama tokenizer: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _llama_special(tokenizer: Any, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer does not define required token {token!r}")
    return int(value)


def _encode_turns(
    tokenizer: Any,
    turns: list[tuple[str, str]],
    think_median_s: float = 0.0,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Encode a conversation as exact Llama-3 agentic sub-requests."""
    bos = _llama_special(tokenizer, "<|begin_of_text|>")
    eot = _llama_special(tokenizer, "<|eot_id|>")
    history = [bos]
    result = []
    for user, assistant in turns:
        user_header = _token_ids(tokenizer, "<|start_header_id|>user<|end_header_id|>\n\n")
        assistant_header = _token_ids(tokenizer, "<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = history + user_header + _token_ids(tokenizer, user) + [eot] + assistant_header
        output = _token_ids(tokenizer, assistant) + [eot]
        tool_duration_ns = 0
        if think_median_s > 0 and rng is not None and len(turns) > len(result) + 1:
            tool_duration_ns = int(
                rng.lognormvariate(math.log(think_median_s), 0.8) * 1_000_000_000
            )
        result.append({
            "input_toks": len(prompt),
            "output_toks": len(output),
            "input_tok_ids": prompt,
            "output_tok_ids": output,
            "tool_duration_ns": tool_duration_ns,
        })
        history = prompt + output
    return result


def _arrivals_ns(
    count: int,
    rate_rps: float,
    rng: random.Random,
    mode: str,
    cv: float,
) -> list[int]:
    if count == 0:
        return []
    if rate_rps <= 0:
        raise ValueError("--rate-rps must be positive")
    if mode == "constant":
        gaps = [1.0 / rate_rps] * count
    elif mode == "gamma":
        if cv <= 0:
            raise ValueError("--arrival-cv must be positive")
        shape = 1.0 / (cv * cv)
        gaps = [rng.gammavariate(shape, (1.0 / rate_rps) / shape) for _ in range(count)]
    else:
        raise ValueError(f"Unsupported arrival mode: {mode}")
    origin = gaps[0]
    elapsed = 0.0
    result = []
    for gap in gaps:
        elapsed += gap
        result.append(round((elapsed - origin) * 1_000_000_000))
    return result


def _unique_ids(count: int, key: str, seed: int) -> list[int]:
    rng = random.Random((seed << 32) ^ zlib.crc32(key.encode()))
    return [rng.randrange(1000, 128000) for _ in range(count)]


def _reservoir(
    items: Iterable[Any],
    k: int,
    rng: random.Random,
) -> list[Any]:
    kept: list[Any] = []
    for index, item in enumerate(items):
        if index < k:
            kept.append(item)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < k:
            kept[replacement] = item
    return kept


def _in_split(key: str, split: str) -> bool:
    return split == "all" or (zlib.crc32(key.encode()) & 1) == (split == "test")


def _request(
    *,
    input_ids: list[int] | None,
    output_ids: list[int] | None,
    arrival_ns: int,
    client_id: int | str,
    source: str,
    source_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input_toks": len(input_ids) if input_ids is not None else 0,
        "output_toks": len(output_ids) if output_ids is not None else 0,
        "arrival_time_ns": max(0, int(arrival_ns)),
        "client_id": client_id,
        "source": source,
        "source_id": source_id,
    }
    if input_ids is not None:
        row["input_tok_ids"] = input_ids
    if output_ids is not None:
        row["output_tok_ids"] = output_ids
    if metadata:
        row.update(metadata)
    return row


def _prompt(history: list[tuple[str, str]], user_text: str) -> str:
    parts = ["<|begin_of_text|>"]
    for role, text in history:
        parts.append(f"{role}: {text}\n")
    parts.append(f"user: {user_text}\nassistant:")
    return "".join(parts)


def _paired_turns(
    conversations: list[dict[str, Any]],
) -> Iterator[tuple[int, str, str, str]]:
    history: list[tuple[str, str]] = []
    turn = 0
    for message in conversations:
        role = str(message.get("from", message.get("role", ""))).lower()
        text = message.get("value", message.get("content"))
        if not isinstance(text, str) or not text.strip():
            continue
        if role in {"human", "user"}:
            if history:
                history = history
            pending = text
        elif role in {"gpt", "assistant", "model"} and "pending" in locals():
            yield turn, _prompt(history, pending), text, str(turn)
            history.extend((("user", pending), ("assistant", text)))
            turn += 1
            del pending


def _client_id(
    strategy: str,
    *,
    dataset: str,
    record_id: str,
    index: int,
    num_clients: int,
    weights: list[float],
    rng: random.Random,
) -> int:
    if strategy == "source":
        return 0
    if strategy == "record":
        return index
    if strategy == "session":
        digest = hashlib.sha256(f"{dataset}:{record_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % num_clients
    if strategy == "field":
        digest = hashlib.sha256(f"{dataset}:{record_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % num_clients
    if strategy == "hash":
        digest = hashlib.sha256(f"{dataset}:{record_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % num_clients
    if strategy == "weighted":
        value = rng.random()
        total = 0.0
        for client, weight in enumerate(weights):
            total += weight
            if value < total:
                return client
        return len(weights) - 1
    return index % num_clients


def _parse_weights(raw: str | None, num_clients: int) -> list[float]:
    if not raw:
        return [1.0 / num_clients] * num_clients
    values = [float(value) for value in raw.split(",")]
    if len(values) != num_clients or any(value < 0 for value in values):
        raise ValueError("--client-weights must contain num_clients non-negative values")
    total = sum(values)
    if total <= 0:
        raise ValueError("--client-weights must have a positive sum")
    return [value / total for value in values]


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value


def _json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        finished = False
        started = False
        while not finished:
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                finished = True
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position == len(buffer):
                    break
                if not started:
                    if buffer[position] != "[":
                        raise ValueError(f"{path} must contain a top-level JSON array")
                    position += 1
                    started = True
                    continue
                if buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if finished:
                        raise ValueError(f"Invalid or truncated JSON array: {path}")
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object array item")
                yield value
                position = end
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) and buffer[position] == ",":
                    position += 1
                elif position < len(buffer) and buffer[position] != "]":
                    raise ValueError(f"Invalid separator in JSON array: {path}")
            if position:
                buffer = buffer[position:]
                position = 0


def _parquet(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for parquet datasets. Install it with: pip install pyarrow"
        ) from exc
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                yield row


def _timestamp_ns(raw: str, origin: datetime | None = None) -> int:
    value = raw.strip()
    if value.replace(".", "", 1).isdigit():
        return int(float(value) * 1_000_000_000)
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if origin is None:
        return int(timestamp.timestamp() * 1_000_000_000)
    return int((timestamp - origin).total_seconds() * 1_000_000_000)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_requests(
    dataset: str,
    records: Iterable[tuple[str, str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = (item for item in records if _in_split(item[0], args.split))
    materialized = (
        _reservoir(records, args.sample_records, rng)
        if args.sample_records
        else records
    )
    output = []
    for index, (record_id, prompt, answer) in enumerate(materialized):
        input_ids = _token_ids(tokenizer, prompt)
        output_ids = _token_ids(tokenizer, answer)
        if args.max_input_tokens and len(input_ids) > args.max_input_tokens:
            if args.truncate:
                input_ids = input_ids[-args.max_input_tokens:]
            else:
                args.dropped += 1
                continue
        if args.max_output_tokens and len(output_ids) > args.max_output_tokens:
            if args.truncate:
                output_ids = output_ids[:args.max_output_tokens]
            else:
                args.dropped += 1
                continue
        if args.max_total_tokens and len(input_ids) + len(output_ids) > args.max_total_tokens:
            if not args.truncate:
                args.dropped += 1
                continue
            input_ids = input_ids[-max(1, args.max_total_tokens - len(output_ids)):]
        client_record_id = record_id
        if args.client_strategy == "session":
            client_record_id = record_id.rsplit("/", 1)[0]
        client = _client_id(
            args.client_strategy,
            dataset=dataset,
            record_id=client_record_id,
            index=index,
            num_clients=args.num_clients,
            weights=args.client_weights,
            rng=rng,
        )
        output.append(
            _request(
                input_ids=input_ids,
                output_ids=output_ids,
                arrival_ns=0,
                client_id=client,
                source=dataset,
                source_id=record_id,
            )
        )
    arrivals = _arrivals_ns(
        len(output), args.rate_rps or 1.0, rng, args.arrival_mode, args.arrival_cv
    )
    for row, arrival in zip(output, arrivals):
        row["arrival_time_ns"] = arrival
    return output


def _session_requests(
    dataset: str,
    records: Iterable[tuple[str, list[tuple[str, str]], dict[str, Any]]],
    tokenizer: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = (item for item in records if _in_split(item[0], args.split))
    sessions = (
        _reservoir(records, args.sample_sessions, random.Random(args.seed))
        if args.sample_sessions
        else list(records)
    )
    field_map: dict[str, int] = {}
    if args.client_strategy == "field":
        field_values = sorted(
            {str(metadata.get(args.client_field, dataset)) for _, _, metadata in sessions}
        )
        field_map = {value: index for index, value in enumerate(field_values)}
        args.field_map = field_map
    arrivals = _arrivals_ns(
        len(sessions), args.rate_rps or 1.0, rng, args.arrival_mode, args.arrival_cv
    )
    output = []
    for index, (session_id, turns, metadata) in enumerate(sessions):
        if args.first_turn_only:
            turns = turns[:1]
        sub_requests = _encode_turns(
            tokenizer, turns, args.think_median_s, rng
        )
        total_tokens = sum(
            sub["input_toks"] + sub["output_toks"] for sub in sub_requests
        )
        if args.max_total_tokens and total_tokens > args.max_total_tokens:
            if not args.truncate:
                args.dropped += 1
                continue
            while sub_requests and total_tokens > args.max_total_tokens:
                sub_requests.pop()
                total_tokens = sum(
                    sub["input_toks"] + sub["output_toks"] for sub in sub_requests
                )
        if not sub_requests:
            args.dropped += 1
            continue
        client_key = session_id
        if field_map:
            client = field_map[str(metadata.get(args.client_field, dataset))]
        else:
            client = _client_id(
                args.client_strategy,
                dataset=dataset,
                record_id=client_key,
                index=index,
                num_clients=args.num_clients,
                weights=args.client_weights,
                rng=rng,
            )
        output.append({
            "session_id": session_id,
            "client_id": client,
            "arrival_time_ns": arrivals[index],
            "sub_requests": sub_requests,
            "source": dataset,
            "source_id": session_id,
            **metadata,
        })
    return output


def _sharegpt(root: Path, tokenizer: Any, args: argparse.Namespace, rng: random.Random):
    path = args.sharegpt_file or (
        root / "sharegpt" / "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    )
    conversations = _json_array(path)
    if args.emit == "sessions":
        records = (
            (
                str(item.get("id", f"record-{index}")),
                [
                    (
                        str(message.get("value", message.get("content", ""))),
                        str(item["conversations"][turn_index + 1].get("value", item["conversations"][turn_index + 1].get("content", ""))),
                    )
                    for turn_index, message in enumerate(item.get("conversations", [])[:-1])
                    if str(message.get("from", message.get("role", ""))).lower() in {"human", "user"}
                    and str(item["conversations"][turn_index + 1].get("from", item["conversations"][turn_index + 1].get("role", ""))).lower() in {"gpt", "assistant", "model"}
                ],
                {},
            )
            for index, item in enumerate(conversations)
        )
        return _session_requests("sharegpt", records, tokenizer, args, rng)
    records = (
        (f"{item.get('id', 'record')}/{turn_id}", prompt, answer)
        for item in conversations
        for turn, prompt, answer, turn_id in _paired_turns(item.get("conversations", []))
    )
    return _text_requests("sharegpt", records, tokenizer, args, rng)


def _lmsys(root: Path, tokenizer: Any, args: argparse.Namespace, rng: random.Random):
    paths = sorted((root / "lmsys_chatbot_arena" / "data").glob("*.parquet"))
    if args.emit == "sessions":
        records = []
        for item in _parquet(paths):
            side = args.side
            if side == "random":
                side = "conversation_a" if rng.randrange(2) == 0 else "conversation_b"
            messages = item.get(side) or []
            turns = []
            pending = None
            for message in messages:
                role, text = message.get("role"), message.get("content")
                if role == "user":
                    pending = text
                elif role == "assistant" and pending:
                    turns.append((pending, text))
                    pending = None
            records.append((
                str(item["question_id"]),
                turns,
                {"model": item.get("model_a" if side == "conversation_a" else "model_b", "unknown")},
            ))
        return _session_requests("lmsys_chatbot_arena", records, tokenizer, args, rng)
    records = []
    for item in _parquet(paths):
        sides = ("conversation_a", "conversation_b")
        if args.side in sides:
            sides = (args.side,)
        elif args.side == "random":
            sides = ("conversation_a" if rng.randrange(2) == 0 else "conversation_b",)
        for side in sides:
            messages = item.get(side) or []
            converted = [
                {"role": message.get("role"), "content": message.get("content")}
                for message in messages
            ]
            history: list[tuple[str, str]] = []
            pending = None
            for turn, message in enumerate(converted):
                role, text = message["role"], message["content"]
                if role == "user":
                    pending = text
                elif role == "assistant" and pending:
                    records.append(
                        (
                            f"{item['question_id']}/{side}/{turn}",
                            _prompt(history, pending),
                            text,
                        )
                    )
                    history.extend((("user", pending), ("assistant", text)))
                    pending = None
    return _text_requests("lmsys_chatbot_arena", records, tokenizer, args, rng)


def _loogle(root: Path, tokenizer: Any, args: argparse.Namespace, rng: random.Random):
    records = (
        (str(item["id"]), item["context"] + "\n\nQuestion: " + item["question"], item["answer"])
        for item in _jsonl(root / "loogle" / "data" / "summarization.jsonl")
    )
    return _text_requests("loogle", records, tokenizer, args, rng)


def _arxiv(root: Path, tokenizer: Any, args: argparse.Namespace, rng: random.Random):
    paths = sorted((root / "arxiv_summarization" / "data").glob("*.parquet"))
    records = (
        (f"{path.stem}/{index}", row["source"], row["summary"])
        for path in paths
        for index, row in enumerate(_parquet([path]))
        if row.get("source") and row.get("summary")
    )
    return _text_requests("arxiv_summarization", records, tokenizer, args, rng)


def _code_reasoning(root: Path, tokenizer: Any, args: argparse.Namespace, rng: random.Random):
    paths = sorted((root / "code_reasoning" / "data").glob("*.parquet"))
    records = (
        (
            f"{path.stem}/{index}",
            row["question"],
            row.get(args.code_output_field, ""),
        )
        for path in paths
        for index, row in enumerate(_parquet([path]))
        if row.get("question") and row.get(args.code_output_field)
    )
    return _text_requests("code_reasoning", records, tokenizer, args, rng)


def _trace_requests(
    dataset: str,
    rows: Iterable[tuple[str, int, int, int]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    materialized = (
        _reservoir(rows, args.sample_records, random.Random(args.seed))
        if args.sample_records
        else list(itertools.islice(rows, args.max_records))
        if args.max_records
        else list(rows)
    )
    materialized.sort(key=lambda row: row[1])
    output = []
    for index, (record_id, arrival_ns, input_toks, output_toks) in enumerate(materialized):
        if args.max_records and index >= args.max_records:
            break
        if input_toks <= 0 or (args.drop_zero_output and output_toks <= 0):
            args.dropped += 1
            continue
        if args.window_seconds and arrival_ns > args.window_seconds * 1_000_000_000:
            continue
        if args.max_input_tokens and input_toks > args.max_input_tokens:
            if args.truncate:
                input_toks = args.max_input_tokens
            else:
                args.dropped += 1
                continue
        if args.max_output_tokens and output_toks > args.max_output_tokens:
            if args.truncate:
                output_toks = args.max_output_tokens
            else:
                args.dropped += 1
                continue
        client = _client_id(
            args.client_strategy,
            dataset=dataset,
            record_id=record_id,
            index=index,
            num_clients=args.num_clients,
            weights=args.client_weights,
            rng=rng,
        )
        input_ids = output_ids = None
        if args.trace_token_ids == "synthetic":
            input_ids = _unique_ids(input_toks, f"{dataset}:{record_id}:input", args.seed)
            output_ids = _unique_ids(output_toks, f"{dataset}:{record_id}:output", args.seed)
        row = _request(
                input_ids=input_ids,
                output_ids=output_ids,
                arrival_ns=arrival_ns,
                client_id=client,
                source=dataset,
                source_id=record_id,
                metadata={"token_ids_kind": "synthetic" if input_ids is not None else "count_only"},
            )
        row["input_toks"] = input_toks
        row["output_toks"] = output_toks
        output.append(row)
    if args.rate_rps and args.arrival_mode != "replay":
        arrivals = _arrivals_ns(len(output), args.rate_rps, rng, args.arrival_mode, args.arrival_cv)
        for row, arrival in zip(output, arrivals):
            row["arrival_time_ns"] = arrival
    elif args.rate_rps and args.arrival_mode == "replay" and output:
        original = [row["arrival_time_ns"] for row in output]
        gaps = [max(0, current - previous) for previous, current in zip([original[0]] + original[:-1], original)]
        mean_gap = sum(gaps) / max(1, len(gaps))
        target_gap = 1_000_000_000 / args.rate_rps
        scale = target_gap / mean_gap if mean_gap else 1.0
        elapsed = 0
        for row, gap in zip(output, gaps):
            elapsed += round(gap * scale)
            row["arrival_time_ns"] = elapsed
    return output


def _azure(root: Path, args: argparse.Namespace, rng: random.Random):
    output = []
    for path in sorted((root / "azure_llm_inference").glob(args.azure_glob)):
        with path.open(newline="", encoding="utf-8") as handle:
            timestamps = (
                datetime.fromisoformat(row["TIMESTAMP"].replace("Z", "+00:00"))
                for row in csv.DictReader(handle)
            )
            origin = min(timestamps, default=None)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = (
                (
                    f"{path.stem}/{index}",
                    _timestamp_ns(row["TIMESTAMP"], origin),
                    int(row["ContextTokens"]),
                    int(row["GeneratedTokens"]),
                )
                for index, row in enumerate(csv.DictReader(handle))
            )
            output.extend(_trace_requests(
                "azure_llm_inference",
                rows,
                args,
                rng,
            ))
    return output


def _burstgpt(root: Path, args: argparse.Namespace, rng: random.Random):
    path = root / "burstgpt" / "BurstGPT_1.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle), None)
        origin = int(float(first["Timestamp"]) * 1_000_000_000) if first else 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = (
            (
                str(index),
                int(float(row["Timestamp"]) * 1_000_000_000) - origin,
                int(row["Request tokens"]),
                int(row["Response tokens"]),
            )
            for index, row in enumerate(csv.DictReader(handle))
        )
        return _trace_requests("burstgpt", rows, args, rng)


def _order(rows: list[dict[str, Any]], strategy: str, rng: random.Random) -> list[dict[str, Any]]:
    if strategy == "arrival":
        return sorted(rows, key=lambda row: (row["arrival_time_ns"], row["source_id"]))
    if strategy == "shuffle":
        rows = rows[:]
        rng.shuffle(rows)
        return rows
    if strategy == "client_interleave":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["client_id"]].append(row)
        result = []
        clients = sorted(groups)
        while clients:
            next_clients = []
            for client in clients:
                if groups[client]:
                    result.append(groups[client].pop(0))
                if groups[client]:
                    next_clients.append(client)
            clients = next_clients
        return result
    if strategy == "burst":
        return sorted(rows, key=lambda row: (row["arrival_time_ns"] // 1_000_000_000, row["arrival_time_ns"]))
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=Path("fairroute_workloads/data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("workloads/fairroute_converted"))
    parser.add_argument("--dataset", nargs="+", choices=("all",) + DATASET_NAMES, default=["all"])
    parser.add_argument(
        "--client-strategy",
        choices=("source", "record", "session", "round_robin", "hash", "weighted", "field"),
        default="round_robin",
        help="How client_id is assigned. session assigns whole sessions to sticky tenants.",
    )
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--client-weights", help="Comma-separated weights for weighted strategy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sample-records", type=int, default=0)
    parser.add_argument("--sample-sessions", type=int, default=0)
    parser.add_argument(
        "--first-turn-only",
        action="store_true",
        help="Keep only the first user/assistant turn in each emitted session.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "tune", "test"),
        default="all",
        help="Deterministic CRC32 split for disjoint tuning and test workloads.",
    )
    parser.add_argument("--emit", choices=("flat", "sessions"), default="flat")
    parser.add_argument("--side", choices=("both", "conversation_a", "conversation_b", "random"), default="both")
    parser.add_argument("--sharegpt-file", type=Path)
    parser.add_argument("--client-field", default="model")
    parser.add_argument("--code-output-field", choices=("solution", "r1_generation"), default="r1_generation")
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument(
        "--think-median-s",
        type=float,
        default=0.0,
        help="Median lognormal tool wait between session turns, in seconds.",
    )
    parser.add_argument(
        "--rate-rps",
        type=float,
        default=None,
        help="Target request rate. Text defaults to 1 rps; traces preserve native gaps unless set.",
    )
    parser.add_argument("--arrival-cv", type=float, default=1.0)
    parser.add_argument("--arrival-mode", choices=("gamma", "constant", "replay"), default="gamma")
    parser.add_argument("--drop-zero-output", action="store_true")
    parser.add_argument("--window-seconds", type=int, default=0)
    parser.add_argument("--azure-glob", default="*.csv")
    parser.add_argument(
        "--trace-token-ids",
        choices=("none", "synthetic"),
        default="none",
        help="Trace sources have no text. Synthetic IDs are placeholders, not tokenizer output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.num_clients < 1:
        raise SystemExit("--num-clients must be positive")
    args.client_weights = _parse_weights(args.client_weights, args.num_clients)
    datasets = list(DATASET_NAMES) if "all" in args.dataset else args.dataset
    root = args.input_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_datasets = {"sharegpt", "lmsys_chatbot_arena", "loogle", "arxiv_summarization", "code_reasoning"}
    tokenizer = _load_tokenizer(root / "llama31_8b_tokenizer") if text_datasets.intersection(datasets) else None
    rng = random.Random(args.seed)
    loaders = {
        "sharegpt": lambda: _sharegpt(root, tokenizer, args, rng),
        "lmsys_chatbot_arena": lambda: _lmsys(root, tokenizer, args, rng),
        "loogle": lambda: _loogle(root, tokenizer, args, rng),
        "arxiv_summarization": lambda: _arxiv(root, tokenizer, args, rng),
        "code_reasoning": lambda: _code_reasoning(root, tokenizer, args, rng),
        "azure_llm_inference": lambda: _azure(root, args, rng),
        "burstgpt": lambda: _burstgpt(root, args, rng),
    }
    manifest = {
        "tokenizer": (
            str(root / "llama31_8b_tokenizer" / "tokenizer.json")
            if tokenizer is not None
            else None
        ),
        "client_strategy": args.client_strategy,
        "num_clients": args.num_clients,
        "emit": args.emit,
        "seed": args.seed,
        "trace_token_ids": args.trace_token_ids,
        "argv": sys.argv[1:],
        "datasets": {},
    }
    args.dropped = 0
    for dataset in datasets:
        dataset_rng = random.Random((args.seed << 32) ^ zlib.crc32(dataset.encode()))
        rng = dataset_rng
        rows = loaders[dataset]()
        rows.sort(key=lambda row: (row["arrival_time_ns"], row.get("source_id", "")))
        output_path = args.output_dir / f"{dataset}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        digest = _sha256_file(output_path)
        manifest["datasets"][dataset] = {
            "records": len(rows), "path": str(output_path), "sha256": digest,
        }
        print(f"{dataset}: {len(rows):,} records -> {output_path}", file=sys.stderr)
    manifest["dropped_records"] = args.dropped
    manifest["field_map"] = getattr(args, "field_map", {})
    client_counts: dict[str, int] = defaultdict(int)
    for dataset in manifest["datasets"]:
        path = Path(manifest["datasets"][dataset]["path"])
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                client_counts[json.loads(line)["client_id"]] += 1
    with (args.output_dir / "client_map.csv").open("w", encoding="utf-8") as handle:
        handle.write("client_id,request_count\n")
        for client_id in sorted(client_counts):
            handle.write(f"{client_id},{client_counts[client_id]}\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
