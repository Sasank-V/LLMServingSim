"""RAG workload builders from arXiv documents and ShareGPT first turns.

The document/question pairing is intentionally a content proxy: token IDs are
real but the question need not be semantically related to the document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


BLOCK = 16


def _rng(seed, *parts):
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def load_docs(arxiv_dir, tok, n_docs, doc_len, seed, max_scan=1500, shards=2):
    """Return distinct, block-aligned document heads."""
    import pyarrow.parquet as pq
    if doc_len % BLOCK:
        raise ValueError("doc_len must be a multiple of 16")
    pool, seen = [], set()
    for path in sorted(Path(arxiv_dir).glob("train-*.parquet"))[:shards]:
        parquet = pq.ParquetFile(path)
        column = next(name for name in ("source", "article", "text")
                      if name in parquet.schema_arrow.names)
        for batch in parquet.iter_batches(batch_size=64, columns=[column]):
            for document in batch.column(column).to_pylist():
                if not document:
                    continue
                ids = tok.encode(document, add_special_tokens=False).ids
                if len(ids) >= doc_len and tuple(ids[:BLOCK]) not in seen:
                    head = tuple(ids[:doc_len])
                    seen.add(head[:BLOCK])
                    pool.append(head)
                if len(pool) >= max_scan:
                    break
            if len(pool) >= max_scan:
                break
    if len(pool) < n_docs:
        raise RuntimeError(f"only {len(pool)} documents with >= {doc_len} tokens found; need {n_docs}")
    return [list(doc) for doc in _rng(seed, "docs").sample(pool, n_docs)]


def load_questions(sharegpt_json, tok, count, seed, q_min=8, q_max=96, out_min=8, out_cap=48):
    with open(sharegpt_json, encoding="utf-8") as handle:
        conversations = json.load(handle)
    order = list(range(len(conversations)))
    _rng(seed, "questions").shuffle(order)
    questions = []
    for index in order:
        turns = conversations[index].get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") not in ("human", "user"):
            continue
        question = tok.encode(str(turns[0].get("value", "")), add_special_tokens=False).ids
        answer = tok.encode(str(turns[1].get("value", "")), add_special_tokens=False).ids
        if len(question) >= q_min and len(answer) >= out_min:
            questions.append((question[:q_max], min(len(answer), out_cap), answer[:out_cap]))
        if len(questions) >= count:
            break
    if len(questions) < count:
        raise RuntimeError(f"only {len(questions)} usable questions; need {count}")
    return questions


def _gaps(seed, count):
    return [_rng(seed, "gap", index).expovariate(1.0) for index in range(count)]


def _write(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["arrival_time_ns"], row["source_id"]))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def build_rag_zipf(out, rate, seed, docs, questions, count=1000, clients=8, alpha=1.0, tag="RAGZIPF"):
    weights = [1.0 / (rank + 1) ** alpha for rank in range(len(docs))]
    elapsed, rows = 0.0, []
    for index, gap in enumerate(_gaps(seed, count)):
        elapsed += gap / rate
        document = _rng(seed, "doc", index).choices(range(len(docs)), weights)[0]
        question, output_length, output_ids = questions[index]
        input_ids = docs[document] + question
        rows.append({"input_toks": len(input_ids), "output_toks": output_length,
                     "arrival_time_ns": int(elapsed * 1e9), "input_tok_ids": input_ids,
                     "output_tok_ids": output_ids, "client_id": index % clients,
                     "prefix_group": f"doc{document}", "source": f"{tag}_proxy",
                     "proxy": "PROXY", "source_id": str(index)})
    return _write(rows, out)


def build_flood_pressure(out, rate, seed, docs, questions, count=1000, clients=8,
                         flood_share=0.35, normal_pool=7, tag="FLOODP"):
    if len(docs) < normal_pool + 1 or clients < 2:
        raise ValueError("need one flood document, normal_pool documents, and at least two clients")
    elapsed, rows = 0.0, []
    for index, gap in enumerate(_gaps(seed, count)):
        elapsed += gap / rate
        chooser = _rng(seed, "client", index)
        if chooser.random() < flood_share:
            client, document = 0, 0
        else:
            client, document = 1 + chooser.randrange(clients - 1), 1 + chooser.randrange(normal_pool)
        question, output_length, output_ids = questions[index]
        input_ids = docs[document] + question
        rows.append({"input_toks": len(input_ids), "output_toks": output_length,
                     "arrival_time_ns": int(elapsed * 1e9), "input_tok_ids": input_ids,
                     "output_tok_ids": output_ids, "client_id": client,
                     "role": "flooder" if client == 0 else "normal",
                     "prefix_group": f"doc{document}", "source": f"{tag}_proxy",
                     "proxy": "PROXY", "source_id": str(index)})
    return _write(rows, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arxiv-dir", required=True, type=Path)
    parser.add_argument("--sharegpt-json", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--kind", choices=("zipf", "flood"), default="flood")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--docs", type=int, default=12)
    parser.add_argument("--doc-len", type=int, default=3072)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    docs = load_docs(args.arxiv_dir, tokenizer, args.docs, args.doc_len, args.seed)
    questions = load_questions(args.sharegpt_json, tokenizer, args.count, args.seed)
    kwargs = dict(out=args.out, rate=args.rate, seed=args.seed, docs=docs, questions=questions,
                  count=args.count, clients=args.clients)
    written = build_rag_zipf(**kwargs) if args.kind == "zipf" else build_flood_pressure(**kwargs)
    print(f"wrote {written} requests to {args.out}")


if __name__ == "__main__":
    main()
