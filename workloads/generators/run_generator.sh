#!/bin/bash
# Generate LLMServingSim workload from the full Nebius dataset (659K rows)
# with 100 Zipfian-distributed clients for FairRoute fairness experiments.

set -euo pipefail

python3 -m pip install datasets transformers numpy tqdm

python3 generate_workload.py \
    --use-all \
    --sps 10 \
    --max-sessions 0 \
    --dataset-dir ./nebius_dataset_cache \
    --tokenizer-dir ./llama_tokenizer_cache \
    --output ./workload-llama8b-full-sps10-100clients.jsonl \
    --checkpoint-file ./workload-llama8b-full-sps10-100clients.tokenize.checkpoint.jsonl \
    --checkpoint-interval 10 \
    --num-workers 16 \
    --add-client-ids \
    --num-clients 100 \
    --zipf-alpha 1.1 \
    --seed 42