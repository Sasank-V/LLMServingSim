#!/usr/bin/env python3
"""Generate a workload designed to maximize the advantage of predictive algorithms over LOAD/RR.

Scenario: Asymmetric Output Lengths + Shared System Prompts (Prefix Caching)
- Type A (Heavy Decoders): Short prompt (128 toks), LONG output (1024 toks).
- Type B (Fast Q&A): Long prompt (1024 toks with 100% prefix match), SHORT output (16 toks).

Why this breaks LOAD:
- LOAD sees 2 Type A requests as 'less load' (count=2) than 3 Type B requests (count=3).
- LOAD routes new requests to the instance running Type A, causing massive queuing behind 1024-token decodes.

Why Predictive Algorithms (BF_IO, PAST_FUTURE, QUARTZ) Win:
- BF_IO models the 1024-token decode trajectory over horizon H and avoids the heavy instance.
- PAST_FUTURE samples conditional output length distributions and predicts peak memory M*.
- QUARTZ adds prefix locality bonus + Pollaczek-Khinchin waiting time.
"""

import json
import random
import os

def generate_predictive_advantage_workload(filename="predictive_advantage_stress.jsonl", num_requests=400, qps=20.0):
    random.seed(1337)
    filepath = os.path.join("workloads/fairroute_hypothesis", filename)
    
    interval_ns = int((1.0 / qps) * 1e9)
    current_time_ns = 0
    
    # 4 distinct 1024-token system prompt prefix pools for prefix caching
    prefix_pools = [
        list(range(i * 2000, i * 2000 + 1024)) for i in range(4)
    ]
    
    with open(filepath, "w") as f:
        for i in range(num_requests):
            # 50% Type A (Heavy Decoder), 50% Type B (Fast Q&A with shared prefix)
            is_heavy_decoder = (i % 2 == 0)
            prefix_group = i % 4
            
            if is_heavy_decoder:
                # Type A: Short prompt, massive output (1024 tokens)
                input_toks = 128
                output_toks = 1024
                input_ids = list(range(90000 + i * 200, 90000 + i * 200 + input_toks))
            else:
                # Type B: Long prompt (shared 1024-token prefix), tiny output (16 tokens)
                input_toks = 1024
                output_toks = 16
                input_ids = list(prefix_pools[prefix_group])  # Reuses prefix!
                
            client_id = prefix_group
            record = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": current_time_ns,
                "input_tok_ids": input_ids,
                "output_tok_ids": list(range(input_toks, input_toks + output_toks)),
                "client_id": client_id,
                "scenario": "predictive_advantage_stress"
            }
            f.write(json.dumps(record) + "\n")
            current_time_ns += interval_ns
            
    print(f"Generated {filepath}: {num_requests} requests at {qps} QPS.")

if __name__ == "__main__":
    os.makedirs("workloads/fairroute_hypothesis", exist_ok=True)
    generate_predictive_advantage_workload()
