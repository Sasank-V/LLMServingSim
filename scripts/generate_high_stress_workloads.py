#!/usr/bin/env python3
"""Generate high-saturation stress workloads to expose predictive routing bottlenecks."""

import json
import random
import os

def create_high_qps_stress(filename, num_requests=10000, qps=100.0):
    """Continuous high arrival rate (QPS=25.0) pushing system utilization above 75%."""
    random.seed(42)
    filepath = os.path.join("workloads/fairroute_hypothesis", filename)
    
    interval_ns = int((1.0 / qps) * 1e9)
    current_time_ns = 0
    
    with open(filepath, "w") as f:
        for i in range(num_requests):
            input_toks = random.randint(256, 768)
            output_toks = random.randint(128, 384)
            client_id = random.randint(0, 7)
            
            record = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": current_time_ns,
                "input_tok_ids": list(range(i * 10, i * 10 + input_toks)),
                "output_tok_ids": list(range(input_toks, input_toks + output_toks)),
                "client_id": client_id,
                "scenario": "high_qps_stress"
            }
            f.write(json.dumps(record) + "\n")
            current_time_ns += interval_ns
    print(f"Generated {filepath}: {num_requests} reqs at {qps} QPS.")

def create_bimodal_length_stress(filename, num_requests=300, qps=20.0):
    """Extreme bimodal output length distribution (25% heavy, 75% light)."""
    random.seed(42)
    filepath = os.path.join("workloads/fairroute_hypothesis", filename)
    
    interval_ns = int((1.0 / qps) * 1e9)
    current_time_ns = 0
    
    with open(filepath, "w") as f:
        for i in range(num_requests):
            is_heavy = (random.random() < 0.25)
            if is_heavy:
                input_toks = random.randint(1024, 1536)
                output_toks = random.randint(768, 1024)
            else:
                input_toks = random.randint(128, 256)
                output_toks = random.randint(32, 96)
                
            client_id = random.randint(0, 7)
            record = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": current_time_ns,
                "input_tok_ids": list(range(i * 10, i * 10 + input_toks)),
                "output_tok_ids": list(range(input_toks, input_toks + output_toks)),
                "client_id": client_id,
                "scenario": "bimodal_length_stress"
            }
            f.write(json.dumps(record) + "\n")
            current_time_ns += interval_ns
    print(f"Generated {filepath}: {num_requests} reqs bimodal length at {qps} QPS.")

def create_burst_heavy_stress(filename, num_requests=300, burst_size=40, burst_interval_s=2.0):
    """Sharp multi-tenant bursts (40 requests arriving at once every 2 seconds)."""
    random.seed(42)
    filepath = os.path.join("workloads/fairroute_hypothesis", filename)
    
    current_time_ns = 0
    burst_interval_ns = int(burst_interval_s * 1e9)
    
    with open(filepath, "w") as f:
        count = 0
        while count < num_requests:
            current_burst = min(burst_size, num_requests - count)
            for j in range(current_burst):
                input_toks = random.randint(256, 512)
                output_toks = random.randint(128, 256)
                client_id = random.randint(0, 7)
                
                # Small jitter (0 to 5ms) within burst
                jitter_ns = random.randint(0, 5000000)
                record = {
                    "input_toks": input_toks,
                    "output_toks": output_toks,
                    "arrival_time_ns": current_time_ns + jitter_ns,
                    "input_tok_ids": list(range(count * 10, count * 10 + input_toks)),
                    "output_tok_ids": list(range(input_toks, input_toks + output_toks)),
                    "client_id": client_id,
                    "scenario": "burst_heavy_stress"
                }
                f.write(json.dumps(record) + "\n")
                count += 1
            current_time_ns += burst_interval_ns
    print(f"Generated {filepath}: {num_requests} reqs in bursts of {burst_size}.")

if __name__ == "__main__":
    os.makedirs("workloads/fairroute_hypothesis", exist_ok=True)
    create_high_qps_stress("high_qps_stress.jsonl")
    create_bimodal_length_stress("bimodal_length_stress.jsonl")
    create_burst_heavy_stress("burst_heavy_stress.jsonl")
