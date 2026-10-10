# FairRoute scaling study: Phase 0 inventory

This inventory is based on the current working tree.  It records wiring facts,
not measured experiment results.

## Request and routing path

`Router.load_requests()` reads JSONL rows and dispatches flat rows to
`_load_flat_request()` and rows containing `sub_requests` to
`_load_agentic_session()` in `serving/core/router.py`.  Both paths copy
`client_id` into the route-data dictionary.  `route_arrived_requests()` passes
that dictionary to FairRoute and to `custom_routing_fn`.

`_custom_select(schedulers, role, req_data)` therefore receives request data,
including `client_id`, input token count, output token count, arrival time, and
prefix hashes.  The default `CUSTOM` CLI path is not wired to a callback:
`serving/__main__.py` constructs `Router` without `custom_routing_fn`, so
selecting `CUSTOM` raises the repository's explicit runtime error.

## Client identity

`serving/core/request.py:Request.__init__` stores `client_id` without
conversion, so the type is whatever the JSON decoder produced (normally an
integer for generated workloads, but strings are also preserved).  Flat rows
and agentic session rows both preserve it.  `Scheduler.save_output()` writes
`client id` in the per-request CSV.

## Policy selection

`--request-routing-policy` is defined in `serving/__main__.py`.  The current
registry includes `RR`, `LOAD`, `RAND`, the FairRoute variants, and `CUSTOM`.
`RR` maps to `_rr_select`, `LOAD` to `_least_load_select`, FairRoute variants
to `_fair_route`, and `CUSTOM` to the callback path described above.  PRISM
and D2LPM are not currently registered.

## Result and diagnostics surfaces

`serving/core/scheduler.py:Scheduler.save_output()` writes request-level
latency, TTFT, TPOT, ITL, routing metrics, cache-hit counters, preemption
counts, and session fields.  `instance id` is the per-replica service
location; `prefix_cache_hit`, `npu_cache_hit`, and `storage_cache_hit` are
the cache-hit columns.  Router-side aggregate score and margin statistics are
available through `Router.get_score_stats()`.  There is currently no
decision-latency histogram sidecar and no compact stage/candidate diagnostic
columns.

## Converter inventory

`workloads/datasets/convert_to_llmservingsim.py` currently exposes
`--client-strategy`, `--rate-rps`, `--num-clients`, `--split`,
`--first-turn-only`, `--sample-records`, `--sample-sessions`,
`--max-records`, `--max-input-tokens`, `--max-output-tokens`,
`--max-total-tokens`, `--truncate`, `--window-seconds`, `--arrival-mode`,
and `--seed`, plus dataset-specific controls.  The converter supports
`gamma`, `constant`, and `replay` arrival modes.  `flood_proxy.py` was absent
before this implementation; it is now provided as a deterministic library.

## Simulator command and cluster configuration

The repository's standard simulator entry point is `python -m serving`.
The checked-in example command is in `serving/run.sh`; it uses
`configs/cluster/single_node_single_instance.json` for the default example.
The FairRoute experiment wrapper defaults to
`configs/cluster/dual_node_multi_instance_policy_study.json` and invokes
the simulator through its configured container.  No single authoritative
scaling-study command or lambda-star file is currently checked in.

## Phase-0 gate

The client identity reaches the router and the request CSV end to end.
FairRoute can see the request data.  The requested PRISM/D2LPM registry and
custom callback wiring are not yet present, so policy-divergence and
RR/LOAD flatness are **unverified** until those additions and smoke tests are
implemented.
