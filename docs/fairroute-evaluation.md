# FairRoute evaluation protocol

This protocol is the acceptance gate for FairRoute results. It prevents a
single overloaded cell, a missing baseline, or unavailable prediction telemetry
from becoming a headline claim.

## Split and matrix

- Seed `7` is tuning-only. Use at least three held-out seeds; the checked-in
  workload builder defaults to `101` through `105`.
- A policy comparison is paired by workload, offered-load cell, seed, and local
  scheduler. Generate each trace once and replay it under every router.
- Keep router and local-scheduler experiments distinct. Report D2LPM as the
  repository's global-router adaptation with its selected local scheduler, not
  as an unqualified reproduction of the published scheduler.
- Run `scripts/validate_fairroute_matrix.py` before analysis. The matrix driver
  does this automatically and exits non-zero for incomplete cells, a tuning
  seed, unequal request sets, missing routing sidecars, or missing predictions.
- Run `scripts/analyze_fairroute_results.py RESULTS --isolated-results BASELINES
  --output-dir ANALYSIS` to generate seed-paired summaries. `BASELINES` must
  be one-request/zero-queue results matched by workload, seed, and request ID;
  RR under load is not an isolation baseline.

## Required validation controls

1. Run a scheduler stress trace in which FCFS, VTC, and DLPM must admit a
   different request. Inspect `local admission order` and `local admission
   time` in the request CSV; equal end-to-end results alone do not validate the
   scheduler wiring.
2. Run a heavy/light or flooder positive control. A fairness metric is accepted
   only if the fairness-oriented scheduler improves its predeclared client
   slowdown result against FCFS on this trace.
3. Fit predictor parameters on tuning traces only, freeze them, and retain the
   fit artifact with its hardware/config hash. A result must report prediction
   coverage, MAE, RMSE, and bias from the routing sidecar. Missing values are
   missing telemetry, never zero predictions.
4. Validate PRISM and D2LPM against a reference/example decision trace before
   comparing their aggregate performance. Record all adaptation and parameter
   choices in the run manifest.

## Metrics and aggregation

- Primary latency metrics are per-workload/per-load-cell TTFT and E2E p50/p95/
  p99 with paired seed confidence intervals. Never average raw p99 values over
  heterogeneous workload or load cells.
- Primary fairness metrics are per-client and per-role slowdown p99, worst
  client/median slowdown, heavy-to-light slowdown ratio, slowdown CoV, and
  time-windowed demand-weighted service debt. Do not use completed request
  share as a fairness metric: it is fixed by a fully completed trace.
- Measure cache hits as hit-token fraction plus latency benefit and memory
  occupancy. Restrict locality conclusions to workloads that actually contain
  reusable prefixes.
- Only treat throughput as discriminating after utilization/queue traces show
  the cell is capacity-bound. Arrival-bound cells should report throughput as a
  non-discriminating control.
- If an overall score is needed, aggregate predeclared per-cell normalized
  regret and also report worst-case regret/rank.

## Claim boundary

The default claim is a characterisation: combination forms trade prediction,
locality, replica balance, and client latency differently by workload. A
general FairRoute superiority claim requires a predeclared metric on which it
wins with confidence intervals against LOAD and the validated baselines.
