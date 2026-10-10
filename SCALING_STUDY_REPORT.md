# FairRoute scaling study implementation status

This report distinguishes implemented wiring from measurements that require a
simulator run. No result values are asserted here without a generated input
file.

## Implemented

- Phase-0 inventory in [phase0_report.md](./phase0_report.md).
- Deterministic FLOOD proxy in
  [flood_proxy.py](./workloads/datasets/flood_proxy.py), including client
  roles, shared/independent prefixes, rate-only arrival scaling, and
  `PROXY` provenance.
- Focused proxy tests in [test_flood_proxy.py](./tests/test_flood_proxy.py).
- Request-level routing diagnostics: decision time, FairRoute stage, gate and
  slack candidate counts, and credit exhaustion.
- Aggregate routing timing sidecars named `<result>.csv.routing.json`.
- `--fairroute-n0`, defaulting to zero, for shrinkage toward the global
  output estimate. The default path retains the prior estimate behavior.
- Policy registry names for `PRISM` and `D2LPM`, mapped to existing
  implementations.
- Paper-shaped policy formulas:
  - NexusSched iteration-time predictor and multiplicative PRISM score.
  - D²LPM-global longest-match selection with per-client/per-replica
    refillable credits and completion charges.
  - FairRoute v2 gate, adaptive slack band, credit-gated locality, replica
    service deficit, and lexicographic selection.
- The implementation explicitly uses the documented FCFS-local-engine
  deviation for D²LPM-global. Predictor constants remain configurable
  fallbacks because no fitted per-instance constants are currently stored in
  the repository.
- Executable workload, runner, metrics, and analysis notebook entry points:
  `notebooks/FairRoute_Scaling_*.ipynb`.

## Verification

- Python syntax compilation passed for every modified Python module.
- The deterministic proxy smoke test passed, including token identity
  preservation and rate dilation.
- Editor diagnostics report no errors in the modified simulator modules.
- A pytest command was attempted, but pytest is not installed in the selected
  system interpreter; the tests remain ordinary pytest-compatible test
  functions and the direct smoke test was used instead.

## Not yet measured

The following require real simulator outputs and are intentionally not
fabricated by this change:

- lambda-star calibration and the 80-run execution matrix;
- W1 content-backed generator and its cards;
- result-derived `metrics.parquet`, bootstrap confidence intervals, and PNG
  figures;
- pairwise policy divergence, K-flatness, interaction residuals, and hardware
  validation.

The runner notebook stops at a manifest with exactly 80 non-reference cells
(`2 workloads × 8 cells × 5 policies`). It does not silently execute failed
or unavailable simulator runs.

## Design choices

All choices in the supplied scaling specification are treated as `[D]`
project design choices. The proxy is labelled `[PROXY]`; its token ids are
synthetic and are not presented as conversation data. The FairRoute prior
default is `n0 = 0`, preserving current behavior.

## Open questions

1. Which generated or recorded W1 trace is the authoritative FAIRMIX source?
2. What measured `lambda_star.json` values should the runner consume?
3. Which cluster configuration is the canonical scaling-study configuration?
4. Should parquet/PNG generation use the repository's existing notebook
   environment or a newly pinned analysis environment?
5. Should `CUSTOM` receive a project callback, or remain an explicit extension
   point for callers?
