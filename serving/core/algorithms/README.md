# Serving algorithms

Runtime request-routing choices are `LOAD`, `RR`, `RAND`, `PRISM`, `D2LPM`,
`FAIRROUTE` (the previous FairRoute-v2 pipeline), `FAIRROUTE_SLACK`,
`FAIRROUTE_GATED`, and `CUSTOM`. `FAIRROUTE_V2` remains an accepted
compatibility alias. The local scheduler choices are `FCFS`, `VTC`, and `DLPM`.

- `fairness.py` tracks service and request latency for analysis.
- `locality.py` and `prediction.py` provide cache and queue estimates.
- `fairroute.py` provides the shared FairRoute gate, slack set, and candidate
  combination primitives used by the three selectable FairRoute policies.
- `published.py` contains the VTC/DLPM policies, D²LPM router state,
  Preble E2 primitives, and NexusSched performance-model/LENS helpers.

Preble E2 and NexusSched LENS are algorithm components, not selectable full
serving policies. PRISM remains an explicit simulator scoring approximation;
its unspecified paper score functions should not be described as a reproduction.
