#!/usr/bin/env bash
# Compatibility entry point for the complete FairRoute workload suite.
#
# Usage:
#   ./scripts/generate_fairroute_workloads.sh <pilot|main|scaling|check>
#
# The underlying builder automatically prefers .venv-fairroute/bin/python,
# then .venv/bin/python. Override it explicitly with PY=/path/to/python.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  cat >&2 <<'USAGE'
Usage: ./scripts/generate_fairroute_workloads.sh <pilot|main|scaling|check>

Examples:
  ./scripts/generate_fairroute_workloads.sh pilot
  LAM_W1=2 LAM_W3=2 ./scripts/generate_fairroute_workloads.sh main
  SEEDS=100 LAM_W1=2 LAM_W3=2 ./scripts/generate_fairroute_workloads.sh scaling
  ./scripts/generate_fairroute_workloads.sh check

The builder automatically uses .venv-fairroute/bin/python or .venv/bin/python.
Set PY=/absolute/path/to/python to override that selection.
USAGE
  exit 2
fi

case "$STAGE" in
  pilot|main|scaling|check) ;;
  *)
    echo "Unknown stage '$STAGE'; expected pilot, main, scaling, or check." >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/build_fairroute_suite.sh" "$STAGE"
