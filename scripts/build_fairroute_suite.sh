#!/usr/bin/env bash
# Build the FairRoute workload suite: 5 workloads, main matrix at K=16, scaling cells for W1 and W3.
#   ./scripts/build_fairroute_suite.sh <pilot|main|scaling|check>
# W1 FAIRMIX (2 heavy + K-2 light) | W2 LENMIX | W3 FLOOD (PROXY) | W4 ARENA sessions | W5 BURST (Azure replay)
# Order: pilot -> run RR in the simulator -> set LAM_W1..LAM_W5 -> main -> scaling.
# DRY=1 prints the commands without running them. Existing traces are skipped unless FORCE=1.
set -euo pipefail
STAGE="${1:-}"
[[ "$STAGE" =~ ^(pilot|main|scaling|check)$ ]] || { echo "usage: $0 <pilot|main|scaling|check>" >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${PY:-}" ]]; then
  if [[ -x "$PWD/.venv-fairroute/bin/python" ]]; then
    PY="$PWD/.venv-fairroute/bin/python"
  elif [[ -x "$PWD/.venv/bin/python" ]]; then
    PY="$PWD/.venv/bin/python"
  else
    PY="python3"
  fi
fi
CONV="${CONV:-workloads/datasets/convert_to_llmservingsim.py}"
RAW="${RAW:-workloads/datasets/raw}"
SG="${SG:-$RAW/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"
AZ_GLOB="${AZ_GLOB:-AzureLLMInferenceTrace_conv*.csv}"
OUT="${OUT:-workloads/fairroute}"
N="${N:-10000}"; NSESS="${NSESS:-8000}"
SEEDS="${SEEDS:-100}"                 # seed 7 -> --split tune, any other seed -> --split test
WL="${WL:-w1 w2 w3 w4 w5}"
LEVELS="${LEVELS:-0.5 0.8 1.0 1.25 1.5}"
DRY="${DRY:-0}"; FORCE="${FORCE:-0}"
PILOT_W1="${PILOT_W1:-2 4 8 16}"; PILOT_W2="${PILOT_W2:-2 4 8 16}"; PILOT_W3="${PILOT_W3:-2 4 8 16}"
PILOT_W4="${PILOT_W4:-0.5 1 2 4}"; PILOT_W5="${PILOT_W5:-4 8 16 32}"   # W4 = session starts/s

calc() { awk "BEGIN{printf \"%.4f\", $1}"; }
cnt()  { awk "BEGIN{printf \"%d\", $1}"; }
nreq() { awk -v r="$1" 'BEGIN{n=300*r; if(n<2000)n=2000; printf "%d", n}'; }
split_for() { [[ "$1" == 7 ]] && echo tune || echo test; }
run()  { if [[ "$DRY" == 1 ]]; then echo "+ $*"; else "$@"; fi; }
conv()  { run "$PY" "$CONV" --input-root "$RAW" "$@"; }
lam()  { local v="LAM_${1^^}"; [[ -n "${!v:-}" ]] || { echo "set $v (RR saturation rate from the pilot)" >&2; exit 3; }; echo "${!v}"; }
in_wl() { [[ " $WL " == *" $1 "* ]]; }

merge() {  # merge OUT part:client_offset:role ... ; clips every part to the shortest part's duration
  if [[ "$DRY" == 1 ]]; then echo "+ merge $*"; return; fi
  local out=$1; shift
  "$PY" - "$out" "$@" <<'PYEOF'
import sys, json
out, parts = sys.argv[1], sys.argv[2:]
rows, ends = [], []
for p in parts:
    path, off, role = p.rsplit(":", 2)
    part = [json.loads(l) for l in open(path)]
    ends.append(max(r["arrival_time_ns"] for r in part))
    for r in part:
        r["client_id"] += int(off); r["role"] = role
    rows += part
T = min(ends)
rows = [r for r in rows if r["arrival_time_ns"] <= T]
rows.sort(key=lambda r: (r["arrival_time_ns"], r["client_id"]))
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r, separators=(",", ":")) + "\n")
print(out, len(rows), "rows,", len({r["client_id"] for r in rows}), "clients, duration", round(T / 1e9, 1), "s")
PYEOF
}

gen_w1() { # R S N O K : heavy group fixed at 12.5% of requests (2 clients); light group = K-2 clients
  local R=$1 S=$2 N=$3 O=$4 K=$5 SP; SP=$(split_for "$S"); local SH=0.125 LC=$(( K - 2 ))
  conv --dataset arxiv_summarization --split "$SP" --sample-records "$(cnt "$N*$SH*4")" \
    --max-input-tokens 8192 --max-output-tokens 1024 --client-strategy round_robin --num-clients 2 \
    --rate-rps "$(calc "$R*$SH")" --seed "$S" --output-dir "$O/heavy"
  conv --dataset sharegpt --sharegpt-file "$SG" --emit sessions --first-turn-only --split "$SP" \
    --client-strategy round_robin --num-clients "$LC" --max-total-tokens 4096 \
    --sample-sessions "$(cnt "$N*(1-$SH)*1.3")" --rate-rps "$(calc "$R*(1-$SH)")" --seed "$S" --output-dir "$O/light"
  merge "$O/trace.jsonl" "$O/heavy/arxiv_summarization.jsonl:0:heavy" "$O/light/sharegpt.jsonl:2:light"
}
gen_w2() { # R S N O : 90% short chat, 5% long-prefill (arXiv), 5% long-decode (code r1_generation); 8 clients see all classes
  local R=$1 S=$2 N=$3 O=$4 SP; SP=$(split_for "$S"); local FS=0.90 FL=0.05
  conv --dataset sharegpt --sharegpt-file "$SG" --emit sessions --first-turn-only --split "$SP" \
    --client-strategy round_robin --num-clients 8 --max-total-tokens 4096 \
    --sample-sessions "$(cnt "$N*$FS*1.3")" --rate-rps "$(calc "$R*$FS")" --seed "$S" --output-dir "$O/short"
  conv --dataset arxiv_summarization --split "$SP" --sample-records "$(cnt "$N*$FL*4")" \
    --max-input-tokens 8192 --max-output-tokens 1024 --client-strategy round_robin --num-clients 8 \
    --rate-rps "$(calc "$R*$FL")" --seed "$S" --output-dir "$O/long_prefill"
  conv --dataset code_reasoning --split "$SP" --sample-records "$(cnt "$N*$FL*1.3")" --truncate \
    --max-input-tokens 4096 --max-output-tokens 2048 --client-strategy round_robin --num-clients 8 \
    --rate-rps "$(calc "$R*$FL")" --seed "$S" --output-dir "$O/long_decode"
  merge "$O/trace.jsonl" "$O/short/sharegpt.jsonl:0:short" "$O/long_prefill/arxiv_summarization.jsonl:0:long_prefill" \
    "$O/long_decode/code_reasoning.jsonl:0:long_decode"
}
gen_w3() { # R S N O K : flooder = 35% of requests on one 4096-token prefix; K-1 normal clients with own 512-token prefixes
  local R=$1 S=$2 N=$3 O=$4 K=$5
  run "$PY" -c "from workloads.datasets.flood_proxy import build_flood as b; print(b($R,$S,$N,'$O/trace.jsonl',K=$K,flood_share=0.35))"
}
gen_w4() { # R S N O : Arena sessions, one side per row, client = model; R = session starts/s, N = sessions
  local R=$1 S=$2 N=$3 O=$4 SP; SP=$(split_for "$S")
  conv --dataset lmsys_chatbot_arena --emit sessions --side random --split "$SP" --client-strategy field \
    --client-field model --max-total-tokens 16384 --truncate --think-median-s 20 \
    --sample-sessions "$N" --rate-rps "$R" --seed "$S" --output-dir "$O/raw"
  run cp "$O/raw/lmsys_chatbot_arena.jsonl" "$O/trace.jsonl"
}
gen_w5() { # R S N O : Azure conversation trace, native burst shape rescaled to R req/s, count-only (synthetic unique ids)
  local R=$1 S=$2 O=$4
  conv --dataset azure_llm_inference --azure-glob "$AZ_GLOB" --window-seconds 3600 --drop-zero-output \
    --max-input-tokens 8192 --max-output-tokens 2048 --trace-token-ids synthetic --client-strategy round_robin \
    --num-clients 8 --arrival-mode replay --rate-rps "$R" --seed "$S" --output-dir "$O/raw"
  run cp "$O/raw/azure_llm_inference.jsonl" "$O/trace.jsonl"
}

built=0; skipped=0
build() { # w K R seed O n
  local w=$1 K=$2 R=$3 S=$4 O=$5 n=$6
  if [[ -s "$O/trace.jsonl" && "$FORCE" != 1 ]]; then echo "skip (exists): $O"; skipped=$((skipped+1)); return; fi
  echo "== $w K=$K rate=$R seed=$S -> $O"
  [[ "$DRY" == 1 ]] || mkdir -p "$O"
  case "$w" in
    w1) gen_w1 "$R" "$S" "$n" "$O" "$K";; w2) gen_w2 "$R" "$S" "$n" "$O";; w3) gen_w3 "$R" "$S" "$n" "$O" "$K";;
    w4) gen_w4 "$R" "$S" "$n" "$O";;      w5) gen_w5 "$R" "$S" "$n" "$O";;
  esac
  built=$((built+1))
}

SCALING_CELLS="4:1.0 8:1.0 32:1.0 64:1.0 4:0.5 8:0.8 32:1.25 64:1.5"   # E1 + E3 cells not at K=16 (E2 lives in main)

case "$STAGE" in
  pilot)
    for w in $WL; do v="PILOT_${w^^}"
      for R in ${!v}; do build "$w" 16 "$R" 7 "$OUT/pilot/$w/r${R}_s7" "$(nreq "$R")"; done
    done;;
  main)
    for S in $SEEDS; do for w in $WL; do L=$(lam "$w"); n=$N; k=16; kd=K16
      [[ "$w" == w4 ]] && n=$NSESS; [[ "$w" == w4 || "$w" == w5 ]] && { k=0; kd=Kna; }
      for rho in $LEVELS; do build "$w" "$k" "$(calc "$L*$rho")" "$S" "$OUT/main/$w/${kd}_L${rho}_s$S" "$n"; done
    done; done;;
  scaling)
    for S in $SEEDS; do for w in w1 w3; do in_wl "$w" || continue; L=$(lam "$w")
      for cell in $SCALING_CELLS; do K=${cell%%:*}; rho=${cell##*:}
        n=$N; [[ "$w" == w3 && $(( K * 200 )) -gt $N ]] && n=$(( K * 200 ))   # keep >=100 requests per normal client at large K
        build "$w" "$K" "$(calc "$L*$rho")" "$S" "$OUT/scaling/$w/K${K}_L${rho}_s$S" "$n"
      done
    done; done;;
  check)
    for d in pilot main scaling; do echo "$d: $(find "$OUT/$d" -name trace.jsonl 2>/dev/null | wc -l) traces"; done
    echo "expected: main=$(( $(wc -w <<<"$WL") * $(wc -w <<<"$LEVELS") * $(wc -w <<<"$SEEDS") ))" \
         "scaling=$(( 8 * $(wc -w <<<"$(for w in w1 w3; do in_wl $w && echo $w; done)") * $(wc -w <<<"$SEEDS") ))";;
esac
echo "done: built=$built skipped=$skipped"