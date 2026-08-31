#!/bin/bash
# Crossover-only generation, to test the five fusion mechanisms introduced on
# 2026-08-09 against the one the pipeline had collapsed to.
#
# Every generated slot is a crossover: pool_size 5 minus 1 survivor leaves four
# generated slots and all four are assigned. The main campaign mixed roughly two
# crossovers into seven slots, which made mechanism diversity impossible to read
# -- 186 crossovers produced 5 true fusions and the rest were noise from the
# mutation majority sharing the same planner.
#
# Seeds are chosen for contrast rather than coverage: g06 is where the
# `universal_supplier` defect appeared three times, proofnet_g02 is where the
# group-theory parents produced four flagged rows, and g08 was clean.
set -uo pipefail
# Repository root, found by walking up to the marker rather than by
# counting directories: this file has moved once already and `..` was
# wrong the moment it did.
cd "$(cd "$(dirname "$0")" && until [ -f pyproject.toml ] || [ "$PWD" = / ]; do cd ..; done; pwd)"
set -a; source .env 2>/dev/null; set +a

OUT="${XO_OUT:-data/certified/ablation}"
SEEDS="${XO_SEEDS:-minif2f_g06 minif2f_g08 proofnet_g02}"
GENERATIONS="${XO_GENERATIONS:-5}"
mkdir -p "$OUT"
LOG="$OUT/driver.log"

GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-5.6-luna}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-gpt-5.6-terra}"
PROBLEM_JUDGE="${PROBLEM_JUDGE:-1}"
PROBLEM_JUDGE_MODEL="${PROBLEM_JUDGE_MODEL:-gpt-5.6-luna}"
echo "[models] gen=$GENERATOR_MODEL orch=$ORCHESTRATOR_MODEL judge=$PROBLEM_JUDGE_MODEL" | tee -a "$LOG"
echo "[config] crossover-only, generations=$GENERATIONS, seeds=$SEEDS" | tee -a "$LOG"

for name in $SEEDS; do
  csv="data/certified/run-a/seeds/$name.csv"
  [ -f "$csv" ] || { echo "[skip] no seeds for $name" | tee -a "$LOG"; continue; }
  [ -f "$OUT/$name.jsonl" ] && { echo "[skip] $name already has output" | tee -a "$LOG"; continue; }
  echo "[start] $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  GENERATION_PROVIDER=codex_cli LEAN_VERIFIER=repl \
  ORCHESTRATOR_MODEL="$ORCHESTRATOR_MODEL" \
  PROBLEM_JUDGE="$PROBLEM_JUDGE" PROBLEM_JUDGE_MODEL="$PROBLEM_JUDGE_MODEL" \
  python3 scripts/generate/run_pool_generation.py \
    --input "$csv" \
    --output "$OUT/$name.jsonl" \
    --summary-output "$OUT/${name}_summary.json" \
    --generation-model "$GENERATOR_MODEL" \
    --pool-size 5 --survivor-count 1 --crossover-count 4 \
    --max-generations "$GENERATIONS" --max-retries 1 --max-parallel 1 \
    --run-name "xo_$name" --tag crossover-focus \
    >> "$OUT/$name.log" 2>&1
  certified=$(python3 -c "
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('status')=='certified'))
except Exception: print(0)
" "$OUT/$name.jsonl" 2>/dev/null || echo 0)
  echo "[done ] $name certified=$certified  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "[all done] $(date '+%H:%M:%S')" | tee -a "$LOG"
