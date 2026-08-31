#!/bin/bash
# Three arms over identical seeds: crossover-only, mutation-only, and the mixed
# pipeline this work proposes.
#
# The arms are only comparable if the operator survives the run, and it did not.
# In a crossover-only group, 14 of 25 slots planned as crossover came back as
# mutations: a crossover that fails is rescued by retrying it as a mutation,
# which is right for the mixed pipeline and fatal for a controlled arm, because
# the failures of one arm are then counted as the output of the other.
# OP_TYPE_LOCK pins the operator for the single-operator arms; the mixed arm runs
# without it, since the rescue is part of what is being proposed.
#
# Everything else is held fixed: seeds, generations, pool size, survivor count,
# models, judge. Only crossover_count and the lock differ.
#
#   ARM=crossover|mutation|mix   which arm to run (default: all three in turn)
#   ABL_SEEDS                    seed group names (default: the ten miniF2F)
#   ABL_GENERATIONS              generations per group (default 5)
set -uo pipefail
# Repository root, found by walking up to the marker rather than by
# counting directories: this file has moved once already and `..` was
# wrong the moment it did.
cd "$(cd "$(dirname "$0")" && until [ -f pyproject.toml ] || [ "$PWD" = / ]; do cd ..; done; pwd)"
# .env is sourced under `set -a`, so anything it exports wins over the caller.
# That silently reverted LANGCHAIN_PROJECT for this run, and did the same to
# MAX_GENERATIONS on an earlier one. Caller values are captured first and put
# back afterwards.
_want_project="${LANGCHAIN_PROJECT:-}"
set -a; source .env 2>/dev/null; set +a
if [ -n "$_want_project" ]; then
  export LANGCHAIN_PROJECT="$_want_project" LANGSMITH_PROJECT="$_want_project"
fi

ROOT="${ABL_OUT:-data/certified/ablation}"
SEEDS="${ABL_SEEDS:-minif2f_g01 minif2f_g02 minif2f_g03 minif2f_g04 minif2f_g05 minif2f_g06 minif2f_g07 minif2f_g08 minif2f_g09 minif2f_g10}"
GENERATIONS="${ABL_GENERATIONS:-5}"
# Raised from 1 after measuring where a group's time goes: the generator is 63%
# of wall clock and Lean is 7.5%, so slots were serialised to avoid a contention
# that the numbers do not show. Two slots at a time roughly halves the group.
ABL_PARALLEL="${ABL_PARALLEL:-2}"
ARMS="${ARM:-crossover mutation mix}"

GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-5.6-luna}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-gpt-5.6-terra}"
PROBLEM_JUDGE="${PROBLEM_JUDGE:-1}"
PROBLEM_JUDGE_MODEL="${PROBLEM_JUDGE_MODEL:-gpt-5.6-luna}"

for arm in $ARMS; do
  case "$arm" in
    crossover) XCOUNT=4; LOCK=1 ;;
    mutation)  XCOUNT=0; LOCK=1 ;;
    mix)       XCOUNT=2; LOCK=0 ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  OUT="$ROOT/$arm"
  mkdir -p "$OUT"
  LOG="$OUT/driver.log"
  echo "[arm] $arm  crossover_count=$XCOUNT op_type_lock=$LOCK generations=$GENERATIONS" | tee -a "$LOG"
  echo "[models] gen=$GENERATOR_MODEL orch=$ORCHESTRATOR_MODEL judge=$PROBLEM_JUDGE_MODEL" | tee -a "$LOG"

  for name in $SEEDS; do
    csv="data/certified/run-a/seeds/$name.csv"
    [ -f "$csv" ] || { echo "[skip] no seeds for $name" | tee -a "$LOG"; continue; }
    [ -f "$OUT/$name.jsonl" ] && { echo "[skip] $name done" | tee -a "$LOG"; continue; }
    echo "[start] $arm/$name  $(date '+%H:%M:%S')" | tee -a "$LOG"
    GENERATION_PROVIDER=codex_cli LEAN_VERIFIER=repl \
    ORCHESTRATOR_MODEL="$ORCHESTRATOR_MODEL" \
    PROBLEM_JUDGE="$PROBLEM_JUDGE" PROBLEM_JUDGE_MODEL="$PROBLEM_JUDGE_MODEL" \
    OP_TYPE_LOCK="$LOCK" \
    POOL_ALIGNMENT_GOAL_AUDIT="${POOL_ALIGNMENT_GOAL_AUDIT:-1}" \
    python3 scripts/generate/run_pool_generation.py \
      --input "$csv" \
      --output "$OUT/$name.jsonl" \
      --summary-output "$OUT/${name}_summary.json" \
      --generation-model "$GENERATOR_MODEL" \
      --pool-size 5 --survivor-count 1 --crossover-count "$XCOUNT" \
      --max-generations "$GENERATIONS" --max-retries 1 --max-parallel "$ABL_PARALLEL" \
      --run-name "abl_${arm}_$name" --tag operator-ablation --tag "arm:$arm" \
      >> "$OUT/$name.log" 2>&1
    certified=$(python3 -c "
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('status')=='certified'))
except Exception: print(0)
" "$OUT/$name.jsonl" 2>/dev/null || echo 0)
    echo "[done ] $arm/$name certified=$certified  $(date '+%H:%M:%S')" | tee -a "$LOG"
  done
  echo "[arm done] $arm  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
