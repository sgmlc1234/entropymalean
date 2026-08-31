#!/bin/bash
# Drive the EML-1 main generation: 20 seed groups, five generations each.
#
# Groups run one at a time. The bottleneck is the Lean REPL, which every slot
# shares, so two concurrent groups do not finish in half the time — they
# contend, and a wedged verifier then takes both runs down instead of one. A
# group that fails is skipped rather than retried: its seeds stay recorded in
# the CSV, so it can be re-run alone afterwards without disturbing the rest.
#
# ProofNet certifies at 54-66% against miniF2F's 84%, so the same five
# generations yield fewer accepted rows there; the shortfall is measured at the
# end rather than papered over by running ProofNet longer, because changing the
# generation budget per benchmark would confound the two arms.
set -uo pipefail
# Repository root, found by walking up to the marker rather than by
# counting directories: this file has moved once already and `..` was
# wrong the moment it did.
cd "$(cd "$(dirname "$0")" && until [ -f pyproject.toml ] || [ "$PWD" = / ]; do cd ..; done; pwd)"
set -a; source .env 2>/dev/null; set +a

# Output directory is a knob so a campaign at a different depth writes beside
# the old one rather than into it: the driver skips any group that already has
# output, so reusing the directory would skip all twenty and do nothing.
OUT="${EML_OUT:-data/certified/run-a}"
mkdir -p "$OUT"
LOG="$OUT/driver.log"

# Planning and filling are separate jobs on separate models. The orchestrator
# decides which parents meet and what the child should attempt; the generator
# writes one artifact against that decision. The planner is the smaller number
# of calls and the larger share of the outcome, so the stronger model goes
# there. Unset ORCHESTRATOR_MODEL to put both roles back on one model.
GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-5.6-luna}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-gpt-5.6-terra}"

# The quality judge stayed off until its verdicts had been checked against
# hand-read rows. On 2026-08-08 they were: all 15 rejections it issued over the
# 31 labelled rows named a redundancy that holds up against the Lean. It judges
# on a model other than the one that wrote the row, so it moves when the
# generator does.
# Depth per group. Groups 01-04 ran at five; the later ones run at three, so
# generation number is not comparable across the whole ProofNet set and any
# per-generation reading has to be taken within a group.
#
# Named EML_ rather than MAX_GENERATIONS because this script sources .env under
# `set -a`, and .env already exports MAX_GENERATIONS=10. A `${MAX_GENERATIONS:-3}`
# therefore reads 10 and the run silently ignores the setting -- which is what
# happened on the first attempt at this change. Any knob here needs a name .env
# does not use.
EML_GENERATIONS="${EML_GENERATIONS:-3}"

PROBLEM_JUDGE="${PROBLEM_JUDGE:-1}"
PROBLEM_JUDGE_MODEL="${PROBLEM_JUDGE_MODEL:-gpt-5.6-luna}"
echo "[models] generator=$GENERATOR_MODEL orchestrator=$ORCHESTRATOR_MODEL " \
     "judge=$PROBLEM_JUDGE_MODEL (enabled=$PROBLEM_JUDGE)" | tee -a "$LOG"

# Group order. The glob runs g01 first, which is only a convention -- and a
# costly one when a campaign is cut short, because whichever groups the budget
# reaches are the ones that get read. Running the other way puts a different
# slice at the front, so two campaigns stopped early do not both end up
# describing the same low-numbered groups.
if [ "${EML_ORDER:-forward}" = "reverse" ]; then
  CSVS=$(ls "$OUT"/seeds/*.csv | sort -r)
else
  CSVS=$(ls "$OUT"/seeds/*.csv | sort)
fi

for csv in $CSVS; do
  name=$(basename "$csv" .csv)
  if [ -f "$OUT/$name.jsonl" ]; then
    echo "[skip] $name already has output" | tee -a "$LOG"
    continue
  fi
  echo "[start] $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  GENERATION_PROVIDER=codex_cli LEAN_VERIFIER=repl \
  ORCHESTRATOR_MODEL="$ORCHESTRATOR_MODEL" \
  PROBLEM_JUDGE="$PROBLEM_JUDGE" PROBLEM_JUDGE_MODEL="$PROBLEM_JUDGE_MODEL" \
  python3 scripts/generate/run_pool_generation.py \
    --input "$csv" \
    --output "$OUT/$name.jsonl" \
    --summary-output "$OUT/${name}_summary.json" \
    --generation-model "$GENERATOR_MODEL" \
    --pool-size 5 --survivor-count 1 --crossover-count 2 \
    --max-generations "$EML_GENERATIONS" --max-retries 1 --max-parallel 2 \
    --run-name "eml1_$name" --tag eml1-main --tag novelty-gate \
    >> "$OUT/$name.log" 2>&1
  status=$?
  rows=$(wc -l < "$OUT/$name.jsonl" 2>/dev/null || echo 0)
  # Counted by parsing, not by grepping. `"status": "certified"` also appears
  # inside nested attempt_history and quality_evidence, so the grep reported 14
  # for a group that had 8 and 16 for one that had fewer -- always high, which
  # is the direction that stops the yield guard below from ever firing.
  certified=$(python3 -c "
import json,sys
try:
    print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('status')=='certified'))
except Exception:
    print(0)
" "$OUT/$name.jsonl" 2>/dev/null || echo 0)
  echo "[done ] $name exit=$status rows=$rows certified=$certified  $(date '+%H:%M:%S')" | tee -a "$LOG"

  # An exhausted quota is not a slot failure and not retryable: every remaining
  # call returns the same refusal, so continuing would burn through the queue
  # writing empty groups and leave no record of where real generation stopped.
  # The first run of this script did exactly that -- 207 of 255 "generation
  # failures" were the provider saying it was out of credit.
  if grep -q "usage_limit_reached" "$OUT/$name.log" 2>/dev/null; then
    echo "[halt ] usage limit reached during $name; stopping. Re-run this " \
         "script after the quota resets -- finished groups are skipped." | tee -a "$LOG"
    exit 3
  fi

  # Yield guard. The marker above needs the provider to say why it failed, and
  # on 2026-08-07 it said nothing at all -- bare exit 1, empty stderr -- so six
  # ProofNet groups "completed" in three minutes each having generated nothing,
  # and were written out as survivor copies of their own seeds. A group that
  # certifies zero rows is not a bad group; it is a broken run, whatever the
  # cause, and the next group will be broken the same way.
  if [ "$certified" -eq 0 ]; then
    echo "[halt ] $name certified 0 rows -- generation is not working. " \
         "Stopping before the rest of the queue is burned. Delete " \
         "$OUT/$name.jsonl before re-running this group." | tee -a "$LOG"
    exit 4
  fi
done

echo "[all done] $(date '+%H:%M:%S')" | tee -a "$LOG"
