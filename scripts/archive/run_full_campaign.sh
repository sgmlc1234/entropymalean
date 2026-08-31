#!/usr/bin/env bash
# Run the main-results 2-benchmark × 2-arm × 3-model × 3-repeat campaign.
#
# Expects:
#   $OPENROUTER_API_KEY
#   data/raw/<benchmark>_seed_control.csv     (20 seeds per benchmark)
#   data/certified/<benchmark>_treatment.jsonl (>=200 certified rows)
#
# Outputs land in data/evaluation/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/evaluation"
mkdir -p "$OUT"

BENCHMARKS=("miniF2F" "proofnet")
CONTROL_CAP="${CONTROL_CAP:-20}"
TREATMENT_CAP="${TREATMENT_CAP:-100}"
REPEATS="${REPEATS:-3}"
PARALLEL="${PARALLEL:-8}"

for bench in "${BENCHMARKS[@]}"; do
  lower="$(echo "$bench" | tr '[:upper:]' '[:lower:]')"
  control="$ROOT/data/raw/${lower}_seed_control.csv"
  treatment="$ROOT/data/certified/${lower}_treatment.jsonl"
  per_cell="$OUT/${lower}_eval.jsonl"
  summary="$OUT/${lower}_summary.json"
  slopes="$OUT/${lower}_slopes.json"

  echo "=== $bench ==="
  echo "  control:   $control"
  echo "  treatment: $treatment"
  echo "  per-cell:  $per_cell"

  python "$ROOT/scripts/archive/run_evaluation.py" \
    --benchmark "$bench" \
    --control "$control" \
    --treatment "$treatment" \
    --output "$per_cell" \
    --summary "$summary" \
    --slopes "$slopes" \
    --control-cap "$CONTROL_CAP" \
    --treatment-cap "$TREATMENT_CAP" \
    --repeats "$REPEATS" \
    --max-parallel-rows "$PARALLEL"
done

echo
echo "campaign done. aggregating..."
python "$ROOT/scripts/archive/aggregate_campaign.py" \
  --output "$OUT/campaign_summary.json" \
  "$OUT"/*_summary.json
echo "wrote $OUT/campaign_summary.json"
