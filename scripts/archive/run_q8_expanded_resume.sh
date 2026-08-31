#!/usr/bin/env bash
# Q8_0 expanded run: pick up 8 new miniF2F treatment cells added to
# accepted.jsonl after the original Q8 campaign launched.
#
# Plan:
#   1. Refresh /tmp/eml_campaign/{minif2f,proofnet}_treatment.jsonl
#      via prepare_campaign_inputs.py. This pulls all 23 unique miniF2F
#      treatments (was 15 at launch). proofnet (11) is unchanged.
#   2. Reuse the same campaign tag `2026-05-20-Q8` so existing cells
#      remain in their JSONLs. `--resume` skips every (problem_id,
#      model_label) already present, so only the 8 new miniF2F unique
#      IDs × {BFS-V2-7B Q8_0, Goedel-V2-8B} = 16 cells get evaluated.
#   3. Summary is regenerated at the end (includes both old and new cells).
#
# Safe to re-run: cells already in the JSONL won't be re-evaluated.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Same model + budget knobs as the original Q8 run for parity.
exec env \
  CAMPAIGN_TAG=2026-05-20-Q8 \
  K=3 \
  T_MAX=4 \
  S_MAX=6 \
  N_PER_STEP=6 \
  N_PARALLEL=1 \
  BFS_TREE_SEARCH=1 \
  BFS_TREE_MAX_NODES=8 \
  CONTROL_CAP=20 \
  TREATMENT_CAP=30 \
  MAX_PARALLEL_CELLS=1 \
  LEAN_TIMEOUT=180 \
  MODEL_TIMEOUT=60 \
  LEAN_CONCURRENCY_PER_EXPANSION=2 \
  bash "$ROOT/scripts/archive/run_eml_campaign.sh"
