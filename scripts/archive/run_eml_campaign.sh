#!/usr/bin/env bash
# Run the EntropyMaLean K=3 main-results campaign across miniF2F and
# ProofNet for both arms (control + treatment)
# against the local LM Studio panel (Goedel-Prover-V2-8B + BFS-Prover-V2-7B).
#
# Resumable: each per-benchmark output JSONL is opened in append mode and the
# orchestrator skips (problem_id, model_label) cells already finished. To
# restart from scratch, delete the per-benchmark output files first.
#
# Required env (or pass via .env):
#   LM_STUDIO_BASE_URL   — defaults to http://127.0.0.1:1234/v1
#   EVAL_PANEL=local     — auto-set below
# Optional overrides:
#   K                    — attempts per cell (default 3)
#   T_MAX                — whole-proof refine turns (default 4)
#   S_MAX                — tactic-step max steps (default 6)
#   N_PER_STEP           — BFS-V2 candidates per step (default 8)
#   CONTROL_CAP          — control rows per benchmark (default 20)
#   TREATMENT_CAP        — treatment rows per benchmark (default 0 = all)
#   MODEL_SEQUENCE       — comma-separated model labels run sequentially
#                          (default BFS-Prover-V2-7B,Goedel-Prover-V2-8B)
#   MAX_PARALLEL_CELLS   — concurrent (problem, model) cells (default 2)
#   LEAN_TIMEOUT         — seconds per Lean verifier call (default 300; bumped
#                          from 90 after 2026-05-20 finding that warm Lean runs
#                          heavy Mathlib tactics in 10–65s but cold/contended
#                          runs hit 100s+. Identity-cache + reduced parallelism
#                          do the rest.)
#   MODEL_TIMEOUT        — seconds per LM Studio call (default 600)
#   CAMPAIGN_TAG         — output sub-directory tag (default 2026-05-19-K3)
#   INPUT_DIR            — pre-built control/treatment dir (default /tmp/eml_campaign)
#   ACCEPTED_JSONL       — accepted treatment ledger used to build inputs

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-/tmp/eml_campaign}"
ACCEPTED_JSONL="${ACCEPTED_JSONL:-$ROOT/data/evaluation/treatment_inventory/final_curated/accepted.jsonl}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-2026-05-19-K3}"
OUT_DIR="$ROOT/data/evaluation/campaign_$CAMPAIGN_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export EVAL_PANEL="${EVAL_PANEL:-local}"
export LM_STUDIO_BASE_URL="${LM_STUDIO_BASE_URL:-http://127.0.0.1:1234/v1}"
export LEAN_VERIFIER="${LEAN_VERIFIER:-repl}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

K="${K:-3}"
T_MAX="${T_MAX:-4}"
S_MAX="${S_MAX:-6}"
N_PER_STEP="${N_PER_STEP:-8}"
N_PARALLEL="${N_PARALLEL:-1}"
BFS_TREE_SEARCH="${BFS_TREE_SEARCH:-0}"
BFS_TREE_MAX_NODES="${BFS_TREE_MAX_NODES:-64}"
CONTROL_CAP="${CONTROL_CAP:-20}"
TREATMENT_CAP="${TREATMENT_CAP:-0}"
MAX_PARALLEL_CELLS="${MAX_PARALLEL_CELLS:-2}"
LEAN_TIMEOUT="${LEAN_TIMEOUT:-300}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-600}"
MODEL_SEQUENCE="${MODEL_SEQUENCE:-BFS-Prover-V2-7B,Goedel-Prover-V2-8B}"
export MODEL_SEQUENCE

PY="${PY:-python3}"
IFS=',' read -r -a MODEL_LABELS <<< "$MODEL_SEQUENCE"

echo "campaign tag: $CAMPAIGN_TAG"
echo "input dir:    $INPUT_DIR"
echo "accepted:     $ACCEPTED_JSONL"
echo "output dir:   $OUT_DIR"
echo "panel:        $EVAL_PANEL  (server: $LM_STUDIO_BASE_URL)"
echo "lean verifier: $LEAN_VERIFIER"
echo "K=$K T_max=$T_MAX S_max=$S_MAX n_per_step=$N_PER_STEP"
echo "control_cap=$CONTROL_CAP treatment_cap=$TREATMENT_CAP"
echo "concurrency: $MAX_PARALLEL_CELLS cells (Lean ≤${LEAN_TIMEOUT}s, model ≤${MODEL_TIMEOUT}s)"
echo "model sequence: $MODEL_SEQUENCE"
echo

# 1. Refresh per-benchmark inputs (idempotent — overwrites in place).
$PY "$ROOT/scripts/generate/prepare_campaign_inputs.py" \
  --out-dir "$INPUT_DIR" \
  --accepted-jsonl "$ACCEPTED_JSONL"
echo

# 2. Sanity-probe the LM Studio panel. Fail-fast if the server is unreachable
#    or either model slot is missing so we don't burn time in vain.
$PY - <<PY_PROBE
import json
import os
import sys
import urllib.request

from src.evaluation.model_runner import MODEL_PANEL

print("resolved panel:")
for m in MODEL_PANEL:
    print(f"  {m.label:25} slug={m.provider_slug:25} paradigm={m.paradigm}")
if not MODEL_PANEL:
    sys.exit("MODEL_PANEL is empty — set EVAL_PANEL=local + LM_STUDIO_BASE_URL")
requested = [item.strip() for item in os.environ["MODEL_SEQUENCE"].split(",") if item.strip()]
available = {m.label for m in MODEL_PANEL}
missing = [label for label in requested if label not in available]
if missing:
    sys.exit(f"MODEL_SEQUENCE has unknown label(s): {missing}; available={sorted(available)}")
if any(m.backend == "lm_studio" for m in MODEL_PANEL):
    base = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        sys.exit(f"LM Studio /v1/models probe failed at {base}/models: {exc}")
    served = {item.get("id") for item in payload.get("data", []) if item.get("id")}
    needed = {m.provider_slug for m in MODEL_PANEL if m.label in requested}
    missing_slugs = sorted(needed - served)
    if missing_slugs:
        sys.exit(
            f"LM Studio is not serving required model slug(s): {missing_slugs}; "
            f"served={sorted(served)}"
        )
PY_PROBE
echo

# 3. Run each benchmark sequentially, and run the two local prover models
#    sequentially within each benchmark. Per-benchmark output JSONLs survive
#    a crash and can be resumed per model.
for bench in miniF2F proofnet; do
  lower="$(echo "$bench" | tr '[:upper:]' '[:lower:]')"
  ctrl_csv="$INPUT_DIR/${lower}_control.csv"
  trt_jsonl="$INPUT_DIR/${lower}_treatment.jsonl"
  out_jsonl="$OUT_DIR/${lower}_proof.jsonl"
  summary="$OUT_DIR/${lower}_summary.json"

  echo "=== $bench ==="
  echo "  control   $ctrl_csv"
  echo "  treatment $trt_jsonl"
  echo "  output    $out_jsonl"
  for model_label in "${MODEL_LABELS[@]}"; do
    model_label="$(echo "$model_label" | sed 's/^ *//;s/ *$//')"
    [ -n "$model_label" ] || continue
    model_log_slug="$(printf '%s' "$model_label" | tr ' /' '__' | tr -c 'A-Za-z0-9_.-' '_')"
    log="$LOG_DIR/${lower}_${model_log_slug}.log"
    echo "  model     $model_label"
    echo "  log       $log"

    $PY "$ROOT/scripts/archive/run_proof_evaluation.py" \
      --benchmark "$bench" \
      --control "$ctrl_csv" \
      --treatment "$trt_jsonl" \
      --output "$out_jsonl" \
      --summary "$summary" \
      --models "$model_label" \
      --K "$K" --T-max "$T_MAX" \
      --S-max "$S_MAX" --n-per-step "$N_PER_STEP" \
      --n-parallel "$N_PARALLEL" \
      $( [ "$BFS_TREE_SEARCH" = "1" ] && echo "--bfs-tree-search" ) \
      --bfs-tree-max-nodes "$BFS_TREE_MAX_NODES" \
      --control-cap "$CONTROL_CAP" --treatment-cap "$TREATMENT_CAP" \
      --max-parallel-cells "$MAX_PARALLEL_CELLS" \
      --lean-timeout "$LEAN_TIMEOUT" --model-timeout "$MODEL_TIMEOUT" \
      --resume \
      2>&1 | tee "$log"
  done
  echo
done

echo "campaign done -> $OUT_DIR"
ls -la "$OUT_DIR"/*.jsonl "$OUT_DIR"/*.json 2>/dev/null
