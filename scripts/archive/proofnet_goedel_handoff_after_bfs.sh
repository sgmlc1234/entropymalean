#!/usr/bin/env bash
# Wait until Mac Studio has a free Goedel lane (miniF2F Goedel complete) or
# until the miniF2F BFS support run finishes, then split the remaining ProofNet
# Goedel work across local + Mac Studio without concurrent writes to the same
# JSONL.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/data/evaluation/campaign_2026-05-25-bfs-verified}"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

PY="${PY:-python3}"
INPUT_DIR="${INPUT_DIR:-/tmp/eml_campaign_final_replfix}"

MINIF2F_BFS_JSONL="${MINIF2F_BFS_JSONL:-$OUT/minif2f_proof.jsonl}"
MINIF2F_GOEDEL_JSONL="${MINIF2F_GOEDEL_JSONL:-$OUT/minif2f_goedel_remote_t2_ref_proof.jsonl}"
PROOFNET_MAIN_JSONL="${PROOFNET_MAIN_JSONL:-$OUT/proofnet_goedel_local_t2_ref_proof.jsonl}"
PROOFNET_MAIN_SUMMARY="${PROOFNET_MAIN_SUMMARY:-$OUT/proofnet_goedel_local_t2_ref_summary.json}"
PROOFNET_MAC_JSONL="${PROOFNET_MAC_JSONL:-$OUT/proofnet_goedel_mac_t2_ref_support_proof.jsonl}"
PROOFNET_MAC_SUMMARY="${PROOFNET_MAC_SUMMARY:-$OUT/proofnet_goedel_mac_t2_ref_support_summary.json}"

SHARD_DIR="${SHARD_DIR:-$OUT/handoff_inputs/proofnet_goedel_after_bfs}"
LOCAL_CONTROL="$SHARD_DIR/proofnet_control.local.csv"
LOCAL_TREATMENT="$SHARD_DIR/proofnet_treatment.local.jsonl"
MAC_CONTROL="$SHARD_DIR/proofnet_control.mac.csv"
MAC_TREATMENT="$SHARD_DIR/proofnet_treatment.mac.jsonl"

LOCAL_SCREEN="${LOCAL_SCREEN:-proofnet_goedel_ref_t2_local_shard}"
MAC_SCREEN="${MAC_SCREEN:-proofnet_goedel_ref_t2_mac_support}"
HANDOFF_MARKER="$OUT/.proofnet_goedel_after_bfs_handoff_started"

LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:1234/v1}"
MAC_BASE_URL="${MAC_BASE_URL:-http://192.168.0.43:1234/v1}"

K="${K:-3}"
T_MAX="${T_MAX:-2}"
S_MAX="${S_MAX:-6}"
N_PER_STEP="${N_PER_STEP:-6}"
N_PARALLEL="${N_PARALLEL:-1}"
CONTROL_CAP="${CONTROL_CAP:-20}"
TREATMENT_CAP="${TREATMENT_CAP:-0}"
MAX_PARALLEL_CELLS="${MAX_PARALLEL_CELLS:-1}"
LEAN_TIMEOUT="${LEAN_TIMEOUT:-60}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-90}"
GOEDEL_MAX_TOKENS="${GOEDEL_MAX_TOKENS:-4096}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$LOG_DIR/campaign.log"
}

jsonl_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  "$PY" - "$path" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
print(sum(1 for line in p.open() if line.strip()))
PY
}

screen_exists() {
  local name="$1"
  ps -axo command | grep -F "SCREEN -dmS ${name}" | grep -Fvq grep
}

wait_for_mac_support_slot() {
  log "handoff watcher active; waiting for miniF2F Goedel or miniF2F BFS to reach 120 rows"
  while true; do
    bfs_rows="$(jsonl_rows "$MINIF2F_BFS_JSONL")"
    goedel_rows="$(jsonl_rows "$MINIF2F_GOEDEL_JSONL")"
    if [[ "$goedel_rows" -ge 120 ]]; then
      log "miniF2F Goedel reached $goedel_rows/120 rows; starting ProofNet Goedel handoff while BFS continues if needed"
      break
    fi
    if [[ "$bfs_rows" -ge 120 ]]; then
      log "miniF2F BFS reached $bfs_rows/120 rows; starting ProofNet Goedel handoff"
      break
    fi
    if ! screen_exists "minif2f_goedel_remote_ref_t2" && ! screen_exists "minif2f_bfs_remote_repl_resume2"; then
      log "neither miniF2F Goedel nor miniF2F BFS screen is running; current rows: Goedel=$goedel_rows/120 BFS=$bfs_rows/120. Continuing to wait unless one is complete."
      if [[ "$goedel_rows" -ge 120 || "$bfs_rows" -ge 120 ]]; then
        break
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

stop_original_proofnet_local() {
  if screen_exists "proofnet_goedel_ref_t2"; then
    log "stopping original single-worker ProofNet Goedel screen after preserving completed rows"
    env -u STY screen -S proofnet_goedel_ref_t2 -X quit || true
    sleep 5
  else
    log "original ProofNet Goedel screen is already absent"
  fi
}

write_shards() {
  mkdir -p "$SHARD_DIR"
  "$PY" - <<PY
import csv
import json
import pathlib
from collections import OrderedDict

input_dir = pathlib.Path("$INPUT_DIR")
main_jsonl = pathlib.Path("$PROOFNET_MAIN_JSONL")
shard_dir = pathlib.Path("$SHARD_DIR")
control_src = input_dir / "proofnet_control.csv"
treatment_src = input_dir / "proofnet_treatment.jsonl"

local_control = pathlib.Path("$LOCAL_CONTROL")
local_treatment = pathlib.Path("$LOCAL_TREATMENT")
mac_control = pathlib.Path("$MAC_CONTROL")
mac_treatment = pathlib.Path("$MAC_TREATMENT")

model_label = "Goedel-Prover-V2-8B"
control_cap = int("$CONTROL_CAP")
treatment_cap = int("$TREATMENT_CAP")

completed = set()
if main_jsonl.exists():
    for line in main_jsonl.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == model_label and row.get("problem_id"):
            completed.add(str(row["problem_id"]))

def read_control():
    with control_src.open() as f:
        rows = list(csv.DictReader(f))
    seen = set()
    out = []
    for row in rows:
        pid = str(row.get("id") or row.get("release_id") or row.get("problem_id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append((pid, row))
        if control_cap > 0 and len(out) >= control_cap:
            break
    return out, rows[0].keys() if rows else []

def read_treatment():
    out = []
    seen = set()
    for line in treatment_src.open():
        if not line.strip():
            continue
        row = json.loads(line)
        bench = str(row.get("benchmark") or "").lower()
        if bench and bench != "proofnet":
            continue
        cert = row.get("certificate") if isinstance(row.get("certificate"), dict) else {}
        ok = row.get("status") == "certified" or cert.get("status") == "certified" or str(row.get("_manual_qa_decision") or "").lower() == "accept"
        if not ok or not str(row.get("statement") or "").strip():
            continue
        formal_src = str(row.get("formal_statement") or row.get("lean_code") or "").strip()
        if not formal_src:
            continue
        pid = str(row.get("eval_problem_id") or row.get("problem_id") or row.get("id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append((pid, row))
        if treatment_cap > 0 and len(out) >= treatment_cap:
            break
    return out

controls, control_fields = read_control()
treatments = read_treatment()
remaining = [("control", pid, row) for pid, row in controls if pid not in completed]
remaining += [("treatment", pid, row) for pid, row in treatments if pid not in completed]

local = []
mac = []
for i, item in enumerate(remaining):
    (local if i % 2 == 0 else mac).append(item)

def write_control(path, items):
    rows = [row for arm, pid, row in items if arm == "control"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(control_fields)
    with path.open("w", newline="") as f:
        if not fields:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_treatment(path, items):
    rows = [row for arm, pid, row in items if arm == "treatment"]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")

write_control(local_control, local)
write_treatment(local_treatment, local)
write_control(mac_control, mac)
write_treatment(mac_treatment, mac)

manifest = {
    "completed_before_handoff": len(completed),
    "remaining": len(remaining),
    "local_rows": len(local),
    "mac_rows": len(mac),
    "local_control": sum(1 for arm, _, _ in local if arm == "control"),
    "local_treatment": sum(1 for arm, _, _ in local if arm == "treatment"),
    "mac_control": sum(1 for arm, _, _ in mac if arm == "control"),
    "mac_treatment": sum(1 for arm, _, _ in mac if arm == "treatment"),
}
(shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
}

start_worker() {
  local screen_name="$1"
  local base_url="$2"
  local control="$3"
  local treatment="$4"
  local output="$5"
  local summary="$6"
  local log_file="$7"

  if screen_exists "$screen_name"; then
    log "screen $screen_name already exists; not starting duplicate"
    return
  fi

  env -u STY screen -dmS "$screen_name" bash -lc "
    set -euo pipefail
    cd '$ROOT'
    export EVAL_PANEL=local
    export LM_STUDIO_BASE_URL='$base_url'
    export EVAL_GOEDEL_V2_SLUG='goedel-prover-v2-8b'
    export EVAL_BFS_PROVER_SLUG='disabled'
    export GOEDEL_PROMPT_STYLE='goedel_v2'
    export GOEDEL_MAX_TOKENS='$GOEDEL_MAX_TOKENS'
    export LEAN_VERIFIER=repl
    export PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH}
    echo '[$screen_name-start] '\$(date '+%Y-%m-%d %H:%M:%S %Z') | tee -a '$LOG_DIR/campaign.log'
    python -u scripts/archive/run_proof_evaluation.py \\
      --benchmark proofnet \\
      --control '$control' \\
      --treatment '$treatment' \\
      --output '$output' \\
      --summary '$summary' \\
      --models Goedel-Prover-V2-8B \\
      --K '$K' \\
      --T-max '$T_MAX' \\
      --S-max '$S_MAX' \\
      --n-per-step '$N_PER_STEP' \\
      --n-parallel '$N_PARALLEL' \\
      --control-cap '$CONTROL_CAP' \\
      --treatment-cap '$TREATMENT_CAP' \\
      --max-parallel-cells '$MAX_PARALLEL_CELLS' \\
      --lean-timeout '$LEAN_TIMEOUT' \\
      --model-timeout '$MODEL_TIMEOUT' \\
      --resume 2>&1 | tee -a '$log_file'
    echo '[$screen_name-done] '\$(date '+%Y-%m-%d %H:%M:%S %Z') | tee -a '$LOG_DIR/campaign.log'
  "
  log "started $screen_name -> $output"
}

merge_mac_support() {
  "$PY" - <<PY
import json
import pathlib
import shutil

from src.evaluation.proof_orchestrator import summarize_proof_jsonl

main = pathlib.Path("$PROOFNET_MAIN_JSONL")
support = pathlib.Path("$PROOFNET_MAC_JSONL")
summary = pathlib.Path("$PROOFNET_MAIN_SUMMARY")

main.parent.mkdir(parents=True, exist_ok=True)
rows = []
seen = set()
if main.exists():
    for line in main.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("problem_id"), row.get("model"))
        if key not in seen:
            rows.append(row)
            seen.add(key)

added = 0
if support.exists():
    for line in support.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("problem_id"), row.get("model"))
        if key not in seen:
            rows.append(row)
            seen.add(key)
            added += 1

backup = main.with_suffix(main.suffix + ".pre_support_merge.bak")
if main.exists():
    shutil.copy2(main, backup)
tmp = main.with_suffix(main.suffix + ".tmp")
with tmp.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
tmp.replace(main)
payload = summarize_proof_jsonl(main, summary)
print(json.dumps({
    "main_rows": len(rows),
    "support_added": added,
    "backup": str(backup),
    "summary": str(summary),
}, indent=2))
PY
}

if [[ -f "$HANDOFF_MARKER" ]]; then
  log "handoff marker already exists at $HANDOFF_MARKER; exiting to avoid duplicate orchestration"
  exit 0
fi

wait_for_mac_support_slot
date '+%Y-%m-%d %H:%M:%S %Z' > "$HANDOFF_MARKER"
stop_original_proofnet_local
write_shards | tee -a "$LOG_DIR/proofnet_goedel_handoff_after_bfs.log"

start_worker "$LOCAL_SCREEN" "$LOCAL_BASE_URL" "$LOCAL_CONTROL" "$LOCAL_TREATMENT" "$PROOFNET_MAIN_JSONL" "$PROOFNET_MAIN_SUMMARY" "$LOG_DIR/proofnet_Goedel-Prover-V2-8B.local_shard_after_bfs.log"
start_worker "$MAC_SCREEN" "$MAC_BASE_URL" "$MAC_CONTROL" "$MAC_TREATMENT" "$PROOFNET_MAC_JSONL" "$PROOFNET_MAC_SUMMARY" "$LOG_DIR/proofnet_Goedel-Prover-V2-8B.mac_support_after_bfs.log"

log "waiting for local/Mac ProofNet Goedel shard screens to finish"
while screen_exists "$LOCAL_SCREEN" || screen_exists "$MAC_SCREEN"; do
  sleep "$POLL_SECONDS"
done

log "both ProofNet Goedel shard screens finished; merging Mac support output"
merge_mac_support | tee -a "$LOG_DIR/proofnet_goedel_handoff_after_bfs.log"
log "ProofNet Goedel handoff complete"
