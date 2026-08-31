#!/usr/bin/env bash
# Wait for the Mac Studio miniF2F Goedel stmtfix run to finish, then stop the
# current local ProofNet stmtfix workers and split the remaining ProofNet work
# across local front/back plus Mac front/back.  All outputs stay separate so the
# final paper table can merge by (problem_id, model) after the run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/data/evaluation/campaign_2026-05-25-bfs-verified}"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

PY="${PY:-python3}"
POLL_SECONDS="${POLL_SECONDS:-60}"

SOURCE_DIR="${SOURCE_DIR:-$OUT/handoff_inputs/goedel_codeonly_front_back_full120/proofnet}"
SHARD_DIR="${SHARD_DIR:-$OUT/handoff_inputs/proofnet_goedel_stmtfix_4way}"

MINIF2F_FRONT="${MINIF2F_FRONT:-$OUT/minif2f_goedel_codeonly_stmtfix_t2_front_proof.jsonl}"
MINIF2F_BACK="${MINIF2F_BACK:-$OUT/minif2f_goedel_codeonly_stmtfix_t2_back_proof.jsonl}"

PROOFNET_FRONT="${PROOFNET_FRONT:-$OUT/proofnet_goedel_codeonly_stmtfix_t2_front_proof.jsonl}"
PROOFNET_BACK="${PROOFNET_BACK:-$OUT/proofnet_goedel_codeonly_stmtfix_t2_back_proof.jsonl}"

HANDOFF_MARKER="${HANDOFF_MARKER:-$OUT/.proofnet_goedel_stmtfix_4way_handoff_started}"
HANDOFF_LOG="$LOG_DIR/proofnet_goedel_stmtfix_4way_handoff.log"

LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:1234/v1}"
MAC_BASE_URL="${MAC_BASE_URL:-http://192.168.0.43:1234/v1}"

K="${K:-3}"
T_MAX="${T_MAX:-2}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-60}"
LEAN_TIMEOUT="${LEAN_TIMEOUT:-60}"
GOEDEL_MAX_TOKENS="${GOEDEL_MAX_TOKENS:-3072}"
GOEDEL_TEMPERATURE="${GOEDEL_TEMPERATURE:-1.0}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$HANDOFF_LOG" "$LOG_DIR/campaign.log"
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
  screen -ls | grep -Fq ".${name}"
}

wait_for_minif2f_done() {
  log "waiting for miniF2F stmtfix Goedel shards to reach 120 rows"
  while true; do
    local front back total
    front="$(jsonl_rows "$MINIF2F_FRONT")"
    back="$(jsonl_rows "$MINIF2F_BACK")"
    total=$((front + back))
    log "miniF2F stmtfix progress: front=$front back=$back total=$total/120"
    if [[ "$total" -ge 120 ]]; then
      break
    fi
    sleep "$POLL_SECONDS"
  done
}

stop_current_proofnet_workers() {
  for name in proofnet_goedel_stmtfix_front proofnet_goedel_stmtfix_back; do
    if screen_exists "$name"; then
      log "stopping $name before 4-way ProofNet handoff"
      env -u STY screen -S "$name" -X quit || true
    else
      log "$name is already absent"
    fi
  done
  sleep 5
}

write_4way_shards() {
  mkdir -p "$SHARD_DIR"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<PY
import csv
import json
import pathlib

from src.evaluation.dataset import cap_rows, load_control_rows, load_treatment_rows

source_dir = pathlib.Path("$SOURCE_DIR")
shard_dir = pathlib.Path("$SHARD_DIR")
current_outputs = [pathlib.Path("$PROOFNET_FRONT"), pathlib.Path("$PROOFNET_BACK")]

completed = set()
for path in current_outputs:
    if not path.exists():
        continue
    for line in path.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == "Goedel-Prover-V2-8B" and row.get("problem_id"):
            completed.add(str(row["problem_id"]))

control_sources = [
    source_dir / "proofnet_control.front.csv",
    source_dir / "proofnet_control.back.csv",
]
treatment_sources = [
    source_dir / "proofnet_treatment.front.jsonl",
    source_dir / "proofnet_treatment.back.jsonl",
]

rows = []
seen = set()
for path in control_sources:
    for row in cap_rows(load_control_rows(path, "proofnet"), 20):
        if row.problem_id in seen:
            continue
        seen.add(row.problem_id)
        rows.append(row)
for path in treatment_sources:
    for row in cap_rows(load_treatment_rows(path, "proofnet"), 0):
        if row.problem_id in seen:
            continue
        seen.add(row.problem_id)
        rows.append(row)

remaining = [row for row in rows if row.problem_id not in completed]
names = ["local_a", "local_b", "mac_a", "mac_b"]
shards = {name: [] for name in names}
for idx, row in enumerate(remaining):
    shards[names[idx % len(names)]].append(row)

control_fields = []
for path in control_sources:
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            control_fields = list(reader.fieldnames or [])
            if control_fields:
                break

def write_control(name, items):
    path = shard_dir / f"proofnet_control.{name}.csv"
    with path.open("w", newline="") as f:
        if not control_fields:
            return
        writer = csv.DictWriter(f, fieldnames=control_fields)
        writer.writeheader()
        for row in items:
            if row.arm != "control":
                continue
            payload = {field: "" for field in control_fields}
            payload["id"] = row.problem_id
            payload["statement"] = row.statement
            payload["answer"] = row.gold_answer
            payload["formal_statement"] = row.formal_statement or ""
            payload["lean_header"] = row.lean_header or ""
            payload["formal_status"] = row.formal_status or ""
            writer.writerow(payload)

def write_treatment(name, items):
    path = shard_dir / f"proofnet_treatment.{name}.jsonl"
    with path.open("w") as f:
        for row in items:
            if row.arm != "treatment":
                continue
            f.write(json.dumps({
                "problem_id": row.problem_id,
                "benchmark": row.benchmark,
                "statement": row.statement,
                "answer": row.gold_answer,
                "generation": row.generation,
                "family": row.family,
                "lean_level": row.lean_level,
                "formal_statement": row.formal_statement,
                "lean_header": row.lean_header,
                "certificate": {"status": "certified"},
            }, ensure_ascii=False) + "\\n")

manifest = {
    "source_total_unique": len(rows),
    "completed_before_handoff": len(completed),
    "remaining": len(remaining),
    "shards": {},
}
for name, items in shards.items():
    write_control(name, items)
    write_treatment(name, items)
    manifest["shards"][name] = {
        "rows": len(items),
        "control": sum(1 for row in items if row.arm == "control"),
        "treatment": sum(1 for row in items if row.arm == "treatment"),
    }

(shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
}

start_worker() {
  local name="$1"
  local base_url="$2"
  local control="$SHARD_DIR/proofnet_control.$name.csv"
  local treatment="$SHARD_DIR/proofnet_treatment.$name.jsonl"
  local output="$OUT/proofnet_goedel_codeonly_stmtfix_t2_handoff_${name}_proof.jsonl"
  local summary="$OUT/proofnet_goedel_codeonly_stmtfix_t2_handoff_${name}_summary.json"
  local log_file="$LOG_DIR/proofnet_Goedel-Prover-V2-8B.codeonly_stmtfix_t2_handoff_${name}.log"
  local screen_name="proofnet_goedel_stmtfix_handoff_${name}"

  if screen_exists "$screen_name"; then
    log "$screen_name already exists; not starting duplicate"
    return
  fi

  env -u STY screen -dmS "$screen_name" zsh -lc "
    cd '$ROOT'
    exec env EVAL_PANEL=local \\
      LM_STUDIO_BASE_URL='$base_url' \\
      EVAL_GOEDEL_V2_SLUG=goedel-prover-v2-8b \\
      EVAL_BFS_PROVER_SLUG=disabled \\
      GOEDEL_PROMPT_STYLE=goedel_v1 \\
      GOEDEL_MAX_TOKENS='$GOEDEL_MAX_TOKENS' \\
      GOEDEL_TEMPERATURE='$GOEDEL_TEMPERATURE' \\
      LEAN_VERIFIER=repl \\
      PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH} \\
      python -u scripts/archive/run_proof_evaluation.py \\
        --benchmark proofnet \\
        --control '$control' \\
        --treatment '$treatment' \\
        --output '$output' \\
        --summary '$summary' \\
        --models Goedel-Prover-V2-8B \\
        --K '$K' \\
        --T-max '$T_MAX' \\
        --max-tokens '$GOEDEL_MAX_TOKENS' \\
        --model-timeout '$MODEL_TIMEOUT' \\
        --lean-timeout '$LEAN_TIMEOUT' \\
        --max-parallel-cells 1 \\
        --resume >> '$log_file' 2>&1
  "
  log "started $screen_name -> $output"
}

if [[ -f "$HANDOFF_MARKER" ]]; then
  log "handoff marker exists at $HANDOFF_MARKER; exiting to avoid duplicate orchestration"
  exit 0
fi

wait_for_minif2f_done
date '+%Y-%m-%d %H:%M:%S %Z' > "$HANDOFF_MARKER"
stop_current_proofnet_workers
write_4way_shards | tee -a "$HANDOFF_LOG"

start_worker "local_a" "$LOCAL_BASE_URL"
start_worker "local_b" "$LOCAL_BASE_URL"
start_worker "mac_a" "$MAC_BASE_URL"
start_worker "mac_b" "$MAC_BASE_URL"

log "ProofNet stmtfix 4-way handoff started"
