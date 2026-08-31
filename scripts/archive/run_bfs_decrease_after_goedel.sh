#!/usr/bin/env bash
# Wait for the treatment-only Goedel decrease campaign to finish, then run the
# treatment-only decrease-direction BFS-Prover campaign across four local
# workers and four Mac Studio workers. Control rows are not rerun for decrease;
# the paper should reuse the increase-campaign control baselines.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/data/evaluation/campaign_2026-05-25-bfs-verified}"
SOURCE_DIR="${SOURCE_DIR:-$OUT/handoff_inputs/goedel_decrease_2way}"
SHARD_DIR="${SHARD_DIR:-$OUT/handoff_inputs/bfs_decrease_treatment_only_8way}"
LOG_DIR="$OUT/logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

PY="${PY:-python3}"
POLL_SECONDS="${POLL_SECONDS:-120}"

LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:1234/v1}"
MAC_BASE_URL="${MAC_BASE_URL:-http://192.168.0.43:1234/v1}"

K="${K:-3}"
T_MAX="${T_MAX:-4}"
S_MAX="${S_MAX:-6}"
N_PER_STEP="${N_PER_STEP:-8}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-60}"
LEAN_TIMEOUT="${LEAN_TIMEOUT:-60}"
LEAN_REPL_POOL_SIZE="${LEAN_REPL_POOL_SIZE:-4}"
BFS_MAX_TOKENS="${BFS_MAX_TOKENS:-256}"
BFS_TEMPERATURE="${BFS_TEMPERATURE:-0.7}"

RUN_TAG="${RUN_TAG:-decrease_bfs_8way}"
ORCH_LOG="$LOG_DIR/${RUN_TAG}.log"
MARKER="$OUT/.${RUN_TAG}_started"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$ORCH_LOG" "$LOG_DIR/campaign.log"
}

screen_exists() {
  local name="$1"
  screen -ls | grep -Fq ".${name}"
}

goedel_done() {
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import json
import pathlib
import sys

base = pathlib.Path("data/evaluation/campaign_2026-05-25-bfs-verified")
expected = {
    ("miniF2F", "treatment"): 18,
    ("proofnet", "treatment"): 21,
}
seen = {}
for path in base.glob("*goedel_decrease_stmtfix*t2*proof.jsonl"):
    for line in path.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") != "Goedel-Prover-V2-8B":
            continue
        key = (row.get("benchmark"), row.get("arm"), row.get("problem_id"))
        seen[key] = row
counts = {}
for bench, arm, _pid in seen:
    counts[(bench, arm)] = counts.get((bench, arm), 0) + 1
done = all(counts.get(key, 0) >= val for key, val in expected.items())
print(json.dumps({
    "done": done,
    "counts": {f"{k[0]}:{k[1]}": counts.get(k, 0) for k in expected},
    "expected": {f"{k[0]}:{k[1]}": v for k, v in expected.items()},
}))
sys.exit(0 if done else 1)
PY
}

wait_for_goedel() {
  while true; do
    if status="$(goedel_done 2>/dev/null)"; then
      log "Goedel decrease complete: $status"
      break
    else
      status="$(goedel_done 2>/dev/null || true)"
      log "waiting for Goedel decrease completion: $status"
    fi
    sleep "$POLL_SECONDS"
  done
}

stop_goedel_screens() {
  for name in local_a local_b local_c local_d mac_a mac_b mac_c mac_d; do
    local screen_name="goedel_decrease_stmtfix_t2_8way_${name}"
    if screen_exists "$screen_name"; then
      log "stopping completed Goedel screen: $screen_name"
      env -u STY screen -S "$screen_name" -X quit || true
    fi
  done
}

reload_local_bfs_parallel8() {
  if ! command -v lms >/dev/null 2>&1; then
    log "lms CLI not found; skipping local BFS reload"
    return
  fi
  log "reloading local BFS with parallel=8 context=2048"
  lms unload goedel-prover-v2-8b >/dev/null 2>&1 || true
  lms unload bytedance-seed.bfs-prover-v2-7b >/dev/null 2>&1 || true
  lms load bytedance-seed.bfs-prover-v2-7b \
    --gpu max \
    --context-length 2048 \
    --parallel 8 \
    --ttl 3600 \
    --identifier bytedance-seed.bfs-prover-v2-7b \
    -y | tee -a "$ORCH_LOG"
  lms ps | tee -a "$ORCH_LOG" || true
}

write_shards() {
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<PY
import csv
import json
import pathlib
from collections import defaultdict

from src.evaluation.dataset import cap_rows, load_control_rows, load_treatment_rows

source_dir = pathlib.Path("$SOURCE_DIR")
shard_dir = pathlib.Path("$SHARD_DIR")
shard_dir.mkdir(parents=True, exist_ok=True)

names = [
    "local_a", "local_b", "local_c", "local_d",
    "mac_a", "mac_b", "mac_c", "mac_d",
]
shards = {name: defaultdict(list) for name in names}
manifest = {"source_dir": str(source_dir), "benchmarks": {}, "shards": {name: {} for name in names}}

control_fields = {}
for benchmark in ["miniF2F", "proofnet"]:
    rows = []
    seen = set()
    for suffix in ["local", "mac"]:
        path = source_dir / f"{benchmark}_treatment.{suffix}.jsonl"
        if path.exists():
            for row in cap_rows(load_treatment_rows(path, benchmark), 0):
                if row.problem_id in seen:
                    continue
                seen.add(row.problem_id)
                rows.append(row)

    manifest["benchmarks"][benchmark] = {
        "rows": len(rows),
        "control": 0,
        "treatment": sum(1 for row in rows if row.arm == "treatment"),
    }
    for idx, row in enumerate(rows):
        shards[names[idx % len(names)]][benchmark].append(row)

    fields = []
    for suffix in ["local", "mac"]:
        path = source_dir / f"{benchmark}_control.{suffix}.csv"
        if path.exists():
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                if fields:
                    break
    control_fields[benchmark] = fields

def write_control(benchmark, name, rows):
    fields = control_fields.get(benchmark, [])
    path = shard_dir / f"{benchmark}_control.{name}.csv"
    with path.open("w", newline="") as f:
        if not fields:
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.arm != "control":
                continue
            payload = {field: "" for field in fields}
            payload["id"] = row.problem_id
            payload["statement"] = row.statement
            payload["answer"] = row.gold_answer
            payload["formal_statement"] = row.formal_statement or ""
            payload["lean_header"] = row.lean_header or ""
            payload["formal_status"] = row.formal_status or ""
            writer.writerow(payload)

def write_treatment(benchmark, name, rows):
    path = shard_dir / f"{benchmark}_treatment.{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
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

for name in names:
    for benchmark in ["miniF2F", "proofnet"]:
        rows = shards[name][benchmark]
        write_control(benchmark, name, rows)
        write_treatment(benchmark, name, rows)
        manifest["shards"][name][benchmark] = {
            "rows": len(rows),
            "control": 0,
            "treatment": sum(1 for row in rows if row.arm == "treatment"),
        }

(shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
}

start_worker() {
  local name="$1"
  local base_url="$2"
  local screen_name="bfs_${RUN_TAG}_${name}"
  local log_file="$LOG_DIR/bfs_${RUN_TAG}_${name}.log"

  if screen_exists "$screen_name"; then
    log "$screen_name already exists; not starting duplicate"
    return
  fi

  env -u STY screen -dmS "$screen_name" zsh -lc "
    set -e
    cd '$ROOT'
    export EVAL_PANEL=local
    export LM_STUDIO_BASE_URL='$base_url'
    export EVAL_BFS_PROVER_SLUG=bytedance-seed.bfs-prover-v2-7b
    export EVAL_GOEDEL_V2_SLUG=disabled
    export BFS_MAX_TOKENS='$BFS_MAX_TOKENS'
    export BFS_TEMPERATURE='$BFS_TEMPERATURE'
    export LEAN_VERIFIER=repl
    export LEAN_REPL_POOL_SIZE='$LEAN_REPL_POOL_SIZE'
    export LEAN_CONCURRENCY_PER_EXPANSION='$LEAN_REPL_POOL_SIZE'
    export PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH}

    python -u scripts/archive/run_proof_evaluation.py \\
      --benchmark miniF2F \\
      --control '$SHARD_DIR/miniF2F_control.$name.csv' \\
      --treatment '$SHARD_DIR/miniF2F_treatment.$name.jsonl' \\
      --output '$OUT/minif2f_bfs_${RUN_TAG}_${name}_proof.jsonl' \\
      --summary '$OUT/minif2f_bfs_${RUN_TAG}_${name}_summary.json' \\
      --models BFS-Prover-V2-7B \\
      --K '$K' \\
      --T-max '$T_MAX' \\
      --S-max '$S_MAX' \\
      --n-per-step '$N_PER_STEP' \\
      --model-timeout '$MODEL_TIMEOUT' \\
      --lean-timeout '$LEAN_TIMEOUT' \\
      --max-parallel-cells 1 \\
      --resume >> '$log_file' 2>&1

    python -u scripts/archive/run_proof_evaluation.py \\
      --benchmark proofnet \\
      --control '$SHARD_DIR/proofnet_control.$name.csv' \\
      --treatment '$SHARD_DIR/proofnet_treatment.$name.jsonl' \\
      --output '$OUT/proofnet_bfs_${RUN_TAG}_${name}_proof.jsonl' \\
      --summary '$OUT/proofnet_bfs_${RUN_TAG}_${name}_summary.json' \\
      --models BFS-Prover-V2-7B \\
      --K '$K' \\
      --T-max '$T_MAX' \\
      --S-max '$S_MAX' \\
      --n-per-step '$N_PER_STEP' \\
      --model-timeout '$MODEL_TIMEOUT' \\
      --lean-timeout '$LEAN_TIMEOUT' \\
      --max-parallel-cells 1 \\
      --resume >> '$log_file' 2>&1
  "
  log "started $screen_name using $base_url"
}

if [[ -f "$MARKER" ]]; then
  log "marker exists at $MARKER; exiting to avoid duplicate BFS start"
  exit 0
fi

wait_for_goedel
date '+%Y-%m-%d %H:%M:%S %Z' > "$MARKER"
stop_goedel_screens
reload_local_bfs_parallel8
write_shards | tee -a "$ORCH_LOG"

for name in local_a local_b local_c local_d; do
  start_worker "$name" "$LOCAL_BASE_URL"
done
for name in mac_a mac_b mac_c mac_d; do
  start_worker "$name" "$MAC_BASE_URL"
done

log "BFS decrease 8-way run started"
