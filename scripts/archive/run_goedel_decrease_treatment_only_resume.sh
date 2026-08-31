#!/usr/bin/env bash
# Resume only the decrease-direction treatment rows for Goedel. Control rows
# are intentionally excluded; use the increase-campaign control baselines.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/data/evaluation/campaign_2026-05-25-bfs-verified}"
SOURCE_DIR="${SOURCE_DIR:-$OUT/handoff_inputs/goedel_decrease_2way}"
SHARD_DIR="${SHARD_DIR:-$OUT/handoff_inputs/goedel_decrease_treatment_only_8way}"
LOG_DIR="$OUT/logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

PY="${PY:-python3}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:1234/v1}"
MAC_BASE_URL="${MAC_BASE_URL:-http://192.168.0.43:1234/v1}"

K="${K:-3}"
T_MAX="${T_MAX:-2}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-60}"
LEAN_TIMEOUT="${LEAN_TIMEOUT:-60}"
GOEDEL_MAX_TOKENS="${GOEDEL_MAX_TOKENS:-3072}"
GOEDEL_TEMPERATURE="${GOEDEL_TEMPERATURE:-1.0}"

RUN_TAG="${RUN_TAG:-decrease_stmtfix_t2_treatment_only}"
ORCH_LOG="$LOG_DIR/goedel_${RUN_TAG}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$ORCH_LOG" "$LOG_DIR/campaign.log"
}

screen_exists() {
  local name="$1"
  screen -ls | grep -Fq ".${name}"
}

reload_local_goedel_parallel4() {
  if ! command -v lms >/dev/null 2>&1; then
    log "lms CLI not found; skipping local model reload"
    return
  fi
  log "ensuring local Goedel parallel=4 context=4096"
  lms unload goedel-prover-v2-8b >/dev/null 2>&1 || true
  lms load goedel-prover-v2-8b --parallel 4 --context-length 4096 --ttl 3600 -y | tee -a "$ORCH_LOG"
  lms ps | tee -a "$ORCH_LOG" || true
}

write_shards() {
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<PY
import csv
import json
import pathlib
from collections import defaultdict

from src.evaluation.dataset import cap_rows, load_treatment_rows

source_dir = pathlib.Path("$SOURCE_DIR")
shard_dir = pathlib.Path("$SHARD_DIR")
shard_dir.mkdir(parents=True, exist_ok=True)
out = pathlib.Path("$OUT")

completed = set()
for path in out.glob("*goedel_decrease_stmtfix*t2*proof.jsonl"):
    for line in path.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") != "Goedel-Prover-V2-8B":
            continue
        if row.get("arm") != "treatment":
            continue
        benchmark = row.get("benchmark")
        pid = row.get("problem_id")
        if benchmark and pid:
            completed.add((str(benchmark).lower(), str(pid)))

names = [
    "local_a", "local_b", "local_c", "local_d",
    "mac_a", "mac_b", "mac_c", "mac_d",
]
shards = {name: defaultdict(list) for name in names}
manifest = {
    "source_dir": str(source_dir),
    "completed_treatment_excluded": len(completed),
    "benchmarks": {},
    "shards": {name: {} for name in names},
}

for benchmark in ["miniF2F", "proofnet"]:
    rows = []
    seen = set()
    for suffix in ["local", "mac"]:
        path = source_dir / f"{benchmark}_treatment.{suffix}.jsonl"
        if not path.exists():
            continue
        for row in cap_rows(load_treatment_rows(path, benchmark), 0):
            key = (benchmark.lower(), row.problem_id)
            if row.problem_id in seen or key in completed:
                continue
            seen.add(row.problem_id)
            rows.append(row)

    manifest["benchmarks"][benchmark] = {
        "remaining_treatment": len(rows),
    }
    for idx, row in enumerate(rows):
        shards[names[idx % len(names)]][benchmark].append(row)

def write_empty_control(benchmark, name):
    path = shard_dir / f"{benchmark}_control.{name}.csv"
    fields = ["id", "statement", "answer", "formal_statement", "lean_header", "formal_status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

def write_treatment(benchmark, name, rows):
    path = shard_dir / f"{benchmark}_treatment.{name}.jsonl"
    with path.open("w") as f:
        for row in rows:
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
        write_empty_control(benchmark, name)
        write_treatment(benchmark, name, rows)
        manifest["shards"][name][benchmark] = {
            "rows": len(rows),
            "control": 0,
            "treatment": len(rows),
        }

(shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
}

start_worker() {
  local name="$1"
  local base_url="$2"
  local screen_name="goedel_${RUN_TAG}_${name}"
  local log_file="$LOG_DIR/goedel_${RUN_TAG}_${name}.log"

  if screen_exists "$screen_name"; then
    log "$screen_name already exists; not starting duplicate"
    return
  fi

  env -u STY screen -dmS "$screen_name" zsh -lc "
    set -e
    cd '$ROOT'
    export EVAL_PANEL=local
    export LM_STUDIO_BASE_URL='$base_url'
    export EVAL_GOEDEL_V2_SLUG=goedel-prover-v2-8b
    export EVAL_BFS_PROVER_SLUG=disabled
    export GOEDEL_PROMPT_STYLE=goedel_v1
    export GOEDEL_MAX_TOKENS='$GOEDEL_MAX_TOKENS'
    export GOEDEL_TEMPERATURE='$GOEDEL_TEMPERATURE'
    export LEAN_VERIFIER=repl
    export PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH}

    python -u scripts/archive/run_proof_evaluation.py \\
      --benchmark miniF2F \\
      --control '$SHARD_DIR/miniF2F_control.$name.csv' \\
      --treatment '$SHARD_DIR/miniF2F_treatment.$name.jsonl' \\
      --output '$OUT/minif2f_goedel_${RUN_TAG}_${name}_proof.jsonl' \\
      --summary '$OUT/minif2f_goedel_${RUN_TAG}_${name}_summary.json' \\
      --models Goedel-Prover-V2-8B \\
      --K '$K' \\
      --T-max '$T_MAX' \\
      --max-tokens '$GOEDEL_MAX_TOKENS' \\
      --model-timeout '$MODEL_TIMEOUT' \\
      --lean-timeout '$LEAN_TIMEOUT' \\
      --max-parallel-cells 1 \\
      --resume >> '$log_file' 2>&1

    python -u scripts/archive/run_proof_evaluation.py \\
      --benchmark proofnet \\
      --control '$SHARD_DIR/proofnet_control.$name.csv' \\
      --treatment '$SHARD_DIR/proofnet_treatment.$name.jsonl' \\
      --output '$OUT/proofnet_goedel_${RUN_TAG}_${name}_proof.jsonl' \\
      --summary '$OUT/proofnet_goedel_${RUN_TAG}_${name}_summary.json' \\
      --models Goedel-Prover-V2-8B \\
      --K '$K' \\
      --T-max '$T_MAX' \\
      --max-tokens '$GOEDEL_MAX_TOKENS' \\
      --model-timeout '$MODEL_TIMEOUT' \\
      --lean-timeout '$LEAN_TIMEOUT' \\
      --max-parallel-cells 1 \\
      --resume >> '$log_file' 2>&1
  "
  log "started $screen_name using $base_url"
}

reload_local_goedel_parallel4
write_shards | tee -a "$ORCH_LOG"

for name in local_a local_b local_c local_d; do
  start_worker "$name" "$LOCAL_BASE_URL"
done
for name in mac_a mac_b mac_c mac_d; do
  start_worker "$name" "$MAC_BASE_URL"
done

log "Goedel decrease treatment-only resume started"
