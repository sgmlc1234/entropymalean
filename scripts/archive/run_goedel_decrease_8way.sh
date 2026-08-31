#!/usr/bin/env bash
# Split the remaining decrease-direction Goedel stmtfix campaign across
# four local LM Studio workers and four Mac Studio workers. Existing 2-way
# decrease outputs are treated as completed provenance and are not overwritten.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/data/evaluation/campaign_2026-05-25-bfs-verified}"
SOURCE_DIR="${SOURCE_DIR:-$OUT/handoff_inputs/goedel_decrease_2way}"
SHARD_DIR="${SHARD_DIR:-$OUT/handoff_inputs/goedel_decrease_8way}"
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

RUN_TAG="${RUN_TAG:-decrease_stmtfix_t2_8way}"
ORCH_LOG="$LOG_DIR/goedel_${RUN_TAG}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$ORCH_LOG" "$LOG_DIR/campaign.log"
}

screen_exists() {
  local name="$1"
  screen -ls | grep -Fq ".${name}"
}

stop_old_2way() {
  for name in goedel_decrease_stmtfix_local goedel_decrease_stmtfix_mac; do
    if screen_exists "$name"; then
      log "stopping old 2-way screen: $name"
      env -u STY screen -S "$name" -X quit || true
    else
      log "old 2-way screen absent: $name"
    fi
  done
}

reload_local_goedel_parallel4() {
  if ! command -v lms >/dev/null 2>&1; then
    log "lms CLI not found; skipping local model reload"
    return
  fi
  log "reloading local Goedel with parallel=4 context=4096"
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

from src.evaluation.dataset import cap_rows, load_control_rows, load_treatment_rows

source_dir = pathlib.Path("$SOURCE_DIR")
shard_dir = pathlib.Path("$SHARD_DIR")
out = pathlib.Path("$OUT")

old_outputs = [
    out / "minif2f_goedel_decrease_stmtfix_t2_local_proof.jsonl",
    out / "minif2f_goedel_decrease_stmtfix_t2_mac_proof.jsonl",
    out / "proofnet_goedel_decrease_stmtfix_t2_local_proof.jsonl",
    out / "proofnet_goedel_decrease_stmtfix_t2_mac_proof.jsonl",
]

completed = set()
for path in old_outputs:
    if not path.exists():
        continue
    for line in path.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = row.get("problem_id")
        benchmark = row.get("benchmark")
        model = row.get("model")
        if pid and benchmark and model == "Goedel-Prover-V2-8B":
            completed.add((str(benchmark).lower(), str(pid)))

def load_benchmark(benchmark):
    control_paths = [
        source_dir / f"{benchmark}_control.local.csv",
        source_dir / f"{benchmark}_control.mac.csv",
    ]
    treatment_paths = [
        source_dir / f"{benchmark}_treatment.local.jsonl",
        source_dir / f"{benchmark}_treatment.mac.jsonl",
    ]
    rows = []
    seen = set()
    for path in control_paths:
        if not path.exists():
            continue
        for row in cap_rows(load_control_rows(path, benchmark), 20):
            key = row.problem_id
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    for path in treatment_paths:
        if not path.exists():
            continue
        for row in cap_rows(load_treatment_rows(path, benchmark), 0):
            key = row.problem_id
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return [row for row in rows if (benchmark.lower(), row.problem_id) not in completed]

names = [
    "local_a", "local_b", "local_c", "local_d",
    "mac_a", "mac_b", "mac_c", "mac_d",
]
shards = {name: defaultdict(list) for name in names}
manifest = {
    "source_dir": str(source_dir),
    "old_outputs_excluded": [str(p) for p in old_outputs],
    "completed_excluded": len(completed),
    "benchmarks": {},
    "shards": {name: {} for name in names},
}

control_fields = {}
for benchmark in ["miniF2F", "proofnet"]:
    rows = load_benchmark(benchmark)
    manifest["benchmarks"][benchmark] = {
        "remaining": len(rows),
        "control": sum(1 for row in rows if row.arm == "control"),
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

def write_control(benchmark, name, items):
    path = shard_dir / f"{benchmark}_control.{name}.csv"
    fields = control_fields.get(benchmark, [])
    with path.open("w", newline="") as f:
        if not fields:
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in items:
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

def write_treatment(benchmark, name, items):
    path = shard_dir / f"{benchmark}_treatment.{name}.jsonl"
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

for name in names:
    for benchmark in ["miniF2F", "proofnet"]:
        items = shards[name][benchmark]
        write_control(benchmark, name, items)
        write_treatment(benchmark, name, items)
        manifest["shards"][name][benchmark] = {
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

stop_old_2way
sleep 3
reload_local_goedel_parallel4
write_shards | tee -a "$ORCH_LOG"

for name in local_a local_b local_c local_d; do
  start_worker "$name" "$LOCAL_BASE_URL"
done
for name in mac_a mac_b mac_c mac_d; do
  start_worker "$name" "$MAC_BASE_URL"
done

log "Goedel decrease 8-way run started"
