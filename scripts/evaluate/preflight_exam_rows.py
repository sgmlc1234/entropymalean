#!/usr/bin/env python3
"""Check every row elaborates before any model is asked to prove it.

The exam environment refuses a statement it cannot elaborate: `reset()` marks
the episode done and no player ever moves. The episode that results has no
tokens, no actions, and zero elapsed time — the same shape a dead serving stack
produces — so an unplayable row is indistinguishable from an outage unless
something asks in advance.

It matters twice over. Counted as failures, unplayable rows understate every
model equally and silently. And three of them in a row trips the run's
dead-server abort, ending a campaign for a reason that has nothing to do with
the server.

So this plays the reset and nothing else: one elaboration per row, no model, no
tactics. Rows that pass are written to `--output` if given; rows that fail are
listed with Lean's complaint, which is usually enough to say whether the fault
is the row or our header.

Usage:
  python3 scripts/evaluate/preflight_exam_rows.py --rows data/evaluation/exam/release309_rows.jsonl
  python3 scripts/evaluate/preflight_exam_rows.py --rows in.jsonl --output playable.jsonl --report r.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.exam_env.environment import LeanExamEnv  # noqa: E402


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the playable rows here, in input order")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--lean-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    playable: List[Dict[str, Any]] = []
    broken: List[Dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, 1):
            env = LeanExamEnv(
                formal_statement=str(row.get("formal_statement") or ""),
                lean_header=str(row.get("lean_header") or "import Mathlib"),
                palette={"tactics": {}, "theorems": {}},
                verifier=verify_lean_proof_repl,
                max_steps=8,
                lean_timeout=args.lean_timeout,
                strict_steps=False,
            )
            observation = await env.reset()
            ok = observation.status != "error"
            (playable if ok else broken).append(
                row if ok else {
                    "name": row.get("name"),
                    "benchmark": row.get("benchmark"),
                    "message": (observation.message or "")[:300],
                }
            )
            if not ok:
                print(f"[BROKEN] {index:4d}/{len(rows)} {str(row.get('name'))[:46]:46s} "
                      f"{(observation.message or '')[:80]}", flush=True)
            elif index % 25 == 0:
                print(f"[ ok   ] {index:4d}/{len(rows)}  broken so far: {len(broken)}", flush=True)
    finally:
        await close_global_repl_verifier()

    by_bench = collections.Counter(str(b.get("benchmark")) for b in broken)
    summary = {
        "rows": len(rows),
        "playable": len(playable),
        "broken": len(broken),
        "broken_by_benchmark": dict(by_bench),
        "broken_rows": broken,
    }
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "broken_rows"},
                            ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in playable:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(playable)} playable rows -> {args.output}")
    if args.report:
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"wrote report -> {args.report}")


if __name__ == "__main__":
    asyncio.run(_run())
