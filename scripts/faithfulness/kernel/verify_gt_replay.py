#!/usr/bin/env python3
"""Replay every ground-truth proof under our pinned toolchain and record whether it holds.

A row inherited from a released benchmark carries a proof that compiled for
*its* authors, against *their* Mathlib. Ours is pinned elsewhere, and Mathlib
renames and restates lemmas continuously, so "this benchmark is verified" is a
claim about a revision we do not run. Treating it as true here would put
uncompilable proofs into the seed set and into the exam's answer key.

So the claim is checked rather than inherited: each row's assembled `lean_code`
is sent to Lean, and the verdict is written back as a column. What comes out is
also a number worth reporting — the share of a *verified* benchmark that still
replays under a different pin.

Writes two fields per row:
  proof_replays        True | False
  proof_replay_error   the first diagnostic, when it does not

Usage:
  LEAN_VERIFIER=repl python scripts/faithfulness/kernel/verify_gt_replay.py \
    --input data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.levels import runtime_pins  # noqa: E402
from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)


def error_kind(summary: str) -> str:
    """Coarse cause, so the failures can be reported as a distribution.

    The distinction that matters is drift (the proof references something this
    Mathlib no longer has, or has differently) versus a proof that is simply
    incomplete or too slow here — the first is a property of the pin, the
    second of the proof.
    """
    text = (summary or "").lower()
    if "unknown identifier" in text or "unknown constant" in text:
        return "unknown_identifier"
    if "invalid field" in text or "type mismatch" in text:
        return "signature_drift"
    if "unsolved goals" in text:
        return "unsolved_goals"
    if "timeout" in text or "maxheartbeats" in text or "deterministic" in text:
        return "resource_limit"
    if "sorry" in text:
        return "contains_sorry"
    return "other"


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output = args.output or args.input

    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    failures: List[Dict[str, str]] = []
    try:
        for index, row in enumerate(rows, 1):
            code = str(row.get("lean_code") or "").strip()
            if not code:
                row["proof_replays"] = False
                row["proof_replay_error"] = "no lean_code"
                continue
            verdict = await verify_lean_proof_repl(code, timeout=args.timeout)
            row["proof_replays"] = bool(verdict.ok)
            row["proof_replay_error"] = None if verdict.ok else verdict.summary()[:400]
            if not verdict.ok:
                failures.append(
                    {
                        "name": str(row.get("name")),
                        "kind": error_kind(row["proof_replay_error"]),
                        "error": row["proof_replay_error"][:200],
                    }
                )
            mark = "ok  " if verdict.ok else "FAIL"
            print(f"[{mark}] {index:3d}/{len(rows)} {str(row.get('name'))[:46]}", flush=True)
    finally:
        await close_global_repl_verifier()

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    replays = sum(1 for row in rows if row.get("proof_replays"))
    pins = runtime_pins(str(REPO_ROOT))
    report = {
        "input": str(args.input),
        "rows": len(rows),
        "replays": replays,
        "replay_rate": round(replays / max(1, len(rows)), 3),
        "lean_toolchain": pins["lean_toolchain"],
        "mathlib_revision": pins["mathlib_revision"],
        "failure_kinds": dict(Counter(f["kind"] for f in failures).most_common()),
        "failures": failures,
    }
    (output.parent / f"{output.stem}_replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print(f"replays {replays}/{len(rows)} -> {output}")


if __name__ == "__main__":
    asyncio.run(_run())
