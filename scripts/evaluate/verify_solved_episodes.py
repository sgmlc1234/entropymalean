#!/usr/bin/env python3
"""Re-check every episode that claims a proof, against its own statement.

A desynchronised verifier can hand an episode a verdict computed for a
different problem. When that verdict happens to be `complete`, the environment
sets `success` and the episode is recorded as solved --- with a proof nothing
ever checked against the theorem it is filed under. That is the only direction
of the corruption that inflates a score, and it is the only one the records can
still settle: a false *failure* looks exactly like an honest one, because a
stale rejection and a real rejection leave the same trace.

So this re-verifies what was kept. Each solved episode stores `solved_code`,
the complete file the environment assembled --- header, statement, and the
accepted proof body. Compiling it again answers the question directly: does
this proof close this statement? An episode whose code does not verify was
never a solve.

Run it against the fixed verifier. Before the fix, `pexpect.TIMEOUT` left the
REPL child in place with the timed-out command's response still in flight, so
the next call read someone else's verdict.

Usage:
  set -a; source .env; set +a
  python3 scripts/evaluate/verify_solved_episodes.py \
      data/evaluation/exam/release309_bfs/episodes_bfs_closed_book.jsonl \
      --report /tmp/false_solves_bfs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

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


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lean-timeout", type=float, default=180.0)
    args = parser.parse_args()

    findings, checked = [], 0
    try:
        for path in args.episodes:
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            solved = [r for r in rows if r.get("success")
                      and (r.get("solved_code") or "").strip()]
            if args.limit:
                solved = solved[: args.limit]
            print(f"{path.name}: {len(solved)} claimed proofs", flush=True)
            for i, row in enumerate(solved, 1):
                verdict = await verify_lean_proof_repl(
                    row["solved_code"], timeout=args.lean_timeout)
                ok = bool(getattr(verdict, "complete", False))
                checked += 1
                if not ok:
                    findings.append({
                        "file": str(path), "seed": row.get("seed"),
                        "attempt": row.get("attempt"), "benchmark": row.get("benchmark"),
                        "why": (getattr(verdict, "system_error", None)
                                or getattr(verdict, "summary", lambda: "")()[:200]),
                    })
                    print(f"  [FALSE SOLVE] {row['seed'][:52]} a{row.get('attempt')}",
                          flush=True)
                elif i % 50 == 0:
                    print(f"  {i}/{len(solved)} re-verified, {len(findings)} false",
                          flush=True)
    finally:
        await close_global_repl_verifier()

    print(f"\n{checked} claimed proofs re-checked, {len(findings)} do not close "
          f"their own statement.")
    if args.report:
        args.report.write_text(json.dumps(findings, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    asyncio.run(_run())
