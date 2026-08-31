#!/usr/bin/env python3
"""Count episodes whose verdict came from a different problem.

The REPL verifier used to keep its child after a `pexpect.TIMEOUT`. The
timed-out command's response was still in flight, so the *next* call read it as
its own: a verdict computed for one problem, recorded against another. This is
not a missing measurement but a wrong one.

It is visible in exactly one place. When the stale verdict lands on an
episode's `reset()` --- which submits only `skip` --- an error mentioning a
tactic or a local hypothesis cannot have come from the statement, so the
episode is recorded as `statement_invalid` carrying a message no statement
could produce. Those are counted here.

What cannot be counted here is the rest. A stale verdict landing mid-episode
accepts or rejects a tactic on someone else's evidence and leaves no trace in
the episode record, because a rejected tactic's message is not stored. So the
number this prints is a floor on the affected episodes, not an estimate of
them, and the only way to clear a cell is to re-run it against a verifier that
drops the child on timeout.

The tell for the boundary case is the preceding episode: a desync needs a
timeout, and a timeout needs a long episode, so these cluster immediately after
the longest searches.

Usage:
  python3 scripts/evaluate/audit_repl_desync.py data/evaluation/exam/*/episodes_*.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

def playable_rows(*paths: Path) -> set:
    """Rows the pre-flight confirmed the environment can elaborate.

    This is the test, not the wording of the error. A row that elaborated
    during pre-flight and is later reported `statement_invalid` did not become
    unplayable; the verdict belongs to something else. Reading the message
    instead misses the desyncs whose stale verdict happens to look like a
    plausible statement complaint --- `Unknown identifier` for a hypothesis
    name, say --- which was half of them on the first pass.
    """
    names = set()
    for p in paths:
        if p.is_file():
            names |= {json.loads(l)["name"]
                      for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return names


PLAYABLE = playable_rows(Path("data/evaluation/exam/release417_playable.jsonl"),
                         Path("data/evaluation/exam/release309_playable.jsonl"),
                         Path("data/evaluation/exam/seeds_all100.jsonl"))


def audit(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    boundary, other_invalid = [], []
    for i, r in enumerate(rows):
        if r.get("outcome") != "statement_invalid":
            continue
        (boundary if r["seed"] in PLAYABLE else other_invalid).append((i, r))
    prev_outcomes = collections.Counter(
        rows[i - 1]["outcome"] for i, _ in boundary if i)
    prev_seconds = sorted(rows[i - 1]["elapsed_seconds"] for i, _ in boundary if i)
    return {
        "file": str(path),
        "episodes": len(rows),
        "desync_at_reset": len(boundary),
        "statement_invalid_genuine": len(other_invalid),
        "preceding_outcome": dict(prev_outcomes),
        "preceding_seconds_median": prev_seconds[len(prev_seconds) // 2] if prev_seconds else None,
        "affected_seeds": sorted({r["seed"] for _, r in boundary}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    reports = [audit(p) for p in args.files if p.is_file()]
    total = 0
    for r in reports:
        if not r["desync_at_reset"] and not r["statement_invalid_genuine"]:
            continue
        total += r["desync_at_reset"]
        name = "/".join(Path(r["file"]).parts[-2:])
        print(f"{name}")
        print(f"    episodes {r['episodes']:5d}   desync-at-reset {r['desync_at_reset']:4d}"
              f"   genuine statement_invalid {r['statement_invalid_genuine']:3d}")
        if r["preceding_outcome"]:
            print(f"    preceded by {r['preceding_outcome']}, "
                  f"median {r['preceding_seconds_median']:.0f}s")
    print(f"\n{total} episodes carry a verdict from another problem — a floor, not an estimate.")
    print("Mid-episode desyncs leave no trace; a cell can only be cleared by re-running it.")
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
