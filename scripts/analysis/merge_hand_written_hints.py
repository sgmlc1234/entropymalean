#!/usr/bin/env python3
"""Fold hand-written proof outlines into the hint ladder.

ProofNet's ground truth is annotated with `-- Step N:` comments, so its level-2
hint falls out of parsing. Gen-0 writes no such comments, which left every
miniF2F row with a ladder that jumps from "here are some lemma names" straight
to "here is the first tactic" — the middle rung, the one that describes the
*shape* of the argument without handing over Lean, simply did not exist. An
ablation over hint strength cannot compare the two benchmarks while one of them
is missing a level.

So the outlines are written by hand from each proof and kept here, apart from
anything derived, because they are the one field in the row a person authored
and that provenance should survive a rebuild.

An outline describes strategy, never syntax: which case split opens the proof,
what each branch has to establish, which fact does the real work. If it can be
pasted into Lean it belongs at level 3, not level 2.

Usage:
  python scripts/analysis/merge_hand_written_hints.py \
    --rows data/benchmarks/minif2f_v2/raw/exam_rows_v2.jsonl \
    --hints data/benchmarks/minif2f_v2/raw/hand_written_hints.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolved one level short after the
    move -- to a directory that exists, so nothing raised."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LEAN_ISH = re.compile(
    r"(^|\s)(rw|simp|nlinarith|linarith|norm_num|rcases|obtain|refine|exact|intro|"
    r"constructor|apply|omega|decide|field_simp|by_contra|induction)\b"
)


def lint(outline: List[str]) -> List[str]:
    """Complaints about an outline, so a lazy one fails loudly rather than shipping.

    The check is for tactic names because that is the specific way a level-2
    hint decays into a level-3 one: paraphrasing the proof script instead of
    describing the argument. A hint that leaks the tactics makes the ladder
    non-monotone and quietly inflates hinted Pass@K.
    """
    problems = []
    if not outline:
        problems.append("empty")
    for step in outline:
        if _LEAN_ISH.search(step):
            problems.append(f"names a tactic: {step[:60]!r}")
        if len(step) < 20:
            problems.append(f"too terse to be a strategy: {step[:40]!r}")
    return problems


def rebuild_ladder(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Level 1 lemma names, level 2 outline, level 3 first concrete tactic."""
    ladder: List[Dict[str, Any]] = []
    names = sorted((row.get("palette") or {}).get("theorems") or {})
    if names:
        ladder.append(
            {"level": 1, "kind": "lemma_names", "content": names, "leaks_proof": False}
        )
    outline = ((row.get("hints") or {}).get("outline")) or []
    if outline:
        ladder.append(
            {"level": 2, "kind": "proof_outline", "content": outline, "leaks_proof": False}
        )
    steps = ((row.get("hints") or {}).get("step_tactics")) or []
    if steps:
        first = steps[0]
        ladder.append(
            {
                "level": 3,
                "kind": "first_step_tactic",
                "content": first.get("tactic") if isinstance(first, dict) else str(first),
                "leaks_proof": True,
            }
        )
    return ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-lint-failures",
        action="store_true",
        help="write anyway; by default a complaint stops the merge",
    )
    args = parser.parse_args()
    output = args.output or args.rows

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    written = json.loads(args.hints.read_text(encoding="utf-8")) if args.hints.is_file() else {}

    complaints: Dict[str, List[str]] = {}
    for name, entry in written.items():
        outline = entry.get("outline") or []
        found = lint(outline)
        if found:
            complaints[name] = found
    if complaints and not args.allow_lint_failures:
        for name, found in list(complaints.items())[:10]:
            print(f"  {name}: {'; '.join(found)}")
        raise SystemExit(f"{len(complaints)} outline(s) failed the lint; nothing written")

    applied = 0
    for row in rows:
        entry = written.get(str(row.get("name")))
        if not entry:
            continue
        hints = dict(row.get("hints") or {})
        hints["outline"] = entry["outline"]
        hints["outline_author"] = entry.get("author", "hand-written")
        row["hints"] = hints
        row["hint_ladder"] = rebuild_ladder(row)
        row["max_hint_level"] = len(row["hint_ladder"])
        applied += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    levels = Counter(r.get("max_hint_level") for r in rows)
    with_outline = sum(1 for r in rows if (r.get("hints") or {}).get("outline"))
    print(f"rows={len(rows)}  outlines applied={applied}  -> {output}")
    print(f"  rows with an outline: {with_outline}/{len(rows)}")
    print(f"  ladder depth: {dict(sorted(levels.items()))}")
    missing = [str(r.get("name")) for r in rows if not (r.get("hints") or {}).get("outline")]
    if missing:
        print(f"  still missing ({len(missing)}): {', '.join(missing[:6])}…")


if __name__ == "__main__":
    main()
