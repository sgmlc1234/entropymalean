#!/usr/bin/env python3
"""Write the narrow view of the exam rows: one level of the game per row.

The analysis CSV carries every column an ablation table might slice by, which
makes it the wrong thing to open when the question is simply "what does this
level look like?". This writes the playable view — the fields a level actually
consists of, in the order they are used: pick it, read it, attempt it, ask for
help, check against the answer.

Thirteen columns, all of them scalar and all of them readable:

  id                 which problem
  topic              what it is about
  difficulty         easy | medium | hard, split at this corpus's own tertiles
  certificate        how far the row got: statement_checked / proof_checked /
                     kernel_replayed (the last one means an independent kernel
                     replayed the exported term, not just that Lean accepted it)
  goal               the problem in prose — what the player is asked to prove
  lean_header        imports and opens the level runs under
  lean_goal          the theorem, ending in `:= by`
  tools              lemmas the palette offers — the level-1 hint, always shown
  max_hint_level     highest rung available (3 = a concrete first step exists)
  hint_caveat        where a rung stops meaning its name: `single_line_proof`
                     (any hint is the answer), `l1_reveals_proof` (the lemma
                     name is the proof). Empty for a row whose ladder holds.
  hint_outline       the proof's shape, without any Lean
  hint_first_step    one concrete tactic; this one gives part of the answer away
  solution           the full compilable file, header + statement + proof

Usage:
  python scripts/archive/exam_rows_to_game_csv.py \
    --input data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl \
    --output data/benchmarks/proofnet_verified/raw/levels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

COLUMNS: List[str] = [
    "id", "topic", "difficulty", "certificate",
    "goal", "lean_header", "lean_goal",
    "tools", "max_hint_level", "hint_caveat", "hint_outline", "hint_first_step",
    "solution",
]


def difficulty_cuts(rows: List[Dict[str, Any]]) -> tuple:
    """Tertiles of this corpus's ground-truth proof lengths.

    Proof length is the only difficulty proxy that does not require a model to
    have attempted the problem first. Fixed edges do not transfer between
    corpora — ProofNet's median proof is 35 lines, a competition problem's is a
    few — so the thirds are cut from whatever file is being converted, which
    keeps the tiers useful for picking a level in either.
    """
    lengths = sorted(int(r.get("gt_step_count") or 0) for r in rows)
    if not lengths:
        return (0, 0)
    return (lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3])


def difficulty_of(steps: Any, cuts: tuple) -> str:
    try:
        count = int(steps)
    except (TypeError, ValueError):
        return "unknown"
    if count <= cuts[0]:
        return "easy"
    if count <= cuts[1]:
        return "medium"
    return "hard"


def _text(value: Any, limit: int = 0) -> str:
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value)
    elif isinstance(value, dict):
        value = "; ".join(sorted(value))
    text = " ".join(str(value or "").split())
    return text if not limit or len(text) <= limit else text[: limit - 1] + "…"


def level_of(row: Dict[str, Any], cuts: tuple) -> Dict[str, Any]:
    ladder = {int(h.get("level", 0)): h for h in (row.get("hint_ladder") or [])}
    palette = (row.get("palette") or {}).get("theorems") or {}
    return {
        "id": row.get("stem") or row.get("seed") or row.get("name"),
        "topic": row.get("topic"),
        "difficulty": difficulty_of(row.get("gt_step_count"), cuts),
        # One column, three claims, ordered — a bare `ready` could not say
        # which rows survived the comparator, and that is the claim worth reading.
        "certificate": row.get("certificate_level")
        or ("statement_checked" if row.get("statement_checked") else "none"),
        "goal": _text(row.get("statement_nl")),
        "lean_header": _text(row.get("lean_header")),
        "lean_goal": str(row.get("formal_statement") or "").strip(),
        # level 1 of the ladder is exactly the palette, so it is not repeated
        "tools": _text(palette, limit=300),
        "max_hint_level": row.get("max_hint_level"),
        # Carried next to the level so a filter on hint strength cannot pick up
        # rows where the level does not mean what it says.
        "hint_caveat": "; ".join(
            k for k, v in (row.get("hint_degeneracy") or {}).items() if v
        ),
        "hint_outline": _text((ladder.get(2) or {}).get("content"), limit=400),
        "hint_first_step": _text((ladder.get(3) or {}).get("content"), limit=200),
        "solution": str(row.get("lean_code") or "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="drop levels that earned no certificate at all",
    )
    parser.add_argument(
        "--no-solution",
        action="store_true",
        help="drop the solution column, for a file meant to be played from",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cuts = difficulty_cuts(rows)
    levels = [level_of(row, cuts) for row in rows]
    if args.ready_only:
        levels = [level for level in levels if level["certificate"] != "none"]

    columns = [c for c in COLUMNS if not (args.no_solution and c == "solution")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(levels)

    spread = {
        tier: sum(1 for level in levels if level["difficulty"] == tier)
        for tier in ("easy", "medium", "hard", "unknown")
    }
    certs = {}
    for level in levels:
        certs[level["certificate"]] = certs.get(level["certificate"], 0) + 1
    solved = sum(1 for level in levels if level.get("solution"))
    print(f"wrote {len(levels)} levels x {len(columns)} columns -> {args.output}")
    print(f"  difficulty: {spread}  (cuts at <={cuts[0]} and <={cuts[1]} proof lines)")
    print(f"  certificate: {certs}")
    print(f"  with solution: {solved}/{len(levels)}")


if __name__ == "__main__":
    main()
