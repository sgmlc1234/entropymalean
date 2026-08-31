#!/usr/bin/env python3
"""Rebuild each row's Lean context from its ground-truth file, keeping the helpers.

The original header extraction stopped at the first declaration and kept only
imports and opens. That is wrong for any ground-truth file that defines a
helper lemma *between* its theorems — and ProofNet-Verified does this often.
The proof body then referenced a lemma that no longer existed in our assembled
file, Lean said `Unknown identifier`, and we recorded it as the released
benchmark failing to replay under our pin.

It was our file, not their proof. 43 of 118 replay failures traced directly to
a dropped helper and 7 more to a dropped `open`; the honest drift number could
not be read off that run at all.

What the context should be is everything in the ground-truth file *except* the
problem's own theorems. Slicing those spans out — rather than collecting
declarations into a new file — preserves `namespace`/`section` structure and
the original ordering, so a helper that depends on an earlier helper still
comes after it.

Usage:
  python scripts/archive/repair_exam_row_context.py \
    --rows data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl \
    --gt-dir /tmp/pnv_gt/proofnet_verified_gt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

_DECL_RE = re.compile(
    r"(?m)^\s*(?:@\[[^\]]*\]\s*)?"
    r"(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_.']*)"
)


def declaration_spans(code: str) -> List[Tuple[str, int, int]]:
    """→ [(name, start, end)] for top-level theorem/lemma declarations."""
    matches = list(_DECL_RE.finditer(code))
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code)
        spans.append((match.group(2), match.start(), end))
    return spans


def context_of(code: str, problem: str) -> str:
    """The file with the problem's own theorems removed.

    Everything else stays where it was: the prelude, any helper lemmas, and the
    `namespace`/`end` lines that scope them. Variants (`_corrected`, `_neg`,
    `_formal`) go too — they are alternative claims about the same problem, not
    context for proving it, and `_neg` in particular proves the original false.
    """
    keep = []
    cursor = 0
    for name, start, end in declaration_spans(code):
        if name == problem or name.startswith(f"{problem}_"):
            keep.append(code[cursor:start])
            cursor = end
    keep.append(code[cursor:])
    context = "".join(keep)
    # Collapse the blank runs left behind by the excisions.
    return re.sub(r"\n{3,}", "\n\n", context).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=Path("/private/tmp/pnv_gt/proofnet_verified_gt"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.rows

    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    stats = Counter()
    grew = 0
    for row in rows:
        gt = args.gt_dir / f"{row.get('stem')}.lean"
        if not gt.is_file():
            stats["gt_missing"] += 1
            continue
        text = _prelint_lean_syntax(gt.read_text(encoding="utf-8", errors="replace"))
        context = context_of(text, str(row.get("name")))
        before = str(row.get("lean_header") or "")
        if len(context) > len(before):
            grew += 1
        row["lean_header"] = context
        statement = str(row.get("formal_statement") or "").strip()
        if not re.search(r":=\s*by\s*$", statement):
            statement = re.sub(r":=\s*$", "", statement).rstrip() + " := by"
        proof = str(row.get("gt_proof_body") or "").rstrip()
        if proof and not proof.startswith(("\n", " ")):
            proof = "\n  " + proof
        row["lean_code"] = f"{context}\n\n{statement}{proof}\n"
        # The replay verdict on the old assembly no longer describes this file.
        row.pop("proof_replays", None)
        row.pop("proof_replay_error", None)
        stats["repaired"] += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows={len(rows)} -> {output}")
    print(f"  repaired: {stats['repaired']}  gt_missing: {stats['gt_missing']}")
    print(f"  context grew for {grew} rows")
    print("  stale proof_replays verdicts dropped — re-run verify_gt_replay.py")


if __name__ == "__main__":
    main()
