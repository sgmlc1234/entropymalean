#!/usr/bin/env python3
"""Give every row the same three-rung ladder, whichever benchmark it came from.

The two benchmarks reached their hints by different routes and ended up with
different ladders. ProofNet's ground truth carries `-- Step N:` comments, so a
first concrete tactic falls out of parsing; Gen-0 writes no comments, so miniF2F
had level 3 on nothing at all — 0 of 50 against 49 of 50. An ablation over hint
strength cannot run its strongest arm on half the corpus.

Two other repairs travel with it.

`max_hint_level` was the *count* of rungs, not the highest one. A ProofNet row
holding levels {2,3} reported 2, and a miniF2F row holding {1,2} reported 2 as
well — the same number naming different conditions, which is exactly the kind of
column that corrupts a comparison without ever looking wrong. It now holds the
maximum level present, and the rungs themselves are listed alongside it.

Level 1 was missing wherever a proof cited no library lemma, which is most
common on the short proofs — `norm_num` closes them without naming anything. The
palette is widened there to the library names appearing in the *statement*: not
lemmas to apply, but the concepts the problem is about, which is what a first
hint should point at when there is nothing else to point at.

Usage:
  python scripts/analysis/fill_hint_ladder.py \
    --rows data/benchmarks/minif2f_v2/raw/exam_rows_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
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

from src.exam_env.palette import (  # noqa: E402
    TACTIC_DOCS,
    build_check_probe,
    candidate_theorem_names,
    parse_check_probe_output,
)

_COMMENT = re.compile(r"^\s*(--|/-)")

#: Tactics that rearrange the proof state without deciding anything about the
#: mathematics: entering classical logic, moving binders across the turnstile,
#: naming a subterm, unfolding a definition. Taking the literal first line as
#: the level-3 hint made 54 of 100 rows open with one of these — `classical`,
#: `revert n hn` — which is a hint that costs the solver a rung and tells it
#: nothing. The first line that is *not* one of these is the first place the
#: proof commits to an idea.
BOOKKEEPING = {
    "classical", "intro", "intros", "rintro", "revert", "set", "let", "letI",
    "haveI", "subst", "rename_i", "change", "show", "dsimp", "ext", "norm_cast",
    "push_cast", "unfold", "delta", "clear", "rcases",
}


def _head(line: str) -> str:
    return line.strip().lstrip("·<;>| ").split(" ", 1)[0].strip()


def first_tactic(proof_body: str) -> tuple:
    """→ (first mathematically committing line, whether we had to settle).

    Falls back to the literal first line when every line is bookkeeping, so a
    row always has a level 3; the second element records that, because such a
    hint is worth excluding from a hint-strength comparison rather than
    silently averaging in.
    """
    lines = [
        line.strip()
        for line in str(proof_body or "").splitlines()
        if line.strip() and not _COMMENT.match(line) and line.strip() not in {"·", "by", "case"}
    ]
    for line in lines:
        if _head(line) not in BOOKKEEPING:
            return line, False
    return (lines[0], True) if lines else ("", True)


#: Notation carries the concept for statements that name nothing explicitly.
#: `2003 % 11 = 1` mentions no library declaration at all, but it is plainly a
#: statement about natural-number remainders, and that is what a first hint
#: should say. Mapping the symbols back to their declarations recovers it.
_NOTATION = {
    "ℝ": "Real", "ℚ": "Rat", "ℂ": "Complex", "ℤ": "Int", "ℕ": "Nat",
    "%": "Nat.mod", "∑": "Finset.sum", "∏": "Finset.prod",
    "∣": "Dvd.dvd", "√": "Real.sqrt", "!": "Nat.factorial",
    "≡": "Nat.ModEq", "⁻¹": "Inv.inv", "∈": "Membership.mem",
}


def statement_concepts(statement: str) -> List[str]:
    """Library names the statement is about — concepts, not lemmas to apply.

    The theorem's own name is dropped: it is neither a library declaration nor
    information, since the solver is already looking at it.
    """
    text = str(statement or "")
    own = re.search(r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_.']*)", text)
    skip = {own.group(1)} if own else set()
    names = [
        name
        for name in candidate_theorem_names(text)
        if name not in TACTIC_DOCS and name not in skip
    ]
    for symbol, declaration in _NOTATION.items():
        if symbol in text and declaration not in names:
            names.append(declaration)
    return names


def run_check(names: List[str], chunk: int = 200) -> Dict[str, str]:
    """Validate names against our Mathlib, keeping those that exist.

    Matching the output back to the query by name fails for types, because
    Lean prints them under their notation: asking about `Real` prints
    `ℝ : Type`, and a name-keyed parser drops it as unknown. Since the probe
    emits one `#check` per name in order, the reply lines can be matched by
    position instead — guarded by a count check, so a batch with any error
    falls back to name matching rather than pairing signatures to the wrong
    names.
    """
    signatures: Dict[str, str] = {}
    for start in range(0, len(names), chunk):
        part = names[start : start + chunk]
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as handle:
            handle.write(build_check_probe("import Mathlib", part))
            path = handle.name
        proc = subprocess.run(
            ["lake", "env", "lean", path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
        )
        raw = proc.stdout + "\n" + proc.stderr
        by_name = parse_check_probe_output(raw, part)
        lines = [line for line in proc.stdout.splitlines() if " : " in line]
        if len(lines) == len(part):
            for name, line in zip(part, lines):
                by_name.setdefault(name, line.strip())
        signatures.update(by_name)
    return signatures


_STEP_PREFIX = re.compile(r"^\s*Step\s+\d+(?:\.\d+)?[a-z]?\s*[:.]\s*")


def clean_outline(outline: List[str]) -> List[str]:
    """Drop the `Step N:` prefixes the ground truth's comments carried.

    They are formatting, not content, and the numbering does not survive
    parsing intact — one row jumps from Step 1 to Step 3 because the
    intervening comment was a sub-step. Presenting a hint with a broken count
    invites the reader to look for the missing piece. The order of the list
    already carries the sequence.
    """
    cleaned = []
    for step in outline:
        text = _STEP_PREFIX.sub("", str(step)).strip()
        if text:
            cleaned.append(text[0].upper() + text[1:] if text[0].islower() else text)
    return cleaned


def build_ladder(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ladder: List[Dict[str, Any]] = []
    names = sorted((row.get("palette") or {}).get("theorems") or {})
    if names:
        ladder.append(
            {"level": 1, "kind": "lemma_names", "content": names, "leaks_proof": False}
        )
    outline = clean_outline(((row.get("hints") or {}).get("outline")) or [])
    if outline:
        ladder.append(
            {"level": 2, "kind": "proof_outline", "content": outline, "leaks_proof": False}
        )
    steps = ((row.get("hints") or {}).get("step_tactics")) or []
    tactic = ""
    if steps:
        head = steps[0]
        tactic = head.get("tactic") if isinstance(head, dict) else str(head)
    settled = False
    if not str(tactic).strip() or _head(str(tactic)) in BOOKKEEPING:
        tactic, settled = first_tactic(row.get("gt_proof_body"))
    row["l3_is_bookkeeping"] = bool(settled)
    if str(tactic).strip():
        ladder.append(
            {
                "level": 3,
                "kind": "first_step_tactic",
                "content": str(tactic).strip(),
                "leaks_proof": True,
            }
        )
    return ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-palette-widening", action="store_true")
    args = parser.parse_args()
    output = args.output or args.rows

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    widened = 0
    if not args.skip_palette_widening:
        need = [r for r in rows if not (r.get("palette") or {}).get("theorems")]
        wanted: Dict[str, List[str]] = {}
        every: List[str] = []
        for row in need:
            names = statement_concepts(row.get("formal_statement"))
            wanted[str(row.get("name"))] = names
            every.extend(names)
        if every:
            # Probe the two kinds separately. Names pulled out of a statement
            # are arbitrary and some will not exist, which is what makes the
            # positional match unsafe for them; the notation list is fixed and
            # every entry is a real declaration, so its batch can rely on it —
            # and has to, since those are exactly the names Lean prints under
            # their notation rather than under the name asked for.
            fixed = sorted({n for n in every if n in set(_NOTATION.values())})
            free = sorted({n for n in every if n not in set(_NOTATION.values())})
            signatures = {}
            if free:
                signatures.update(run_check(free))
            if fixed:
                signatures.update(run_check(fixed))
            print(f"widening probe: {len(signatures)}/{len(set(every))} names exist", flush=True)
            for row in need:
                found = {
                    n: signatures[n]
                    for n in wanted[str(row.get("name"))]
                    if n in signatures
                }
                if found:
                    palette = dict(row.get("palette") or {})
                    palette["theorems"] = found
                    palette["theorems_source"] = "statement_concepts"
                    row["palette"] = palette
                    row["palette_theorem_count"] = len(found)
                    widened += 1

    def degeneracy(row: Dict[str, Any]) -> Dict[str, Any]:
        """Where a rung stops meaning what its name says.

        Recorded rather than repaired: a one-line proof cannot be hinted at
        without being given away, and a lemma name that *is* the proof is real
        information about the problem, not a defect to hide. Analysis can hold
        these out; averaging over them would report the ladder working where it
        had collapsed.
        """
        body = re.sub(r"^\s*(--|/-).*$", "", str(row.get("gt_proof_body") or ""), flags=re.M)
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        palette = sorted((row.get("palette") or {}).get("theorems") or {})
        return {
            "single_line_proof": len(lines) <= 1,
            "l1_reveals_proof": (
                len(lines) <= 3 and 0 < len(palette) <= 2
                and any(name in body for name in palette)
            ),
            "l3_is_bookkeeping": bool(row.get("l3_is_bookkeeping")),
        }

    filled = 0
    for row in rows:
        before = {h["level"] for h in (row.get("hint_ladder") or [])}
        ladder = build_ladder(row)
        row["hint_ladder"] = ladder
        levels = sorted(h["level"] for h in ladder)
        row["hint_levels"] = levels
        row["max_hint_level"] = max(levels) if levels else 0
        row["hint_degeneracy"] = degeneracy(row)
        if 3 in levels and 3 not in before:
            filled += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    have = Counter()
    for row in rows:
        for level in row["hint_levels"]:
            have[level] += 1
    combos = Counter(tuple(r["hint_levels"]) for r in rows)
    print(f"rows={len(rows)} -> {output}")
    print(f"  level 3 filled: {filled}   palette widened: {widened}")
    print(f"  per level: 1={have[1]}  2={have[2]}  3={have[3]}")
    print(f"  combinations: {dict(combos)}")
    print(f"  max_hint_level: {dict(sorted(Counter(r['max_hint_level'] for r in rows).items()))}")
    deg = Counter()
    for row in rows:
        for key, flag in row["hint_degeneracy"].items():
            if flag:
                deg[key] += 1
    print(f"  degeneracy flags: {dict(deg) or 'none'}")


if __name__ == "__main__":
    main()
