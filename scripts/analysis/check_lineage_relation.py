#!/usr/bin/env python3
"""Ask Lean how a generated problem relates to the parent it was bred from.

The certificate layer answers "is this a theorem". It cannot answer "is this a
*new* theorem", and for a bred corpus that is the question a reader will ask:
250 certified rows are compatible with 250 restatements of the same fact. Our
own seed set contains the cautionary case — `mathd_algebra_275` has
contradictory hypotheses, so it is provable by anything, and the certificate
layer stamped it `proof_checked` without complaint.

Comparing a child to prose cannot settle this. Comparing it to its *parent*
can, because breeding leaves both statements in the same formal language and
Lean can be asked directly:

    parent ⟹ child  and  child ⟹ parent

  both        the child restates the parent — not a new problem
  child⟹parent only   the child is strictly stronger — harder, keep
  parent⟹child only   the child is strictly weaker — demote
  neither     incomparable — a genuinely different direction
  unproved    undetermined; see below

The last outcome is the honest one and will dominate on hard pairs. A failed
proof search is not a proof of independence: it means our tactic budget did not
close the implication, which is a statement about the budget. `undetermined` is
therefore reported as its own category and never folded into `incomparable`.

Usage:
  LEAN_VERIFIER=repl python scripts/analysis/check_lineage_relation.py \
    --rows release/huggingface/EML-1/accepted.jsonl \
    --output data/evaluation/lineage/relations.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)

_DECL = re.compile(r"^\s*(?:theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_.']*\s*", re.S)

#: Deliberately modest. A wide search would turn "undetermined" into "proved"
#: for pairs where the implication is true but hard, at the cost of minutes per
#: pair and of a category whose meaning then depends on how long we waited.
TACTICS = (
    "  intro hP\n"
    "  first\n"
    "  | exact hP\n"
    "  | (intros; exact hP ..)\n"
    "  | (intros; apply hP <;> assumption)\n"
    "  | (intros; solve_by_elim [hP])\n"
    "  | (intros; simp_all)\n"
    "  | (intros; omega)\n"
    "  | (intros; linarith)\n"
)


def closed_prop(statement: str) -> str:
    """`theorem f (x : T) (h : P) : C` → `∀ (x : T) (h : P), C`.

    That ∀-form is the declaration's own type, so nothing is being asserted
    here beyond what the theorem already says — the binders simply have to be
    written out to talk about the statement as a proposition.
    """
    text = str(statement or "").strip()
    text = re.sub(r":=\s*(by)?\s*$", "", text).rstrip()
    text = _DECL.sub("", text, count=1)
    depth = 0
    for index, char in enumerate(text):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and depth == 0:
            binders = text[:index].strip()
            goal = text[index + 1 :].strip()
            return f"(∀ {binders}, {goal})" if binders else f"({goal})"
    return f"({text})"


def probe(header: str, source: str, target: str) -> str:
    head = str(header or "import Mathlib").rstrip()
    if "import" not in head:
        head = "import Mathlib\n" + head
    return (
        f"{head}\nset_option autoImplicit false\nset_option maxHeartbeats 1000000\n\n"
        f"example : {source} → {target} := by\n{TACTICS}"
    )


def classify(forward: bool, backward: bool) -> str:
    if forward and backward:
        return "restatement"
    if backward:
        return "child_stronger"
    if forward:
        return "child_weaker"
    return "undetermined"


async def relate(
    child: Dict[str, Any], parent: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    c_prop = closed_prop(child.get("formal_statement"))
    p_prop = closed_prop(parent.get("formal_statement"))
    header = child.get("lean_header") or parent.get("lean_header") or "import Mathlib"
    forward = await verify_lean_proof_repl(probe(header, p_prop, c_prop), timeout=timeout)
    backward = await verify_lean_proof_repl(probe(header, c_prop, p_prop), timeout=timeout)
    return {
        "child": child.get("problem_id") or child.get("id"),
        "parent": parent.get("problem_id") or parent.get("id"),
        "parent_implies_child": bool(forward.ok),
        "child_implies_parent": bool(backward.ok),
        "relation": classify(bool(forward.ok), bool(backward.ok)),
    }


def parent_statement(entry: Any, corpus: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resolve a parent reference to something with a full Lean statement."""
    if isinstance(entry, str):
        return corpus.get(entry)
    if not isinstance(entry, dict):
        return None
    text = str(entry.get("formal_statement") or "")
    if "theorem" in text or "lemma" in text:
        return entry
    return corpus.get(str(entry.get("parent_id") or entry.get("id") or ""))


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus = {str(r.get("problem_id") or r.get("id")): r for r in rows}

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unresolved = 0
    for row in rows:
        for entry in row.get("parents") or row.get("parent_ids") or []:
            parent = parent_statement(entry, corpus)
            if parent and str(parent.get("formal_statement") or "").strip():
                pairs.append((row, parent))
            else:
                unresolved += 1
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"pairs={len(pairs)}  unresolved_parents={unresolved}", flush=True)

    results: List[Dict[str, Any]] = []
    try:
        for index, (child, parent) in enumerate(pairs, 1):
            try:
                verdict = await relate(child, parent, args.timeout)
            except Exception as error:
                verdict = {
                    "child": child.get("problem_id"),
                    "parent": parent.get("problem_id") or parent.get("parent_id"),
                    "relation": "probe_error",
                    "error": f"{type(error).__name__}: {error}"[:200],
                }
            results.append(verdict)
            print(f"[{verdict['relation']:14s}] {index:3d}/{len(pairs)} "
                  f"{str(verdict.get('child'))[:52]}", flush=True)
    finally:
        await close_global_repl_verifier()

    tally = Counter(r["relation"] for r in results)
    report = {
        "rows": len(rows),
        "pairs_attempted": len(pairs),
        "unresolved_parents": unresolved,
        "relations": dict(tally.most_common()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    asyncio.run(_run())
