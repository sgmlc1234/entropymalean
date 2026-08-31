#!/usr/bin/env python3
"""Select a diverse miniF2F-v2 seed set and emit it as a Gen-0 seed CSV.

miniF2F ground-truth proofs are deliberately not published — the maintainers
refuse proof contributions to keep the test set uncontaminated — so seeds must
carry proofs we produce ourselves. This picks the seeds; Gen-0 proof completion
fills them in.

Diversity is enforced on three axes rather than left to a uniform draw:
problem family (mathd_algebra … imo), split, and a statement-shape proxy for
how much structure a mutation operator has to work with (hypothesis count and
whether the conclusion is an equation, inequality, divisibility, …).

Usage:
  python scripts/analysis/select_minif2f_v2_seeds.py --count 50 \
    --output data/benchmarks/minif2f_v2/raw/seeds_50.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
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

from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

SOURCE = REPO_ROOT / "references/miniF2F_v2/datasets/miniF2F_v2s.jsonl"

FAMILY_PREFIXES = (
    "mathd_algebra",
    "mathd_numbertheory",
    "amc12",
    "aime",
    "imo",
    "induction",
    "algebra",
    "numbertheory",
)


def family(name: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if str(name).startswith(prefix):
            return prefix
    return "other"


def _conclusion(statement: str) -> str:
    """Text after the binders' closing colon, with the `:= by` tail removed.

    Splitting on the last colon is wrong: these statements end in `:= by`, so
    the last colon is the one in `:=` and every goal looks like an equation.
    Walk the string instead, tracking bracket depth, and take the first colon
    that sits outside every binder group.
    """
    body = re.sub(r":=\s*by\s*$", "", statement.strip()).rstrip()
    depth = 0
    for index, char in enumerate(body):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and depth == 0:
            if body[index : index + 2] == ":=":
                continue
            return body[index + 1 :].strip()
    return body


def conclusion_shape(statement: str) -> str:
    """Coarse shape of the goal — what a mutation operator has to grip."""
    tail = _conclusion(statement)
    if re.search(r"∣|Nat\.gcd|Nat\.lcm|Prime|%", tail):
        return "divisibility"
    if re.search(r"[<>]|≤|≥", tail):
        return "inequality"
    if re.search(r"∃", tail):
        return "existential"
    if re.search(r"∀", tail):
        return "universal"
    if "=" in tail:
        return "equation"
    return "other"


def hypothesis_bucket(statement: str) -> str:
    count = len(re.findall(r"\(h[₀-₉0-9]*\s*:", statement))
    if count == 0:
        return "closed"          # no hypotheses: a bare computation
    if count <= 2:
        return "light"
    return "rich"                # more structure for crossover to combine


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statement = _prelint_lean_syntax(str(row["formal_statement"]))
        rows.append(
            {
                "name": str(row["name"]),
                "split": str(row["split"]),
                "statement": statement,
                "header": _prelint_lean_syntax(str(row.get("header") or "")),
                "informal": str(row.get("informal statement") or row.get("informal_statement") or ""),
                "informal_proof": str(row.get("informal_proof") or ""),
                "family": family(row["name"]),
                "shape": conclusion_shape(statement),
                "hyps": hypothesis_bucket(statement),
            }
        )
    return rows


def select(rows: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    """Proportional over family, then round-robin over (shape, hypotheses).

    Families are allocated in proportion to the benchmark so the seed set is
    not a caricature of it; within a family, cells are visited round-robin so
    no single statement shape dominates a small allocation.
    """
    rng = random.Random(seed)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    total = len(rows)
    quota = {f: max(1, round(count * len(v) / total)) for f, v in by_family.items()}
    # trim/expand to hit `count` exactly, largest families absorbing the diff
    order = sorted(quota, key=lambda f: -len(by_family[f]))
    while sum(quota.values()) > count:
        for f in reversed(order):
            if sum(quota.values()) == count:
                break
            if quota[f] > 1:
                quota[f] -= 1
    while sum(quota.values()) < count:
        for f in order:
            if sum(quota.values()) == count:
                break
            if quota[f] < len(by_family[f]):
                quota[f] += 1

    picked: List[Dict[str, Any]] = []
    for fam_name, want in quota.items():
        cells: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in by_family[fam_name]:
            cells[(row["shape"], row["hyps"])].append(row)
        for bucket in cells.values():
            rng.shuffle(bucket)
        keys = sorted(cells, key=lambda k: -len(cells[k]))
        taken, index = 0, 0
        while taken < want:
            progressed = False
            for key in keys:
                if taken >= want:
                    break
                if index < len(cells[key]):
                    picked.append(cells[key][index])
                    taken += 1
                    progressed = True
            if not progressed:
                break
            index += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help=(
            "CSV of seeds to keep out of the draw. Used to top the set back up "
            "after a seed is dropped: a replacement must not be one already "
            "tried, or the same defect comes back under the same name."
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data/benchmarks/minif2f_v2/raw/seeds_50.csv",
    )
    args = parser.parse_args()

    rows = load_rows()
    if args.exclude and args.exclude.is_file():
        used = {r["id"] for r in csv.DictReader(args.exclude.open(encoding="utf-8"))}
        before = len(rows)
        rows = [row for row in rows if row["name"] not in used]
        print(f"excluding {before - len(rows)} already-used seeds")
    picked = select(rows, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "statement", "answer", "formal_statement", "lean_header", "informal_proof", "split"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in picked:
            writer.writerow(
                {
                    "id": row["name"],
                    "statement": row["informal"],
                    "answer": "",
                    "formal_statement": row["statement"],
                    "lean_header": row["header"],
                    "informal_proof": row["informal_proof"],
                    "split": row["split"],
                }
            )
    summary = {
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "selected": len(picked),
        "by_family": dict(Counter(r["family"] for r in picked)),
        "by_shape": dict(Counter(r["shape"] for r in picked)),
        "by_hypotheses": dict(Counter(r["hyps"] for r in picked)),
        "by_split": dict(Counter(r["split"] for r in picked)),
        "benchmark_family_shares": {
            f: round(c / len(rows), 3)
            for f, c in Counter(r["family"] for r in rows).most_common()
        },
        "names": [r["name"] for r in picked],
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "names"}, indent=2))
    print(f"wrote {len(picked)} seeds -> {args.output}")


if __name__ == "__main__":
    main()
