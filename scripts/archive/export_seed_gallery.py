#!/usr/bin/env python3
"""Emit the seed sets as one JSON payload for the workspace gallery.

The gallery has to show every seed — 100 of them — with enough per-problem
detail to be worth opening: the prose, the Lean goal, all three hints, the
certificate, and the caveat when a rung does not mean what it says. That is
too much to hand-maintain in the page, and duplicating it would let the page
and the CSV drift apart, which is the specific failure the CSV exists to
prevent.

So the page gets its data from here, and here reads the same CSV a person
would open. Proofs travel in full — an excerpt is enough to see that a solution
exists but not to check it, and checking is the point. They are carried once:
the column dump refers to the same string rather than repeating it, which is
worth roughly a third of the payload.

Usage:
  python scripts/archive/export_seed_gallery.py --output ICLR_2027/seed_gallery.json
"""

from __future__ import annotations

import argparse
import csv
import json
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
SOURCES = [
    ("ProofNet-Verified", "proofnet_verified"),
    ("miniF2F-v2", "minif2f_v2"),
]


def split_list(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(";") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT.parent / "ICLR_2027" / "seed_gallery.json",
    )
    args = parser.parse_args()

    seeds: List[Dict[str, Any]] = []
    for label, folder in SOURCES:
        path = REPO_ROOT / "data" / "benchmarks" / folder / "seeds_50_levels.csv"
        for row in csv.DictReader(path.open(encoding="utf-8")):
            seeds.append(
                {
                    "id": row["id"],
                    "benchmark": label,
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                    "certificate": row["certificate"],
                    "max_hint_level": row["max_hint_level"],
                    "caveat": split_list(row["hint_caveat"]),
                    "goal_nl": row["goal"],
                    "lean": row["lean_goal"],
                    "header": row["lean_header"],
                    "hints": {
                        "1": split_list(row["tools"]),
                        "2": split_list(row["hint_outline"]),
                        "3": row["hint_first_step"],
                    },
                    "solution": row["solution"],
                    "solution_lines": len(row["solution"].splitlines()),
                    # Fields the page does not already carry under another
                    # name. The column *order* is shipped once at the top level
                    # and the values are read back off the seed, because a
                    # per-seed copy of every column duplicated the whole payload
                    # and pushed the page past what the publisher would accept.
                    "extra": {
                        k: v for k, v in row.items()
                        if k not in {
                            "id", "topic", "difficulty", "certificate", "goal",
                            "lean_header", "lean_goal", "tools", "hint_caveat",
                            "hint_outline", "hint_first_step", "solution",
                        }
                    },
                }
            )

    def tally(field: str, where=lambda s: True) -> Dict[str, int]:
        return dict(Counter(s[field] for s in seeds if where(s)).most_common())

    payload = {
        "column_order": [
            "id", "topic", "difficulty", "certificate", "goal", "lean_header",
            "lean_goal", "tools", "max_hint_level", "hint_caveat",
            "hint_outline", "hint_first_step", "solution",
        ],
        "seeds": seeds,
        "summary": {
            "total": len(seeds),
            "by_benchmark": tally("benchmark"),
            "by_certificate": tally("certificate"),
            "by_difficulty": tally("difficulty"),
            "topics_proofnet": tally("topic", lambda s: s["benchmark"] == "ProofNet-Verified"),
            "topics_minif2f": tally("topic", lambda s: s["benchmark"] == "miniF2F-v2"),
            "clean_ladder": sum(1 for s in seeds if not s["caveat"]),
            "caveats": dict(Counter(c for s in seeds for c in s["caveat"]).most_common()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    size = args.output.stat().st_size
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {len(seeds)} seeds ({size / 1024:.0f} KB) -> {args.output}")


if __name__ == "__main__":
    main()
