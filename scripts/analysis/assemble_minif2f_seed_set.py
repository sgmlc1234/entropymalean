#!/usr/bin/env python3
"""Assemble the final miniF2F seed set from every proof run, to a fixed size.

The set was built over several passes — an initial run, a retry with a larger
budget, and a pool of spares drawn to replace what could not be proved — and
which pass a seed came from is an accident of scheduling, not a property worth
recording. This collapses them into one file of exactly `--count` seeds.

Only proved seeds are eligible, because a seed without a proof cannot be bred
from. Where more are available than needed, spares are taken in an order that
keeps the stratification the original draw was designed for: the seed set
should still look like miniF2F, not like whichever problems happened to be
easy enough to prove.

Usage:
  python scripts/analysis/assemble_minif2f_seed_set.py --count 50 \
    --output data/benchmarks/minif2f_v2/raw/seeds_50_final.csv
"""

from __future__ import annotations

import argparse
import csv
import json
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

RAW = REPO_ROOT / "data/benchmarks/minif2f_v2/raw"
SOURCES = ["seeds_49_proved.csv", "retry_11_proved.csv", "spares_16_proved.csv"]
_SORRY = re.compile(r"(?<![A-Za-z_])(sorry|admit)(?![A-Za-z_])")

FAMILY_PREFIXES = (
    "mathd_algebra", "mathd_numbertheory", "amc12", "aime", "imo",
    "induction", "algebra", "numbertheory",
)


def family(name: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if str(name).startswith(prefix):
            return prefix
    return "other"


def load_proved(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Later files win: a retry that succeeded supersedes the run that did not."""
    proved: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8")):
            code = (row.get("lean_code") or "").strip()
            if code and not _SORRY.search(code):
                proved[str(row["id"])] = row
    return proved


def take_balanced(rows: List[Dict[str, Any]], want: int, have: Counter) -> List[Dict[str, Any]]:
    """Fill `want` slots, each time from the family currently most under-drawn.

    Taking spares in file order would let one family absorb the whole top-up,
    which is exactly the skew the stratified draw exists to prevent — and the
    families that need topping up are the ones whose seeds were hardest to
    prove, so the skew would correlate with difficulty.
    """
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[family(row["id"])].append(row)
    picked: List[Dict[str, Any]] = []
    while len(picked) < want:
        candidates = [f for f, rs in by_family.items() if rs]
        if not candidates:
            break
        weakest = min(candidates, key=lambda f: (have[f] + sum(1 for p in picked if family(p["id"]) == f)))
        picked.append(by_family[weakest].pop(0))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, default=RAW / "seeds_50_final.csv")
    parser.add_argument(
        "--core",
        type=Path,
        default=RAW / "seeds_50_screened.csv",
        help="the original draw; its proved members are kept before any spare",
    )
    args = parser.parse_args()

    proved = load_proved([RAW / name for name in SOURCES])
    core_ids = [r["id"] for r in csv.DictReader(args.core.open(encoding="utf-8"))]
    core = [proved[i] for i in core_ids if i in proved]
    spares = [r for k, r in proved.items() if k not in set(core_ids)]

    have = Counter(family(r["id"]) for r in core)
    chosen = core[: args.count]
    if len(chosen) < args.count:
        chosen += take_balanced(spares, args.count - len(chosen), have)

    if len(chosen) < args.count:
        raise SystemExit(
            f"only {len(chosen)} proved seeds available, need {args.count} — "
            f"draw more spares with select_minif2f_v2_seeds.py --exclude"
        )

    fields = list(chosen[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)

    summary = {
        "count": len(chosen),
        "from_original_draw": len(core[: args.count]),
        "from_spares": len(chosen) - len(core[: args.count]),
        "proved_pool": len(proved),
        "by_family": dict(Counter(family(r["id"]) for r in chosen).most_common()),
        "by_split": dict(Counter(r.get("split") for r in chosen)),
        "names": [r["id"] for r in chosen],
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "names"}, indent=2))
    print(f"wrote {len(chosen)} proved seeds -> {args.output}")


if __name__ == "__main__":
    main()
