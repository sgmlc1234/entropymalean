#!/usr/bin/env python3
"""Drop seeds whose hypotheses are contradictory, before any proof is attempted.

miniF2F carries a known formalization defect: numeric literals written as `1/4`
elaborate as *natural* division and collapse to `0`, so a statement like

    (h : ((11 : ℝ) ^ (1 / 4)) ^ (3 * x - 3) = 1 / 5)

has `11 ^ 0 = 1` in it, making `h` say `1 = 1/5`. The hypotheses are
contradictory, every goal follows, and `norm_num at h` "proves" the theorem
without touching the mathematics the prose asked about.

Such a row is worthless as a seed twice over: the proof teaches nothing, and
every mutation of it inherits the contradiction. Our alignment gate already
rejects these downstream, but by then a proof attempt has been spent on each —
so this screens them out first, by asking Lean directly whether `False` follows
from the hypotheses alone.

The test is one-sided on purpose. A seed that survives is not thereby faithful;
it merely is not vacuous, which is the cheap half of the check.

Usage:
  LEAN_VERIFIER=repl python scripts/faithfulness/lean/screen_vacuous_seeds.py \
    --input data/benchmarks/minif2f_v2/raw/seeds_50.csv \
    --output data/benchmarks/minif2f_v2/raw/seeds_50_screened.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
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

# Tactics that will find a contradiction if the arithmetic makes one available.
# Kept short and total: this is a screen, not a prover, and a miss costs only a
# seed that the alignment gate will catch later anyway.
# Shared with the generation-time gate so a seed and its descendants are judged
# by one probe. The earlier local copy lacked `done` on each branch, which let
# `first` stop at a branch that had not closed the goal.
from src.certification.vacuity import REFUTATION, binders_of, vacuity_probe  # noqa: E402,F401






async def screen(rows: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
    verdicts: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        probe = vacuity_probe(row.get("lean_header"), row.get("formal_statement"))
        result = await verify_lean_proof_repl(probe, timeout=timeout)
        # ok == True means `False` was derived from the hypotheses: vacuous.
        vacuous = bool(result.ok)
        verdicts.append({"id": row.get("id"), "vacuous": vacuous})
        mark = "VACUOUS" if vacuous else "ok     "
        print(f"[{mark}] {index:3d}/{len(rows)} {str(row.get('id'))[:46]}", flush=True)
    return verdicts


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    try:
        verdicts = await screen(rows, args.timeout)
    finally:
        await close_global_repl_verifier()

    vacuous = {v["id"] for v in verdicts if v["vacuous"]}
    kept = [row for row in rows if row.get("id") not in vacuous]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)

    report = {
        "input": str(args.input),
        "seeds": len(rows),
        "vacuous": sorted(vacuous),
        "vacuous_count": len(vacuous),
        "kept": len(kept),
        "rate": round(len(vacuous) / max(1, len(rows)), 3),
    }
    (args.output.parent / f"{args.output.stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "vacuous"}, indent=2))
    print(f"kept {len(kept)}/{len(rows)} -> {args.output}")


if __name__ == "__main__":
    asyncio.run(_run())
