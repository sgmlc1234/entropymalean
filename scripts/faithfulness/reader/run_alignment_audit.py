#!/usr/bin/env python3
"""Batch goal_roundtrip alignment audit over certified rows (release-gate evidence).

Reads a JSONL of generated rows, computes the elaborated-goal alignment
signal for each (docs/semantic_alignment_plan.md), and writes the rows back
with an ``alignment_evidence`` field plus a summary. Rows whose signal says
``equivalent == false`` are the audit queue — they are flagged, not deleted.

Usage:
  python scripts/faithfulness/reader/run_alignment_audit.py \
    --input release/huggingface/EML-1/accepted.jsonl \
    --output data/evaluation/alignment_audit/accepted_annotated.jsonl \
    --summary data/evaluation/alignment_audit/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

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

from src.certification.alignment import elaborated_goal_alignment  # noqa: E402
from src.certification.generation import default_generation_config  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lean-timeout", type=float, default=300.0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--include-uncertified",
        action="store_true",
        help="Audit every row, not only status=certified ones.",
    )
    return parser.parse_args()


async def _run() -> None:
    args = _parse()
    config = default_generation_config(args.model, args.temperature)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not args.include_uncertified:
        eligible = [
            row
            for row in rows
            if (row.get("status") or (row.get("certificate") or {}).get("status"))
            == "certified"
        ]
    else:
        eligible = list(rows)
    if args.limit > 0:
        eligible = eligible[: args.limit]
    eligible_ids = {id(row) for row in eligible}

    semaphore = asyncio.Semaphore(max(1, args.max_parallel))

    async def _audit(row: dict) -> None:
        statement = str(row.get("statement") or "").strip()
        formal = str(row.get("formal_statement") or "").strip()
        header = str(row.get("lean_header") or "").strip() or "import Mathlib"
        if not statement or not formal:
            row["alignment_evidence"] = {
                "source": "elaborated_goal_informalization",
                "status": "missing_fields",
                "equivalent": None,
            }
            return
        async with semaphore:
            row["alignment_evidence"] = await elaborated_goal_alignment(
                statement_nl=statement,
                formal_statement=formal,
                lean_header=header,
                config=config,
                lean_timeout=args.lean_timeout,
            )

    await asyncio.gather(*(_audit(row) for row in eligible))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if id(row) not in eligible_ids:
                row.setdefault(
                    "alignment_evidence",
                    {"source": "elaborated_goal_informalization", "status": "skipped"},
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audited = [row for row in rows if id(row) in eligible_ids]
    statuses = Counter(
        str((row.get("alignment_evidence") or {}).get("status")) for row in audited
    )
    verdicts = Counter(
        str((row.get("alignment_evidence") or {}).get("equivalent")) for row in audited
    )
    flagged = [
        {
            "problem_id": row.get("problem_id"),
            "mismatches": (row.get("alignment_evidence") or {}).get("mismatches"),
        }
        for row in audited
        if (row.get("alignment_evidence") or {}).get("equivalent") is False
    ]
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "audited": len(audited),
        "signal_status_counts": dict(statuses),
        "equivalent_counts": dict(verdicts),
        "flagged_rows": flagged,
        "model": config.model,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"audited={len(audited)} statuses={dict(statuses)} "
        f"equivalent={dict(verdicts)} flagged={len(flagged)}"
    )


if __name__ == "__main__":
    asyncio.run(_run())
