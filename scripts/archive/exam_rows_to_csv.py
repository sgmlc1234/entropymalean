#!/usr/bin/env python3
"""Flatten enriched exam rows into a spreadsheet-friendly CSV.

The JSONL is the record of truth — it keeps the ground-truth proof, the full
palette with signatures, and the per-episode results. This writes the analysis
view of it: one row per problem, scalar columns only, so the ablation tables
can be built in a spreadsheet or a dataframe without unpacking JSON.

Nested fields are handled rather than dropped: dicts become prefixed columns,
lists of names become semicolon-joined strings, and the hint ladder becomes one
column per level plus a flag for the level that leaks part of the proof. The
ground-truth proof body is excluded by default — it is the one field that would
make the file unreadable, and it stays one join away in the JSONL.

Usage:
  python scripts/archive/exam_rows_to_csv.py \
    --input data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl \
    --output data/benchmarks/proofnet_verified/raw/exam_rows_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# Order matters: this is the column order a reader will scan left to right.
COLUMNS: List[str] = [
    # identity
    "row_id", "name", "benchmark", "split_role", "exam_theorem", "used_corrected",
    # exam material
    "statement_nl", "formal_statement", "lean_header", "lean_code",
    # readiness
    "statement_checked", "statement_check_error",
    # stratification
    "topic", "textbook", "conclusion_shape", "hypothesis_bucket",
    # audit (semantic axis)
    "audit_faithfulness", "audit_provability", "audit_error_type",
    # difficulty proxies
    "gt_step_count", "gt_char_length", "gt_uses_induction", "gt_uses_cases", "gt_tactics",
    # help available
    "palette_theorem_count", "palette_theorems", "palette_tactic_count",
    "max_hint_level", "hint_l1_lemma_names", "hint_l2_outline", "hint_l3_first_tactic",
    "hint_l3_leaks_proof",
    # provenance
    "gt_proof_public", "gt_proof_source", "lean_toolchain", "mathlib_revision",
    "schema_version", "license",
    # results (filled by evaluation)
    "episode_count", "arms_run",
]


def _join(values: Any, limit: int = 400) -> str:
    if not values:
        return ""
    if isinstance(values, dict):
        values = sorted(values)
    text = "; ".join(str(v) for v in values)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    audit = row.get("audit") or {}
    palette = row.get("palette") or {}
    ladder = {int(h.get("level", 0)): h for h in (row.get("hint_ladder") or [])}
    episodes = row.get("episodes") or []

    out: Dict[str, Any] = {
        "row_id": row.get("stem") or row.get("seed") or row.get("name"),
        "name": row.get("name"),
        "benchmark": row.get("benchmark"),
        "split_role": row.get("lineage_role") or "seed",
        "exam_theorem": row.get("exam_theorem"),
        "used_corrected": row.get("used_corrected"),
        "statement_nl": " ".join(str(row.get("statement_nl") or "").split()),
        "formal_statement": str(row.get("formal_statement") or "").strip(),
        "lean_header": " ".join(str(row.get("lean_header") or "").split()),
        "lean_code": str(row.get("lean_code") or "").strip(),
        "statement_checked": row.get("statement_checked"),
        "statement_check_error": row.get("statement_check_error") or "",
        "topic": row.get("topic"),
        "textbook": row.get("textbook"),
        "conclusion_shape": row.get("conclusion_shape"),
        "hypothesis_bucket": row.get("hypothesis_bucket"),
        "audit_faithfulness": audit.get("faithfulness"),
        "audit_provability": audit.get("provability"),
        "audit_error_type": audit.get("error_type"),
        "gt_step_count": row.get("gt_step_count"),
        "gt_char_length": row.get("gt_char_length"),
        "gt_uses_induction": row.get("gt_uses_induction"),
        "gt_uses_cases": row.get("gt_uses_cases"),
        "gt_tactics": _join(row.get("gt_tactics")),
        "palette_theorem_count": row.get("palette_theorem_count"),
        "palette_theorems": _join(palette.get("theorems")),
        "palette_tactic_count": row.get("palette_tactic_count"),
        "max_hint_level": row.get("max_hint_level"),
        "hint_l1_lemma_names": _join((ladder.get(1) or {}).get("content")),
        "hint_l2_outline": _join((ladder.get(2) or {}).get("content"), limit=600),
        "hint_l3_first_tactic": " ".join(
            str((ladder.get(3) or {}).get("content") or "").split()
        ),
        "hint_l3_leaks_proof": (ladder.get(3) or {}).get("leaks_proof", ""),
        "gt_proof_public": row.get("gt_proof_public"),
        "gt_proof_source": row.get("gt_proof_source"),
        "lean_toolchain": row.get("lean_toolchain"),
        "mathlib_revision": (row.get("mathlib_revision") or "")[:12],
        "schema_version": row.get("schema_version"),
        "license": row.get("license"),
        "episode_count": len(episodes),
        "arms_run": _join(sorted({str(e.get("arm")) for e in episodes if e.get("arm")})),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="drop lean_code, for a narrow file meant only for reading",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    columns = list(COLUMNS)
    if args.no_proof:
        columns.remove("lean_code")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten(row))

    ready = sum(1 for r in rows if r.get("statement_checked"))
    print(f"wrote {len(rows)} rows x {len(columns)} columns -> {args.output}")
    print(f"  statement_checked: {ready}/{len(rows)}")
    print(f"  columns: {', '.join(columns[:10])} …")


if __name__ == "__main__":
    main()
