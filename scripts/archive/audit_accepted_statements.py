#!/usr/bin/env python3
"""Audit public accepted-ledger statements for workflow/internal wording."""

from __future__ import annotations

import argparse
import json
import re
import sys
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
DEFAULT_INPUT = REPO_ROOT / "data/evaluation/treatment_inventory/final_curated/accepted.jsonl"

sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.quality import informal_statement_internal_term_hits  # noqa: E402


LEAN_SURFACE_PATTERNS = {
    "lean_namespace_nat": re.compile(r"\bNat\."),
    "lean_namespace_finset": re.compile(r"\bFinset\."),
    "lean_namespace_int": re.compile(r"\bInt\."),
    "lean_cast": re.compile(r"\(\s*[A-Za-z][A-Za-z0-9_']*\s*:\s*[ℕℤℚℝℂ][^)]*\)"),
    "lean_percent_mod": re.compile(r"\b[A-Za-z][A-Za-z0-9_']*\s*%\s*[A-Za-z0-9_']+"),
    "ascii_le": re.compile(r"<="),
    "ascii_ge": re.compile(r">="),
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit(path: Path, *, long_statement_chars: int) -> int:
    rows = _read_jsonl(path)
    term_hits = []
    long_rows = []
    for index, row in enumerate(rows, start=1):
        statement = str(row.get("statement") or "")
        hits = list(informal_statement_internal_term_hits(statement))
        hits.extend(name for name, pattern in LEAN_SURFACE_PATTERNS.items() if pattern.search(statement))
        if hits:
            term_hits.append((index, row, sorted(set(hits))))
        if len(statement) > long_statement_chars:
            long_rows.append((index, row, len(statement)))

    print(f"rows={len(rows)}")
    print(f"internal_term_hits={len(term_hits)}")
    for index, row, hits in term_hits:
        print(
            f"- line={index} benchmark={row.get('benchmark')} "
            f"problem_id={row.get('problem_id')} hits={','.join(hits)}"
        )
        print(f"  statement={row.get('statement')}")

    print(f"long_statements>{long_statement_chars}={len(long_rows)}")
    for index, row, length in long_rows[:25]:
        print(
            f"- line={index} len={length} benchmark={row.get('benchmark')} "
            f"problem_id={row.get('problem_id')}"
        )
    return 1 if term_hits else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--long-statement-chars", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    raise SystemExit(audit(input_path, long_statement_chars=args.long_statement_chars))


if __name__ == "__main__":
    main()
