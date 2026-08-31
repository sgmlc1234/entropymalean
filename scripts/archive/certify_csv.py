#!/usr/bin/env python3
"""Certify CSV problems, optionally generating harder LLM children first."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolved one level short after the
    move -- to a directory that exists, so nothing raised."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certification import certify_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify CSV problems with Lean L2+ templates.")
    parser.add_argument("--input", required=True, type=Path, help="Input v1-style CSV file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process.")
    parser.add_argument("--project", default=None, help="LangSmith project name.")
    parser.add_argument("--run-name", default=None, help="LangSmith root run name.")
    parser.add_argument(
        "--generate-harder",
        action="store_true",
        help="Use an LLM node to generate one harder child per input row before certification.",
    )
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Model for harder problem generation. Defaults to GENERATION_MODEL, OPENAI_MODEL, or gpt-4o-mini.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=None,
        help="Temperature for harder problem generation. Defaults to GENERATION_TEMPERATURE or 0.3.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Additional LangSmith tag. May be repeated.",
    )
    args = parser.parse_args()

    results = certify_csv(
        args.input,
        args.output,
        limit=args.limit,
        generate_harder=args.generate_harder,
        generation_model=args.generation_model,
        generation_temperature=args.generation_temperature,
        project_name=args.project,
        run_name=args.run_name,
        tags=args.tag,
    )
    counts = Counter(result.status for result in results)
    print(
        "summary "
        f"total={len(results)} "
        f"certified={counts.get('certified', 0)} "
        f"unsupported={counts.get('unsupported', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"generation_failed={counts.get('generation_failed', 0)} "
        f"lean_unavailable={counts.get('lean_unavailable', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
