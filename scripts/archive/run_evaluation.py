#!/usr/bin/env python3
"""Run the EntropyMaG-2 direct no-tool evaluation campaign.

Usage:
    python scripts/archive/run_evaluation.py \
        --benchmark miniF2F \
        --control data/raw/minif2f_seed_control.csv \
        --treatment data/certified/minif2f_treatment.jsonl \
        --output data/evaluation/minif2f_eval.jsonl \
        --summary data/evaluation/minif2f_summary.json \
        --control-cap 20 --treatment-cap 100 --repeats 3

The script schedules ``3 models x 3 repeats`` per row through OpenRouter and
writes one JSONL line per (row, model, repeat) cell plus a JSON summary with
control/treatment accuracies, drops, Pass@3, and bootstrap CIs.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from src.evaluation.benchmarks import (
    load_benchmark_seeds_by_name,
    load_catalog,
)
from src.evaluation.dataset import (
    BENCHMARKS,
    EvalRow,
    cap_rows,
    load_control_rows,
    load_treatment_rows,
)
from src.evaluation.orchestrator import (
    generation_slope_table,
    run_evaluation_async,
    summarize_jsonl,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument(
        "--control",
        type=Path,
        default=None,
        help="seed control CSV (legacy); when omitted, controls are loaded "
        "from the HF snapshot via config/benchmarks.yaml",
    )
    p.add_argument(
        "--benchmarks-config",
        type=Path,
        default=Path("config/benchmarks.yaml"),
        help="catalog used when --control is omitted",
    )
    p.add_argument(
        "--treatment", type=Path, required=True, help="quality-gated treatment JSONL"
    )
    p.add_argument("--output", type=Path, required=True, help="per-cell JSONL")
    p.add_argument("--summary", type=Path, required=True, help="aggregated JSON")
    p.add_argument(
        "--slopes", type=Path, default=None, help="optional generation-slope JSON"
    )
    p.add_argument("--control-cap", type=int, default=20)
    p.add_argument("--treatment-cap", type=int, default=100,
                   help="evaluation treatment cap (default 100; revised down "
                        "from earlier 200 to match observed certification yield)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-parallel-rows", type=int, default=8)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    return p.parse_args()


def _load_controls(args: argparse.Namespace) -> List[EvalRow]:
    if args.control is not None:
        return cap_rows(
            load_control_rows(args.control, args.benchmark), args.control_cap
        )
    catalog = load_catalog(args.benchmarks_config)
    return load_benchmark_seeds_by_name(
        catalog, args.benchmark, limit=args.control_cap
    )


def main() -> None:
    args = _parse()

    control = _load_controls(args)
    treatment = cap_rows(
        load_treatment_rows(args.treatment, args.benchmark), args.treatment_cap
    )
    rows: List[EvalRow] = control + treatment
    if not rows:
        raise SystemExit("no rows to evaluate; check input paths and quality gating")
    print(
        f"benchmark={args.benchmark} control={len(control)} treatment={len(treatment)} "
        f"cells={len(rows) * args.repeats * 3}"
    )

    asyncio.run(
        run_evaluation_async(
            rows,
            args.output,
            repeats=args.repeats,
            max_parallel_rows=args.max_parallel_rows,
            timeout_seconds=args.timeout_seconds,
        )
    )

    summary = summarize_jsonl(
        args.output,
        args.summary,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"wrote summary -> {args.summary}")
    for cell in summary["cells"]:
        ci = cell["drop_ci_pp"]
        print(
            f"  {cell['benchmark']:<12} {cell['model']:<22} "
            f"ctrl={cell['control_accuracy']*100:5.1f}% "
            f"trt={cell['treatment_accuracy']*100:5.1f}% "
            f"drop={cell['drop_pp']:+5.1f}pp "
            f"CI=[{ci[0]:+5.1f},{ci[1]:+5.1f}]"
        )

    if args.slopes is not None:
        slopes = generation_slope_table(args.output)
        args.slopes.parent.mkdir(parents=True, exist_ok=True)
        args.slopes.write_text(__import__("json").dumps(slopes, indent=2))
        print(f"wrote slopes -> {args.slopes}")


if __name__ == "__main__":
    main()
