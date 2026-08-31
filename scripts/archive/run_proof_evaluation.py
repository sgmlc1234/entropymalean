#!/usr/bin/env python3
"""Run the EntropyMaLean multi-turn proof evaluation campaign.

For every (problem, model) pair the evaluator runs K independent attempts,
each up to T_max turns. Every turn is one chat completion followed by a
Lean type-check; on failure the verifier diagnostics are folded back into
the next turn as the refinement prompt.

Usage:
    python scripts/archive/run_proof_evaluation.py \
        --benchmark miniF2F \
        --control data/raw/minif2f_pilot_5.csv \
        --treatment data/certified/minif2f_pilot_gen1.jsonl \
        --output data/evaluation/minif2f_proof.jsonl \
        --summary data/evaluation/minif2f_proof_summary.json \
        --K 3 --T-max 4 --control-cap 20 --treatment-cap 0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List

# Keep the documented ``python scripts/archive/run_proof_evaluation.py`` invocation
# working from a source checkout without requiring an editable installation.
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
from src.evaluation.proof_orchestrator import (
    run_proof_evaluation,
    summarize_proof_jsonl,
)
from src.evaluation.model_runner import MODEL_PANEL, ModelConfig


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument("--control", type=Path, default=None,
                   help="seed control CSV; if omitted, pulled from "
                        "config/benchmarks.yaml")
    p.add_argument("--benchmarks-config", type=Path,
                   default=Path("config/benchmarks.yaml"))
    p.add_argument("--treatment", type=Path, required=True,
                   help="quality-gated EMG-2 treatment JSONL")
    p.add_argument("--output", type=Path, required=True,
                   help="per-(problem, model) JSONL with K attempts each")
    p.add_argument("--summary", type=Path, required=True)

    p.add_argument("--K", type=int, default=3,
                   help="independent attempts per (problem, model) cell")
    p.add_argument("--T-max", type=int, default=4,
                   help="max verifier-driven turns per attempt "
                        "(whole-proof / chat-paradigm models)")
    p.add_argument("--S-max", type=int, default=6,
                   help="max tactic steps per attempt "
                        "(tactic-step / completion-paradigm models)")
    p.add_argument("--n-per-step", type=int, default=8,
                   help="candidate tactics sampled per step "
                        "(tactic-step models; mirrors BFS-V2 "
                        "n_sampling_search, scaled to a workshop budget)")
    p.add_argument("--n-parallel", type=int, default=1,
                   help="best-of-N parallel completions per (attempt, turn) "
                        "(whole-proof / chat-paradigm models; mirrors "
                        "Goedel-V1 / DSP-V1.5 best-of-N protocol)")
    p.add_argument("--bfs-tree-search", action="store_true",
                   help="use proper best-first tree search with priority queue "
                        "+ backtracking instead of greedy step expansion "
                        "(tactic-step / completion-paradigm models)")
    p.add_argument("--bfs-tree-max-nodes", type=int, default=64,
                   help="node-expansion budget per attempt when "
                        "--bfs-tree-search is set")
    p.add_argument("--lean-timeout", type=float, default=300.0)
    p.add_argument(
        "--lean-comparator",
        action="store_true",
        help=(
            "require leanprover/comparator acceptance for every candidate "
            "the fast verifier considers complete "
            "(Linux/landrun/systemd-run only)"
        ),
    )
    p.add_argument(
        "--comparator-bin",
        default=os.getenv("COMPARATOR_BIN", "comparator"),
        help="comparator executable name or absolute path",
    )
    p.add_argument(
        "--comparator-work-root",
        type=Path,
        default=None,
        help="generated comparator workspaces (default: beside --output)",
    )
    p.add_argument(
        "--comparator-mathlib-dir",
        type=Path,
        default=None,
        help="trusted local mathlib checkout (default: .lake/packages/mathlib)",
    )
    p.add_argument(
        "--comparator-timeout",
        type=float,
        default=600.0,
        help="seconds for lake update and each final comparator invocation",
    )
    p.add_argument("--model-timeout", type=float, default=180.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional completion cap. Omit by default so the provider/model supplies its own limit.",
    )
    p.add_argument("--max-parallel-cells", type=int, default=4)
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model labels to run, matched against MODEL_PANEL. "
            "Omit for the full resolved panel. Example: BFS-Prover-V2-7B"
        ),
    )

    p.add_argument("--resume", action="store_true",
                   help="append to --output and skip (problem_id, model) "
                        "cells already present; use to recover from "
                        "a mid-campaign crash without re-running them")
    p.add_argument("--control-cap", type=int, default=20)
    p.add_argument("--treatment-cap", type=int, default=0,
                   help="evaluation treatment cap; <=0 keeps all accepted rows "
                        "after problem_id dedup (default: all)")
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument(
        "--run-purpose",
        choices=("smoke", "pilot", "paper"),
        default="pilot",
        help="label the evidential scope so smoke results are not mistaken "
             "for paper-level estimates",
    )
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


def _select_models(labels: str | None, panel: List[ModelConfig]) -> List[ModelConfig]:
    """Filter the resolved model panel by exact label while preserving order."""
    if not labels:
        return list(panel)
    requested = [item.strip() for item in labels.split(",") if item.strip()]
    if not requested:
        return list(panel)
    by_label = {model.label: model for model in panel}
    missing = [label for label in requested if label not in by_label]
    if missing:
        available = ", ".join(model.label for model in panel) or "<empty>"
        raise SystemExit(
            f"unknown --models label(s): {', '.join(missing)}; "
            f"available labels: {available}"
        )
    return [by_label[label] for label in requested]


def main() -> None:
    args = _parse()
    models = _select_models(args.models, list(MODEL_PANEL))
    control = _load_controls(args)
    treatment = cap_rows(
        load_treatment_rows(args.treatment, args.benchmark), args.treatment_cap
    )
    rows: List[EvalRow] = control + treatment
    if not rows:
        raise SystemExit(
            "no rows to evaluate; check --control / --treatment paths"
        )
    print(
        f"benchmark={args.benchmark} control={len(control)} "
        f"treatment={len(treatment)} K={args.K} T_max={args.T_max} "
        f"models={[m.label for m in models]} "
        f"comparator={'on' if args.lean_comparator else 'off'}"
    )

    asyncio.run(
        run_proof_evaluation(
            rows,
            args.output,
            K=args.K,
            T_max=args.T_max,
            S_max=args.S_max,
            n_per_step=args.n_per_step,
            n_parallel=args.n_parallel,
            bfs_tree_search=args.bfs_tree_search,
            bfs_tree_max_nodes=args.bfs_tree_max_nodes,
            max_parallel_cells=args.max_parallel_cells,
            lean_timeout=args.lean_timeout,
            model_timeout=args.model_timeout,
            max_tokens=args.max_tokens,
            models=models,
            resume=args.resume,
            comparator_enabled=args.lean_comparator,
            comparator_bin=args.comparator_bin,
            comparator_work_root=args.comparator_work_root,
            comparator_mathlib_dir=args.comparator_mathlib_dir,
            comparator_timeout=args.comparator_timeout,
            run_purpose=args.run_purpose,
        )
    )
    summary = summarize_proof_jsonl(
        args.output, args.summary, bootstrap_iterations=args.bootstrap_iterations
    )
    print(f"wrote summary -> {args.summary}")
    for cell in summary["cells"]:
        ci = cell["pass_at_k_ci"]
        print(
            f"  {cell['benchmark']:<12} {cell['model']:<22} "
            f"{cell['arm']:<9} pass@{args.K}={cell['pass_at_k']*100:5.1f}% "
            f"CI=[{ci[0]*100:5.1f},{ci[1]*100:5.1f}] N={cell['rows']}"
        )
    for d in summary["drops"]:
        ci = d["drop_ci_pp"]
        print(
            f"  drop  {d['benchmark']:<12} {d['model']:<22} "
            f"ctrl={d['control_pass_at_k']*100:5.1f}% "
            f"trt={d['treatment_pass_at_k']*100:5.1f}% "
            f"drop={d['drop_pp']:+5.1f}pp CI=[{ci[0]:+5.1f},{ci[1]:+5.1f}]"
        )


if __name__ == "__main__":
    main()
