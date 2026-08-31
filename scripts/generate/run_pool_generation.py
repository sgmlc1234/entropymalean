#!/usr/bin/env python3
"""Run one generation of the central 5-pool LangGraph orchestrator."""

from __future__ import annotations

import argparse
import sys
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


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestration import run_pool_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5-pool LLM generation + Lean certification.")
    parser.add_argument("--input", required=True, type=Path, help="Input v1-style seed CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output per-slot JSONL.")
    parser.add_argument(
        "--summary-output", type=Path, default=None, help="Output generation summary JSON."
    )
    parser.add_argument("--max-generations", type=int, default=1, help="MVP supports 1.")
    parser.add_argument("--pool-size", type=int, default=5, help="MVP fixed pool size, default 5.")
    parser.add_argument("--survivor-count", type=int, default=1, help="Default 1 survivor.")
    parser.add_argument(
        "--crossover-count",
        type=int,
        default=2,
        help="Maximum crossover slots. Default 2 for 1 survivor + up to 2 crossover + mutation backfill.",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=5, help="Max LangGraph slot concurrency."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Generated-slot retry budget after the initial attempt. Default 3.",
    )
    parser.add_argument(
        "--target-accepted-per-generation",
        type=int,
        default=3,
        help="Accepted-proxy generated candidates target per generation. Default 3.",
    )
    parser.add_argument(
        "--reserve-budget",
        type=int,
        default=3,
        help="Maximum reserve generated slots when accepted-grade proxy yield is below target. Default 3.",
    )
    parser.add_argument(
        "--disable-reserve-slots",
        action="store_true",
        help="Disable reserve slots even when accepted-grade proxy yield is below target.",
    )
    parser.add_argument(
        "--gen0-proof-k",
        type=int,
        default=4,
        help="Independent Gen0 seed proof attempts for missing theorem proof bodies. Default 4.",
    )
    parser.add_argument(
        "--gen0-proof-turns",
        type=int,
        default=6,
        help="Verifier-refinement turns per Gen0 proof attempt. Default 6.",
    )
    parser.add_argument(
        "--gen0-max-seed-seconds",
        type=float,
        default=3600.0,
        help="Wall-clock cap per Gen0 seed proof completion. Default 3600.",
    )
    parser.add_argument(
        "--gen0-max-parallel",
        type=int,
        default=3,
        help="Max concurrent Gen0 seed proof completions. Default 3.",
    )
    parser.add_argument(
        "--skip-gen0",
        action="store_true",
        help="Skip Gen0 seed proof completion even if theorem seeds lack proof bodies.",
    )
    parser.add_argument(
        "--planner-memory-dir",
        type=Path,
        default=Path("data/certified"),
        help="Directory of prior JSONL outputs used for planner case memory. Default data/certified.",
    )
    parser.add_argument(
        "--planner-memory-limit",
        type=int,
        default=24,
        help="Maximum cross-run raw semantic planner cases to inject. Default 24.",
    )
    parser.add_argument(
        "--disable-planner-memory",
        action="store_true",
        help="Disable cross-run planner case memory.",
    )
    parser.add_argument(
        "--disable-leansearch",
        action="store_true",
        help="Disable LeanSearch premise retrieval for theorem proof workers.",
    )
    parser.add_argument(
        "--leansearch-limit",
        type=int,
        default=8,
        help="LeanSearch premise candidates per theorem query. Default 8.",
    )
    parser.add_argument(
        "--generation-model", default=None, help="OpenAI/OpenRouter-compatible model."
    )
    parser.add_argument(
        "--generation-temperature", type=float, default=None, help="Generation temperature."
    )
    parser.add_argument("--project", default=None, help="LangSmith project name.")
    parser.add_argument("--run-name", default=None, help="LangSmith run name suffix.")
    parser.add_argument("--tag", action="append", default=[], help="Additional LangSmith tag.")
    args = parser.parse_args()

    result = run_pool_generation(
        args.input,
        args.output,
        args.summary_output,
        max_generations=args.max_generations,
        pool_size=args.pool_size,
        survivor_count=args.survivor_count,
        crossover_count=args.crossover_count,
        max_parallel=args.max_parallel,
        max_retries=args.max_retries,
        target_accepted_per_generation=args.target_accepted_per_generation,
        reserve_budget=args.reserve_budget,
        disable_reserve_slots=args.disable_reserve_slots,
        gen0_proof_k=args.gen0_proof_k,
        gen0_proof_turns=args.gen0_proof_turns,
        gen0_max_seed_seconds=args.gen0_max_seed_seconds,
        gen0_max_parallel=args.gen0_max_parallel,
        skip_gen0=args.skip_gen0,
        planner_memory_dir=args.planner_memory_dir,
        planner_memory_limit=args.planner_memory_limit,
        disable_planner_memory=args.disable_planner_memory,
        disable_leansearch=args.disable_leansearch,
        leansearch_limit=args.leansearch_limit,
        generation_model=args.generation_model,
        generation_temperature=args.generation_temperature,
        project_name=args.project,
        run_name=args.run_name,
        tags=args.tag,
    )
    print(
        "summary "
        f"generation={result.generation_count} "
        f"pool_size={result.pool_size} "
        f"saved={len(result.saved_pool)} "
        f"failed={len(result.failed_slots)} "
        f"status={result.generation_save_status} "
        f"counts={result.counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
