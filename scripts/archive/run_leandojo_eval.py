#!/usr/bin/env python3
"""Run the LeanDojo-backed BFS arm of the EntropyMaLean evaluation.

Takes the Theorem-record JSONL produced by
``scripts/archive/prepare_leandojo_theorems.py``, runs
``src.evaluation.leandojo_bfs.prove_with_leandojo_bfs`` against the LM
Studio panel, and writes per-(theorem, model) JSONL records compatible
with the existing summary aggregator.

Usage:
    python scripts/archive/run_leandojo_eval.py \
        --theorems /tmp/eml_campaign/minif2f_control_leandojo.jsonl \
        --output data/evaluation/campaign_*/minif2f_leandojo_bfs.jsonl \
        --summary data/evaluation/campaign_*/minif2f_leandojo_bfs_summary.json \
        --K 3 --n-sampling 16 --timeout 300 --max-parallel 2 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from lean_dojo import LeanGitRepo, Theorem

from src.evaluation.leandojo_bfs import (
    LeanDojoTacticGenerator,
    SearchResult,
    prove_with_leandojo_bfs,
)
from src.evaluation.model_runner import MODEL_PANEL, ModelConfig, _client_for


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theorems", type=Path, required=True,
                   help="JSONL from prepare_leandojo_theorems.py")
    p.add_argument("--output", type=Path, required=True,
                   help="Per-(theorem, model) result JSONL")
    p.add_argument("--summary", type=Path, default=None,
                   help="Optional roll-up JSON with Pass@K per (benchmark, arm)")
    p.add_argument("--K", type=int, default=3,
                   help="Independent attempts per (theorem, model) cell")
    p.add_argument("--n-sampling", type=int, default=16,
                   help="Tactic candidates per expansion (BFS-V2 default 16)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-attempt wall-clock timeout (seconds)")
    p.add_argument("--tactic-timeout", type=float, default=10.0,
                   help="Per-tactic Dojo timeout (seconds)")
    p.add_argument("--max-parallel", type=int, default=2,
                   help="Concurrent (theorem, model) cells")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--resume", action="store_true",
                   help="Skip (problem_id, model_label) cells already in --output")
    return p.parse_args()


def _read_theorems(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _completed_cells(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    seen: Set[Tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("problem_id")
            model = r.get("model")
            if pid and model:
                seen.add((pid, model))
    return seen


def _filter_completion_models(panel: List[ModelConfig]) -> List[ModelConfig]:
    """LeanDojo BFS only makes sense for tactic-step (completion) models.

    Goedel-Prover-V2 etc. are whole-proof chat models — they emit the
    full proof body in one shot, not single tactics. Drop them from the
    LeanDojo arm so we don't waste GPU hours.
    """
    keep = [m for m in panel if m.paradigm == "completion"]
    if not keep:
        raise SystemExit(
            "no completion-paradigm models in MODEL_PANEL; "
            "the LeanDojo arm requires a tactic-step prover (e.g. BFS-Prover-V2)"
        )
    return keep


async def _one_cell(
    *,
    theorem_record: Dict[str, Any],
    model: ModelConfig,
    client,
    K: int,
    timeout_per_attempt_s: float,
    n_sampling: int,
    tactic_timeout_s: float,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    repo = LeanGitRepo(theorem_record["repo_url"], theorem_record["repo_commit"])
    theorem = Theorem(repo, theorem_record["file_path"], theorem_record["full_name"])
    generator = LeanDojoTacticGenerator(
        client,
        model=model.provider_slug,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result: SearchResult = await prove_with_leandojo_bfs(
        theorem=theorem,
        generator=generator,
        K=K,
        timeout_per_attempt_s=timeout_per_attempt_s,
        n_sampling=n_sampling,
        tactic_timeout_s=tactic_timeout_s,
    )
    record = {
        "problem_id": theorem_record["problem_id"],
        "benchmark": theorem_record.get("benchmark"),
        "arm": theorem_record.get("arm"),
        "model": model.label,
        "provider_slug": model.provider_slug,
        "paradigm": "leandojo_bfs",
        "pass_at_k": result.status.value == "Proved",
        "min_turns_to_success": (
            len(result.proof) if result.proof else None
        ),
        "total_elapsed_seconds": result.total_time,
        "search_result": result.to_summary(),
    }
    return record


async def _main_async(args: argparse.Namespace) -> None:
    theorems = _read_theorems(args.theorems)
    panel = _filter_completion_models(list(MODEL_PANEL))
    print(f"theorems: {len(theorems)}  models: {[m.label for m in panel]}")
    print(f"K={args.K}  n_sampling={args.n_sampling}  timeout={args.timeout}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skip = _completed_cells(args.output) if args.resume else set()
    if args.resume and skip:
        print(f"resume: skipping {len(skip)} already-completed cells")
    open_mode = "a" if args.resume and args.output.exists() else "w"

    semaphore = asyncio.Semaphore(args.max_parallel)
    clients: Dict[str, Any] = {m.label: _client_for(m) for m in panel}
    started = time.monotonic()
    written = 0

    async def _bounded(thm_rec: Dict[str, Any], model: ModelConfig) -> Dict[str, Any]:
        async with semaphore:
            return await _one_cell(
                theorem_record=thm_rec,
                model=model,
                client=clients[model.label],
                K=args.K,
                timeout_per_attempt_s=args.timeout,
                n_sampling=args.n_sampling,
                tactic_timeout_s=args.tactic_timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    try:
        with args.output.open(open_mode) as out:
            tasks = [
                _bounded(thm, model)
                for thm in theorems
                for model in panel
                if (thm["problem_id"], model.label) not in skip
            ]
            for fut in asyncio.as_completed(tasks):
                record = await fut
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
    finally:
        for cli in clients.values():
            close = getattr(cli, "close", None)
            if close is not None:
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe

    elapsed = time.monotonic() - started
    print(f"done: cells={written}  elapsed={elapsed:.1f}s -> {args.output}")

    if args.summary:
        _write_summary(args.output, args.summary)


def _write_summary(jsonl_path: Path, summary_path: Path) -> None:
    """Pass@K per (benchmark, model, arm) — same schema as
    proof_orchestrator.summarize_proof_jsonl, simplified.
    """
    from collections import defaultdict

    cells: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["benchmark"], r["model"], r["arm"])
            cells[key].append(1.0 if r["pass_at_k"] else 0.0)

    summary_payload: List[Dict[str, Any]] = []
    for (bench, model, arm), row_acc in sorted(cells.items()):
        n = len(row_acc)
        summary_payload.append(
            {
                "benchmark": bench,
                "model": model,
                "arm": arm,
                "rows": n,
                "pass_at_k": sum(row_acc) / n if n else 0.0,
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"cells": summary_payload}, indent=2))
    print(f"wrote summary -> {summary_path}")
    for cell in summary_payload:
        print(
            f"  {cell['benchmark']:13} {cell['model']:25} "
            f"{cell['arm']:9} pass@K={cell['pass_at_k']*100:5.1f}% N={cell['rows']}"
        )


def main() -> None:
    asyncio.run(_main_async(_parse()))


if __name__ == "__main__":
    main()
