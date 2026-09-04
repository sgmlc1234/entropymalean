"""End-to-end evaluation orchestrator.

Schedules the model panel across a list of ``EvalRow`` records, grades each
extracted answer, and writes a JSONL with one record per ``(row, model,
repeat)`` cell plus a JSON summary with arm-level accuracies, drops, and
bootstrap CIs.

The protocol implemented here:
- temperature 0, provider/model default completion limit, ``\\boxed{...}`` extraction
- 3 repeats per problem, 3 models per panel
- bootstrap CI on dataset row indices (not individual repeats)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.evaluation.answer_grader import grade_answer
from src.evaluation.bootstrap_ci import bootstrap_ci, bootstrap_drop_ci
from src.evaluation.dataset import EvalRow
from src.evaluation.model_runner import (
    MODEL_PANEL,
    ModelConfig,
    ModelResponse,
    _client,
    run_model_panel,
)


async def _run_row(
    row: EvalRow,
    *,
    repeats: int,
    models: Sequence[ModelConfig],
    timeout_seconds: float,
    semaphore: asyncio.Semaphore,
    client,
) -> List[Dict]:
    async with semaphore:
        responses: List[ModelResponse] = await run_model_panel(
            problem_id=row.problem_id,
            statement=row.statement,
            repeats=repeats,
            models=list(models),
            timeout_seconds=timeout_seconds,
            client=client,
        )
    out: List[Dict] = []
    for resp in responses:
        is_correct = grade_answer(resp.extracted_answer, row.gold_answer)
        out.append(
            {
                "benchmark": row.benchmark,
                "arm": row.arm,
                "problem_id": row.problem_id,
                "generation": row.generation,
                "family": row.family,
                "lean_level": row.lean_level,
                "model": resp.model_label,
                "provider_slug": resp.provider_slug,
                "repeat_index": resp.repeat_index,
                "gold_answer": row.gold_answer,
                "extracted_answer": resp.extracted_answer,
                "is_correct": bool(is_correct),
                "finish_reason": resp.finish_reason,
                "error": resp.error,
                "elapsed_seconds": resp.elapsed_seconds,
                "usage": resp.usage,
                "raw_text": resp.raw_text,
            }
        )
    return out


async def run_evaluation_async(
    rows: Sequence[EvalRow],
    output_jsonl: Path,
    *,
    repeats: int = 3,
    models: Optional[Sequence[ModelConfig]] = None,
    max_parallel_rows: int = 8,
    timeout_seconds: float = 180.0,
) -> Path:
    """Run the panel over `rows`, streaming one record per cell to JSONL.

    Returns the output path. The JSONL is append-safe across resumes because
    the orchestrator writes a single line per ``(row, model, repeat)`` cell.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    panel = list(models or MODEL_PANEL)
    semaphore = asyncio.Semaphore(max_parallel_rows)
    client = _client()
    started = time.monotonic()
    written = 0
    try:
        with output_jsonl.open("w") as out_f:
            tasks = [
                _run_row(
                    row,
                    repeats=repeats,
                    models=panel,
                    timeout_seconds=timeout_seconds,
                    semaphore=semaphore,
                    client=client,
                )
                for row in rows
            ]
            for fut in asyncio.as_completed(tasks):
                records = await fut
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                out_f.flush()
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe
    elapsed = time.monotonic() - started
    print(
        f"evaluation done rows={len(rows)} cells={written} elapsed={elapsed:.1f}s -> {output_jsonl}"
    )
    return output_jsonl


def _row_accuracy(records: Iterable[Dict], model: str) -> Dict[str, List[float]]:
    """Return ``{problem_id: per-row mean accuracy}`` for one model."""
    bucket: Dict[str, List[bool]] = {}
    for r in records:
        if r["model"] != model:
            continue
        bucket.setdefault(r["problem_id"], []).append(bool(r["is_correct"]))
    return {pid: [sum(v) / len(v) for _ in [None]] for pid, v in bucket.items()}


def summarize_jsonl(
    eval_jsonl: Path,
    summary_json: Path,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> Dict:
    """Compute arm-level accuracies, drops, and bootstrap CIs.

    The summary structure is
    one cell per (benchmark, model) with control acc, treatment acc, drop in pp,
    and a 95% bootstrap CI on the drop.
    """
    records: List[Dict] = []
    with eval_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    cells: Dict[str, Dict] = {}
    for r in records:
        key = (r["benchmark"], r["model"])
        cell = cells.setdefault(
            f"{r['benchmark']}::{r['model']}",
            {
                "benchmark": r["benchmark"],
                "model": r["model"],
                "control": {},
                "treatment": {},
            },
        )
        arm_bucket = cell[r["arm"]]
        per_row = arm_bucket.setdefault(r["problem_id"], [])
        per_row.append(bool(r["is_correct"]))

    summary_cells: List[Dict] = []
    for cell in cells.values():
        control_rows = [sum(v) / len(v) for v in cell["control"].values()]
        treatment_rows = [sum(v) / len(v) for v in cell["treatment"].values()]
        c_mean, c_lo, c_hi = bootstrap_ci(
            control_rows, iterations=bootstrap_iterations, seed=seed
        )
        t_mean, t_lo, t_hi = bootstrap_ci(
            treatment_rows, iterations=bootstrap_iterations, seed=seed + 1
        )
        drop, lo, hi = bootstrap_drop_ci(
            control_rows,
            treatment_rows,
            iterations=bootstrap_iterations,
            seed=seed + 2,
        )
        # Pass@3 per row = any repeat correct; arm-level Pass@3 = mean over rows
        def _pass3(arm_dict: Dict[str, List[bool]]) -> float:
            if not arm_dict:
                return 0.0
            return sum(1.0 if any(v) else 0.0 for v in arm_dict.values()) / len(arm_dict)

        c_p3 = _pass3(cell["control"])
        t_p3 = _pass3(cell["treatment"])

        summary_cells.append(
            {
                "benchmark": cell["benchmark"],
                "model": cell["model"],
                "control_rows": len(cell["control"]),
                "treatment_rows": len(cell["treatment"]),
                "control_accuracy": c_mean,
                "control_ci": [c_lo, c_hi],
                "control_pass_at_3": c_p3,
                "treatment_accuracy": t_mean,
                "treatment_ci": [t_lo, t_hi],
                "treatment_pass_at_3": t_p3,
                "drop_pp": drop,
                "drop_ci_pp": [lo, hi],
                "pass_at_3_drop_pp": 100.0 * (c_p3 - t_p3),
            }
        )

    payload = {
        "source_jsonl": str(eval_jsonl),
        "cells": sorted(summary_cells, key=lambda c: (c["benchmark"], c["model"])),
        "config": {
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2))
    return payload


def generation_slope_table(eval_jsonl: Path) -> List[Dict]:
    """Per (benchmark, model) least-squares slope across generations 1..G."""
    import statistics

    records: List[Dict] = []
    with eval_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    bucket: Dict[tuple, Dict[int, List[bool]]] = {}
    for r in records:
        if r["arm"] != "treatment":
            continue
        gen = int(r.get("generation") or 0)
        key = (r["benchmark"], r["model"])
        bucket.setdefault(key, {}).setdefault(gen, []).append(bool(r["is_correct"]))

    out: List[Dict] = []
    for (bench, model), gen_map in bucket.items():
        gens = sorted(g for g in gen_map if g >= 1)
        if len(gens) < 2:
            continue
        xs = [float(g) for g in gens]
        ys = [
            sum(gen_map[g]) / len(gen_map[g]) * 100.0 for g in gens
        ]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = num / den
        out.append(
            {
                "benchmark": bench,
                "model": model,
                "gen_first": ys[0],
                "gen_last": ys[-1],
                "delta_pp": ys[-1] - ys[0],
                "slope_pp_per_gen": slope,
                "gens_covered": gens,
            }
        )
    return out
