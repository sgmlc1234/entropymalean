#!/usr/bin/env python3
"""miniF2F seed grid experiment: BFS-Prover plays the exam environment.

Arms:
  closed_book — live goal-state prompt only (already an upgrade over the
                prefix-approximation prompt of the legacy harness);
  palette     — the goal-state prompt additionally lists sufficient lemma
                names extracted from a locally verified proof of the seed.

Seeds: miniF2F seeds that have a locally verified proof (Gen0 completions
and certified seed rows) play BOTH arms (paired comparison); extra seeds
without proofs play closed_book only (and successful episodes mine new
proofs → future palettes).

Uses the warm REPL verifier: one ~25s warmup, then ~seconds per step.

Usage:
  set -a; source .env; set +a
  EVAL_BFS_PROVER_SLUG=bytedance-seed.bfs-prover-v2-7b \
  python scripts/archive/run_exam_grid.py --output tmp/exam_grid \
    --paired-limit 18 --extra-seeds 7
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.bfs_step_prover import (  # noqa: E402
    default_tactic_sampler,
)
from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.evaluation.model_runner import ModelConfig, _client_for  # noqa: E402
from src.exam_env.bfs_player import (  # noqa: E402
    BFSExamPlayer,
    llama_cpp_scored_sampler,
)
from src.exam_env.environment import LeanExamEnv  # noqa: E402
from src.exam_env.palette import build_palette  # noqa: E402
from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

MINIF2F_PATH = REPO_ROOT / "data" / "benchmarks" / "minif2f" / "minif2f.jsonl"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="output dir")
    parser.add_argument("--paired-limit", type=int, default=18)
    parser.add_argument("--extra-seeds", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=90)
    # Reference n_sampling_search default is 16 (main.py:111).
    parser.add_argument("--n-per-step", type=int, default=16)
    parser.add_argument("--resample-rounds", type=int, default=1)
    parser.add_argument("--max-rollbacks", type=int, default=3)
    parser.add_argument("--lean-timeout", type=float, default=180.0)
    parser.add_argument("--seed-concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--arms",
        default="closed_book,palette",
        help="comma-separated arms to run on paired seeds",
    )
    parser.add_argument("--tag", default="", help="label recorded on each episode")
    parser.add_argument(
        "--llama-cpp",
        default=None,
        metavar="BASE_URL",
        help=(
            "serve from a llama.cpp server (e.g. http://127.0.0.1:8080/v1) to "
            "get token log-probabilities; LM Studio does not return them"
        ),
    )
    return parser.parse_args()


def harvest_seed_proofs() -> Dict[str, str]:
    """Locally verified miniF2F seed proofs (Gen0 completions et al.)."""
    proofs: Dict[str, str] = {}
    for path in glob.glob(str(REPO_ROOT / "data/certified/minif2f*gen0_seeds.csv")):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            code = (row.get("lean_code") or "").strip()
            completed = str(row.get("gen0_proof_completed") or "").lower()
            if code and "sorry" not in code and completed in {"true", "1", ""}:
                proofs.setdefault(str(row.get("id")), code)
    for path in glob.glob(str(REPO_ROOT / "data/certified/minif2f*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("op_type") == "seed_proof_completion"
                and row.get("status") == "certified"
            ):
                code = (row.get("lean_code") or "").strip()
                seed_id = str(row.get("source_problem_id") or row.get("problem_id"))
                if code and "sorry" not in code:
                    proofs.setdefault(seed_id, code)
    return proofs


def load_minif2f() -> Dict[str, dict]:
    rows = {}
    for line in open(MINIF2F_PATH, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            # Upstream miniF2F statements use the pre-4.x big-operator binder
            # syntax (`∑ k in s`); normalize for the pinned toolchain.
            row["formal_statement"] = _prelint_lean_syntax(
                str(row.get("formal_statement") or "")
            )
            rows[str(row["name"])] = row
    return rows


def _bfs_model_config(
    temperature: Optional[float] = None, top_p: Optional[float] = None
) -> ModelConfig:
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("BFS_TEMPERATURE", "0.7"))
    )
    top_p = top_p if top_p is not None else float(os.getenv("BFS_TOP_P", "1.0"))
    slug = os.getenv("EVAL_BFS_PROVER_SLUG", "bytedance-seed.bfs-prover-v2-7b")
    return ModelConfig(
        label="BFS-Prover-V2-7B",
        provider_slug=slug,
        backend="lm_studio",
        paradigm="completion",
        temperature=temperature,
        top_p=top_p,
        # Reference default is 2048 (main.py:129); 256 truncates tactics.
        max_tokens=int(os.getenv("BFS_MAX_TOKENS", "2048")),
        seed=int(os.getenv("BFS_SEED", "30")),
        stop=[":::", "\n\n"],
    )


async def _run() -> None:
    args = _parse()
    args.output.mkdir(parents=True, exist_ok=True)
    proofs = harvest_seed_proofs()
    bench = load_minif2f()

    paired = [name for name in sorted(proofs) if name in bench][: args.paired_limit]
    extras = [
        name
        for name in sorted(bench)
        if name not in proofs
        and any(name.startswith(p) for p in ("mathd_numbertheory", "mathd_algebra"))
    ][: args.extra_seeds]
    print(f"paired seeds (both arms): {len(paired)}; extras (closed_book): {len(extras)}")

    config = _bfs_model_config(args.temperature, args.top_p)
    client = _client_for(config)

    if args.llama_cpp:
        async def sampler(prompt: str, n: int):
            return await llama_cpp_scored_sampler(
                prompt,
                n,
                base_url=args.llama_cpp,
                temperature=config.temperature,
                top_p=config.top_p if config.top_p is not None else 1.0,
                max_tokens=config.max_tokens or 2048,
                stop=tuple(config.stop or (":::", "\n\n")),
            )
    else:
        async def sampler(prompt: str, n: int):
            return await default_tactic_sampler(
                config, client, prompt, n, timeout_seconds=120.0
            )

    episodes_path = args.output / "episodes.jsonl"
    semaphore = asyncio.Semaphore(max(1, args.seed_concurrency))
    write_lock = asyncio.Lock()

    async def run_episode(name: str, arm: str) -> dict:
        row = bench[name]
        header = str(row.get("header") or "import Mathlib")
        palette: Dict[str, Dict[str, str]] = {"tactics": {}, "theorems": {}}
        if arm == "palette":
            palette = await build_palette(
                lean_code=proofs[name],
                formal_statement=str(row["formal_statement"]),
                lean_header=header,
                repo_root=REPO_ROOT,
            )
        async with semaphore:
            env = LeanExamEnv(
                formal_statement=str(row["formal_statement"]),
                lean_header=header,
                palette=palette,
                verifier=verify_lean_proof_repl,
                max_steps=args.max_steps,
                lean_timeout=args.lean_timeout,
                strict_steps=False,  # BFS emits single tactics natively
            )
            player = BFSExamPlayer(
                sampler,
                n_per_step=args.n_per_step,
                resample_rounds=args.resample_rounds,
                max_rollbacks=args.max_rollbacks,
                use_palette=(arm == "palette"),
            )
            result = await player.play(env, max_actions=args.max_actions)
        record = {
            "seed": name,
            "arm": arm,
            "tag": args.tag,
            "n_per_step": args.n_per_step,
            "temperature": config.temperature,
            "sampler": "llama_cpp_logprob" if args.llama_cpp else "lm_studio",
            "palette_theorems": sorted(palette["theorems"]),
            **{k: v for k, v in result.items() if k != "solved_code"},
            "solved_code": result["solved_code"],
        }
        async with write_lock:
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{arm}] {name}: success={result['success']} steps={len(result['steps'])} "
            f"actions={result['actions']} rejected={result['rejected']} "
            f"rollbacks={result['rollbacks']}"
        )
        return record

    episodes_path.write_text("", encoding="utf-8")
    records: List[dict] = []
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    try:
        for name in paired:
            for arm in arms:
                records.append(await run_episode(name, arm))
        for name in extras:
            records.append(await run_episode(name, arms[0]))
    finally:
        await close_global_repl_verifier()
        close = getattr(client, "close", None)
        if close is not None:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe

    def _arm_stats(arm: str, names: Optional[List[str]] = None) -> dict:
        rows = [
            r
            for r in records
            if r["arm"] == arm and (names is None or r["seed"] in names)
        ]
        if not rows:
            return {"episodes": 0}
        return {
            "episodes": len(rows),
            "solved": sum(1 for r in rows if r["success"]),
            "solve_rate": round(
                sum(1 for r in rows if r["success"]) / len(rows), 3
            ),
            "mean_actions": round(
                sum(r["actions"] for r in rows) / len(rows), 1
            ),
            "mean_rejected": round(
                sum(r["rejected"] for r in rows) / len(rows), 1
            ),
            "rollback_episodes": sum(1 for r in rows if r["rollbacks"]),
        }

    summary = {
        "paired_seeds": paired,
        "extra_seeds": extras,
        "config": {
            "n_per_step": args.n_per_step,
            "max_steps": args.max_steps,
            "max_actions": args.max_actions,
            "resample_rounds": args.resample_rounds,
            "max_rollbacks": args.max_rollbacks,
            "model": config.provider_slug,
            "prompt": "live_goal_state",
        },
        "arms": {
            "closed_book_paired": _arm_stats("closed_book", paired),
            "palette_paired": _arm_stats("palette", paired),
            "closed_book_extras": _arm_stats("closed_book", extras),
        },
        "paired_outcomes": {
            name: {
                "closed_book": next(
                    (r["success"] for r in records if r["seed"] == name and r["arm"] == "closed_book"),
                    None,
                ),
                "palette": next(
                    (r["success"] for r in records if r["seed"] == name and r["arm"] == "palette"),
                    None,
                ),
            }
            for name in paired
        },
        "status_counts": dict(
            Counter(f"{r['arm']}:{r['success']}" for r in records)
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["arms"], indent=2))


if __name__ == "__main__":
    asyncio.run(_run())
