#!/usr/bin/env python3
"""Smoke-run the Lean exam environment on certified rows.

For each input row (needs formal_statement + lean_code GT proof):
  1. build the palette from the certified proof (validated via #check);
  2. let an LLM agent play the game-style loop (tactic / inspect / rollback);
  3. record the episode transcript.

Usage:
  python scripts/archive/run_exam_smoke.py --input tmp/smoke/a2_input.jsonl \
    --output tmp/smoke/exam_episodes.jsonl --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.generation import default_generation_config  # noqa: E402
from src.exam_env.agent import ChatExamAgent, run_exam_episode  # noqa: E402
from src.exam_env.environment import LeanExamEnv  # noqa: E402
from src.exam_env.palette import build_palette  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lean-timeout", type=float, default=300.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="one tactic per action: reject ';' chained mega-tactics",
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
    rows = [
        row
        for row in rows
        if (row.get("formal_statement") or "").strip()
        and (row.get("lean_code") or "").strip()
    ][: max(1, args.limit)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            header = str(row.get("lean_header") or "import Mathlib")
            palette = await build_palette(
                lean_code=str(row.get("lean_code") or ""),
                formal_statement=str(row.get("formal_statement") or ""),
                lean_header=header,
                repo_root=REPO_ROOT,
            )
            env = LeanExamEnv(
                formal_statement=str(row.get("formal_statement") or ""),
                lean_header=header,
                palette=palette,
                max_steps=args.max_steps,
                lean_timeout=args.lean_timeout,
                strict_steps=args.strict,
            )
            episode = await run_exam_episode(
                env, ChatExamAgent(config), max_actions=args.max_actions
            )
            record = {
                "problem_id": row.get("problem_id"),
                "success": episode["success"],
                "actions": episode["actions"],
                "steps": episode["steps"],
                "palette_tactics": sorted(palette["tactics"]),
                "palette_theorems": sorted(palette["theorems"]),
                "solved_code": episode["solved_code"],
                "transcript": episode["transcript"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{row.get('problem_id')}: success={episode['success']} "
                f"actions={episode['actions']} steps={len(episode['steps'])} "
                f"palette={len(palette['theorems'])} theorems"
            )


if __name__ == "__main__":
    asyncio.run(_run())
