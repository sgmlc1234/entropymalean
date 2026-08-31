#!/usr/bin/env python3
"""Replay recorded exam episodes and export their goal states for the demo.

Episode logs record which tactics were accepted, not what the goal looked like
in between. This replays the accepted steps through the real environment and
captures the observation before each one, so the published demo shows states
Lean actually produced rather than states we wrote by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.exam_env.bfs_player import first_goal  # noqa: E402
from src.exam_env.environment import LeanExamEnv  # noqa: E402
from src.exam_env.palette import build_palette  # noqa: E402
from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

MINIF2F = REPO_ROOT / "data" / "benchmarks" / "minif2f" / "minif2f.jsonl"


def load_bench() -> Dict[str, dict]:
    rows = {}
    for line in MINIF2F.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["formal_statement"] = _prelint_lean_syntax(row["formal_statement"])
            rows[str(row["name"])] = row
    return rows


def load_episodes(seeds: List[str]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for path in glob.glob(str(REPO_ROOT / "tmp/exam_*/episodes.jsonl")):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episode = json.loads(line)
            if not episode.get("success") or not episode.get("steps"):
                continue
            name = episode["seed"]
            if seeds and name not in seeds:
                continue
            episode["_src"] = Path(path).parent.name
            # Prefer the episode with the most steps: more states to show.
            if name not in best or len(episode["steps"]) > len(best[name]["steps"]):
                best[name] = episode
    return best


def harvest_proof(seed: str) -> str:
    import csv

    for path in glob.glob(str(REPO_ROOT / "data/certified/minif2f*gen0_seeds.csv")):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            if str(row.get("id")) == seed:
                code = (row.get("lean_code") or "").strip()
                if code and "sorry" not in code:
                    return code
    return ""


async def replay(seed: str, episode: dict, bench: Dict[str, dict]) -> Dict[str, Any]:
    row = bench[seed]
    header = str(row.get("header") or "import Mathlib")
    env = LeanExamEnv(
        formal_statement=row["formal_statement"],
        lean_header=header,
        verifier=verify_lean_proof_repl,
        max_steps=40,
        lean_timeout=240.0,
    )
    observation = await env.reset()
    states: List[Dict[str, Any]] = []
    for tactic in episode["steps"]:
        states.append(
            {
                "goals": list(observation.goals),
                "prompted": first_goal(observation.goals),
                "accepts": tactic,
            }
        )
        observation = await env.step({"type": "tactic", "tactic": tactic})
        if observation.status not in {"accepted", "solved"}:
            states[-1]["replay_note"] = f"diverged: {observation.message[:120]}"
            break
    solved = env.success
    palette: Dict[str, Dict[str, str]] = {"theorems": {}, "tactics": {}}
    proof = harvest_proof(seed)
    if proof:
        palette = await build_palette(
            lean_code=proof,
            formal_statement=row["formal_statement"],
            lean_header=header,
            repo_root=REPO_ROOT,
        )
    print(
        f"{seed}: replayed {len(states)}/{len(episode['steps'])} steps "
        f"solved={solved} palette={len(palette['theorems'])}"
    )
    return {
        "seed": seed,
        "source_run": episode["_src"],
        "statement": row["formal_statement"].strip(),
        "informal": str(row.get("informal_prefix") or "").strip(),
        "states": states,
        "solved": solved,
        "recorded": {
            "actions": episode["actions"],
            "rejected": episode["rejected"],
            "rollbacks": episode["rollbacks"],
            "steps": len(episode["steps"]),
        },
        "palette": {
            "theorems": palette["theorems"],
            "tactics": palette["tactics"],
        },
    }


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", default=[])
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "tmp/exam_demo_episodes.json"
    )
    args = parser.parse_args()

    bench = load_bench()
    episodes = load_episodes(args.seeds)
    order = args.seeds or sorted(episodes)
    out = []
    try:
        for seed in order:
            if seed in episodes and seed in bench:
                out.append(await replay(seed, episodes[seed], bench))
    finally:
        await close_global_repl_verifier()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(out)} episodes -> {args.output}")


if __name__ == "__main__":
    asyncio.run(_run())
