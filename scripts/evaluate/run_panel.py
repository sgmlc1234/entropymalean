#!/usr/bin/env python3
"""Run the panel, one group at a time, resuming onto whatever is already there.

Eight models across two arms is sixteen cells, and the last time cells were
launched by hand two of them drifted apart on `max_actions` and the drop that
came out carried the difference inside it. This drives them all from
`config/exam_cells.json` through `run_exam_cell.py`, which refuses a cell whose
budget differs from the control it will be compared against.

Resuming is the normal case, not the exception. `run_seed_exam.py --resume`
keys on `(seed, attempt)`, so a treatment cell built when the corpus was 415
rows plays only the 120 rows added since; a cell interrupted halfway plays only
what is missing. Nothing already measured is replayed, and the only way to make
a cell start over is to delete its episodes.

Groups exist because the panel is read in groups (see `groups` in the config),
and because they fail differently: the Lean provers are local and serialise on
one GPU, the general SLMs and the frontier models are hosted and can run
several at once.

Usage:
  set -a; source .env; set +a

  # what would run, and what is already done
  python3 scripts/evaluate/run_panel.py --plan

  # one group, controls first
  python3 scripts/evaluate/run_panel.py --group lean_provers --arm control
  python3 scripts/evaluate/run_panel.py --group lean_provers --arm treatment

  # the hosted groups, three cells at a time
  python3 scripts/evaluate/run_panel.py --group general_slms --arm both --parallel 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO = _repo_root()
CONFIG = REPO / "config" / "exam_cells.json"

#: Which corpus each arm plays. The control is the certified seeds; the
#: treatment is the admitted release.
ARMS = {"control": "seeds100", "treatment": "release535"}


def load() -> Dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def output_dir(model: str, arm: str, cfg: Dict) -> Path:
    """Where a cell's episodes live.

    The control path is whatever the config already names, because three of
    these cells are finished and their directories predate any convention.
    """
    if arm == "control":
        return REPO / cfg["controls"][model]
    return REPO / cfg["treatments"][model]


def done_count(directory: Path) -> Tuple[int, int]:
    """Episodes on file and how many of them are empty replies.

    Ignores the pre-replay backups. The second number is what a plain count
    hides: an episode that came back empty is on file and counts toward the
    target, so a cell reports `done` while carrying failures that belong to the
    serving stack rather than to the model. Goedel's treatment arm read
    1605/1605 with three ProofNet rows in exactly that state, each one scored
    as the prover failing to prove a theorem it was never shown. They are
    recoverable -- drop_unmeasured_episodes.py plus --resume replays them --
    but only if the count makes them visible in the first place.
    """
    total = empty = 0
    for path in directory.glob("episodes_*.jsonl"):
        if "before-replay" in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                if json.loads(line).get("outcome") == "generator_empty":
                    empty += 1
    return total, empty


def expected(arm: str, cfg: Dict, attempts: int) -> int:
    rows = REPO / cfg["corpora"][ARMS[arm]]["rows"]
    if not rows.is_file():
        return 0
    with rows.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) * attempts


def cell_command(model: str, arm: str, cfg: Dict, args) -> List[str]:
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate" / "run_exam_cell.py"),
           "--model", model, "--corpus", ARMS[arm],
           "--output", str(output_dir(model, arm, cfg)),
           "--attempts", str(args.attempts), "--skip-preflight"]
    budget = cfg["budgets"][model]
    if budget["player"] == "whole_proof":
        cmd += ["--whole-proof-url", budget.get("url") or args.whole_proof_url]
        if budget.get("model"):
            cmd += ["--whole-proof-model", budget["model"]]
    if budget["player"] == "bfs":
        cmd += ["--llama-cpp", args.llama_cpp]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default="all",
                        help="lean_provers | general_slms | frontier_llms | all")
    parser.add_argument("--model", default="", help="one cell, overriding --group")
    parser.add_argument("--arm", default="both", choices=("control", "treatment", "both"))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1,
                        help="hosted cells only; local cells always serialise")
    parser.add_argument("--plan", action="store_true", help="print, do not run")
    parser.add_argument("--llama-cpp", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--whole-proof-url", default="http://127.0.0.1:8081/v1")
    args = parser.parse_args()

    cfg = load()
    if args.model:
        models = [args.model]
    elif args.group == "all":
        models = [m for g in cfg["groups"].values() for m in g]
    else:
        if args.group not in cfg["groups"]:
            raise SystemExit(f"unknown group {args.group!r}; known: {sorted(cfg['groups'])}")
        models = cfg["groups"][args.group]
    arms = ["control", "treatment"] if args.arm == "both" else [args.arm]

    jobs = []
    print(f"{'cell':34s} {'done':>6} {'target':>7}  {'remaining':>9}  {'empty':>6}")
    for model in models:
        for arm in arms:
            directory = output_dir(model, arm, cfg)
            have, empty = done_count(directory) if directory.is_dir() else (0, 0)
            want = expected(arm, cfg, args.attempts)
            left = max(want - have, 0)
            # A cell whose remaining count is zero is not finished while empty
            # replies are still on file; say so rather than printing `done`.
            mark = ("done" if not empty else "replay") if left == 0 and want else f"{left}"
            print(f"  {model + '/' + arm:32s} {have:>6} {want:>7}  {mark:>9}  "
                  f"{(str(empty) if empty else ''):>6}")
            if left:
                jobs.append((model, arm))

    if args.plan or not jobs:
        if not jobs:
            print("\nnothing left to play.")
        return

    # Local cells share one GPU, so they run one at a time whatever --parallel
    # says; a second llama.cpp on the same box swaps and both crawl.
    local = {m for m in models if not str(cfg["budgets"][m].get("url", "")).startswith("http")}
    running: List[subprocess.Popen] = []
    for model, arm in jobs:
        cmd = cell_command(model, arm, cfg, args)
        serial = model in local or args.parallel <= 1
        print("\n$ " + " ".join(cmd), flush=True)
        if serial:
            code = subprocess.run(cmd, cwd=REPO).returncode
            if code != 0:
                raise SystemExit(f"{model}/{arm} exited {code}; nothing after it was started")
        else:
            while len([p for p in running if p.poll() is None]) >= args.parallel:
                for p in running:
                    if p.poll() is None:
                        p.wait()
                        break
            running.append(subprocess.Popen(cmd, cwd=REPO))
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
