#!/usr/bin/env python3
"""What the revision attempts after the first one are worth, per model.

An episode gives a whole-proof model up to four attempts. The first is a cold
try; the other three are revisions, each handed Lean's own diagnostics on the
proof that just failed. The question this answers is what that loop recovers,
and the denominator is the part that is easy to get wrong: asking which attempt
closed a *solved* episode says almost nothing, because a row a model can prove
is usually proved immediately and a row it cannot is not proved at four
attempts either. The loop's yield is over the episodes whose first attempt did
not close: of those, how many did a later attempt rescue.

The tactic-step cell is excluded. Its episodes record steps rather than
attempts -- one action is a tactic, not a proof -- so there is no revision loop
to price.

    python3 scripts/evaluate/revision_yield.py            # per model and total
    python3 scripts/evaluate/revision_yield.py --latex    # the table body

Reads the shipped episode bundle; needs no model and no Lean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.exam_records import cell_episodes  # noqa: E402

LABEL = {
    "bfs": "BFS-V2-7B", "goedel": "Goedel-V2-8B", "pythagoras": "Pythagoras-4B",
    "leanstral": "Leanstral-1.5", "muse": "Muse-Glimmer-30B", "qwen3_14b": "Qwen3-14B",
    "nemotron_nano_9b": "Nemotron-nano-9B", "nemotron": "Nemotron-3-nano",
    "qwen36": "Qwen3.6-35B-A3B", "gptoss": "gpt-oss-20b", "luna": "gpt-5.6-luna",
    "muse_spark": "Muse-Spark-1.2", "gemini_flash": "gemini-3.7-flash",
}


def yield_for(model: str, cfg: dict) -> tuple[int, int, int]:
    """(episodes with an attempt count, first attempt failed, rescued later)."""
    seen = failed_first = rescued = 0
    for arm, registry in (("control", "controls"), ("treatment", "treatments")):
        for row in cell_episodes(cfg[registry].get(model), model, arm):
            attempts = row.get("attempts")
            if not attempts:            # tactic-step player: records steps
                continue
            seen += 1
            if row.get("success") and int(attempts) == 1:
                continue
            failed_first += 1
            if row.get("success"):
                rescued += 1
    return seen, failed_first, rescued


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config/exam_cells.json"))
    ap.add_argument("--latex", action="store_true", help="emit the table body rows")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    order = [m for g in ("lean_provers", "reasoning_slms", "frontier_llms")
             for m in (cfg.get("groups") or {}).get(g, [])]

    rows, tot_first, tot_rescued = [], 0, 0
    for model in order:
        seen, first, rescued = yield_for(model, cfg)
        if not seen:
            print(f"  {LABEL.get(model, model):20s} no revision loop (tactic-step cell)")
            continue
        rows.append((model, seen, first, rescued))
        tot_first += first
        tot_rescued += rescued

    print(f"\n  {'model':20s}{'episodes':>9}{'a1 failed':>11}{'rescued':>9}{'yield':>9}")
    for model, seen, first, rescued in rows:
        print(f"  {LABEL.get(model, model):20s}{seen:9d}{first:11d}{rescued:9d}"
              f"{100 * rescued / first:8.1f}%")
    print(f"\n  {len(rows)} models that revise: {tot_rescued} of {tot_first} episodes whose "
          f"first attempt did not close were rescued, {100 * tot_rescued / tot_first:.1f}%.")

    if args.latex:
        print("\n% --- revision yield rows ---")
        for model, _, first, rescued in rows:
            print(f"    {LABEL.get(model, model)} & {first} & {rescued} & "
                  f"{100 * rescued / first:.1f} \\\\")


if __name__ == "__main__":
    main()
