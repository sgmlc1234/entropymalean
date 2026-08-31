"""Count what the planner actually chooses, by calling it and reading the plan.

The prompt described six operator variants and the planner used four. Reading
the prompt did not explain it; printing the prompt did. One line enumerated the
legal variants and omitted `mutation_silent`, a second told the planner to spend
the whole crossover budget on `crossover_easy` for theorem-only pools, and a
third offered `pipeline_composite` as the escape whenever fusion looked hard.
Across 1,281 slots the planner chose `mutation_silent` zero times and
`crossover_hard` zero times; the handful that exist came from the replan ladder.

So the measurement that matters is not whether the prompt mentions a variant. It
is what comes back. This calls the planner on a real pool and tallies
`operator_variant` over the returned work items.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import List

from src.orchestration.pool_generation import CertificationInput, llm_plan_generation


def load_pool(path: Path, size: int) -> List[CertificationInput]:
    with path.open(encoding="utf-8") as handle:
        seeds = list(csv.DictReader(handle))[:size]
    return [
        CertificationInput(
            id=str(seed.get("id") or seed.get("problem_id")),
            statement=str(seed.get("statement") or seed.get("informal_statement") or ""),
            answer=str(seed.get("answer") or ""),
            metadata={
                "formal_statement": str(seed.get("formal_statement") or ""),
                "lean_code": str(seed.get("lean_code") or seed.get("solution") or ""),
                "lean_header": str(seed.get("lean_header") or ""),
                "problem_style": "theorem_proof",
            },
        )
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=Path("data/certified/run-a/seeds/minif2f_g01.csv"))
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--survivor-count", type=int, default=1)
    parser.add_argument("--crossover-count", type=int, default=2)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/release/planner_variants.json"))
    args = parser.parse_args()

    pool = load_pool(args.seeds, args.pool_size)
    print(f"pool {len(pool)} from {args.seeds.name} · {args.trials} trials")

    plans, variants, mechanisms = [], collections.Counter(), collections.Counter()
    for trial in range(1, args.trials + 1):
        try:
            plan = llm_plan_generation(
                pool,
                pool_size=args.pool_size,
                survivor_count=args.survivor_count,
                crossover_count=args.crossover_count,
                generation_model=args.model,
                generation_temperature=None,
            )
        except Exception as error:
            print(f"  trial {trial}: planner failed — {error}")
            continue
        items = [dict(item) for item in (plan.get("work_items") or [])] if isinstance(plan, dict) else [
            dict(getattr(item, "__dict__", {})) for item in (getattr(plan, "work_items", []) or [])
        ]
        chosen = [str(item.get("operator_variant") or "") for item in items]
        variants.update(chosen)
        mechanisms.update(
            str(item.get("fusion_mechanism") or "") for item in items
            if str(item.get("op_type") or "").startswith("crossover")
        )
        plans.append({"trial": trial, "variants": chosen,
                      "source": str(plan.get("planner_source") or "") if isinstance(plan, dict) else ""})
        print(f"  trial {trial}: {', '.join(chosen)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"trials": plans, "variants": dict(variants), "fusion_mechanisms": dict(mechanisms)},
        ensure_ascii=False, indent=1), encoding="utf-8")

    total = sum(variants.values())
    print(f"\n{total} slots planned across {len(plans)} successful trials")
    for name in ("survivor", "mutation_easy", "mutation_hard", "mutation_silent",
                 "crossover_easy", "crossover_hard"):
        count = variants.get(name, 0)
        bar = "#" * count
        print(f"  {name:16s} {count:3d}  {bar}")
    other = {k: v for k, v in variants.items() if k not in {
        "survivor", "mutation_easy", "mutation_hard", "mutation_silent",
        "crossover_easy", "crossover_hard"}}
    if other:
        print(f"  unexpected: {other}")
    if mechanisms:
        print(f"\nfusion mechanisms chosen: {dict(mechanisms)}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
