"""Pick five miniF2F groups for a ten-generation run, on yield alone.

Topic fit is deliberately absent. It was the basis of the ProofNet allocation
and it has now been measured three times on three sets of crossovers, giving
three different answers: on the first ProofNet pass, no shared vocabulary
certified 7% and a little certified 41%; replayed with the parents' real
statements, the same bands read 40% and 6%; on miniF2F the best band is the
most-overlapping one at 54%. Bins of ~28 that reverse their ordering every time
are noise, and an objective built on them optimises noise.

What survives measurement is per-root yield. Across the run-a miniF2F
groups and both ablation arms, roots range from 72% of attempts certifying down
to 13%, and the ordering is stable in a way the fit bands are not.

So the objective is: put high-yield roots in every group, spread the low-yield
ones out rather than letting them pool, and keep the five groups distinct.
Reuse is allowed -- a seed may seed two groups -- because a strict partition
forces awkward company for the sake of spending every seed, and 50 seeds do not
have to fill 25 slots.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.orchestration.problem_ids import roots_of

RUNS = (
    "data/certified/run-a/minif2f_g*.jsonl",
    "data/certified/ablation/mutation/minif2f_g*.jsonl",
    "data/certified/ablation/crossover/minif2f_g*.jsonl",
)


def yields(patterns=RUNS) -> Dict[str, Tuple[int, int]]:
    attempted, certified = collections.Counter(), collections.Counter()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if ".pre_" in path or ".partial" in path:
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                # A survivor is a carry-over, not an attempt; counting it would
                # reward roots for staying in the pool rather than for producing.
                if row.get("status") == "survivor":
                    continue
                for root in roots_of(row.get("problem_id") or ""):
                    attempted[root] += 1
                    if row.get("status") == "certified":
                        certified[root] += 1
    return {k: (certified[k], attempted[k]) for k in attempted}


def rate(y: Dict[str, Tuple[int, int]], name: str, prior=3.0, base=0.40) -> float:
    """Certification rate pulled toward the corpus mean.

    Raw rates are unusable at these counts -- a root seen twice and certified
    twice reads 1.00 and outranks the best-measured root in the corpus. The
    prior is worth three attempts at the observed miniF2F rate.
    """
    certified, attempted = y.get(name, (0, 0))
    return (certified + prior * base) / (attempted + prior)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/certified/run-a/seeds/minif2f_g*.csv")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--reuse-cap", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("data/certified/run-c/seeds"))
    parser.add_argument("--plan", type=Path, default=Path("data/certified/run-c/seed_plan.json"))
    args = parser.parse_args()

    rows: Dict[str, dict] = {}
    fields: List[str] = []
    for path in sorted(glob.glob(args.seeds)):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or fields
            for row in reader:
                if row.get("id"):
                    rows[row["id"]] = row

    y = yields()
    names = sorted(rows)
    score = {n: rate(y, n) for n in names}
    ranked = sorted(names, key=lambda n: -score[n])
    slots = args.groups * args.size

    # Greedy, then a local pass. With reuse capped at 2 the best 25 slots are
    # simply the top ranks taken twice each, so the interesting part is only how
    # they are distributed across groups: every group should get a share of the
    # strongest roots rather than one group taking them all.
    pool: List[str] = []
    for n in ranked:
        pool.extend([n] * min(args.reuse_cap, max(1, slots // len(names) + 1)))
        if len(pool) >= slots:
            break
    pool = pool[:slots]

    # Deal round-robin down the ranking so each group's mean yield is close to
    # every other's, and no group is left holding only the weak end.
    groups: List[List[str]] = [[] for _ in range(args.groups)]
    for index, name in enumerate(pool):
        target = index % args.groups
        # A seed must not appear twice inside one group.
        for offset in range(args.groups):
            candidate = (target + offset) % args.groups
            if name not in groups[candidate] and len(groups[candidate]) < args.size:
                groups[candidate].append(name)
                break

    order = [f"minif2f_h{i:02d}" for i in range(1, args.groups + 1)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    used = collections.Counter(n for g in groups for n in g)
    print(f"{len(rows)} seeds available · {len(used)} placed · max reuse {max(used.values())}")
    print()
    for name, group in zip(order, groups):
        mean = sum(score[s] for s in group) / len(group)
        print(f"{name}  mean yield {mean:.3f}")
        for s in group:
            c, a = y.get(s, (0, 0))
            print(f"    {s[:52]:52s} {c:3d}/{a:<3d} {100*c//max(a,1):3d}%")
        target = args.out_dir / f"{name}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for s in group:
                writer.writerow(rows[s])

    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps({
        "basis": "per-root certification yield, smoothed; topic fit excluded as unreplicated",
        "groups": dict(zip(order, groups)),
        "yield": {n: list(y.get(n, (0, 0))) for n in used},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {args.out_dir}/  and  {args.plan}")


if __name__ == "__main__":
    main()
