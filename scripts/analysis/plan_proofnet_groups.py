"""Ten fresh ProofNet groups for a ten-generation run.

Same basis as the miniF2F planner: per-root certification yield, measured over
every ProofNet campaign so far, and nothing else. Topic fit is excluded --
measured three times on three sets of crossovers, it gave three different
orderings, and an objective built on bins that reverse every time optimises
noise.

Yield here means **two-pass admission**, not certification. Those rank the
seeds differently and the difference is large: `Munkres_exercise_13_1`
certifies 63% of its attempts and only 20% survive two independent re-judge
passes; `Axler_exercise_5_20` certifies 42% and admits 5%. Selecting on
certification puts those at the top. The k campaign made the same error
visible at group scale -- k05 certified the most rows of any group and admitted
the fewest.

Seeds are selected, not partitioned. Reuse is capped, low-admission seeds are
dropped rather than spread, and no two seeds that already shared a group may
share one again: 186 pairs have been run across the twenty groups of the gen5
and gen10 campaigns, and repeating them would be new labels on the same
matings.

Yield is counted for seeds only. A first pass counted every root id, which
includes generated descendants -- `Rudin_exercise_4_12__mh__fcde3616` appeared
as a root with 1/8 -- and those are not seeds and cannot be placed in a group.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.orchestration.problem_ids import roots_of

RUNS = ("data/certified/run-a/proofnet_g*.jsonl",
        "data/certified/run-b/proofnet_g*.jsonl")
SEEDS = ("data/certified/run-a/seeds/proofnet_g*.csv",
         "data/certified/run-b/seeds/proofnet_g*.csv")


def load_seeds() -> Tuple[Dict[str, dict], List[str]]:
    rows: Dict[str, dict] = {}
    fields: List[str] = []
    for pattern in SEEDS:
        for path in sorted(glob.glob(pattern)):
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or fields
                for row in reader:
                    if row.get("id"):
                        rows[row["id"]] = row
    return rows, fields


def _admitted() -> Set[str]:
    """Rows that both re-judge passes rated strong and voted to keep."""
    r1 = {r["problem_id"]: r for r in json.loads(
        Path("data/release/rejudged.json").read_text(encoding="utf-8"))}
    r2 = {r["problem_id"]: r for r in json.loads(
        Path("data/release/rejudged_run2.json").read_text(encoding="utf-8"))}
    out = set()
    for pid, a in r1.items():
        b = r2.get(pid)
        if (b and a.get("new_quality") == "strong" and b.get("new_quality") == "strong"
                and a.get("new_verdict") == "keep" and b.get("new_verdict") == "keep"):
            out.add(pid)
    return out


def yields(seed_ids: Set[str]) -> Dict[str, Tuple[int, int]]:
    """(admitted, attempted) per seed. Certification is the wrong numerator --
    see the module docstring."""
    admitted = _admitted()
    attempted, certified = collections.Counter(), collections.Counter()
    for pattern in RUNS:
        for path in glob.glob(pattern):
            if ".pre_" in path or "partial" in path:
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "survivor":
                    continue
                for root in roots_of(row.get("problem_id") or ""):
                    if root not in seed_ids:
                        continue
                    attempted[root] += 1
                    if row.get("status") == "certified" and row["problem_id"] in admitted:
                        certified[root] += 1
    return {k: (certified[k], attempted[k]) for k in seed_ids}


def rate(y, name, prior=4.0, base=0.18) -> float:
    """Smoothed toward the corpus rate; a root seen twice must not outrank one
    measured thirty times."""
    c, a = y.get(name, (0, 0))
    return (c + prior * base) / (a + prior)


def used_pairs() -> Set[frozenset]:
    pairs = set()
    for pattern in SEEDS:
        for path in glob.glob(pattern):
            g = [r["id"] for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r.get("id")]
            for i, a in enumerate(g):
                for b in g[i + 1:]:
                    pairs.add(frozenset((a, b)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("data/certified/run-e/seeds"))
    parser.add_argument("--plan", type=Path, default=Path("data/certified/run-e/seed_plan.json"))
    parser.add_argument("--floor", type=float, default=0.18,
                        help="minimum smoothed admission rate for a seed to be used at all")
    parser.add_argument("--pool", type=int, default=30,
                        help="how many seeds to draw the groups from")
    parser.add_argument("--reuse-cap", type=int, default=2)
    args = parser.parse_args()

    rows, fields = load_seeds()
    names = sorted(rows)
    y = yields(set(names))
    score = {n: rate(y, n) for n in names}
    tried = used_pairs()
    print(f"{len(names)} seeds · {len(tried)} pairs already run")

    ranked = sorted(names, key=lambda n: -score[n])
    third = max(1, len(ranked) // 3)
    top, low = set(ranked[:third]), set(ranked[-third:])

    def cost(groups):
        c = 0.0
        for g in groups:
            for i, a in enumerate(g):
                for b in g[i + 1:]:
                    if frozenset((a, b)) in tried:
                        c += 5.0
            c += 2.0 * max(0, 2 - sum(1 for s in g if s in top))
            c += 1.0 * max(0, sum(1 for s in g if s in low) - 2)
            c += 4.0 * (len(g) - len(set(g)))          # a seed twice in one group
        used = collections.Counter(s for gg in groups for s in gg)
        c += 6.0 * sum(max(0, n - args.reuse_cap) for n in used.values())
        return c

    # Selection, not partition. A seed whose descendants have never survived two
    # judge passes is not improved by better company -- the ProofNet measurement
    # showed those die at statement and proof failure, before any judge sees
    # them -- so they are left out rather than spread around.
    keep = [n for n in ranked if score[n] >= args.floor][:args.pool]
    dropped = [n for n in ranked if n not in set(keep)]
    print(f"selected {len(keep)} of {len(names)} seeds "
          f"(admission floor {args.floor:.2f}); dropped {len(dropped)}")
    if dropped:
        print("  dropped: " + ", ".join(d.split('_exercise')[0][:12] + '..' for d in dropped[:8])
              + (" …" if len(dropped) > 8 else ""))

    rng = random.Random(20260818)
    best, best_cost = None, float("inf")
    slots = args.groups * args.size
    for _ in range(60):
        pool = (keep * (slots // len(keep) + 1))[:slots]
        rng.shuffle(pool)
        groups = [pool[i:i + args.size] for i in range(0, slots, args.size)]
        c = cost(groups)
        for _ in range(40000):
            g1, g2 = rng.randrange(args.groups), rng.randrange(args.groups)
            if g1 == g2:
                continue
            i, j = rng.randrange(args.size), rng.randrange(args.size)
            groups[g1][i], groups[g2][j] = groups[g2][j], groups[g1][i]
            n = cost(groups)
            if n <= c:
                c = n
            else:
                groups[g1][i], groups[g2][j] = groups[g2][j], groups[g1][i]
        if c < best_cost:
            best, best_cost = [g[:] for g in groups], c
        if best_cost == 0:
            break

    repeats = sum(1 for g in best for i, a in enumerate(g) for b in g[i + 1:]
                  if frozenset((a, b)) in tried)
    thin = sum(1 for g in best if sum(1 for s in g if s in top) < 2)
    heavy = sum(1 for g in best if sum(1 for s in g if s in low) > 2)
    placed = {s for g in best for s in g}
    print(f"cost {best_cost:.1f} · repeated pairs {repeats}/{args.groups * 10} · "
          f"groups short of 2 top roots {thin} · groups with 3+ low {heavy}")
    print(f"seeds placed {len(placed)}/{len(names)}")

    order = [f"proofnet_p{i:02d}" for i in range(1, args.groups + 1)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in zip(order, best):
        m = sum(score[s] for s in g) / len(g)
        marks = "".join("+" if s in top else ("-" if s in low else ".") for s in g)
        print(f"\n{name}  yield {m:.3f}  [{marks}]")
        for s in g:
            c_, a_ = y.get(s, (0, 0))
            print(f"   {s[:46]:46s} {c_:3d}/{a_:<3d}")
        with (args.out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            for s in g:
                w.writerow(rows[s])
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(
        {"basis": "per-root yield over all ProofNet campaigns; no pair repeated from the 20 groups already run",
         "groups": dict(zip(order, best)),
         "yield": {n: list(y.get(n, (0, 0))) for n in placed}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {args.out_dir}/")


if __name__ == "__main__":
    main()
