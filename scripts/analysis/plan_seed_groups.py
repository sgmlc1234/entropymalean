"""Redistribute the 50 seeds into 10 groups, using what the corpus measured.

A group is the mating pool: crossover picks both parents from inside it, so the
grouping decides which pairs ever get tried. The current allocation was made
before there was anything to base one on, and it leaves a third of the possible
pairs with no shared mathematics at all.

Three measurements drive the objective, and each is stated with what it is
worth rather than as a rule of thumb:

  topic fit   Over the 116 crossovers this corpus produced, parents sharing no
              vocabulary certified 7% of the time; parents sharing a little
              certified 41%. More overlap is not better -- the top band falls
              back to 24%, which is `recall` territory. Pair value is read
              straight off those bins instead of from a threshold someone
              picked.

  yield       Roots split three ways by how often their descendants certify,
              from 55% down to 18%. A group of only low-yield roots produces
              shallow survivors, so each group is asked to carry at least two
              from the top tier.

  where they die
              The low tier fails before the judge -- statement and proof
              failures at three times the top tier's rate. Those roots are not
              rescued by better company, so the objective does not try; it only
              avoids concentrating them.

Local search with pair swaps. The space is 50 seeds into 10 labelled groups of
5, which is far too large to enumerate and much too easy for hill climbing to
get stuck in, so this restarts many times and keeps the best.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from src.orchestration.problem_ids import roots_of
from scripts.analysis.measure_topic_fit import overlap, signature

#: Certification rate observed in each topic-fit band, from measure_topic_fit
#: over all 116 crossovers. Used as the value of putting two seeds together.
BANDS: Sequence[Tuple[float, float]] = ((0.0, 0.07), (0.08, 0.41), (0.17, 0.21), (1.01, 0.24))


def pair_value(fit: float) -> float:
    for upper, value in BANDS:
        if fit <= upper:
            return value
    return BANDS[-1][1]


def yields(pattern: str) -> Dict[str, Tuple[int, int]]:
    """Certified and attempted descendants, per root."""
    attempted: Dict[str, int] = collections.Counter()
    certified: Dict[str, int] = collections.Counter()
    for path in sorted(glob.glob(pattern)):
        if ".pre_" in path:
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "survivor":
                continue
            for root in roots_of(row.get("problem_id") or ""):
                attempted[root] += 1
                if row.get("status") == "certified":
                    certified[root] += 1
    return {k: (certified[k], attempted[k]) for k in attempted}


def score(groups: List[List[str]], sig: Dict[str, Set[str]], top: Set[str],
          bottom: Set[str], *, need_top: int = 1, cap_bottom: int = 2) -> float:
    """Topic fit, less what an unbalanced group costs.

    `need_top` was 2 and could not be met: the top tier holds 14 roots and ten
    groups would need 20, so the penalty applied uniformly to every arrangement
    and dropped out of the comparison entirely. One per group is satisfiable
    with four to spare, which is what makes it a constraint rather than a
    constant.

    The bottom-tier cap is the other half. Left to topic fit alone the search
    puts low-yield roots together -- they share vocabulary with each other as
    readily as anything else -- and those roots fail before the judge, so such
    a group never gets deep enough to matter. Both penalties are priced in the
    same units as pair value, so trading one against the other is explicit.
    """
    total = 0.0
    for group in groups:
        for index, a in enumerate(group):
            for b in group[index + 1:]:
                total += pair_value(overlap(sig[a], sig[b]))
        total -= 1.5 * max(0, need_top - sum(1 for s in group if s in top))
        total -= 1.0 * max(0, sum(1 for s in group if s in bottom) - cap_bottom)
    return total


def report(groups: List[List[str]], sig, yielded, top, bottom, label: str) -> dict:
    zero = band = 0
    pairs = 0
    for group in groups:
        for index, a in enumerate(group):
            for b in group[index + 1:]:
                fit = overlap(sig[a], sig[b])
                pairs += 1
                zero += fit == 0
                band += 0 < fit <= 0.17
    thin = sum(1 for g in groups if sum(1 for s in g if s in top) < 1)
    heavy = sum(1 for g in groups if sum(1 for s in g if s in bottom) > 2)
    expected = sum(pair_value(overlap(sig[a], sig[b]))
                   for g in groups for i, a in enumerate(g) for b in g[i + 1:])
    print(f"\n{label}")
    print(f"  pairs with no shared mathematics : {zero:3d} / {pairs}  ({100*zero/pairs:.0f}%)")
    print(f"  pairs in the productive band     : {band:3d} / {pairs}  ({100*band/pairs:.0f}%)")
    print(f"  groups with no top-tier root            : {thin}")
    print(f"  groups carrying 3+ bottom-tier roots    : {heavy}")
    print(f"  expected crossover yield (sum of pair rates) : {expected:.1f}")
    return {"zero": zero, "band": band, "thin": thin, "heavy": heavy, "expected": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/certified/run-a/seeds/proofnet_g*.csv")
    parser.add_argument("--runs", default="data/certified/run-a/proofnet_g*.jsonl")
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("data/certified/run-b/seed_plan.json"))
    args = parser.parse_args()

    current: List[List[str]] = []
    order: List[str] = []
    statements: Dict[str, str] = {}
    for path in sorted(glob.glob(args.seeds)):
        group = []
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("id"):
                    group.append(row["id"])
                    statements[row["id"]] = row.get("formal_statement") or ""
        current.append(group)
        order.append(Path(path).stem)

    sig = {k: signature(v) for k, v in statements.items()}
    y = yields(args.runs)
    eligible = sorted((k for k in sig if y.get(k, (0, 0))[1] >= 5),
                      key=lambda k: y[k][0] / y[k][1])
    top = set(eligible[2 * len(eligible) // 3:])
    bottom = set(eligible[:len(eligible) // 3]) | {k for k in sig if y.get(k, (0, 0))[1] < 5 and y.get(k, (0, 0))[0] == 0}
    print(f"{len(sig)} seeds · top tier {len(top)} · bottom tier {len(bottom)}")

    before = report(current, sig, y, top, bottom, "current allocation")

    flat = [s for g in current for s in g]
    best, best_score = None, float("-inf")
    rng = random.Random(20260812)
    for _ in range(args.restarts):
        shuffled = flat[:]
        rng.shuffle(shuffled)
        groups = [shuffled[i:i + 5] for i in range(0, len(shuffled), 5)]
        value = score(groups, sig, top, bottom)
        for _ in range(args.steps):
            g1, g2 = rng.randrange(len(groups)), rng.randrange(len(groups))
            if g1 == g2:
                continue
            i, j = rng.randrange(5), rng.randrange(5)
            groups[g1][i], groups[g2][j] = groups[g2][j], groups[g1][i]
            new = score(groups, sig, top, bottom)
            if new >= value:
                value = new
            else:
                groups[g1][i], groups[g2][j] = groups[g2][j], groups[g1][i]
        if value > best_score:
            best, best_score = [g[:] for g in groups], value

    after = report(best, sig, y, top, bottom, "optimised allocation")

    print("\nproposed groups")
    for name, group in zip(order, best):
        marks = "".join("+" if s in top else ("-" if s in bottom else ".") for s in group)
        fits = [overlap(sig[a], sig[b]) for i, a in enumerate(group) for b in group[i + 1:]]
        print(f"  {name}  [{marks}]  fit min {min(fits):.2f} mean {sum(fits)/len(fits):.2f}")
        for s in group:
            c, a = y.get(s, (0, 0))
            print(f"      {s[:44]:44s} {c:2d}/{a:2d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"groups": dict(zip(order, best)), "before": before, "after": after},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- selection mode
#
# The partition above spends all 50 seeds exactly once, which forces awkward
# company: one group ended up mixing number theory with group theory at a mean
# fit of 0.04 because someone had to take those seeds. Selecting instead of
# partitioning lets a productive seed appear in several pools and lets a seed
# that never produced anything sit out.
#
# Two limits keep that from collapsing. Without a reuse cap the best answer is
# the single strongest five repeated ten times, which is one experiment run ten
# times over; without an overlap penalty the ten groups drift toward each other
# and the corpus-wide dedup then throws away the repeats after paying for them.

def smoothed_yield(y, name, prior=3.0, rate=0.30):
    """A seed's certification rate, pulled toward the corpus average.

    Raw rates are unusable at these counts: a root seen once and certified once
    reads 1.00 and outranks the best-measured root in the corpus at 14/22. The
    prior is worth three attempts at the observed corpus rate, so a single
    lucky or unlucky draw moves the estimate a little and twenty draws move it
    a lot.
    """
    certified, attempted = y.get(name, (0, 0))
    return (certified + prior * rate) / (attempted + prior)


def select(sig, top, bottom, y, *, groups=10, size=5, reuse_cap=2,
           overlap_penalty=0.6, yield_weight=1.0,
           restarts=40, steps=20000, seed=20260812):
    import random
    names = sorted(sig)
    rng = random.Random(seed)
    rate = {n: smoothed_yield(y, n) for n in names}

    def group_value(group):
        total = sum(pair_value(overlap(sig[a], sig[b]))
                    for i, a in enumerate(group) for b in group[i + 1:])
        # Yield enters per seed and continuously. As a threshold ("at least one
        # top-tier root") it only had to be cleared once, after which the other
        # four seats went to whatever shared vocabulary -- which put three roots
        # that have never certified anything into two groups each while leaving
        # 7/12 and 6/9 roots unused entirely.
        total += yield_weight * sum(rate[s] for s in group)
        total -= 1.5 * max(0, 1 - sum(1 for s in group if s in top))
        total -= 1.0 * max(0, sum(1 for s in group if s in bottom) - 2)
        return total

    def total_value(chosen):
        value = sum(group_value(g) for g in chosen)
        counts = collections.Counter(s for g in chosen for s in g)
        value -= 100.0 * sum(max(0, c - reuse_cap) for c in counts.values())
        for i, a in enumerate(chosen):
            for b in chosen[i + 1:]:
                shared = len(set(a) & set(b))
                value -= overlap_penalty * max(0, shared - 1)
        return value

    best, best_score = None, float("-inf")
    for _ in range(restarts):
        chosen = [rng.sample(names, size) for _ in range(groups)]
        value = total_value(chosen)
        for _ in range(steps):
            g = rng.randrange(groups)
            i = rng.randrange(size)
            old = chosen[g][i]
            new = rng.choice(names)
            if new in chosen[g]:
                continue
            chosen[g][i] = new
            fresh = total_value(chosen)
            if fresh >= value:
                value = fresh
            else:
                chosen[g][i] = old
        if value > best_score:
            best, best_score = [g[:] for g in chosen], value
    return best
