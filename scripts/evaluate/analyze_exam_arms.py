#!/usr/bin/env python3
"""Compare two exam arms on the seeds they both played.

Every seed plays every arm, so the comparison is paired and the tests have to
be too: an unpaired proportion test would throw away the pairing and lose most
of the power that running both arms on the same 50 seeds bought.

Two statistics, in the order they should be read.

`rejected` first. It is the count of candidates Lean threw out before the
episode ended, and it moves even when the verdict does not — a seed can go from
35 rejections to 0 and from 12 to 9 without either crossing the pass boundary.
Compared with a Wilcoxon signed-rank test over per-seed medians.

Pass@k second, by McNemar's exact test on the discordant pairs. It is the
number a reader wants, but with 50 seeds it only sees a handful of flips, so it
will usually be the weaker evidence even when the effect is real. Reported with
the discordant counts visible so its precision is not overstated.

Usage:
  python scripts/evaluate/analyze_exam_arms.py --dir data/evaluation/exam/stage3 \
    --arms official_parity palette
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    plays: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            plays[record["seed"]].append(record)
    return plays


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(diffs: List[float]) -> Dict[str, Any]:
    """Signed-rank statistic with an exact two-sided p for small n.

    Exact rather than normal-approximated: with a few dozen non-zero pairs the
    approximation is not obviously safe, and the exact enumeration is cheap
    below ~20 pairs. Above that it falls back and says so.
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "p": 1.0, "method": "no non-zero pairs"}
    ranks = {}
    for rank, (_, index) in enumerate(
        sorted(((abs(d), i) for i, d in enumerate(nonzero))), start=1
    ):
        ranks[index] = rank
    w_plus = sum(ranks[i] for i, d in enumerate(nonzero) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(nonzero) if d < 0)
    stat = min(w_plus, w_minus)
    if n <= 20:
        total = 0
        count = 0
        for mask in range(1 << n):
            s = sum(ranks[i] for i in range(n) if mask >> i & 1)
            total += 1
            if s <= stat:
                count += 1
        return {
            "n": n, "W+": w_plus, "W-": w_minus,
            "p": min(1.0, 2 * count / total), "method": "exact",
        }
    mean = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (stat - mean) / sd if sd else 0.0
    # survival of |z| under the normal, without scipy
    from math import erfc
    return {
        "n": n, "W+": w_plus, "W-": w_minus,
        "p": min(1.0, erfc(abs(z) / 2 ** 0.5)), "method": "normal approximation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--arms", nargs=2, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    a, b_arm = args.arms
    plays = {arm: load(args.dir / f"episodes_{arm}.jsonl") for arm in (a, b_arm)}
    shared = sorted(set(plays[a]) & set(plays[b_arm]))
    if not shared:
        raise SystemExit("the two arms share no seeds")

    solved = {
        arm: {s: any(p["outcome"] == "solved" for p in plays[arm][s]) for s in shared}
        for arm in (a, b_arm)
    }
    solved_first = {
        arm: {
            s: any(p["outcome"] == "solved" for p in plays[arm][s] if p["attempt"] == 1)
            for s in shared
        }
        for arm in (a, b_arm)
    }
    rejected = {
        arm: {
            s: statistics.median(int(p.get("rejected") or 0) for p in plays[arm][s])
            for s in shared
        }
        for arm in (a, b_arm)
    }

    only_b = [s for s in shared if not solved[a][s] and solved[b_arm][s]]
    only_a = [s for s in shared if solved[a][s] and not solved[b_arm][s]]
    diffs = [rejected[a][s] - rejected[b_arm][s] for s in shared]

    report = {
        "dir": str(args.dir),
        "arms": [a, b_arm],
        "seeds": len(shared),
        "attempts_per_seed": max(len(plays[a][s]) for s in shared),
        "pass_at_k": {arm: sum(solved[arm].values()) for arm in (a, b_arm)},
        "pass_at_1": {arm: sum(solved_first[arm].values()) for arm in (a, b_arm)},
        "discordant": {
            f"only_{b_arm}": len(only_b),
            f"only_{a}": len(only_a),
            "both": sum(1 for s in shared if solved[a][s] and solved[b_arm][s]),
            "neither": sum(
                1 for s in shared if not solved[a][s] and not solved[b_arm][s]
            ),
        },
        "mcnemar_p": round(mcnemar_exact(len(only_b), len(only_a)), 4),
        "rejected_median_per_seed": {
            arm: statistics.median(rejected[arm].values()) for arm in (a, b_arm)
        },
        "rejected_total": {
            arm: sum(rejected[arm].values()) for arm in (a, b_arm)
        },
        "wilcoxon_rejected": wilcoxon(diffs),
        f"rescued_by_{b_arm}": [
            {"seed": s, "rejected": [rejected[a][s], rejected[b_arm][s]]} for s in only_b
        ],
        f"lost_under_{b_arm}": [
            {"seed": s, "rejected": [rejected[a][s], rejected[b_arm][s]]} for s in only_a
        ],
        "outcomes": {
            arm: dict(Counter(p["outcome"] for s in shared for p in plays[arm][s]))
            for arm in (a, b_arm)
        },
    }

    out = args.output or args.dir / f"comparison_{a}_vs_{b_arm}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
