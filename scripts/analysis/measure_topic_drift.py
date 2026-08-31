"""Does the corpus keep its seed set's topic proportions, and if not, where does it drift?

Two measurements, both from data already on disk.

1. Trajectory. Total variation distance between the seed topic distribution and
   the distribution of rows produced at each generation. A corpus that preserves
   proportions is flat and near zero from generation one; one that converges
   starts high and falls; one that drifts climbs.

2. Decomposition. Proportions can move through exactly three channels: which
   topics get chosen as parents, which certify once attempted, and which survive
   two judge passes. Each is measured separately, per topic, so a drift can be
   attributed rather than described.

Topic belongs to the seed, so every generated row is attributed through its
lineage. A crossover reaches two roots and is counted under both.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

from src.orchestration.problem_ids import roots_of

CAMPAIGNS = {
    "ProofNet": ["data/certified/run-a/proofnet_g*.jsonl",
                 "data/certified/run-b/proofnet_g*.jsonl",
                 "data/certified/run-e/proofnet_p*.jsonl"],
    "miniF2F": ["data/certified/run-a/minif2f_g*.jsonl",
                "data/certified/run-c/minif2f_h*.jsonl",
                "data/certified/run-d/minif2f_k*.jsonl"],
}


def seed_topics() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(glob.glob("data/benchmarks/*/seeds_50_levels.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                m = re.search(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)",
                              f"{row.get('lean_goal') or ''} {row.get('solution') or ''}")
                if m and row.get("topic"):
                    out[m.group(1)] = str(row["topic"]).replace("_", " ")
    return out


def seed_mix(topic: Dict[str, str], bench: str) -> collections.Counter:
    """The seed distribution, taken from the seed CSVs the campaigns actually read."""
    pat = "proofnet" if bench == "ProofNet" else "minif2f"
    seen = set()
    for path in glob.glob(f"data/certified/*/seeds/{pat}_*.csv"):
        for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if row.get("id"):
                seen.add(row["id"])
    return collections.Counter(topic[s] for s in seen if s in topic)


def tvd(a: collections.Counter, b: collections.Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys) / 2


def admitted_ids() -> set:
    r1 = {r["problem_id"]: r for r in json.loads(Path("data/release/rejudged.json").read_text(encoding="utf-8"))}
    r2 = {r["problem_id"]: r for r in json.loads(Path("data/release/rejudged_run2.json").read_text(encoding="utf-8"))}
    return {p for p, a in r1.items()
            if (b := r2.get(p)) and a.get("new_quality") == "strong" and b.get("new_quality") == "strong"
            and a.get("new_verdict") == "keep" and b.get("new_verdict") == "keep"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/release/topic_drift.json"))
    args = parser.parse_args()

    topic = seed_topics()
    admitted = admitted_ids()
    report: Dict[str, dict] = {}

    for bench, patterns in CAMPAIGNS.items():
        rows: List[dict] = []
        for pattern in patterns:
            for path in glob.glob(pattern):
                if ".pre_" in path or "partial" in path:
                    continue
                rows += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

        seeds = seed_mix(topic, bench)
        by_gen: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        attempted = collections.Counter()
        certified = collections.Counter()
        admitted_c = collections.Counter()
        parent_slots = collections.Counter()

        for row in rows:
            if row.get("status") == "survivor":
                continue
            topics = {topic[r] for r in roots_of(str(row.get("problem_id") or "")) if r in topic}
            if not topics:
                continue
            gen = int(row.get("generation") or 0)
            for t in topics:
                attempted[t] += 1
                parent_slots[t] += 1
                if row.get("status") == "certified":
                    certified[t] += 1
                    by_gen[gen][t] += 1
                    if row.get("problem_id") in admitted:
                        admitted_c[t] += 1

        traj = [{"generation": g, "n": sum(by_gen[g].values()), "tvd": round(tvd(seeds, by_gen[g]), 4)}
                for g in sorted(by_gen) if sum(by_gen[g].values()) >= 5]
        channels = {}
        for t in sorted(seeds, key=lambda k: -seeds[k]):
            a = attempted.get(t, 0)
            channels[t] = {
                "seed_share": round(seeds[t] / sum(seeds.values()), 4),
                "parent_share": round(parent_slots.get(t, 0) / max(sum(parent_slots.values()), 1), 4),
                "certify_rate": round(certified.get(t, 0) / a, 4) if a else None,
                "admit_rate_given_certified": (
                    round(admitted_c.get(t, 0) / certified[t], 4) if certified.get(t) else None),
                "attempted": a, "certified": certified.get(t, 0), "admitted": admitted_c.get(t, 0),
            }
        report[bench] = {
            "seed_mix": dict(seeds),
            "tvd_final": round(tvd(seeds, admitted_c), 4),
            "trajectory": traj,
            "channels": channels,
        }

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for bench, r in report.items():
        print(f"\n{'='*74}\n{bench}   seed mix {dict(r['seed_mix'])}")
        print(f"  TVD(seed, admitted) = {r['tvd_final']}")
        print(f"\n  {'gen':>4s} {'n':>5s} {'TVD':>7s}")
        for t in r["trajectory"]:
            bar = "#" * int(round(t["tvd"] * 100))
            print(f"  {t['generation']:4d} {t['n']:5d} {t['tvd']:7.3f}  {bar}")
        print(f"\n  {'topic':20s} {'seed%':>7s} {'parent%':>8s} {'cert':>7s} {'adm|cert':>9s} {'att':>5s}")
        for t, c in r["channels"].items():
            print(f"  {t:20s} {100*c['seed_share']:6.1f}% {100*c['parent_share']:7.1f}% "
                  f"{('%5.1f%%' % (100*c['certify_rate'])) if c['certify_rate'] is not None else '    —':>7s} "
                  f"{('%7.1f%%' % (100*c['admit_rate_given_certified'])) if c['admit_rate_given_certified'] is not None else '      —':>9s} "
                  f"{c['attempted']:5d}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
