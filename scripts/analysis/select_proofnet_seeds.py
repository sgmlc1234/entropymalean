#!/usr/bin/env python3
"""Select a diverse ProofNet-Verified seed set, matching the miniF2F seed schema.

367 verified rows is more than a seed set needs — breeding widens the corpus, so
the seeds only have to span it. This picks 50 that do, and emits them in the
same CSV schema `complete_seed_proofs.py` reads, so both benchmarks enter the
campaign through one entry point.

Unlike miniF2F, these rows arrive with ground-truth proofs, so eligibility can
be checked rather than assumed — and it has to be. A quarter of the released
proofs do not replay under our pin, because Mathlib moved under them. So a row
is eligible only if all three hold:

  * its statement elaborates here,
  * its ground-truth proof still compiles here (`proof_replays`),
  * its audit did not mark it unfaithful to the prose.

The first two keep uncompilable material out of the exam's answer key; the
third matters because a seed that misstates its own problem breeds a lineage of
problems that all misstate it.

Run `verify_gt_replay.py` first — without `proof_replays` on the rows this
refuses to guess, because emitting `gen0_proof_completed=True` for a proof
nobody checked is exactly the claim this script exists to stop making.

Diversity is enforced on the same axes as the miniF2F selection (topic, goal
shape, hypothesis richness) plus proof length, so the set is not accidentally
all short proofs.

Usage:
  python scripts/analysis/select_proofnet_seeds.py --count 50 \
    --output data/benchmarks/proofnet_verified/raw/seeds_50.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
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

SOURCE = REPO_ROOT / "data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl"

# Faithfulness verdicts that disqualify a seed. "stronger"/"weaker" mean the
# Lean says something other than the prose; "unaudited" only means we have not
# looked, which is not itself a reason to exclude.
UNFAITHFUL = {"stronger", "weaker", "incomparable", "nl_ambiguous"}


def length_band(steps: Any) -> str:
    try:
        count = int(steps)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 19:
        return "short"
    if count <= 64:
        return "medium"
    return "long"


def eligible(row: Dict[str, Any]) -> bool:
    if not row.get("statement_checked"):
        return False
    if not str(row.get("lean_code") or "").strip():
        return False
    if not row.get("proof_replays"):
        return False
    verdict = str((row.get("audit") or {}).get("faithfulness") or "").lower()
    return verdict not in UNFAITHFUL


def select(rows: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    """Proportional over topic, then round-robin over (shape, hypotheses, length).

    Topics get shares of the seed set proportional to their share of the corpus,
    so the seeds are a scale model of it rather than a flat draw that would
    over-represent the small topics. Inside a topic, cells are visited
    round-robin so one goal shape cannot consume a small allocation.
    """
    rng = random.Random(seed)
    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[str(row.get("topic") or "other")].append(row)

    total = len(rows)
    quota = {t: max(1, round(count * len(v) / total)) for t, v in by_topic.items()}
    order = sorted(quota, key=lambda t: -len(by_topic[t]))
    while sum(quota.values()) > count:
        for topic in reversed(order):
            if sum(quota.values()) == count:
                break
            if quota[topic] > 1:
                quota[topic] -= 1
    while sum(quota.values()) < count:
        for topic in order:
            if sum(quota.values()) == count:
                break
            if quota[topic] < len(by_topic[topic]):
                quota[topic] += 1

    picked: List[Dict[str, Any]] = []
    for topic, want in quota.items():
        cells: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in by_topic[topic]:
            cells[
                (
                    row.get("conclusion_shape"),
                    row.get("hypothesis_bucket"),
                    length_band(row.get("gt_step_count")),
                )
            ].append(row)
        for bucket in cells.values():
            rng.shuffle(bucket)
        keys = sorted(cells, key=lambda k: -len(cells[k]))
        taken, index = 0, 0
        while taken < want:
            progressed = False
            for key in keys:
                if taken >= want:
                    break
                if index < len(cells[key]):
                    picked.append(cells[key][index])
                    taken += 1
                    progressed = True
            if not progressed:
                break
            index += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data/benchmarks/proofnet_verified/raw/seeds_50.csv",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not any("proof_replays" in row for row in rows):
        raise SystemExit(
            "rows carry no `proof_replays` field — run scripts/faithfulness/kernel/verify_gt_replay.py "
            "first, or the seed set would claim proofs that were never replayed"
        )
    pool = [row for row in rows if eligible(row)]
    picked = select(pool, args.count, args.seed)

    # Same columns as the miniF2F seed CSV, plus the proof these rows already
    # have — so Gen-0 completion is a no-op here and the two seed files can be
    # concatenated without reconciliation.
    fields = [
        "id", "statement", "answer", "formal_statement", "lean_header",
        "informal_proof", "split", "lean_code", "gen0_proof_completed",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in picked:
            writer.writerow(
                {
                    "id": row.get("name"),
                    "statement": row.get("statement_nl") or "",
                    "answer": "",
                    "formal_statement": row.get("formal_statement") or "",
                    "lean_header": row.get("lean_header") or "",
                    "informal_proof": "",
                    "split": "test",
                    "lean_code": row.get("lean_code") or "",
                    "gen0_proof_completed": True,
                }
            )

    summary = {
        "source": str(args.input.relative_to(REPO_ROOT)),
        "corpus_rows": len(rows),
        "eligible_rows": len(pool),
        "excluded_statement_unchecked": sum(1 for r in rows if not r.get("statement_checked")),
        "excluded_proof_did_not_replay": sum(1 for r in rows if not r.get("proof_replays")),
        "excluded_unfaithful": sum(
            1
            for r in rows
            if str((r.get("audit") or {}).get("faithfulness") or "").lower() in UNFAITHFUL
        ),
        "selected": len(picked),
        "by_topic": dict(Counter(r.get("topic") for r in picked).most_common()),
        "by_textbook": dict(Counter(r.get("textbook") for r in picked).most_common()),
        "by_shape": dict(Counter(r.get("conclusion_shape") for r in picked).most_common()),
        "by_hypotheses": dict(Counter(r.get("hypothesis_bucket") for r in picked)),
        "by_length": dict(Counter(length_band(r.get("gt_step_count")) for r in picked)),
        "corpus_topic_shares": {
            t: round(c / len(pool), 3)
            for t, c in Counter(r.get("topic") for r in pool).most_common()
        },
        "names": [r.get("name") for r in picked],
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "names"}, indent=2))
    print(f"wrote {len(picked)} seeds -> {args.output}")


if __name__ == "__main__":
    main()
