"""Re-judge released candidates under the current judge.

The corpus was judged by three different versions of the judge. 291 of 340 rows
passed a judge that could not see the plan the slot was given, could not see the
siblings already kept from the same parents, and applied one standard to every
mutation tier. Each of those three additions caught real defects the same day it
went in -- a plan that had itself specified a trivial change, two crossovers that
were one problem written twice, and an `easy` slot being held to the `hard` bar.

So the corpus is not one corpus. This re-runs every row through the judge as it
stands now, in the order the pipeline produced them, so that the sibling context
each row sees is the context it would have seen.

Verdicts are recorded, not applied. Removing rows is a separate decision.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
from src.orchestration.pool_generation import (  # noqa: E402
    _review_problem_quality,
    note_kept_row,
)


#: Where the rows behind the candidates live. A campaign missing from this list
#: is not a partial re-judge -- it is a silent one: the candidate carries the
#: statement and the Lean, but the prose, the operator card and both parents'
#: statements are read from here, so every one of them arrives empty and the
#: judge is asked about a child with no parents. Eighty rows came back
#: `ran: false` that way, and the summary line read them as eighty rejections.
DEFAULT_INDEX = (
    "data/certified/run-a/*.jsonl",
    "data/certified/run-b/*.jsonl",
    "data/certified/run-c/*.jsonl",
    "data/certified/run-d/*.jsonl",
    "data/certified/run-e/*.jsonl",
    "data/certified/ablation/*/*.jsonl",
)


def build_index(patterns=DEFAULT_INDEX) -> dict:
    index: dict = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("problem_id")
                if key and (key not in index or row.get("status") == "certified"):
                    index[key] = row
    for path in glob.glob("data/certified/run-a/seeds/*.csv"):
        if path.endswith(".pre_p6_fix"):
            continue
        with open(path, encoding="utf-8") as handle:
            for seed in csv.DictReader(handle):
                key = seed.get("id") or seed.get("problem_id")
                if key and key not in index:
                    index[key] = seed
    return index


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/release/eml_candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/release/rejudged_1.json"))
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--seed-siblings", type=Path, default=None,
                        help="Release JSONL whose rows are treated as already kept, so a "
                             "re-judged subset is measured against the corpus it would join "
                             "rather than against an empty one.")
    args = parser.parse_args()

    index = build_index()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    # Original production order, so each row is judged against the siblings that
    # preceded it rather than against the whole corpus at once.
    order = {"run-a": 0, "ablation/crossover": 1, "ablation/mutation": 2}
    candidates.sort(key=lambda c: (order.get(c["campaign"], 9), str(c["problem_id"])))

    if args.seed_siblings:
        judged_ids = {str(c["problem_id"]) for c in candidates}
        for line in args.seed_siblings.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            kept = json.loads(line)
            if str(kept.get("problem_id")) in judged_ids:
                continue
            note_kept_row(
                kept.get("problem_id"),
                [p.get("parent_id") for p in (kept.get("parents") or [])] or kept.get("parent_ids") or [],
                kept.get("formal_statement") or "",
                "strong",
            )

    results = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def one(cand: dict) -> dict:
        row = index.get(cand["problem_id"]) or {}
        parents = [
            SimpleNamespace(
                id=str(pid),
                metadata={
                    "formal_statement": str((index.get(str(pid)) or {}).get("formal_statement") or ""),
                    "lean_code": str((index.get(str(pid)) or {}).get("lean_code") or ""),
                },
            )
            for pid in (cand.get("parent_ids") or [])
        ]
        generated = SimpleNamespace(
            id=cand["problem_id"],
            formal_statement=cand.get("formal_statement"),
            lean_code=cand.get("lean_code"),
            statement=row.get("statement") or "",
            op_type=cand.get("op_type"),
            llm_model=row.get("llm_model") or "",
            source_problem_id=row.get("source_problem_id") or "",
        )
        item = dict(row.get("operator_card") or {})
        item.update(
            {
                "op_type": cand.get("op_type"),
                "operator_variant": cand.get("operator_variant"),
                "parent_ids": cand.get("parent_ids") or [],
                "novelty_evidence": (row.get("quality_evidence") or {}).get("structural_novelty") or {},
                "redundancy_evidence": (row.get("quality_evidence") or {}).get("redundancy") or {},
                "retry_count": 0,
            }
        )
        async with semaphore:
            verdict = await _review_problem_quality(generated, parents, item)
        return {
            "problem_id": cand["problem_id"],
            "campaign": cand["campaign"],
            "op_type": cand.get("op_type"),
            "operator_variant": cand.get("operator_variant"),
            "evidence_depth": cand.get("evidence_depth"),
            "old_quality": cand.get("judge_quality"),
            "new_verdict": verdict.get("verdict"),
            "new_quality": verdict.get("quality"),
            "new_failure": verdict.get("failure"),
            "new_reason": verdict.get("reason"),
            "fix_scope": verdict.get("fix_scope"),
            "ran": bool(verdict.get("ran")),
        }

    for index_i, cand in enumerate(candidates, 1):
        record = await one(cand)
        results.append(record)
        if record["new_verdict"] == "keep":
            note_kept_row(
                cand["problem_id"],
                cand.get("parent_ids") or [],
                cand.get("formal_statement") or "",
                str(record.get("new_quality") or ""),
            )
        if index_i % 20 == 0 or index_i == len(candidates):
            print(f"  {index_i}/{len(candidates)}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    kept = sum(1 for r in results if r["new_verdict"] == "keep")
    print(f"\nre-judged {len(results)}  keep {kept}  reject {len(results)-kept}")


if __name__ == "__main__":
    asyncio.run(main())
