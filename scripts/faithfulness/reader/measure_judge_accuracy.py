"""Score the mutation and crossover judges against hand labels.

The judges replaced a registry of deterministic gates that turned out not to
work: `parallel_crossover` rejected rows that were genuine fusions, and 46 of
the 58 rules had never fired. Replacing them with an LLM is only an improvement
if the LLM agrees with a careful reader, and nothing so far has measured that.

So the 31 rows labelled by hand are the test set. Each is run through the same
path production uses — deterministic evidence computed first, then handed to the
judge as context rather than as a verdict — and the answers are compared.

The judge model must not be the model that wrote the rows. These 31 were written
by `gpt-5.6-terra`, so the judge here is `gpt-5.6-luna`; for rows written by
luna the roles swap. `JUDGE_MODEL` overrides it, and the script refuses to run
if the judge matches the generator recorded on the rows.

What matters is not overall agreement. The labels are 22 keep against 9 reject,
so a judge that never rejects anything scores 71%. The number to read is recall
on the rejects: of the rows a person threw out, how many does the judge catch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.certification.judges import (
    JUDGE_SYSTEM,
    crossover_prompt,
    mutation_prompt,
    parse_verdict,
)
from src.certification.novelty import judge_crossover, judge_mutation, tactic_skeleton
from src.exam_env.palette import TACTIC_DOCS
from src.utils.codex_cli import call_codex_cli, is_usage_limit

LABELS = Path("data/evaluation/exam/judge_labels.json")
CORPUS = Path("data/certified/run-a/minif2f_dedup.json")
CAMPAIGN = Path("data/certified/run-a")


def _build_index() -> Dict[str, Dict[str, Any]]:
    """Every row the labels or their parents could refer to.

    The deduped corpus holds the children but not their parents: a first
    generation's parent is a benchmark seed, which lives in the seed CSVs, and
    a later generation's parent is a row from an earlier group's output. Indexing
    only the corpus left 15 of 31 rows without the parent proof their judge
    prompt needs, which would have been scored as judge failures.
    """
    import csv
    import glob

    index: Dict[str, Dict[str, Any]] = {}

    def add(row: Dict[str, Any]) -> None:
        for key in ("problem_id", "release_id", "release_uid", "id"):
            value = row.get(key)
            if value and str(value) not in index:
                index[str(value)] = row

    for row in json.loads(CORPUS.read_text(encoding="utf-8")):
        add(row)
    for path in glob.glob(str(CAMPAIGN / "*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                add(json.loads(line))
            except json.JSONDecodeError:
                continue
    for path in glob.glob(str(CAMPAIGN / "seeds" / "*.csv")):
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add(dict(row))
    return index


def _statement(row: Dict[str, Any]) -> str:
    return str(row.get("formal_statement") or row.get("statement") or "")


def _proof(row: Dict[str, Any]) -> str:
    return str(row.get("lean_code") or "")


def _parents(row: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for parent_id in row.get("parent_ids") or []:
        parent = index.get(str(parent_id))
        if parent:
            out.append(parent)
    return out


def _evidence(row: Dict[str, Any], parents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The deterministic measurements, kept as context rather than as a gate.

    These are the same numbers the old registry decided on. They are still worth
    computing — coupling depth and skeleton distance are facts — but they are
    passed to the judge to weigh, which is the whole point of the change.
    """
    child_lean = _proof(row)
    if not child_lean:
        return {"measured": False, "why": "no lean_code"}
    if str(row.get("op_type")) == "crossover":
        pack = [
            {"key": str(p.get("problem_id") or ""), "lean_code": _proof(p)} for p in parents
        ]
        if len([p for p in pack if p["lean_code"]]) < 2:
            return {"measured": False, "why": "fewer than two parent proofs"}
        verdict = judge_crossover(
            child_lean, pack, difficulty=str(row.get("difficulty_label") or "")
        )
    else:
        parent = next((p for p in parents if _proof(p)), None)
        if parent is None:
            return {"measured": False, "why": "no parent proof"}
        tactics = tactic_skeleton(_proof(parent), list(TACTIC_DOCS))
        verdict = judge_mutation(child_lean, tactics, list(TACTIC_DOCS))
    detail = {"measured": True, **verdict.detail}
    if not verdict.ok:
        detail["deterministic_flag"] = verdict.kind or "rejected"
        detail["deterministic_reason"] = verdict.reason
    return detail


async def _judge(
    row: Dict[str, Any],
    parents: List[Dict[str, Any]],
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    evidence = _evidence(row, parents)
    if str(row.get("op_type")) == "crossover":
        prompt = crossover_prompt(
            [
                {
                    "name": str(p.get("problem_id") or "?"),
                    "statement": _statement(p),
                    "proof": _proof(p),
                }
                for p in parents
            ],
            _statement(row),
            _proof(row),
            evidence=evidence,
        )
    else:
        parent = parents[0] if parents else {}
        prompt = mutation_prompt(
            _statement(parent),
            _proof(parent),
            _statement(row),
            _proof(row),
            evidence=evidence,
        )
    reply = await call_codex_cli(
        model=model, system=JUDGE_SYSTEM, user=prompt, timeout_seconds=timeout
    )
    if reply.error:
        if is_usage_limit(reply.error):
            raise SystemExit("judge halted: provider usage limit reached")
        return {"ran": False, "why": str(reply.error)[:200], "evidence": evidence}
    verdict = parse_verdict(reply.raw_text)
    verdict["evidence"] = evidence
    return verdict


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("data/cache/judge_accuracy.json"))
    args = parser.parse_args()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    index = _build_index()

    generators = {
        str(index[uid].get("llm_model"))
        for uid in (row["release_uid"] for row in labels)
        if uid in index and index[uid].get("llm_model")
    }
    if args.model in generators:
        raise SystemExit(
            f"refusing to run: judge {args.model} also wrote these rows "
            f"(generators seen: {sorted(generators)}). A judge grading its own "
            "output measures self-consistency, not accuracy."
        )
    print(f"judge={args.model}  rows_written_by={sorted(generators) or ['unrecorded']}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def one(label: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = index.get(label["release_uid"])
        if row is None:
            return None
        async with semaphore:
            verdict = await _judge(row, _parents(row, index), args.model, args.timeout)
        return {
            "release_uid": label["release_uid"],
            "op_type": label["op_type"],
            "label": label["label_verdict"],
            "label_quality": label.get("label_quality", ""),
            "note": label.get("note", ""),
            "judge": verdict.get("verdict", ""),
            "judge_quality": verdict.get("quality", ""),
            "judge_failure": verdict.get("failure", ""),
            "judge_reason": verdict.get("reason", ""),
            "ran": bool(verdict.get("ran", True)),
            "evidence": verdict.get("evidence", {}),
        }

    results = [r for r in await asyncio.gather(*(one(l) for l in labels)) if r]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _report(results)


def _report(results: List[Dict[str, Any]]) -> None:
    scored = [r for r in results if r["ran"] and r["judge"]]
    print(f"\njudged {len(scored)} of {len(results)} rows")
    if not scored:
        return
    matrix = Counter((r["label"], r["judge"]) for r in scored)
    keeps = sum(1 for r in scored if r["label"] == "keep")
    rejects = len(scored) - keeps

    print("\n            judge:keep  judge:reject")
    for label in ("keep", "reject"):
        print(
            f"  label:{label:<7}{matrix[(label,'keep')]:>7}{matrix[(label,'reject')]:>13}"
        )

    agree = matrix[("keep", "keep")] + matrix[("reject", "reject")]
    print(f"\nagreement      {agree}/{len(scored)} = {agree/len(scored):.0%}")
    # The baseline that matters: 22 keeps against 9 rejects means a judge that
    # accepts everything already scores 71%, so agreement alone says nothing.
    print(f"always-keep    {keeps}/{len(scored)} = {keeps/len(scored):.0%}  (baseline)")
    if rejects:
        recall = matrix[("reject", "reject")] / rejects
        print(f"reject recall  {matrix[('reject','reject')]}/{rejects} = {recall:.0%}")
    caught = matrix[("keep", "reject")]
    if keeps:
        print(f"false reject   {caught}/{keeps} = {caught/keeps:.0%}")

    print("\ndisagreements:")
    for r in scored:
        if r["label"] != r["judge"]:
            print(
                f"  [{r['op_type']}] {r['release_uid'][:58]}\n"
                f"    human={r['label']} ({r['note'][:60]})\n"
                f"    judge={r['judge']} ({r['judge_failure']}) {r['judge_reason'][:110]}"
            )


if __name__ == "__main__":
    asyncio.run(main())
