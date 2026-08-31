"""Ask, of every released row, whether one parent already carried it.

The inline probe runs the tactic ladder and no prover, because during
generation the case worth catching cheaply is a parent that assumes nothing and
falls to `omega`. That leaves the question open wherever the ladder cannot
reach: over the release it settled 37 of 165 rows, and a row it could not settle
is a row about which nothing is known, not a row that passed.

The question -- does the child still hold with that hypothesis removed, does one
parent prove it alone -- is theorem proving, so this scan gives it to a prover
and checks the answer in Lean. The model only proposes; a hallucinated proof
fails to compile and the probe reports nothing, which is the same as not knowing.

Two provers run in sequence, the second only on what the first left open. That
is sound because the probe is one-sided: a found proof is Lean-verified and
decisive, and a failure to find one asserts nothing. Adding an attempt can
therefore only add findings, never false ones -- and a second model is worth
having because the first also generated many of these rows, and asking a model
to undercut its own work is a bias that suppresses findings rather than
inventing them.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, ".")
from src.certification.redundancy import brief, check_mutation, check_parents
from src.evaluation.lean_verifier import verify_lean_proof
from src.utils.codex_cli import call_codex_cli


def make_prover(model: str):
    async def prover(system: str, user: str) -> str:
        reply = await call_codex_cli(model=model, system=system, user=user, timeout_seconds=300)
        return reply.raw_text
    return prover


def parent_index() -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for path in sorted(glob.glob("data/certified/**/*.jsonl", recursive=True)):
        if ".pre_" in path or ".partial" in path:
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                key = row.get("problem_id")
                if key and (key not in index or row.get("status") == "certified"):
                    index[key] = row
    for path in sorted(glob.glob("data/certified/run-a/seeds/*.csv")):
        if path.endswith(".pre_p6_fix"):
            continue
        with open(path, encoding="utf-8") as handle:
            for seed in csv.DictReader(handle):
                key = seed.get("id") or seed.get("problem_id")
                if key and key not in index:
                    index[key] = seed
    return index


async def probe(row: dict, index: Dict[str, dict], prover, timeout: float) -> Dict[str, Any]:
    header = row.get("lean_header") or "import Mathlib"
    parents = [p.get("parent_id") for p in (row.get("parents") or []) if p.get("parent_id")]
    if str(row.get("op_type") or "").startswith("crossover"):
        pack = [{"name": str(p),
                 "statement": str((index.get(str(p)) or {}).get("formal_statement") or "")}
                for p in parents]
        return await check_parents(verify_lean_proof, header, row.get("formal_statement") or "",
                                   pack, prover=prover, timeout=timeout)
    parent = str((index.get(str(parents[0])) or {}).get("formal_statement") or "") if parents else ""
    return await check_mutation(verify_lean_proof, header, row.get("formal_statement") or "",
                                parent, prover=prover,
                                variant=str(row.get("operator_variant") or ""), timeout=timeout)


def settled(evidence: Dict[str, Any]) -> bool:
    """A finding, either way, is a settled row. An empty measurement is not."""
    return bool(evidence.get("redundant")
                or evidence.get("universal_parents") or evidence.get("free_hypotheses"))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml1_release.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/release/redundancy_scan.json"))
    parser.add_argument("--provers", nargs="+", default=["gpt-5.6-luna", "gpt-5.6-terra"])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--op", default="", help="rescan only this op_type, keeping the rest")
    parser.add_argument("--only-missing", action="store_true",
                        help="scan only rows with no record on file. The scan is the expensive "
                             "check in the pipeline -- two provers and a Lean call per row -- and "
                             "a release that grew by 62 rows does not need the other 262 re-proved.")
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.release.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = parent_index()

    # Results already on disk are kept unless this run covers them. The
    # crossover probes were never touched by the change that invalidated the
    # mutation one, so rescanning them would spend prover calls to reproduce
    # answers already held.
    # Everything ever measured is carried forward, not just what this run
    # covers. The scan reads the *current* release, and a row it convicted on an
    # earlier run is no longer in that release -- so it is absent from the input
    # and, on a full rewrite, its verdict is erased. Eight convicted rows came
    # back into the release that way: with no record, nothing objected to them.
    # A gate must not delete its own findings.
    results: Dict[str, Dict[str, Any]] = {}
    if args.output.is_file():
        results = {r["problem_id"]: r for r in json.loads(args.output.read_text(encoding="utf-8"))}
    if args.op:
        rows = [r for r in rows if str(r.get("op_type") or "").startswith(args.op)]
    if args.only_missing:
        rows = [r for r in rows if r["problem_id"] not in results]
    # A row about to be rescanned starts from nothing, so a stale finding does
    # not survive a pass that no longer makes it. Rows *not* in this run keep
    # whatever was measured before -- that is the difference between rescanning
    # and forgetting.
    for row in rows:
        results.pop(row["problem_id"], None)
    print(f"{len(rows)} rows to scan · {len(results)} kept from disk · provers {', '.join(args.provers)}")

    remaining = list(rows)
    for stage, model in enumerate(args.provers, 1):
        if not remaining:
            break
        prover = make_prover(model)
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        print(f"\npass {stage}: {model} on {len(remaining)} row(s)")

        async def one(row: dict) -> tuple:
            async with semaphore:
                try:
                    return row, await probe(row, index, prover, args.timeout)
                except Exception as error:
                    return row, {"measured": False, "why": str(error)[:160]}

        done = 0
        still: List[dict] = []
        for coro in asyncio.as_completed([one(r) for r in remaining]):
            row, evidence = await coro
            done += 1
            found = settled(evidence)
            record = results.get(row["problem_id"], {})
            if found or not record:
                results[row["problem_id"]] = {
                    "problem_id": row["problem_id"],
                    "op_type": row.get("op_type"),
                    "operator_variant": row.get("operator_variant"),
                    "settled_by": model if found else "",
                    **evidence,
                    "brief": brief(evidence) if found else "",
                }
            if not found:
                still.append(row)
            if done % 20 == 0 or done == len(remaining):
                print(f"  {done}/{len(remaining)}  findings so far "
                      f"{sum(1 for r in results.values() if r.get('settled_by'))}", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        remaining = still

    args.output.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=1),
                           encoding="utf-8")
    found = [r for r in results.values() if r.get("settled_by")]
    print(f"\nscanned {len(rows)}   rows with a finding {len(found)}   unsettled {len(remaining)}")
    for record in found:
        print(f"  {record['problem_id']}\n     [{record['settled_by']}] {record['brief'][:150]}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
