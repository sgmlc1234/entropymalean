"""Run the goal round-trip over rows that were certified before it was switched on.

`POOL_ALIGNMENT_GOAL_AUDIT` was unset for most of the campaigns that produced
this corpus, so the check that answers "is the released goal the problem the
prose states" ran on 5 of 146 released rows. Every other check reads 146/146 and
that one reads 5, which is the honest number and a bad one: it is the only gate
that can catch a statement which type-checks, proves, and means something else.

The check is three steps with the roles kept apart. Lean elaborates the goal
with `extract_goal; sorry`, so what is compared is what Lean actually built
rather than the source text. An informalizer renders that back to prose seeing
only the Lean. A separate judge compares that prose with the problem's own,
seeing only the two texts and neither the Lean nor which is which.

One Lean call and two model calls per row, so this is run once over the corpus
rather than per generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from src.certification.alignment import elaborated_goal_alignment
from src.evaluation.lean_verifier import LeanVerifyResult
from src.orchestration.pool_generation import default_generation_config


async def lean_verifier(code: str, timeout: float = 300.0) -> Any:
    """Compile a probe and hand back what Lean printed.

    The alignment module reads `raw_output` for the elaborated goal, which
    arrives as an *error* message: `extract_goal` reports the goal it found, so
    a non-zero exit is the expected path and not a failure.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=".", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        path = Path(handle.name)
    try:
        result = await asyncio.to_thread(
            subprocess.run, ["lake", "env", "lean", path.name],
            capture_output=True, text=True, timeout=timeout,
        )
        # `extract_goal` reports the goal it found, and the probe closes with
        # `sorry`, so Lean exits 0 with warnings and the goal text arrives on the
        # message stream. The alignment module reads it out of raw stdout/stderr.
        return LeanVerifyResult(
            ok=result.returncode == 0,
            complete=result.returncode == 0,
            raw_stdout=result.stdout or "",
            raw_stderr=result.stderr or "",
        )
    except subprocess.TimeoutExpired:
        return LeanVerifyResult(ok=False, complete=False, system_error="timeout")
    finally:
        path.unlink(missing_ok=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml1_release.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/release/goal_roundtrip.json"))
    parser.add_argument("--concurrency", type=int, default=3)
    # `.env` sets GENERATION_MODEL to a model codex has no metadata for, and a
    # config built with model=None falls through to it: the first full run came
    # back `signal_error` on all 146 rows because the comparison step could not
    # reach a model, while Lean and the informalizer had both worked.
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--only-missing", action="store_true",
                        help="skip rows that already carry a round-trip result")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.release.read_text(encoding="utf-8").splitlines() if line.strip()]
    done: Dict[str, dict] = {}
    if args.output.is_file():
        done = {r["problem_id"]: r for r in json.loads(args.output.read_text(encoding="utf-8"))}
    todo = [
        row for row in rows
        if not (args.only_missing and (
            done.get(row["problem_id"], {}).get("equivalent") is not None
            or (row["checks"].get("goal_roundtrip") or {}).get("ran")
        ))
    ]
    print(f"{len(rows)} released rows · {len(todo)} to audit · concurrency {args.concurrency}")

    config = default_generation_config(model=args.model, temperature=None)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: List[dict] = list(done.values())

    async def audit(row: dict) -> dict:
        async with semaphore:
            try:
                evidence = await elaborated_goal_alignment(
                    statement_nl=row.get("statement") or "",
                    formal_statement=row.get("formal_statement") or "",
                    lean_header=row.get("lean_header") or "",
                    config=config,
                    verifier=lean_verifier,
                )
            except Exception as error:  # pragma: no cover
                evidence = {"status": "error", "why": str(error)[:200]}
        return {"problem_id": row["problem_id"], **(evidence or {})}

    for index, coro in enumerate(asyncio.as_completed([audit(r) for r in todo]), 1):
        record = await coro
        results = [r for r in results if r["problem_id"] != record["problem_id"]] + [record]
        if index % 10 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ran = [r for r in results if r.get("equivalent") is not None]
    agree = [r for r in ran if r.get("equivalent")]
    print(f"\naudited {len(ran)}/{len(rows)}")
    print(f"  goal reads back as the stated problem : {len(agree)}")
    print(f"  mismatch                              : {len(ran) - len(agree)}")
    for record in ran:
        if not record.get("equivalent"):
            print(f"    {record['problem_id']}\n      {str(record.get('rationale') or '')[:180]}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
