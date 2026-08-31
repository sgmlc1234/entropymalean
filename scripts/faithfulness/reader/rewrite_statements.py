"""Rewrite released prose in the register of the seed it descends from, and gate it.

The released statements are Lean read back as English. They carry Lean names in
running prose -- `the set of integers b satisfying Int.gcd b x = x + 3` -- use no
mathematical notation at all across 146 rows, and pack every binder into one
sentence. Their parents, from the source benchmarks, look nothing like that:
"What is the remainder when 2003 is divided by 11?" The cause was upstream, in a
seed loader that read a `statement` column holding `Prove the theorem <id>.` for
all 100 seeds, so the worker never saw what a benchmark problem sounds like.

That is fixed for future generations. This repairs the corpus already written.

A rewrite is only allowed to change the words. Whether it still describes the
same theorem is decided by the check that exists for exactly this question: the
goal round-trip, where Lean elaborates the statement, an informalizer renders
that back to prose seeing only the Lean, and a judge compares the two texts.
The elaborated goal and its informalization depend on the Lean alone, so both
are reused from the corpus-wide audit and only the comparison is re-run. A
rewrite the judge does not accept is discarded and the original kept: a prettier
statement that describes a different theorem is worse than an ugly accurate one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.certification.alignment import judge_nl_equivalence
from src.orchestration.pool_generation import default_generation_config
from src.utils.codex_cli import call_codex_cli

SYSTEM = (
    "You restate Lean theorems as the problems they are, in the voice of the "
    "benchmark they were bred from. You never change what is being claimed. "
    "Return JSON only."
)

INSTRUCTION = """Rewrite the CHILD statement as a problem a person would read.

Rules, in order of importance:
1. Say exactly what the Lean says. Every hypothesis, every quantifier, the same
   conclusion. If you cannot phrase a condition without naming a Lean construct,
   keep the condition and phrase it mathematically.
2. No Lean names, ever. `Int.gcd b x` is "the greatest common divisor of $b$ and
   $x$"; `Nat.choose n k` is "$\\binom{{n}}{{k}}$"; `Set.univ`, `IsExtrOn` and
   hypothesis labels like `h0` must not appear.
3. Mathematical notation in `$...$`, the way the parent below does it.
4. Match the PARENT's register. Terse competition question, or textbook
   exercise -- whichever the parent is. One or two sentences. A question is
   often the natural form.

PARENT, from the source benchmark -- this is the voice to write in:
{parent_prose}

PARENT in Lean:
{parent_lean}

CHILD in Lean -- this is what you are describing:
{child_lean}

CHILD's current statement, which reads as a transcription:
{child_prose}

Return exactly:
{{"statement": "the rewritten problem", "note": "anything you could not express without a Lean name"}}"""


def _parse(text: str) -> Dict[str, Any]:
    """First complete JSON object in the reply.

    A greedy `{...}` match spans from the first brace to the last, which fails
    the moment the model writes anything after the object -- or writes `$\\{x\\}$`
    inside the statement it is returning. Decoding from each opening brace stops
    at the first thing that actually parses.
    """
    raw = str(text or "")
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and "statement" in value:
            return value
    raise ValueError("rewriter returned no JSON object with a statement")


def _looks_transcribed(text: str) -> List[str]:
    """Lean constructs surviving in prose, which is what this is fixing."""
    return sorted(set(re.findall(
        r"\b(?:Nat|Int|Real|Set|Finset|Complex|Polynomial|Matrix|Filter|Sylow"
        r"|Subgroup|Submodule|Module|Function|Fintype|ZMod|IsExtrOn|IsMinOn"
        r"|IsOpenMap|Continuous|Metric|Topology)\.[A-Za-z_'.]+", text or "")))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml_v1_release.jsonl"))
    parser.add_argument("--roundtrip", type=Path, default=Path("data/release/goal_roundtrip.json"))
    parser.add_argument("--output", type=Path, default=Path("data/release/statement_rewrites.json"))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.release.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = {r["problem_id"]: r for r in json.loads(args.roundtrip.read_text(encoding="utf-8"))} \
        if args.roundtrip.is_file() else {}
    config = default_generation_config(model=args.model, temperature=None)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    print(f"{len(rows)} rows · {sum(1 for r in rows if cache.get(r['problem_id'], {}).get('informalized_statement'))}"
          f" with a cached round-trip to gate against")

    async def one(row: dict) -> dict:
        record: Dict[str, Any] = {
            "problem_id": row["problem_id"],
            "before": row.get("statement") or "",
            "lean_names_before": _looks_transcribed(row.get("statement") or ""),
        }
        parent = (row.get("parents") or [{}])[0]
        async with semaphore:
            try:
                response = await call_codex_cli(
                    model=config.model,
                    system=SYSTEM,
                    user=INSTRUCTION.format(
                        parent_prose=parent.get("statement") or "(none available)",
                        parent_lean=(parent.get("formal_statement") or "")[:900],
                        child_lean=(row.get("formal_statement") or "")[:1400],
                        child_prose=row.get("statement") or "",
                    ),
                    timeout_seconds=300.0,
                )
                rewritten = str(_parse(getattr(response, "text", response)).get("statement") or "").strip()
            except Exception as error:
                record["error"] = str(error)[:200]
                return record
        if not rewritten:
            record["error"] = "empty rewrite"
            return record
        record["after"] = rewritten
        record["lean_names_after"] = _looks_transcribed(rewritten)

        # The gate: does the rewrite still describe the goal Lean elaborated?
        # Compared against the informalization of that goal, which was produced
        # from the Lean alone and knows nothing about either prose.
        readback = (cache.get(row["problem_id"]) or {}).get("informalized_statement")
        if not readback:
            record["gate"] = "no cached round-trip"
            return record
        async with semaphore:
            try:
                verdict = await judge_nl_equivalence(rewritten, readback, config=config)
            except Exception as error:
                record["gate"] = f"judge error: {str(error)[:120]}"
                return record
        record["gate"] = "accepted" if verdict.get("equivalent") else "rejected"
        record["gate_reason"] = str(verdict.get("rationale") or "")[:400]
        return record

    results: List[dict] = []
    for index, coro in enumerate(asyncio.as_completed([one(r) for r in rows]), 1):
        results.append(await coro)
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    accepted = [r for r in results if r.get("gate") == "accepted"]
    rejected = [r for r in results if r.get("gate") == "rejected"]
    print(f"\nrewritten {sum(1 for r in results if r.get('after'))}/{len(rows)}")
    print(f"  gate accepted : {len(accepted)}")
    print(f"  gate rejected : {len(rejected)}  (original kept)")
    print(f"  errors        : {sum(1 for r in results if r.get('error'))}")
    before = sum(1 for r in results if r["lean_names_before"])
    after = sum(1 for r in accepted if r.get("lean_names_after"))
    print(f"  statements naming a Lean construct: {before} -> {after} (accepted rows)")
    with_math = sum(1 for r in accepted if "$" in r.get("after", ""))
    print(f"  accepted rewrites using $...$ notation: {with_math}/{len(accepted)}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
