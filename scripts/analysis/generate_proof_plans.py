#!/usr/bin/env python3
"""Write a proof plan for every row, by one procedure, from the row's own proof.

The evaluation compares seed problems against the problems bred from them, and
the open-book condition hands each solver a plan. If the seeds' plans were
written by a person and the generated rows' plans by a model, the gap between
the two arms would confound problem difficulty with plan quality — and it would
do so exactly where the measurement is taken. So both arms get their plans from
here: same model, same prompt, same input (the row's verified Lean proof).

The hand-written plans stay where they are. They are better prose and they are
what the gallery shows; they are simply not usable as an experimental variable,
because nothing produces their equal for a hundred generated rows.

A plan describes strategy, never syntax. The same lint the hand-written ladder
used applies here — an outline that names tactics is a weaker copy of the
opening-move hint, which would make the aid conditions non-monotone.

Usage:
  set -a; source .env; set +a
  GENERATION_PROVIDER=codex_cli GENERATION_MODEL=gpt-5.6-luna \
  python scripts/analysis/generate_proof_plans.py \
    --rows data/benchmarks/minif2f_v2/raw/exam_rows_v2.jsonl \
    --output data/benchmarks/minif2f_v2/raw/machine_plans.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
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

from src.certification.generation import default_generation_config  # noqa: E402
from src.utils.codex_cli import call_codex_cli  # noqa: E402

SYSTEM = (
    "You explain Lean 4 proofs to someone who has to reconstruct them. "
    "You describe the mathematical strategy, never the tactic script."
)

USER = """Here is a theorem and a proof of it that compiles.

Statement:
{statement}

Proof:
{proof}

Write the strategy of this proof as {n} numbered steps, as prose a
mathematician would recognise. Say which case split opens the argument, what
each branch has to establish, and which fact does the real work.

Rules:
- Do not name any Lean tactic (no rw, simp, nlinarith, induction, intro, ...).
- Do not quote Lean syntax.
- Each step is one sentence, at least 20 words.
- Return only a JSON array of strings, nothing else.
"""

#: Tokens that are Lean and nothing else — seeing one is enough.
_LEAN_ONLY = re.compile(
    r"(^|\s)(rw|simp|simpa|nlinarith|linarith|norm_num|rcases|refine|omega|"
    r"field_simp|by_contra|aesop|ring_nf|interval_cases|simp_all)\b"
)
#: Tokens that are also ordinary English — "use the given values", "apply the
#: theorem", "by induction", "verify positivity", "a bounded case split". Six
#: plans were rejected for writing mathematics in the words mathematicians use;
#: `positivity` in particular is a Lean tactic *and* the ordinary name for the
#: property. They only count as leakage when written as code.
_AMBIGUOUS = re.compile(
    r"`[^`]*\b(use|exact|apply|intro|intros|calc|induction|cases|constructor|decide|"
    r"obtain|subst|revert|positivity|omega|linarith)\b[^`]*`"
    r"|(^|\s)(use|exact|apply|intro|calc|obtain|constructor|positivity)\s*[\[⟨(<]"
)


def steps_for(proof: str) -> int:
    """Plan length scaled to the proof, so a one-liner does not get four steps."""
    lines = [l for l in str(proof or "").splitlines() if l.strip()]
    return 1 if len(lines) <= 2 else (2 if len(lines) <= 8 else 4)


def lint(plan: List[str]) -> List[str]:
    problems = []
    if not plan:
        problems.append("empty")
    for step in plan:
        if _LEAN_ONLY.search(step) or _AMBIGUOUS.search(step):
            problems.append(f"names a tactic: {step[:50]!r}")
        elif len(step.split()) < 12:
            problems.append(f"too terse: {step[:40]!r}")
    return problems


def parse_plan(text: str) -> List[str]:
    match = re.search(r"\[.*\]", text or "", re.S)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    plans: Dict[str, Any] = {}
    if args.resume and args.output.is_file():
        plans = json.loads(args.output.read_text(encoding="utf-8"))
        print(f"resuming: {len(plans)} already written", flush=True)

    config = default_generation_config(None, None)
    failures: List[str] = []
    for index, row in enumerate(rows, 1):
        name = str(row.get("name"))
        if name in plans:
            continue
        proof = str(row.get("gt_proof_body") or "").strip()
        statement = str(row.get("formal_statement") or "").strip()
        if not proof or not statement:
            failures.append(f"{name}: no proof")
            continue

        plan: List[str] = []
        complaints: List[str] = ["not attempted"]
        for _ in range(args.retries + 1):
            reply = await call_codex_cli(
                model=config.model,
                system=SYSTEM,
                user=USER.format(
                    statement=statement, proof=proof[:6000], n=steps_for(proof)
                ),
                timeout_seconds=args.timeout,
            )
            candidate = parse_plan(reply.raw_text)
            complaints = lint(candidate)
            if not complaints:
                plan = candidate
                break

        if plan:
            plans[name] = {"plan": plan, "author": "machine", "model": config.model}
        else:
            failures.append(f"{name}: {'; '.join(complaints)[:90]}")
        mark = "ok  " if plan else "FAIL"
        print(f"[{mark}] {index:3d}/{len(rows)} {name[:42]:42s} {len(plan)} steps", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plans, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"\nwrote {len(plans)}/{len(rows)} -> {args.output}")
    if failures:
        print(f"failed {len(failures)}:")
        for line in failures[:10]:
            print(f"  {line}")


if __name__ == "__main__":
    asyncio.run(_run())
