"""Decide whether a hypothesis is actually dead by removing it and recompiling.

The flag this replaces searched the proof body for the hypothesis's name and
called it dead when the name did not appear. That is wrong for a whole family of
tactics -- `omega`, `simp_all`, `interval_cases`, `linarith`, `norm_num`,
`decide`, `aesop` -- which scan the entire local context and consume hypotheses
without ever naming one. On a release row whose binder read
`(hp_bound : 25 ≤ p ∧ p ≤ 100)`, the name appears nowhere in the proof and
deleting it makes `omega` fail: the hypothesis was load-bearing all along.

Deleting the binder and recompiling is the test that settles it. A proof that
still closes without the hypothesis genuinely did not need it; one that breaks
did. Unlike the name search, this has no false positives -- though it can miss a
hypothesis that is used only to make some *other* tactic converge faster.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

CONTEXT_SCANNING = ("omega", "simp_all", "interval_cases", "decide", "norm_num",
                    "linarith", "nlinarith", "aesop", "tauto", "positivity")
_BINDER_NAME = re.compile(r"\((h[A-Za-z0-9_'₀-₉]*)\s*[:\s]")


def unmentioned(row: dict) -> List[str]:
    """Hypotheses whose name never appears in the proof body."""
    body = str(row.get("lean_code") or "").split(":= by", 1)[-1]
    return [
        name
        for name in _BINDER_NAME.findall(str(row.get("formal_statement") or ""))
        if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_'₀-₉])", body)
    ]


def drop_binder(code: str, name: str) -> Optional[str]:
    """The Lean file with `(name : …)` removed, counting parens so nesting is exact."""
    opening = code.find("(" + name)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "(":
            depth += 1
        elif code[index] == ")":
            depth -= 1
            if depth == 0:
                return code[:opening] + code[index + 1:]
    return None


def compiles(source: str, slot: int, timeout: int) -> bool:
    path = Path(f"_deadhyp{slot}.lean")
    path.write_text(source, encoding="utf-8")
    try:
        result = subprocess.run(["lake", "env", "lean", path.name],
                                capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        path.unlink(missing_ok=True)


def examine(job: Tuple[int, dict, int]) -> dict:
    slot, row, timeout = job
    code = str(row.get("lean_code") or "")
    findings = []
    for name in unmentioned(row):
        without = drop_binder(code, name)
        if without is None:
            findings.append({"hypothesis": name, "verdict": "unparsed"})
            continue
        findings.append({
            "hypothesis": name,
            # The proof closing without it is the only evidence that it is dead.
            "verdict": "dead" if compiles(without, slot, timeout) else "used silently",
        })
    return {
        "problem_id": row.get("problem_id"),
        "tactics": sorted({t for t in CONTEXT_SCANNING if t in code.split(":= by", 1)[-1]}),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml1_release.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/release/dead_hypotheses.json"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.release.read_text(encoding="utf-8").splitlines() if line.strip()]
    flagged = [row for row in rows if unmentioned(row)]
    print(f"{len(flagged)}/{len(rows)} rows carry a hypothesis the proof never names")

    results = []
    jobs = [(index % args.workers, row, args.timeout) for index, row in enumerate(flagged)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, record in enumerate(pool.map(examine, jobs), 1):
            results.append(record)
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    dead = [(r["problem_id"], f["hypothesis"]) for r in results for f in r["findings"] if f["verdict"] == "dead"]
    silent = sum(1 for r in results for f in r["findings"] if f["verdict"] == "used silently")
    unparsed = sum(1 for r in results for f in r["findings"] if f["verdict"] == "unparsed")
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nhypotheses tested: {sum(len(r['findings']) for r in results)}")
    print(f"  used silently by a context-scanning tactic: {silent}")
    print(f"  genuinely dead (proof closes without them): {len(dead)}")
    print(f"  binder could not be parsed: {unparsed}")
    for problem_id, name in dead:
        print(f"    {problem_id}  ->  {name}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
