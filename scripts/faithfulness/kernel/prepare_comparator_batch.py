"""Build one comparator workspace per released row, and dry-run what it will consume.

Comparator is the top of the certificate ladder: it derives a Challenge from the
trusted statement alone, takes the model's file verbatim as the Solution, and
uses lean4export to establish that the named declaration proves *the same
statement*, uses only the permitted axioms, and is accepted by the Lean kernel.
That is the difference between `proof_checked` and `kernel_replayed`.

It cannot run here. The hardened sandbox is Landlock through `landrun`, so
`validate_comparator_runtime` refuses on anything but Linux, and this machine has
neither `comparator`, `landrun`, `lean4export` nor `systemd-run`. Running it is a
Linux CI job.

What this script does instead is the half that is portable, and it is the half
that decides whether the CI job will be worth starting:

  * materialise every workspace, so the job is `comparator config.json` per
    directory and nothing else;
  * check the two preconditions comparator enforces before it ever reaches the
    kernel -- that the Challenge (statement closed with `sorry`) elaborates on
    its own, and that the declaration named in `theorem_names` is the one the
    Solution actually declares.

A row failing either of those fails comparator for a reason that has nothing to
do with its proof, and is worth knowing before renting a Linux runner. Passing
both is *not* the comparator guarantee and this script does not claim it: the
kernel replay and the axiom comparison are exactly what is still missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.lean_comparator import (
    DEFAULT_PERMITTED_AXIOMS,
    declaration_name,
    prepare_comparator_workspace,
)


def statement_prefix(row: dict) -> str:
    """The statement, without any proof, as comparator's Challenge needs it."""
    formal = str(row.get("formal_statement") or "").strip()
    return formal.split(":=", 1)[0].rstrip() if ":=" in formal else formal


def header_of(row: dict) -> str:
    header = str(row.get("lean_header") or "").strip()
    if header:
        return header
    code = str(row.get("lean_code") or "")
    return code.split("theorem", 1)[0].strip()


def elaborates(source: str, slot: int, timeout: int) -> tuple[bool, str]:
    # One file per job, not per worker slot: slots are reused while a job is
    # still running, and the second job's cleanup deletes the first one's file
    # out from under Lean.
    path = Path(f"_cmp{slot}.lean")
    path.write_text(source, encoding="utf-8")
    try:
        result = subprocess.run(["lake", "env", "lean", path.name],
                                capture_output=True, text=True, timeout=timeout)
        message = (result.stdout + result.stderr).strip()
        # `sorry` is the whole point of a Challenge; the warning is not a failure.
        clean = result.returncode == 0
        return clean, message[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        path.unlink(missing_ok=True)


def examine(job) -> Dict[str, Any]:
    slot, row, root, mathlib, toolchain, timeout = job
    problem_id = str(row.get("problem_id"))
    record: Dict[str, Any] = {"problem_id": problem_id, "prepared": False,
                              "challenge_elaborates": False, "name_matches": False, "why": ""}
    prefix, header = statement_prefix(row), header_of(row)
    try:
        expected = declaration_name(prefix, header)
    except ValueError as error:
        record["why"] = str(error)
        return record
    record["theorem_name"] = expected

    directory = root / problem_id.replace("/", "_")
    try:
        workspace = prepare_comparator_workspace(
            directory,
            header=header,
            formal_prefix=prefix,
            candidate_code=str(row.get("lean_code") or ""),
            mathlib_dir=mathlib,
            lean_toolchain=toolchain,
            permitted_axioms=DEFAULT_PERMITTED_AXIOMS,
        )
    except Exception as error:  # pragma: no cover
        record["why"] = f"prepare failed: {error}"[:200]
        return record
    record["prepared"] = True
    record["workspace"] = str(directory)

    ok, message = elaborates((directory / "Challenge.lean").read_text(encoding="utf-8"), slot, timeout)
    record["challenge_elaborates"] = ok
    if not ok:
        record["why"] = message
    # Comparator looks the declaration up by name; a Solution that renames it
    # fails there rather than in the kernel, which is a preparation bug.
    record["name_matches"] = expected in (directory / "Solution.lean").read_text(encoding="utf-8")
    if not record["name_matches"] and not record["why"]:
        record["why"] = f"Solution does not declare {expected}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml1_release.jsonl"))
    parser.add_argument("--workspaces", type=Path, default=Path("data/release/comparator"))
    parser.add_argument("--output", type=Path, default=Path("data/release/comparator_batch.json"))
    parser.add_argument("--mathlib", type=Path, default=Path(".lake/packages/mathlib"))
    parser.add_argument("--toolchain", type=Path, default=Path("lean-toolchain"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.release.read_text(encoding="utf-8").splitlines() if line.strip()]
    toolchain = args.toolchain.read_text(encoding="utf-8").strip()
    args.workspaces.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} rows · mathlib {args.mathlib} · toolchain {toolchain}")

    jobs = [(index, row, args.workspaces, args.mathlib, toolchain, args.timeout)
            for index, row in enumerate(rows)]
    results: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, record in enumerate(pool.map(examine, jobs), 1):
            results.append(record)
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    prepared = sum(1 for r in results if r["prepared"])
    elaborated = sum(1 for r in results if r["challenge_elaborates"])
    named = sum(1 for r in results if r["name_matches"])
    ready = sum(1 for r in results if r["prepared"] and r["challenge_elaborates"] and r["name_matches"])
    print(f"\nworkspaces prepared           {prepared}/{len(results)}")
    print(f"Challenge elaborates alone    {elaborated}/{len(results)}")
    print(f"declaration name resolves     {named}/{len(results)}")
    print(f"ready for a Linux comparator  {ready}/{len(results)}")
    for record in results:
        if not (record["prepared"] and record["challenge_elaborates"] and record["name_matches"]):
            print(f"  NOT READY {record['problem_id']}\n     {record['why']}")
    print(f"\nwritten: {args.output}")
    print("This is preparation, not certification: the kernel replay and the axiom")
    print("comparison still require `comparator` on Linux.")


if __name__ == "__main__":
    main()
