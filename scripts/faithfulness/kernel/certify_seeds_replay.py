"""Drive the real comparator over the seed set to earn `kernel_replayed`.

Runs inside the Lima Linux VM, because comparator's sandbox is Landlock and
Landlock is Linux-only. Mathlib is not rebuilt there: the host's built packages
are visible over virtiofs and Lean 4 oleans load across OS on the same
architecture, so the VM elaborates against the same pinned artifacts the host
does.

This is the last certificate level. The levels below it ask Lean to accept the
proof and then ask what axioms it leaned on; this one exports the finished term
and replays it through an independent kernel, which is what makes the claim
survive a bug in the elaborator rather than merely a bug in the proof.

The fast verifier is stubbed to "complete" on purpose — every row here already
passed `proof_replays`, so re-elaborating before each comparator run would
double the cost to re-learn something recorded in the file.

Usage (from the host):
  limactl shell eml -- bash -lc 'cd <repo> && python3 scripts/faithfulness/kernel/certify_seeds_replay.py \
    --input data/benchmarks/proofnet_verified/raw/seeds_50_rows.jsonl \
    --output data/benchmarks/proofnet_verified/raw/seeds_50_replay_cert.json'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
HOME = Path.home()

# The VM has no Python environment for the full package — importing `src`
# from the repo pulls in langsmith and the rest of the orchestration stack,
# none of which the comparator needs. `emlmod` is a stripped copy carrying
# only the two evaluation modules, refreshed from the host before each run.
_STRIPPED = HOME / "emlmod"
sys.path.insert(0, str(_STRIPPED if _STRIPPED.is_dir() else REPO_ROOT))

os.environ["PATH"] = os.pathsep.join(
    [
        str(HOME / ".elan" / "bin"),
        str(HOME / "go" / "bin"),
        str(HOME / "lean4export" / ".lake" / "build" / "bin"),
        os.environ.get("PATH", ""),
    ]
)

from src.evaluation.lean_comparator import (  # noqa: E402
    LeanComparatorGate,
    validate_comparator_runtime,
)
from src.evaluation.lean_verifier import LeanVerifyResult  # noqa: E402

COMPARATOR_BIN = HOME / "lean-comparator" / ".lake" / "build" / "bin" / "comparator"
MATHLIB_DIR = REPO_ROOT / ".lake" / "packages" / "mathlib"
TOOLCHAIN = (REPO_ROOT / "lean-toolchain").read_text().strip()

_STATEMENT_END = re.compile(r":=\s*by\b")


def split_row(row: Dict[str, Any]) -> Dict[str, str]:
    """Header and `theorem … := by` prefix, which is what the gate trusts.

    The comparator certifies a candidate *against a trusted statement*, so the
    prefix has to come from the row rather than from the candidate — otherwise
    a candidate that quietly weakened the theorem would certify itself.
    """
    statement = str(row.get("formal_statement") or "").strip()
    if not _STATEMENT_END.search(statement):
        statement = re.sub(r":=\s*$", "", statement).rstrip() + " := by"
    return {
        "header": str(row.get("lean_header") or "").strip(),
        "prefix": statement,
        "code": str(row.get("lean_code") or "").strip(),
    }


async def fast_complete(code: str, *, timeout: float = 300.0, **kwargs):
    return LeanVerifyResult(ok=True, complete=True, verify_time=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=HOME / "eml-cert")
    parser.add_argument(
        "--mathlib-dir",
        type=Path,
        default=None,
        help=(
            "Mathlib package to build against. Defaults to the host's checkout, "
            "which works for elaboration but not here: the host tree is mounted "
            "read-only and Lake needs to take a lock file inside it, so point "
            "this at a writable copy in the VM."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    mathlib_dir = args.mathlib_dir or MATHLIB_DIR
    runtime = validate_comparator_runtime(COMPARATOR_BIN)
    print(f"mathlib: {mathlib_dir}", flush=True)
    print(f"runtime: {json.dumps(runtime)}", flush=True)

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    results: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        name = str(row.get("name"))
        parts = split_row(row)
        workspace = args.workspace_root / re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        gate = LeanComparatorGate(
            fast_verifier=fast_complete,
            header=parts["header"],
            formal_prefix=parts["prefix"],
            workspace_dir=workspace,
            mathlib_dir=mathlib_dir,
            lean_toolchain=TOOLCHAIN,
            comparator_bin=str(COMPARATOR_BIN),
            comparator_timeout=args.timeout,
        )
        try:
            verdict = asyncio.run(gate(parts["code"], timeout=args.timeout))
            certified = bool(verdict.ok and verdict.complete)
            detail = verdict.system_error or (
                verdict.errors[0].body[:300] if verdict.errors else ""
            )
        except Exception as error:  # a crashed run is a failed certification
            certified, detail = False, f"{type(error).__name__}: {error}"[:300]
        results.append(
            {"name": name, "kernel_replayed": certified, "detail": detail}
        )
        print(
            f"[{'CERT' if certified else 'FAIL'}] {index:3d}/{len(rows)} "
            f"{name[:44]} {'' if certified else detail[:110]!r}",
            flush=True,
        )

    certified = sum(1 for r in results if r["kernel_replayed"])
    report = {
        "input": str(args.input),
        "rows": len(rows),
        "kernel_replayed": certified,
        "rate": round(certified / max(1, len(rows)), 3),
        "lean_toolchain": TOOLCHAIN,
        "comparator": str(COMPARATOR_BIN),
        "mathlib_dir": str(mathlib_dir),
        "failure_details": dict(
            Counter(
                r["detail"][:60] for r in results if not r["kernel_replayed"]
            ).most_common(10)
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"kernel_replayed {certified}/{len(rows)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
