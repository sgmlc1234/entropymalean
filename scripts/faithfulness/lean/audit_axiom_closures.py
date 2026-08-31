#!/usr/bin/env python3
"""Retroactive axiom audit of already-certified rows.

`proof_checked` now requires the declaration's axiom closure to lie within
{propext, Quot.sound, Classical.choice}. Rows certified before that rule was
introduced were only checked for elaborator acceptance, which does not see a
smuggled `axiom`, `sorryAx`, or `native_decide`'s `Lean.ofReduceBool`.

This replays the axiom probe over an existing corpus and reports which rows
would still earn `proof_checked` today. Proofs are deduplicated by content
hash, and the warm REPL verifier is used, so the cost is roughly one
elaboration per distinct proof.

Usage:
  python scripts/faithfulness/lean/audit_axiom_closures.py \
    --input release/huggingface/EML-1/accepted.jsonl \
    --output data/evaluation/axiom_audit/released.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.levels import PERMITTED_AXIOMS, axiom_audit  # noqa: E402
# The file verifier gives each probe a fresh process. The REPL shares one
# environment across commands, so a theorem name reused by another row can
# shadow the target and `#print axioms` reports that row's closure instead.
from src.evaluation.lean_verifier import verify_lean_proof  # noqa: E402
from src.orchestration.pool_generation import (  # noqa: E402
    parse_axiom_closure,
    theorem_name_of,
)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "release/huggingface/EML-1/accepted.jsonl"),
        help="JSONL path or glob over certified rows",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/evaluation/axiom_audit/report.json"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-parallel", type=int, default=1)
    return parser.parse_args()


def load_rows(pattern: str, limit: int) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(pattern)) or [pattern]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            status = row.get("status") or (row.get("certificate") or {}).get("status")
            code = str(row.get("lean_code") or "").strip()
            if status != "certified" or not code:
                continue
            rows.append(
                {
                    "problem_id": row.get("problem_id"),
                    "source_file": path,
                    "benchmark": row.get("benchmark"),
                    "lean_code": code,
                    "formal_statement": str(row.get("formal_statement") or ""),
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


async def _run() -> None:
    args = _parse()
    rows = load_rows(args.input, args.limit)
    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        digest = hashlib.sha256(row["lean_code"].encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append(row)
    print(f"rows={len(rows)} distinct_proofs={len(by_hash)}")

    semaphore = asyncio.Semaphore(max(1, args.max_parallel))
    results: Dict[str, Dict[str, Any]] = {}

    async def probe(digest: str, group: List[Dict[str, Any]]) -> None:
        row = group[0]
        decl = theorem_name_of(row["formal_statement"]) or theorem_name_of(
            row["lean_code"]
        )
        if not decl:
            results[digest] = {"status": "no_declaration", "audit": axiom_audit(None)}
            return
        async with semaphore:
            verdict = await verify_lean_proof(
                f"{row['lean_code'].rstrip()}\n\n#print axioms {decl}",
                timeout=args.timeout,
            )
        closure = parse_axiom_closure(
            "\n".join(p for p in (verdict.raw_stdout, verdict.raw_stderr) if p), decl
        )
        results[digest] = {
            "status": "probed" if closure is not None else "closure_unavailable",
            "declaration": decl,
            "audit": axiom_audit(closure),
            "verifier_ok": verdict.ok,
            "system_error": verdict.system_error,
        }
        mark = {True: "ok", False: "FAIL", None: "?"}[results[digest]["audit"]["passed"]]
        print(f"[{mark:4s}] {decl[:52]:54s} {results[digest]['audit']['axioms']}")

    await asyncio.gather(*(probe(d, g) for d, g in by_hash.items()))

    per_row = []
    for digest, group in by_hash.items():
        record = results.get(digest, {})
        for row in group:
            per_row.append(
                {
                    "problem_id": row["problem_id"],
                    "benchmark": row["benchmark"],
                    "source_file": row["source_file"],
                    "passed": record.get("audit", {}).get("passed"),
                    "axioms": record.get("audit", {}).get("axioms"),
                    "disallowed": record.get("audit", {}).get("disallowed"),
                    "status": record.get("status"),
                }
            )

    disallowed = Counter(
        name for r in per_row for name in (r["disallowed"] or [])
    )
    summary = {
        "input": args.input,
        "rows": len(per_row),
        "distinct_proofs": len(by_hash),
        "permitted_axioms": list(PERMITTED_AXIOMS),
        "row_verdicts": dict(Counter(str(r["passed"]) for r in per_row)),
        "probe_status": dict(Counter(str(r["status"]) for r in per_row)),
        "disallowed_axiom_counts": dict(disallowed.most_common()),
        "failing_rows": [r for r in per_row if r["passed"] is False],
        "unresolved_rows": [r for r in per_row if r["passed"] is None],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / f"{args.output.stem}_per_row.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_row) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"failing_rows", "unresolved_rows"}}, indent=2))
    print(f"failing={len(summary['failing_rows'])} unresolved={len(summary['unresolved_rows'])}")


if __name__ == "__main__":
    asyncio.run(_run())
