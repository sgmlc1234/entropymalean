#!/usr/bin/env python3
"""Turn the admitted release into exam rows the episodic evaluation can play.

The release is a publication artefact: it carries the certificate, the review
history, the lineage, and the hashes that justify the row's admission. The exam
harness wants none of that. It wants a name, a statement, a header, and — for
the aided arms — a palette and a ladder. This projects one onto the other and
nothing else.

`benchmark` is normalised to the values the rest of the pipeline already uses
(`proofnet_verified`, `minif2f_v2`) so cells over this set slice by the same
key as the seed and treatment cells. Reporting one number over both would mix
populations whose difficulty differs by more than the effect being measured:
on the seed set, BFS solves 28% of ProofNet and 60% of miniF2F.

Only `admission.admitted` rows are taken. The file should contain nothing else,
so a mismatch is reported rather than silently trusted.

Closed-book cells need no palette, which is the expensive part (a batched
`#check` against Mathlib). Pass `--with-aid` to build one; otherwise the rows
carry empty aid, and `run_seed_exam.py` will refuse the open-book arms rather
than run them without lemmas — which is the behaviour we want, since an aided
arm over empty palettes is a closed-book run wearing the wrong label.

Usage:
  python3 scripts/evaluate/release_to_exam_rows.py \
    --input data/release/eml_v1_release.jsonl \
    --output data/evaluation/exam/release309_rows.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate.build_minif2f_exam_rows import split_lean_code  # noqa: E402
from scripts.evaluate.enrich_exam_rows import proof_metrics  # noqa: E402

#: Release labels -> the keys every other exam file and cell already uses.
BENCHMARK = {"ProofNet": "proofnet_verified", "miniF2F": "minif2f_v2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-aid", action="store_true",
                        help="also derive palette and hints (needs Lean; closed-book does not)")
    args = parser.parse_args()

    src = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    admitted = [r for r in src if (r.get("admission") or {}).get("admitted")]
    if len(admitted) != len(src):
        print(f"note: {len(src) - len(admitted)} of {len(src)} rows are not admitted; skipping them")

    rows: List[Dict[str, Any]] = []
    unknown_benchmark: collections.Counter = collections.Counter()
    unsplittable: List[str] = []
    for row in admitted:
        raw_bench = str(row.get("benchmark") or "")
        bench = BENCHMARK.get(raw_bench)
        if bench is None:
            unknown_benchmark[raw_bench] += 1
            continue
        parts = split_lean_code(
            str(row.get("lean_code") or ""), str(row.get("lean_header") or "")
        )
        body = parts["proof"]
        if not body.strip():
            unsplittable.append(str(row.get("problem_id")))
        out = {
            # The release's id is the exam's name: episodes are keyed on it, and
            # keeping it identical is what lets a result be traced back to the
            # published row it was measured on.
            "name": str(row.get("problem_id")),
            "problem_id": str(row.get("problem_id")),
            "benchmark": bench,
            "formal_statement": str(row.get("formal_statement") or ""),
            "lean_header": str(row.get("lean_header") or "import Mathlib"),
            "lean_code": str(row.get("lean_code") or ""),
            "gt_proof_body": body,
            "generation": row.get("lineage_depth") or 0,
            "op_type": row.get("op_type"),
            # The stratum a row belongs to. Dropping it forces every later
            # analysis to re-join against the release, and an episode file on
            # its own then cannot answer "does the drop differ between a silent
            # mutation and a hard one?" — which is the question the operator
            # tiers exist to pose.
            "operator_variant": row.get("operator_variant"),
            "slot": row.get("slot"),
            "campaign": row.get("campaign"),
            "certificate_level": (row.get("certificate") or {}).get("level"),
            # Empty until --with-aid; the runner refuses aided arms on these.
            "palette": {"tactics": {}, "theorems": {}},
            "hint_ladder": [],
            "hints": {"outline": [], "step_tactics": []},
        }
        out.update(proof_metrics(body))
        rows.append(out)

    if unknown_benchmark:
        print(f"note: unmapped benchmark labels skipped: {dict(unknown_benchmark)}")
    if unsplittable:
        print(f"note: {len(unsplittable)} rows whose proof would not split "
              f"(statement still playable; no aid derivable)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_bench = collections.Counter(r["benchmark"] for r in rows)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "admitted_in": len(admitted),
        "rows_out": len(rows),
        "by_benchmark": dict(by_bench),
        "by_lineage_depth": dict(sorted(collections.Counter(r["generation"] for r in rows).items())),
        "unsplittable_proofs": len(unsplittable),
    }, ensure_ascii=False, indent=2))

    if args.with_aid:
        print("\n--with-aid: now run\n"
              f"  python3 scripts/evaluate/build_treatment_aid_columns.py --input {args.output} "
              f"--output {args.output.with_name(args.output.stem + '_aid.jsonl')}\n"
              f"  python3 scripts/analysis/fill_hint_ladder.py --rows {args.output.with_name(args.output.stem + '_aid.jsonl')}")


if __name__ == "__main__":
    main()
