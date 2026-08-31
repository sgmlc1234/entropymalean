#!/usr/bin/env python3
"""Give treatment exam rows the aid columns the seed rows already carry.

A treatment row is built to *run* an episode — statement, header, certified
proof — and the closed-book cells need nothing more. The open-book arm does:
it reads lemma names out of `palette.theorems` and a first tactic out of
`hint_ladder`, and on the current treatment file both are empty on all 114
rows. Run as-is, `--arm open_book` is `closed_book` wearing a different label,
and the episode still records `aid_delivered: ["lemmas"]`, because the runner
reads the arm's config flag rather than the palette's contents.

The material is already there. Every treatment row carries a `sorry`-free
`lean_code`, which is the same input `build_minif2f_exam_rows.py` derives the
seed palettes from, so this reuses that builder's functions rather than
inventing a second procedure — a palette built two ways would confound the aid
comparison it exists to support.

What this deliberately does not touch: `name`, `formal_statement`,
`lean_header`, `lean_code`, `problem_id`, `generation`, `op_type`. Episodes
already recorded against these rows — the BFS treatment cell and the Goedel
re-run — are keyed on the name and play the statement, so changing either would
silently break comparability with runs already on disk. The script asserts they
came through unchanged rather than trusting that it left them alone.

`enrich_exam_rows.py` is not used here even though it fills similar columns: it
stamps every row `lineage_role: "seed"`, `generation: 0`, and overwrites
`lean_code` with a reassembled file. Those are right for a seed and wrong for a
generation-N child.

Afterwards, run `fill_hint_ladder.py --rows <output>` for the ladder,
`hint_levels`, `max_hint_level`, and `hint_degeneracy`; and
`generate_proof_plans.py` for the open-book plan channel.

Usage:
  python3 scripts/evaluate/build_treatment_aid_columns.py \
    --input data/evaluation/exam/treatment_minif2f_clean114.jsonl \
    --output data/evaluation/exam/treatment_minif2f_clean114_aid.jsonl
"""

from __future__ import annotations

import argparse
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

from scripts.evaluate.build_minif2f_exam_rows import (  # noqa: E402
    _IDENT_ONLY_RE,
    hint_ladder as parse_hints,
    run_check_probe,
    split_lean_code,
)
from scripts.evaluate.enrich_exam_rows import proof_metrics  # noqa: E402
from src.exam_env.palette import (  # noqa: E402
    CORE_TACTICS,
    TACTIC_DOCS,
    build_check_probe,
    candidate_theorem_names,
    parse_check_probe_output,
    tactics_in_proof,
)

#: Columns an already-played episode is keyed on or plays. Changing any of them
#: invalidates the cells already on disk, so they are checked, not assumed.
FROZEN = ("name", "formal_statement", "lean_header", "lean_code", "problem_id",
          "generation", "op_type", "benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-chunk", type=int, default=200)
    parser.add_argument("--skip-palette", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original = [{k: r.get(k) for k in FROZEN} for r in rows]

    unsplittable: List[str] = []
    for row in rows:
        parts = split_lean_code(
            str(row.get("lean_code") or ""), str(row.get("lean_header") or "")
        )
        body = parts["proof"]
        if not body.strip():
            # Recorded, not silently emptied: a row whose proof will not split
            # has no aid to offer, and the ladder must say so rather than
            # present an empty level as a delivered one.
            unsplittable.append(str(row.get("name")))
        row["gt_proof_body"] = body
        row["gt_proof_source"] = "EML generation (certified)"
        row["gt_proof_public"] = False
        row["hints"] = parse_hints(body)
        row.update(proof_metrics(body))

    # ---- palette: one batched #check over every candidate, as the seeds got --
    signatures: Dict[str, str] = {}
    per_row: Dict[str, List[str]] = {}
    if not args.skip_palette:
        every: List[str] = []
        for row in rows:
            in_statement = set(_IDENT_ONLY_RE.findall(str(row.get("formal_statement") or "")))
            names = [
                name
                for name in candidate_theorem_names(row["gt_proof_body"])
                if name not in in_statement
            ]
            per_row[str(row.get("name"))] = names
            every.extend(names)
        unique = sorted(dict.fromkeys(every))
        print(f"{len(unique)} candidate names to validate against our Mathlib", flush=True)
        for start in range(0, len(unique), args.check_chunk):
            chunk = unique[start : start + args.check_chunk]
            raw = run_check_probe(build_check_probe("import Mathlib", chunk))
            signatures.update(parse_check_probe_output(raw, chunk))
            print(
                f"  chunk {start // args.check_chunk + 1}: "
                f"{len(signatures)}/{len(unique)} validated",
                flush=True,
            )

    for row in rows:
        names = per_row.get(str(row.get("name")), [])
        row["palette"] = {
            "theorems": {n: signatures[n] for n in names if n in signatures},
            "tactics": {
                t: TACTIC_DOCS[t]
                for t in sorted(set(tactics_in_proof(row["gt_proof_body"])) | set(CORE_TACTICS))
                if t in TACTIC_DOCS
            },
        }
        row["palette_theorem_count"] = len(row["palette"]["theorems"])
        row["palette_tactic_count"] = len(row["palette"]["tactics"])

    for before, row in zip(original, rows):
        for key in FROZEN:
            if before[key] != row.get(key):
                raise SystemExit(
                    f"refusing to write: {key} changed on {before['name']!r}. "
                    "Episodes already recorded against these rows are keyed on "
                    "these fields."
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    empty_palette = sum(1 for r in rows if not r["palette"]["theorems"])
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "validated_names": len(signatures),
        "rows_with_theorems": len(rows) - empty_palette,
        "rows_with_empty_theorem_palette": empty_palette,
        "rows_with_outline": sum(1 for r in rows if r["hints"]["outline"]),
        "rows_with_step_tactics": sum(1 for r in rows if r["hints"]["step_tactics"]),
        "unsplittable_proofs": unsplittable,
        "median_theorems": sorted(r["palette_theorem_count"] for r in rows)[len(rows) // 2],
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
