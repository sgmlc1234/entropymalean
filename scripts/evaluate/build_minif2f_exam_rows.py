#!/usr/bin/env python3
"""Build exam rows from proved miniF2F seeds, in the ProofNet row schema.

The two benchmarks reach this point by different routes — ProofNet ships proofs
we replay, miniF2F withholds them so we generate our own — but an exam row must
not remember which route it took, or every downstream table has to branch on
benchmark. This converts the Gen-0 output into the same shape
`build_pnv_exam_rows.py` produces, so `enrich_exam_rows.py` and both CSV views
run over either file unchanged.

Two fields necessarily differ in substance, and say so rather than pretending:
  gt_proof_source   "EML Gen-0 (<model>)", not a released answer key
  audit             unaudited — miniF2F has no faithfulness taxonomy to inherit

The palette is built the same way as ProofNet's: candidate names are collected
from the proof and the statement, then validated in one batched `#check` probe
against *our* Mathlib, so a level never offers a lemma that does not exist here.

Usage:
  python scripts/evaluate/build_minif2f_exam_rows.py \
    --input data/benchmarks/minif2f_v2/raw/seeds_49_proved.csv \
    --output data/benchmarks/minif2f_v2/raw/exam_rows.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
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

from src.exam_env.palette import (  # noqa: E402
    CORE_TACTICS,
    TACTIC_DOCS,
    build_check_probe,
    candidate_theorem_names,
    parse_check_probe_output,
    tactics_in_proof,
)
from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

_STEP_RE = re.compile(r"^\s*(?:--|/-)\s*(Step\s+\d+[a-z]?(?:\.\d+)?)\s*[:.]\s*(.+?)\s*(?:-/)?\s*$")
_IDENT_ONLY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*")
_HEADER_LINE_RE = re.compile(r"^\s*(import|open|set_option|noncomputable section|section)\b")


def split_lean_code(code: str, fallback_header: str = "") -> Dict[str, str]:
    """Take a compilable file apart into header, statement, and proof body.

    Gen-0 returns one file; the exam row needs the three pieces separately,
    because the environment shows the statement, runs under the header, and
    keeps the body as the answer key.

    Gen-0 does not always repeat the imports — 14 of 50 replies came back as a
    bare theorem — so an empty extraction falls back to the header the seed
    already carries. Without that, the reassembled file has no `import Mathlib`
    and cannot compile in a fresh process. It *appears* to compile under the
    warm REPL, which keeps one environment across commands and had Mathlib
    loaded from an earlier row, so this failure hides from exactly the check
    meant to catch it.
    """
    lines = str(code or "").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if re.match(r"^\s*(theorem|lemma)\b", line)),
        None,
    )
    if start is None:
        return {"header": "", "statement": "", "proof": ""}
    header = "\n".join(
        line for line in lines[:start] if _HEADER_LINE_RE.match(line) or not line.strip()
    ).strip()
    if "import" not in header:
        header = str(fallback_header or "").strip() or "import Mathlib"
    rest = "\n".join(lines[start:])
    match = re.search(r":=\s*by\b", rest)
    if not match:
        return {"header": header, "statement": rest.strip(), "proof": ""}
    return {
        "header": header,
        "statement": rest[: match.end()].strip(),
        "proof": rest[match.end():].rstrip(),
    }


def hint_ladder(proof_body: str) -> Dict[str, Any]:
    """Parse `-- Step N:` structure, exactly as the ProofNet builder does.

    Gen-0 is prompted to write these, but nothing forces it to, and a one-line
    proof has no steps to describe. A row with an empty outline is a row whose
    level-2 hint does not exist — which the ladder records rather than invents.
    """
    outline: List[str] = []
    step_tactics: List[Dict[str, str]] = []
    current = None
    pending: List[str] = []

    def flush() -> None:
        nonlocal pending
        if current is not None:
            code = "\n".join(line for line in pending if line.strip()).strip()
            if code:
                step_tactics.append({"step": current, "tactic": code})
        pending = []

    for line in proof_body.splitlines():
        step = _STEP_RE.match(line)
        if step:
            flush()
            current = step.group(1)
            if re.fullmatch(r"Step\s+\d+", step.group(1)):
                outline.append(f"{step.group(1)}: {step.group(2).strip()}")
        elif line.strip().startswith("--"):
            continue
        else:
            pending.append(line)
    flush()
    return {"outline": outline, "step_tactics": step_tactics}


def run_check_probe(code: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as handle:
        handle.write(code)
        path = handle.name
    proc = subprocess.run(
        ["lake", "env", "lean", path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    return proc.stdout + "\n" + proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--skip-palette", action="store_true")
    parser.add_argument("--check-chunk", type=int, default=200)
    args = parser.parse_args()

    seeds = list(csv.DictReader(args.input.open(encoding="utf-8")))
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for index, seed in enumerate(seeds, 1):
        code = str(seed.get("lean_code") or "").strip()
        if not code:
            skipped.append({"id": seed.get("id"), "reason": "no_proof"})
            continue
        parts = split_lean_code(code, str(seed.get("lean_header") or ""))
        if not parts["statement"] or not parts["proof"].strip():
            skipped.append({"id": seed.get("id"), "reason": "unsplittable"})
            continue
        rows.append(
            {
                "stem": f"minif2f-{index}",
                "name": str(seed.get("id")),
                "benchmark": "minif2f_v2",
                "exam_theorem": str(seed.get("id")),
                "used_corrected": False,
                "statement_nl": str(seed.get("statement") or ""),
                "formal_statement": _prelint_lean_syntax(parts["statement"]),
                "lean_header": parts["header"],
                "gt_proof_body": parts["proof"],
                "lean_code": code,
                "hints": hint_ladder(parts["proof"]),
                # No faithfulness taxonomy exists for miniF2F; claiming one
                # would be worse than recording its absence.
                "audit": {
                    "faithfulness": "unaudited",
                    "provability": "proved_by_eml_gen0",
                    "error_type": None,
                },
                "split": str(seed.get("split") or ""),
                "license": "MIT (miniF2F-v2, Ospanov et al.)",
            }
        )

    # ---- palette: one batched #check over every candidate name ------------
    signatures: Dict[str, str] = {}
    per_row: Dict[str, List[str]] = {}
    if not args.skip_palette:
        every: List[str] = []
        for row in rows:
            in_statement = set(_IDENT_ONLY_RE.findall(row["formal_statement"]))
            names = [
                name
                for name in candidate_theorem_names(row["gt_proof_body"])
                if name not in in_statement
            ]
            per_row[row["stem"]] = names
            every.extend(names)
        unique = sorted(dict.fromkeys(every))
        # Probed in chunks: one `#check` file with every candidate would be a
        # single long elaboration whose failure loses the whole batch.
        for start in range(0, len(unique), args.check_chunk):
            chunk = unique[start : start + args.check_chunk]
            raw = run_check_probe(build_check_probe("import Mathlib", chunk))
            signatures.update(parse_check_probe_output(raw, chunk))
            print(
                f"  palette chunk {start // args.check_chunk + 1}: "
                f"{len(signatures)}/{len(unique)} validated",
                flush=True,
            )

    for row in rows:
        names = per_row.get(row["stem"], [])
        row["palette"] = {
            "theorems": {n: signatures[n] for n in names if n in signatures},
            "tactics": {
                t: TACTIC_DOCS[t]
                for t in sorted(set(tactics_in_proof(row["gt_proof_body"])) | set(CORE_TACTICS))
                if t in TACTIC_DOCS
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "seeds": len(seeds),
        "rows": len(rows),
        "skipped": skipped,
        "gt_proof_source": f"EML Gen-0 ({args.model})",
        "rows_with_outline": sum(1 for r in rows if r["hints"]["outline"]),
        "palette_validated_names": len(signatures),
        "median_palette_size": (
            sorted(len(r["palette"]["theorems"]) for r in rows)[len(rows) // 2] if rows else 0
        ),
        "by_split": dict(Counter(r["split"] for r in rows)),
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
