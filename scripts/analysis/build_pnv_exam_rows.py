#!/usr/bin/env python3
"""Build exam-environment rows from ProofNet-Verified ground truth.

For each of the 367 PNV problems this produces one exam row containing:
  - the verified exam statement (the `_corrected` theorem when the original
    formalization was erroneous, else the audited original), proof stripped;
  - the trusted header (file prelude + PNV `helper` declarations);
  - the GT proof body (LOCAL-ONLY — PNV ships proofs in a separate zip
    precisely to avoid contamination; do not publish this field);
  - a palette (theorem names+signatures validated against OUR pinned
    Mathlib in batched `#check` probes, plus tactic docs);
  - a hint ladder parsed from the GT's structured `-- Step N:` comments;
  - the PNV audit verdict for the ORIGINAL lineage (ancestry context).

Usage:
  python scripts/analysis/build_pnv_exam_rows.py \
    --gt-dir /tmp/pnv_gt/proofnet_verified_gt \
    --output data/benchmarks/proofnet_verified/raw/exam_rows.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from src.exam_env.palette import (  # noqa: E402
    CORE_TACTICS,
    TACTIC_DOCS,
    build_check_probe,
    candidate_theorem_names,
    parse_check_probe_output,
    tactics_in_proof,
)
from src.orchestration.pool_generation import _prelint_lean_syntax  # noqa: E402

PNV_ROOT = REPO_ROOT / "references" / "ProofNet-Verified"

_DECL_RE = re.compile(r"(?m)^(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_.']*)")
_SORRY_RE = re.compile(r"(?<![A-Za-z_])sorry(?![A-Za-z_])")
_STEP_RE = re.compile(r"^\s*--\s*(Step\s+[\d.]+):?\s*(.*)$")
_IDENT_ONLY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def _strip_comments(code: str) -> str:
    without_line = re.sub(r"--.*?$", "", code, flags=re.M)
    return re.sub(r"/-.*?-/", "", without_line, flags=re.S)


def _split_declarations(code: str) -> List[Tuple[str, str, str]]:
    """→ [(kind, name, decl_text)] for top-level theorem/lemma declarations."""
    matches = list(_DECL_RE.finditer(code))
    declarations = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code)
        declarations.append(
            (match.group(1), match.group(2), code[match.start():end].rstrip())
        )
    return declarations


def _select_main(name: str, declarations) -> Optional[Tuple[str, str]]:
    """Pick the exam declaration: `_corrected` > exact name > any proved."""
    def proved(text: str) -> bool:
        return not _SORRY_RE.search(_strip_comments(text))

    by_name = {decl_name: text for _, decl_name, text in declarations}
    for candidate in (f"{name}_corrected", name):
        if candidate in by_name and proved(by_name[candidate]):
            return candidate, by_name[candidate]
    for _, decl_name, text in declarations:
        if decl_name.endswith("_neg") or decl_name.endswith("_formal"):
            continue
        if proved(text):
            return decl_name, text
    return None


def _split_statement_proof(decl_text: str) -> Optional[Tuple[str, str]]:
    match = re.search(r":=\s*by\b", decl_text)
    if not match:
        match = re.search(r":=", decl_text)
        if not match:
            return None
        return decl_text[: match.start()].rstrip(), decl_text[match.end():]
    return decl_text[: match.start()].rstrip() + " := by", decl_text[match.end():]


def _hint_ladder(proof_body: str) -> Dict[str, object]:
    """Parse the GT's `-- Step N:` structure into a two-level hint ladder."""
    outline: List[str] = []
    step_tactics: List[Dict[str, str]] = []
    current: Optional[str] = None
    pending_lines: List[str] = []

    def flush() -> None:
        nonlocal pending_lines
        if current is not None:
            code = "\n".join(
                line for line in pending_lines if line.strip()
            ).strip()
            if code:
                step_tactics.append({"step": current, "tactic": code})
        pending_lines = []

    for line in proof_body.splitlines():
        step = _STEP_RE.match(line)
        if step:
            flush()
            current = step.group(1)
            description = step.group(2).strip()
            if re.fullmatch(r"Step\s+\d+", step.group(1)):
                outline.append(f"{step.group(1)}: {description}")
        elif line.strip().startswith("--"):
            continue
        else:
            pending_lines.append(line)
    flush()
    return {"outline": outline, "step_tactics": step_tactics}


def _prelude(file_text: str) -> str:
    """Imports/opens before the first doc comment or declaration."""
    lines: List[str] = []
    for line in file_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "open ", "open scoped", "set_option ")):
            lines.append(stripped)
        elif stripped.startswith(("/-", "theorem ", "lemma ", "def ", "noncomputable")):
            break
    return "\n".join(lines)


def _run_check_probe(code: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as handle:
        handle.write(code)
        path = handle.name
    proc = subprocess.run(
        ["lake", "env", "lean", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.stdout + "\n" + proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-dir", type=Path, required=True, help="unzipped proofnet_verified_gt dir"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/benchmarks/proofnet_verified/raw/exam_rows.jsonl",
    )
    parser.add_argument("--check-chunk", type=int, default=400)
    parser.add_argument(
        "--skip-palette", action="store_true", help="skip #check validation"
    )
    args = parser.parse_args()

    jsonl_rows = {
        f"proofnet-{row['index']}": row
        for row in (
            json.loads(line)
            for line in (PNV_ROOT / "data" / "proofnet-verified.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    taxonomy = {
        record["stem"]: record
        for record in (
            json.loads(line)
            for line in (
                PNV_ROOT / "error_taxonomy" / "proofnet#" / "results.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    parsed_rows = []
    skipped = []
    for stem in sorted(jsonl_rows, key=lambda s: int(s.split("-")[1])):
        meta = jsonl_rows[stem]
        gt_path = args.gt_dir / f"{stem}.lean"
        if not gt_path.is_file():
            skipped.append((stem, "missing_gt_file"))
            continue
        text = gt_path.read_text(encoding="utf-8")
        declarations = _split_declarations(text)
        selected = _select_main(str(meta["name"]), declarations)
        if selected is None:
            skipped.append((stem, "no_proved_declaration"))
            continue
        decl_name, decl_text = selected
        split = _split_statement_proof(decl_text)
        if split is None:
            skipped.append((stem, "no_proof_body"))
            continue
        statement, proof_body = split
        header_parts = [_prelude(text)]
        helper = str(meta.get("helper") or "").strip()
        if helper:
            header_parts.append(helper)
        parsed_rows.append(
            {
                "stem": stem,
                "name": str(meta["name"]),
                "exam_theorem": decl_name,
                "used_corrected": decl_name.endswith("_corrected"),
                "statement_nl": str(meta.get("informal_stmt") or ""),
                "formal_statement": _prelint_lean_syntax(statement),
                "lean_header": "\n".join(part for part in header_parts if part),
                "gt_proof_body": proof_body.rstrip(),
                "hints": _hint_ladder(proof_body),
                "audit": {
                    "faithfulness": (taxonomy.get(stem) or {}).get("q2_faithfulness"),
                    "provability": (taxonomy.get(stem) or {}).get("q3_provability"),
                    "error_type": (taxonomy.get(stem) or {}).get("q4_error_type"),
                },
                "license": "Apache-2.0 (marcusm117/ProofNet-Verified)",
            }
        )

    # ---- palette: batched #check validation across ALL rows ---------------
    signatures: Dict[str, str] = {}
    if not args.skip_palette:
        all_candidates: List[str] = []
        per_row_candidates: Dict[str, List[str]] = {}
        for row in parsed_rows:
            statement_names = set(
                _IDENT_ONLY_RE.findall(row["formal_statement"])
            )
            candidates = [
                name
                for name in candidate_theorem_names(
                    _strip_comments(row["gt_proof_body"])
                )
                if name.split(".")[0] not in statement_names
                and name not in statement_names
            ]
            per_row_candidates[row["stem"]] = candidates
            all_candidates.extend(candidates)
        unique = sorted(dict.fromkeys(all_candidates))
        print(f"palette candidates: {len(unique)} unique names")
        for start in range(0, len(unique), args.check_chunk):
            chunk = unique[start : start + args.check_chunk]
            raw = _run_check_probe(
                build_check_probe("import Mathlib", chunk)
            )
            signatures.update(parse_check_probe_output(raw, chunk))
            print(
                f"  chunk {start // args.check_chunk + 1}: "
                f"{len(signatures)} validated so far"
            )
        for row in parsed_rows:
            names = per_row_candidates[row["stem"]]
            row["palette"] = {
                "theorems": {
                    name: signatures[name] for name in names if name in signatures
                },
                "tactics": {
                    name: TACTIC_DOCS[name]
                    for name in dict.fromkeys(
                        [
                            *tactics_in_proof(row["gt_proof_body"]),
                            *CORE_TACTICS,
                        ]
                    )
                },
            }
    else:
        for row in parsed_rows:
            row["palette"] = {"theorems": {}, "tactics": {}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in parsed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(parsed_rows),
        "skipped": skipped,
        "used_corrected": sum(1 for row in parsed_rows if row["used_corrected"]),
        "with_helper": sum(
            1 for row in parsed_rows if "def " in row["lean_header"]
        ),
        "hint_outline_rows": sum(
            1 for row in parsed_rows if row["hints"]["outline"]
        ),
        "mean_palette_theorems": (
            round(
                sum(len(row["palette"]["theorems"]) for row in parsed_rows)
                / max(1, len(parsed_rows)),
                2,
            )
        ),
        "validated_names": len(signatures),
        "audit_counts": dict(
            Counter(str(row["audit"]["faithfulness"]) for row in parsed_rows)
        ),
    }
    (args.output.parent / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
