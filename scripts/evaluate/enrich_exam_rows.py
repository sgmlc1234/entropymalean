#!/usr/bin/env python3
"""Fill in the columns the episodic evaluation and its ablations will need.

An exam row currently carries what is needed to *run* an episode. Analysing
episodes needs more: the variables every ablation table will slice by, a hint
ladder with graded levels rather than a flat outline, and — because a row that
does not elaborate under our pinned toolchain is not an exam at all — a
recorded statement check.

Nothing here is inferred from the model's behaviour; every field is derived
from the row itself or from Lean.

Usage:
  python scripts/evaluate/enrich_exam_rows.py \
    --input data/benchmarks/proofnet_verified/raw/exam_rows.jsonl \
    --output data/benchmarks/proofnet_verified/raw/exam_rows_v2.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from src.certification.levels import runtime_pins  # noqa: E402
from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.exam_env.palette import TACTIC_DOCS  # noqa: E402
from src.orchestration.pool_generation import _statement_sorry_code  # noqa: E402

SCHEMA_VERSION = "eml-exam-row-v2"

# miniF2F names its problems by contest/family rather than by textbook, so the
# same `topic` axis has to be read off a different prefix vocabulary. Both maps
# land in the same topic names where the subjects coincide, which is what makes
# the two benchmarks sliceable together.
FAMILY_TOPIC = {
    "mathd_algebra": "algebra",
    "mathd_numbertheory": "number_theory",
    "numbertheory": "number_theory",
    "algebra": "algebra",
    "induction": "induction",
    "amc12": "competition",
    "aime": "competition",
    "imo": "competition",
}

TEXTBOOK_TOPIC = {
    "Artin": "abstract_algebra",
    "Dummit_Foote": "abstract_algebra",
    "Herstein": "abstract_algebra",
    "Ireland_Rosen": "number_theory",
    "Axler": "linear_algebra",
    "Rudin": "real_analysis",
    "Pugh": "real_analysis",
    "Shakarchi": "complex_analysis",
    "Munkres": "topology",
    "Putnam": "competition",
}


def topic_of(name: str, benchmark: str) -> str:
    table = FAMILY_TOPIC if benchmark == "minif2f_v2" else TEXTBOOK_TOPIC
    # Longest prefix first, so `mathd_numbertheory` is not eaten by `numbertheory`
    for prefix in sorted(table, key=len, reverse=True):
        if str(name).startswith(prefix):
            return table[prefix]
    return "other"


def source_of(name: str, benchmark: str) -> str:
    """Where the problem came from — a textbook for ProofNet, a contest for miniF2F."""
    if benchmark == "minif2f_v2":
        for prefix in sorted(FAMILY_TOPIC, key=len, reverse=True):
            if str(name).startswith(prefix):
                return prefix
        return "unknown"
    marker = str(name).find("exercise")
    return str(name)[: marker].rstrip("_") if marker > 0 else "unknown"


def _conclusion(statement: str) -> str:
    body = re.sub(r":=\s*by\s*$", "", str(statement or "").strip()).rstrip()
    depth = 0
    for index, char in enumerate(body):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and depth == 0 and body[index : index + 2] != ":=":
            return body[index + 1 :].strip()
    return body


def conclusion_shape(statement: str) -> str:
    tail = _conclusion(statement)
    if re.search(r"∣|gcd|lcm|Prime|%", tail):
        return "divisibility"
    if re.search(r"[<>]|≤|≥", tail):
        return "inequality"
    if "↔" in tail:
        return "iff"
    if "∃" in tail:
        return "existential"
    if "∀" in tail:
        return "universal"
    if "=" in tail:
        return "equation"
    return "other"


def hypothesis_bucket(statement: str) -> str:
    count = len(re.findall(r"\([A-Za-z_][A-Za-z0-9_₀-₉']*\s*:", str(statement or "")))
    if count == 0:
        return "closed"
    if count <= 2:
        return "light"
    return "rich"


def assemble_lean_code(header: str, statement: str, body: str) -> str:
    """Header + statement + proof body, as one compilable file.

    The ground truth ships as a bare proof body; a seed has to carry the whole
    file, because that is what gets verified, bred from, and shown in an exam.
    """
    head = str(header or "").rstrip()
    stmt = str(statement or "").strip()
    proof = str(body or "").rstrip()
    if not stmt:
        return ""
    if not re.search(r":=\s*by\s*$", stmt):
        stmt = re.sub(r":=\s*$", "", stmt).rstrip() + " := by"
    if proof and not proof.startswith(("\n", " ")):
        proof = "\n  " + proof
    return f"{head}\n\n{stmt}{proof}\n"


def proof_metrics(body: str) -> Dict[str, Any]:
    """Structure of the ground-truth proof: a difficulty proxy and a normalizer."""
    code = re.sub(r"--.*?$", "", str(body or ""), flags=re.M)
    code = re.sub(r"/-.*?-/", "", code, flags=re.S)
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    tactics = []
    for line in lines:
        head = line.lstrip("·<;>| ").split(" ", 1)[0].strip()
        if head in TACTIC_DOCS and head not in tactics:
            tactics.append(head)
    return {
        "gt_step_count": len(lines),
        "gt_tactics": tactics,
        "gt_uses_induction": bool(re.search(r"\binduction\b", code)),
        "gt_uses_cases": bool(re.search(r"\b(cases|rcases|obtain|constructor)\b", code)),
        "gt_char_length": len(code),
    }


def hint_ladder(row: Dict[str, Any], proof: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Graded hints, weakest first.

    Level 1 names the lemmas without saying how; level 2 gives the proof's
    shape; level 3 gives one concrete step. `hinted Pass@K` records which level
    a solver needed, so the ladder has to be ordered and comparable across rows.
    """
    ladder: List[Dict[str, Any]] = []
    names = sorted((row.get("palette") or {}).get("theorems") or {})
    if names:
        ladder.append(
            {
                "level": 1,
                "kind": "lemma_names",
                "content": names,
                "leaks_proof": False,
            }
        )
    outline = ((row.get("hints") or {}).get("outline")) or []
    if outline:
        ladder.append(
            {"level": 2, "kind": "proof_outline", "content": outline, "leaks_proof": False}
        )
    steps = ((row.get("hints") or {}).get("step_tactics")) or []
    if steps:
        ladder.append(
            {
                "level": 3,
                "kind": "first_step_tactic",
                "content": steps[0].get("tactic") if isinstance(steps[0], dict) else str(steps[0]),
                "leaks_proof": True,
            }
        )
    return ladder


async def check_statement(row: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    code = _statement_sorry_code(
        str(row.get("lean_header") or ""), str(row.get("formal_statement") or "")
    )
    verdict = await verify_lean_proof_repl(code, timeout=timeout)
    return {
        "statement_checked": bool(verdict.ok),
        "statement_check_error": None if verdict.ok else verdict.summary()[:300],
    }


async def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data/benchmarks/proofnet_verified/raw/exam_rows.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--skip-lean", action="store_true", help="derive columns without the statement probe"
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    pins = runtime_pins(str(REPO_ROOT))

    enriched: List[Dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, 1):
            statement = str(row.get("formal_statement") or "")
            proof = proof_metrics(row.get("gt_proof_body"))
            lean_code = assemble_lean_code(
                row.get("lean_header"), statement, row.get("gt_proof_body")
            )
            palette = row.get("palette") or {}
            benchmark = str(row.get("benchmark") or "proofnet_verified")
            record = {
                **row,
                "schema_version": SCHEMA_VERSION,
                "benchmark": benchmark,
                # the full compilable file — what a seed is actually bred from
                "lean_code": lean_code,
                "lineage_role": "seed",
                "generation": 0,
                # --- stratification axes (shared with the miniF2F seed set) ---
                "topic": topic_of(row.get("name"), benchmark),
                "textbook": source_of(row.get("name"), benchmark),
                "conclusion_shape": conclusion_shape(statement),
                "hypothesis_bucket": hypothesis_bucket(statement),
                # --- difficulty proxies / normalizers ---
                **proof,
                "palette_theorem_count": len((palette.get("theorems") or {})),
                "palette_tactic_count": len((palette.get("tactics") or {})),
                # --- ablation material ---
                "hint_ladder": hint_ladder(row, proof),
                "max_hint_level": len(hint_ladder(row, proof)),
                # --- provenance for contamination accounting ---
                # miniF2F withholds its proofs, so ours is the only one there is
                # — and it is ours, which the source has to say out loud.
                "gt_proof_public": benchmark != "minif2f_v2",
                "gt_proof_source": (
                    "EML Gen-0"
                    if benchmark == "minif2f_v2"
                    else "ProofNet-Verified gt.zip"
                ),
                "lean_toolchain": pins["lean_toolchain"],
                "mathlib_revision": pins["mathlib_revision"],
                # --- episode result slots, filled by the evaluation ---
                "episodes": [],
            }
            if not args.skip_lean:
                record.update(await check_statement(row, args.timeout))
                mark = "ok " if record["statement_checked"] else "FAIL"
                print(f"[{mark}] {index:3d}/{len(rows)} {row['name'][:44]}")
            else:
                record["statement_checked"] = None
                record["statement_check_error"] = None
            enriched.append(record)
    finally:
        if not args.skip_lean:
            await close_global_repl_verifier()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in enriched:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(enriched),
        "statement_checked": sum(1 for r in enriched if r["statement_checked"]),
        "statement_failed": [
            r["name"] for r in enriched if r["statement_checked"] is False
        ],
        "by_topic": dict(Counter(r["topic"] for r in enriched).most_common()),
        "by_shape": dict(Counter(r["conclusion_shape"] for r in enriched).most_common()),
        "by_hypotheses": dict(Counter(r["hypothesis_bucket"] for r in enriched)),
        "by_max_hint_level": dict(Counter(r["max_hint_level"] for r in enriched)),
        "gt_step_count": {
            "median": sorted(r["gt_step_count"] for r in enriched)[len(enriched) // 2],
            "max": max(r["gt_step_count"] for r in enriched),
        },
        "palette_theorems": {
            "rows_with_palette": sum(1 for r in enriched if r["palette_theorem_count"]),
            "median": sorted(r["palette_theorem_count"] for r in enriched)[len(enriched) // 2],
        },
        "audit_faithfulness": dict(
            Counter(str((r.get("audit") or {}).get("faithfulness")) for r in enriched)
        ),
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "statement_failed"}, indent=2))
    print(f"statement_failed: {len(summary['statement_failed'])}")


if __name__ == "__main__":
    asyncio.run(_run())
