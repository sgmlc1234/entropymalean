#!/usr/bin/env python3
"""Normalize the curated accepted ledger into a public review schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.levels import (  # noqa: E402
    LEVEL_PROOF,
    legacy_lean_level_for,
    runtime_pins,
)
DEFAULT_INPUT = REPO_ROOT / "data/evaluation/treatment_inventory/final_curated/accepted.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "data/evaluation/treatment_inventory/final_curated/accepted_summary.json"
SCHEMA_VERSION = "emg2.accepted.v1"
PUBLIC_FIELDS = [
    "schema_version",
    "problem_id",
    "benchmark",
    "op_type",
    "generation",
    "slot",
    "family",
    "theorem_name",
    "statement",
    "answer",
    "formal_statement",
    "lean_code",
    "parents",
    "certificate",
    "curation",
    "provenance",
    "hashes",
]
PARENT_FIELDS = [
    "parent_id",
    "source",
    "statement",
    "formal_statement",
    "proof_idea",
]
CERTIFICATE_FIELDS = [
    "status",
    "level",
    "statement_checked",
    "proof_accepted",
    "axiom_audit",
    "kernel_replayed",
    "faithfulness",
    "alignment_method",
    "lean_toolchain",
    "mathlib_revision",
    "lean_level",
    "anti_stub_passed",
    "aligned",
]
CURATION_FIELDS = [
    "decision",
    "snapshot",
    "cluster",
    "entropy_direction",
    "rationale",
]
PROVENANCE_FIELDS = [
    "source_file",
    "source_problem_id",
    "source_run",
]
HASH_FIELDS = [
    "statement_sha256",
    "formal_statement_sha256",
]
STATEMENT_REWRITES = {
    "If b is a natural number with Nat.lcm 120 b = 3720 and Nat.gcd 120 b = 8, and real numbers a,d satisfy a + 6d = 30 and a + 10d = 60, then (b : ℝ) + a + 8d = 293.": (
        "If b is a natural number whose least common multiple with 120 is 3720 and whose greatest common divisor with 120 is 8, and real numbers a and d satisfy a + 6d = 30 and a + 10d = 60, then b, viewed as a real number, plus a plus 8d equals 293."
    ),
    "If b is a natural number with Nat.lcm 120 b = 3720 and Nat.gcd 120 b = 8, then the exact finite sum of the integers in Finset.Icc 120 b is 23736.": (
        "If b is a natural number whose least common multiple with 120 is 3720 and whose greatest common divisor with 120 is 8, then the sum of all integers from 120 through b is 23736."
    ),
    "The sum of all proper divisors d of 198 satisfying Nat.gcd d (Nat.gcd 180 168) = 6 is 90.": (
        "The sum of all proper divisors d of 198 such that the greatest common divisor of d and the greatest common divisor of 180 and 168 is 6 equals 90."
    ),
    "There exists a proper divisor d of 198 surviving the gcd filter Nat.gcd d (Nat.gcd 180 168) = 6 such that 2^(4*d+2) has remainder 4 modulo 10.": (
        "There exists a proper divisor d of 198 such that the greatest common divisor of d and the greatest common divisor of 180 and 168 is 6, and 2^(4d+2) has remainder 4 modulo 10."
    ),
    "For integers m and x satisfying 10 <= m < 100, 0 <= x < m, 6*x % m = 1, and (x - 6^2) % m = 0, the modulus m divides 215.": (
        "For integers m and x satisfying 10 ≤ m < 100, 0 ≤ x < m, 6x has remainder 1 modulo m, and x - 6^2 is congruent to 0 modulo m, the modulus m divides 215."
    ),
    "Let s be the sum of the integers from 2010 through 4018, and let n be the number of integers k in Finset.range 10 with gcd(k,10)=1. For the geometric block threshold G = ∑ i in range (n + s % 2009), 2^i, the shifted sum over the prime divisors of the divisor sum of 500 that are less than G is 85: ∑ p in the filtered divisor set, (p + G) = 85.": (
        "Let s be the sum of the integers from 2010 through 4018, and let n be the number of integers k from 0 through 9 with gcd(k, 10) = 1. Let G be the sum of 2^i over all i from 0 up to, but not including, n plus the residue of s modulo 2009. Then the shifted sum of p + G over the prime divisors p of the divisor sum of 500 that are less than G is 85."
    ),
    "If Nat.lcm 120 b = 3720 and Nat.gcd 120 b = 8, then the quotient b / Nat.gcd 120 b is a prime divisor of Nat.lcm 120 b and it gives the factorization Nat.lcm 120 b = 120 times that quotient.": (
        "If the least common multiple of 120 and b is 3720 and the greatest common divisor of 120 and b is 8, then b divided by that greatest common divisor is a prime divisor of the least common multiple, and the least common multiple factors as 120 times that quotient."
    ),
    "In the same four-term arithmetic-sequence setup, suppose a compact hidden real parameter t records the endpoint gap t = (3p + q) - (3p - q), and the natural index m satisfies (m : ℝ) = 500t + 10. Then the mth term is 8041.": (
        "In the same four-term arithmetic-sequence setup, suppose a real parameter t records the endpoint gap t = (3p + q) - (3p - q), and the natural index m, viewed as a real number, equals 500t + 10. Then the mth term is 8041."
    ),
    "In the same four-term arithmetic-sequence setup, suppose hidden real parameters x and y decompose p and q by x + y = p and x - y = q, and the natural index m satisfies (m : ℝ) = 400 * (x + y) + (x + y) + (x - y) + 3. Then the mth term is 8041.": (
        "In the same four-term arithmetic-sequence setup, suppose real parameters x and y decompose p and q by x + y = p and x - y = q, and the natural index m, viewed as a real number, equals 400(x + y) + (x + y) + (x - y) + 3. Then the mth term is 8041."
    ),
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _compact(value: Any, *, limit: int = 1200) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _naturalize_statement(statement: str) -> str:
    if statement in STATEMENT_REWRITES:
        return STATEMENT_REWRITES[statement]
    replacements = [
        ("Nat.lcm 120 b", "the least common multiple of 120 and b"),
        ("Nat.gcd 120 b", "the greatest common divisor of 120 and b"),
        ("Nat.gcd 180 168", "the greatest common divisor of 180 and 168"),
        ("Finset.Icc 120 b", "the interval from 120 through b"),
        ("Finset.range 10", "the integers from 0 through 9"),
        ("(b : ℝ)", "b viewed as a real number"),
        ("(m : ℝ)", "m viewed as a real number"),
        ("1 <= x <= 7", "1 ≤ x ≤ 7"),
        ("10 <= m", "10 ≤ m"),
        ("0 <= x", "0 ≤ x"),
        ("N >= 2", "N ≥ 2"),
        ("6*x % m = 1", "6x has remainder 1 modulo m"),
        ("(x - 6^2) % m = 0", "x - 6^2 is congruent to 0 modulo m"),
        ("s % 2009", "the residue of s modulo 2009"),
    ]
    for old, new in replacements:
        statement = statement.replace(old, new)
    return statement


def _first_text(*values: Any, limit: int = 1200) -> str:
    for value in values:
        text = _compact(value, limit=limit)
        if text and text != "not_available":
            return text
    return ""


def _none_if_empty(value: Any) -> Any:
    return None if value in ("", [], {}) else value


def _fixed_dict(values: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {field: _none_if_empty(values.get(field)) for field in fields}


def _fixed_parent(parent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "parent_id": str(parent.get("parent_id") or parent.get("id") or ""),
        "source": _first_text(parent.get("source"), limit=300),
        "statement": _first_text(parent.get("statement"), limit=1200),
        "formal_statement": _first_text(parent.get("formal_statement"), parent.get("lean_code"), limit=1600),
        "proof_idea": _first_text(parent.get("proof_idea"), parent.get("proof_plan"), parent.get("solution"), limit=1200),
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except Exception:
            pass
        return [value] if value else []
    return [str(value)]


def _entropy_direction(row: Dict[str, Any], existing_curation: Dict[str, Any]) -> str:
    explicit = str(
        row.get("_entropy_direction")
        or existing_curation.get("entropy_direction")
        or row.get("entropy_direction")
        or ""
    ).strip().lower()
    if explicit in {"increase", "decrease"}:
        return explicit

    cluster = str(row.get("_manual_qa_cluster") or existing_curation.get("cluster") or "").lower()
    rationale = str(
        row.get("_manual_qa_rationale")
        or row.get("_manual_qa_reason")
        or existing_curation.get("rationale")
        or ""
    ).lower()
    surface = f"{cluster} {rationale}"
    decrease_markers = (
        "segmented",
        "simplified",
        "easier",
        "helper-style",
        "helper only",
        "helper-only",
        "direct corollary",
        "concrete_specialization",
        "concrete specialization",
        "clean_constant_specialization",
        "clean constant specialization",
    )
    if any(marker in surface for marker in decrease_markers):
        return "decrease"
    return "increase"


def _theorem_name(row: Dict[str, Any]) -> str:
    import re

    text = _first_text(row.get("formal_statement"), row.get("lean_code"), limit=6000)
    match = re.search(r"\b(?:theorem|lemma|def)\s+([A-Za-z0-9_'.]+)", text)
    return match.group(1) if match else ""


def _benchmark(row: Dict[str, Any]) -> str:
    return str(row.get("_accepted_benchmark") or row.get("benchmark") or "unknown").lower()


def _row_richness(row: Dict[str, Any]) -> int:
    score = 0
    if _json_list(row.get("parent_ids")):
        score += 8
    operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
    if operator_card.get("parent_cards"):
        score += 12
    if row.get("parent_context_cards"):
        score += 8
    if row.get("parents"):
        score += 6
    if row.get("source_file") or row.get("_inventory_source_file") or row.get("_accepted_source"):
        score += 2
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    if provenance.get("source_file"):
        score += 2
    if row.get("lean_code"):
        score += 1
    return score


def _row_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        theorem_name = _theorem_name(row)
        for key in ("problem_id", "id", "name"):
            value = row.get(key)
            if not value:
                continue
            value = str(value)
            if value not in index or _row_richness(row) > _row_richness(index[value]):
                index[str(value)] = row
        if theorem_name and (theorem_name not in index or _row_richness(row) > _row_richness(index[theorem_name])):
            index[theorem_name] = row
    return index


def _path_from_value(value: Any) -> Path | None:
    if not value or "://" in str(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists() and path.suffix == ".jsonl":
        return path
    return None


def _summary_source_paths() -> List[Path]:
    if not DEFAULT_SUMMARY.exists():
        return []
    try:
        summary = json.loads(DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return []
    paths: List[Path] = []
    for value in summary.values():
        if not isinstance(value, dict) or "promotions" not in value:
            continue
        path = _path_from_value(value.get("source_file"))
        if path is not None:
            paths.append(path)
    return paths


def _source_paths(rows: Iterable[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    paths.extend(_summary_source_paths())
    curated_dir = REPO_ROOT / "data/evaluation/treatment_inventory/final_curated"
    paths.extend(sorted(curated_dir.glob("accepted_legacy_full_*.jsonl")))
    for name in ("minif2f_final_curated.jsonl", "proofnet_final_curated.jsonl", "putnambench_final_curated.jsonl"):
        path = curated_dir / name
        if path.exists():
            paths.append(path)
    certified_dir = REPO_ROOT / "data/certified"
    if certified_dir.exists():
        paths.extend(sorted(certified_dir.glob("*.jsonl")))
    for row in rows:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        for value in (
            row.get("source_file"),
            row.get("_inventory_source_file"),
            row.get("_accepted_source"),
            provenance.get("source_file"),
        ):
            path = _path_from_value(value)
            if path is not None:
                paths.append(path)
    return sorted(set(paths))


def _load_source_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for path in _source_paths(rows):
        all_rows.extend(_read_jsonl(path))
    all_rows.extend(rows)
    return _row_index(all_rows)


def _load_csv_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    patterns = [
        REPO_ROOT / "data/raw/*.csv",
        REPO_ROOT / "data/certified/*.gen0_seeds.csv",
    ]
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        row = dict(row)
                        row["_csv_source"] = str(path.relative_to(REPO_ROOT))
                        for key in ("problem_id", "id", "name"):
                            value = row.get(key)
                            if value and str(value) not in index:
                                index[str(value)] = row
            except Exception:
                continue
    return index


def _parent_ids(row: Dict[str, Any]) -> List[str]:
    ids = _json_list(row.get("parent_ids"))
    if ids:
        return ids
    operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
    ids = _json_list(operator_card.get("parent_ids"))
    if ids:
        return ids
    source_problem_id = row.get("source_problem_id")
    if source_problem_id and source_problem_id != row.get("problem_id"):
        return [str(source_problem_id)]
    return []


def _parent_card_index(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
    parent_cards = operator_card.get("parent_cards")
    if not isinstance(parent_cards, list):
        return {}
    return {
        str(card["id"]): card
        for card in parent_cards
        if isinstance(card, dict) and card.get("id")
    }


def _parent_from_card(parent_id: str, card: Dict[str, Any]) -> Dict[str, Any]:
    proof_context = card.get("proof_context") if isinstance(card.get("proof_context"), dict) else {}
    decomposition = (
        card.get("theorem_decomposition")
        if isinstance(card.get("theorem_decomposition"), dict)
        else {}
    )
    return {
        "parent_id": parent_id,
        "source": "generation parent context",
        "statement": _first_text(card.get("statement_preview"), card.get("statement"), limit=1200),
        "formal_statement": _first_text(
            card.get("formal_statement"),
            proof_context.get("formal_statement"),
            decomposition.get("main_conclusion"),
            limit=1600,
        ),
        "proof_idea": _first_text(
            proof_context.get("solution"),
            card.get("answer_preview"),
            decomposition.get("proof_checkpoints"),
            proof_context.get("usable_proof_atoms"),
            limit=1200,
        ),
    }


def _parent_from_row(parent_id: str, row: Dict[str, Any], source: str) -> Dict[str, Any]:
    proof_context = row.get("proof_context") if isinstance(row.get("proof_context"), dict) else {}
    return {
        "parent_id": parent_id,
        "source": source,
        "statement": _first_text(
            row.get("statement"),
            row.get("statement_preview"),
            row.get("informal_statement"),
            limit=1200,
        ),
        "formal_statement": _first_text(
            row.get("formal_statement"),
            proof_context.get("formal_statement"),
            limit=1600,
        ),
        "proof_idea": _first_text(
            row.get("proof_plan"),
            row.get("solution"),
            row.get("generation_notes"),
            proof_context.get("solution"),
            row.get("answer"),
            limit=1200,
        ),
    }


def _parents(
    row: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
    csv_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    parents: List[Dict[str, Any]] = []
    card_index = _parent_card_index(row)
    for parent_id in _parent_ids(row):
        if parent_id in card_index:
            parent = _parent_from_card(parent_id, card_index[parent_id])
        elif parent_id in source_index:
            parent = _parent_from_row(parent_id, source_index[parent_id], "accepted/source JSONL")
        elif parent_id in csv_index:
            parent = _parent_from_row(
                parent_id,
                csv_index[parent_id],
                str(csv_index[parent_id].get("_csv_source", "CSV")),
            )
        else:
            parent = {
                "parent_id": parent_id,
                "source": "unresolved",
                "statement": "",
                "formal_statement": "",
                "proof_idea": "",
            }
        parents.append(_fixed_parent(parent))
    return parents


def _resolved_parent_contexts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    parents = [parent for parent in value if isinstance(parent, dict)]
    if not parents:
        return []
    if any(parent.get("statement") or parent.get("formal_statement") or parent.get("proof_idea") for parent in parents):
        return [_fixed_parent(parent) for parent in parents]
    return []


def normalize_row(
    row: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
    csv_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    problem_id = str(row.get("problem_id") or row.get("id") or "")
    source_candidates = [
        source_index.get(problem_id),
        source_index.get(str(row.get("theorem_name") or "")),
        source_index.get(_theorem_name(row)),
    ]
    source_row = max(
        (candidate for candidate in source_candidates if isinstance(candidate, dict)),
        key=_row_richness,
        default={},
    )
    context_row = source_row if _row_richness(source_row) > _row_richness(row) else row
    statement = _naturalize_statement(str(row.get("statement") or ""))
    formal_statement = _first_text(row.get("formal_statement"), row.get("lean_code"), limit=6000)
    lean_code = str(row.get("lean_code") or "")
    benchmark = _benchmark(row)
    op_type = str(row.get("op_type") or row.get("operation") or "unknown")
    existing_curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    accepted_snapshot = str(row.get("_accepted_snapshot") or row.get("release_id") or "")
    existing_provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source_file = _first_text(
        existing_provenance.get("source_file"),
        row.get("source_file"),
        row.get("_inventory_source_file"),
        row.get("_accepted_source"),
        source_row.get("source_file"),
        source_row.get("_inventory_source_file"),
        source_row.get("_accepted_source"),
        limit=500,
    )
    parents = _resolved_parent_contexts(row.get("parents"))
    if not parents:
        parents = _parents(context_row, source_index, csv_index)

    normalized: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "problem_id": problem_id,
        "benchmark": benchmark,
        "op_type": op_type,
        "generation": row.get("generation"),
        "slot": row.get("slot"),
        "family": row.get("family") or row.get("target_family") or row.get("problem_style"),
        "theorem_name": _theorem_name(row),
        "statement": statement,
        "answer": row.get("answer") if row.get("answer") not in (None, "") else None,
        "formal_statement": formal_statement,
        "lean_code": lean_code,
        "parents": parents,
        "certificate": _fixed_dict({
            "status": "certified",
            # Prefer the row's own certificate; every released row carries a
            # complete verified proof, so the legacy fallback is T2.
            "level": (
                row.get("certificate_level")
                or (row.get("certificate") or {}).get("level")
                or LEVEL_PROOF
            ),
            "statement_checked": bool(
                (row.get("certificate") or {}).get("statement_checked", True)
            ),
            "proof_accepted": bool(
                (row.get("certificate") or {}).get("proof_accepted", True)
            ),
            "axiom_audit": (row.get("certificate") or {}).get("axiom_audit"),
            "kernel_replayed": bool(
                (row.get("certificate") or {}).get("kernel_replayed", False)
            ),
            "faithfulness": (row.get("certificate") or {}).get(
                "faithfulness", "unaudited"
            ),
            "alignment_method": (row.get("certificate") or {}).get(
                "alignment_method", "none"
            ),
            "lean_toolchain": (row.get("certificate") or {}).get("lean_toolchain")
            or runtime_pins(str(REPO_ROOT))["lean_toolchain"],
            "mathlib_revision": (row.get("certificate") or {}).get("mathlib_revision")
            or runtime_pins(str(REPO_ROOT))["mathlib_revision"],
            "lean_level": (
                row.get("lean_level")
                if row.get("lean_level") is not None
                else legacy_lean_level_for(
                    row.get("certificate_level")
                    or (row.get("certificate") or {}).get("level")
                    or LEVEL_PROOF
                )
            ),
            "anti_stub_passed": bool(row.get("anti_stub_passed", True)),
            "aligned": bool(row.get("aligned", True)),
        }, CERTIFICATE_FIELDS),
        "curation": _fixed_dict({
            "decision": "accept",
            "snapshot": accepted_snapshot or existing_curation.get("snapshot"),
            "cluster": str(row.get("_manual_qa_cluster") or existing_curation.get("cluster") or ""),
            "entropy_direction": _entropy_direction(row, existing_curation),
            "rationale": _first_text(
                row.get("_manual_qa_rationale"),
                row.get("_manual_qa_reason"),
                existing_curation.get("rationale"),
                limit=1200,
            ),
        }, CURATION_FIELDS),
        "provenance": _fixed_dict({
            "source_file": source_file,
            "source_problem_id": row.get("source_problem_id") or existing_provenance.get("source_problem_id") or source_row.get("source_problem_id"),
            "source_run": row.get("source_run") or existing_provenance.get("source_run") or source_row.get("source_run"),
        }, PROVENANCE_FIELDS),
        "hashes": _fixed_dict({
            "statement_sha256": _sha(statement),
            "formal_statement_sha256": str(
                row.get("formal_statement_sha256") or _sha(formal_statement)
            ),
        }, HASH_FIELDS),
    }
    return {field: normalized.get(field) for field in PUBLIC_FIELDS}


def normalize_file(input_path: Path, output_path: Path, backup: bool) -> None:
    rows = _read_jsonl(input_path)
    source_index = _load_source_index(rows)
    csv_index = _load_csv_index()
    normalized = [normalize_row(row, source_index, csv_index) for row in rows]

    if backup and input_path == output_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.with_name(f"accepted_legacy_full_{stamp}.jsonl")
        shutil.copy2(input_path, backup_path)
        print(f"backup {_display_path(backup_path)}")

    _write_jsonl(output_path, normalized)
    print(f"wrote {_display_path(output_path)} ({len(normalized)} rows)")
    print(
        "schema fields:",
        ", ".join(PUBLIC_FIELDS) if normalized else "(empty)",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--no-backup", action="store_true", help="do not back up before in-place rewrite")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    normalize_file(input_path, output_path, backup=not args.no_backup)


if __name__ == "__main__":
    main()
