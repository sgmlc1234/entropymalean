"""Benchmark-aware dataset loaders for the EntropyMaLean evaluation arms.

A *cell* in the evaluation cube is `(benchmark, arm)`:
- benchmark: one of {miniF2F, putnambench, proofnet}
- arm:       control (seed) or treatment (quality-gated generated)

Each loader yields ``EvalRow`` records with the four fields needed by the
model runner and grader: ``problem_id, statement, gold_answer, generation``.
The generation field is only meaningful for treatment rows; controls report 0.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


BENCHMARKS = ("miniF2F", "putnambench", "proofnet")
ARMS = ("control", "treatment")


@dataclass(frozen=True)
class EvalRow:
    """One problem-row scheduled for the direct no-tool panel."""

    benchmark: str
    arm: str
    problem_id: str
    statement: str
    gold_answer: str
    generation: int = 0
    family: Optional[str] = None
    lean_level: Optional[int] = None
    # Optional Lean formalisation context (populated for miniF2F /
    # PutnamBench rows that ship a native Lean 4 theorem statement, and for
    # certified EMG-2 treatment rows that carry a rendered template).
    formal_statement: Optional[str] = None
    lean_header: Optional[str] = None
    formal_status: Optional[str] = None


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_csv(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for row in csv.DictReader(f):
            yield row


def lean_theorem_prefix(source: object) -> str:
    """Return the theorem/lemma prefix to ask the prover to complete.

    Accepted ledgers store both ``formal_statement`` and ``lean_code``. Some
    rows historically let a complete proof body leak into one of those fields
    (for example ``:= by native_decide``). Evaluation must present the target
    declaration only; otherwise treatment rows can be solved by replaying the
    released certificate instead of by the prover under test.
    """
    text = str(source or "").strip()
    if not text:
        return ""
    by_match = re.search(r":=\s*by\b", text)
    if by_match:
        return text[: by_match.end()].rstrip()
    assign_match = re.search(r":=", text)
    if assign_match:
        return text[: assign_match.end()].rstrip() + " by"
    return text


_LEAN_DECLARATION_RE = re.compile(
    r"(?m)^[ \t]*(?:theorem|lemma|example)\b"
)
_AUTO_IMPLICIT_FALSE_RE = re.compile(
    r"(?m)^[ \t]*set_option[ \t]+autoImplicit[ \t]+false\b"
)


def lean_header_from_source(source: object) -> str:
    """Recover trusted setup commands that precede the target declaration.

    Public treatment ledgers historically stored ``lean_code`` with imports,
    ``open`` commands, and scoped notation, but omitted the separate
    ``lean_header`` field. Dropping that prefix can make Lean reinterpret
    unresolved identifiers as implicit variables. Keep the entire trusted
    prefix before the first theorem-like declaration.
    """
    text = str(source or "").strip()
    if not text:
        return ""
    declaration = _LEAN_DECLARATION_RE.search(text)
    if declaration is None:
        return ""
    return text[: declaration.start()].rstrip()


def ensure_auto_implicit_false(header: object) -> str:
    """Make unknown identifiers fail instead of becoming implicit variables."""
    text = str(header or "").strip()
    if _AUTO_IMPLICIT_FALSE_RE.search(text):
        return text
    strict = "set_option autoImplicit false"
    return f"{text}\n{strict}" if text else strict


def load_treatment_rows(jsonl_path: Path, benchmark: str) -> List[EvalRow]:
    """Load quality-gated treatment rows from an EntropyMaLean certified JSONL.

    The JSONL is the output of ``run_pool_generation`` or ``certify_csv``.
    A row is kept iff:
      - ``status == "certified"``,
      - ``statement`` is non-empty,
      - the row carries a Lean prefix (``lean_code`` or ``formal_statement``)
        — required by the EntropyMaLean proof protocol.
    A non-empty ``answer`` is *not* required: proof benchmarks (ProofNet,
    PutnamBench, and proof-style miniF2F entries) have no numeric answer
    by design, and the historical answer-recall filter silently dropped
    them. The fallback ``gold_answer`` value is the empty string when
    absent.
    """
    rows: List[EvalRow] = []
    for record in _read_jsonl(jsonl_path):
        row_benchmark = record.get("benchmark")
        if row_benchmark and str(row_benchmark).lower() != benchmark.lower():
            continue
        # Two accept paths:
        # 1. Original cert pipeline: ``status == "certified"``.
        # 2. Manual QA recall pipeline (Tier-1/Tier-2 promotion from the
        #    reject pool): rows lack a ``status`` field but carry
        #    ``_manual_qa_decision == "accept"`` set by the curator. We
        #    treat both as accepted treatment.
        certificate = record.get("certificate") if isinstance(record.get("certificate"), dict) else {}
        is_certified = record.get("status") == "certified" or certificate.get("status") == "certified"
        is_manual_accept = (record.get("_manual_qa_decision") or "").lower() == "accept"
        if not (is_certified or is_manual_accept):
            continue
        statement = (record.get("statement") or "").strip()
        if not statement:
            continue
        lean_code = (record.get("lean_code") or "").strip()
        formal_src = (record.get("formal_statement") or lean_code).strip()
        formal_stmt = lean_theorem_prefix(formal_src)
        if not formal_stmt:
            continue
        lean_header = (
            (record.get("lean_header") or "").strip()
            or lean_header_from_source(lean_code)
        )
        answer = (record.get("answer") or "").strip()
        rows.append(
            EvalRow(
                benchmark=benchmark,
                arm="treatment",
                problem_id=str(
                    record.get("eval_problem_id")
                    or record.get("problem_id")
                    or record.get("id")
                ),
                statement=statement,
                gold_answer=answer,
                generation=int(record.get("generation") or 0),
                family=record.get("family"),
                lean_level=record.get("lean_level") or certificate.get("lean_level"),
                formal_statement=formal_stmt or None,
                lean_header=lean_header or None,
                formal_status=("certified" if formal_stmt else "none"),
            )
        )
    return rows


def load_control_rows(csv_path: Path, benchmark: str) -> List[EvalRow]:
    """Load control-arm seed rows from a CSV.

    Accepts the extended EMG-2 pilot schema
    (``id, statement, answer, formal_statement, lean_header, formal_status, ...``)
    and falls back to the legacy EntropyMaG-1 schema when the Lean columns are
    absent.
    """
    rows: List[EvalRow] = []
    for record in _read_csv(csv_path):
        statement = (record.get("statement") or "").strip()
        answer = (record.get("answer") or "").strip()
        if not statement or not answer:
            continue
        rid = record.get("id") or record.get("release_id") or record.get("problem_id")
        formal_stmt = lean_theorem_prefix(record.get("formal_statement") or "")
        rows.append(
            EvalRow(
                benchmark=benchmark,
                arm="control",
                problem_id=str(rid),
                statement=statement,
                gold_answer=answer,
                generation=0,
                formal_statement=formal_stmt or None,
                lean_header=(record.get("lean_header") or None),
                formal_status=(record.get("formal_status") or None),
            )
        )
    return rows


def cap_rows(rows: List[EvalRow], cap: int) -> List[EvalRow]:
    """Deterministically dedup rows, then apply ``cap`` when positive.

    A non-positive cap means "keep all deduplicated rows"; this is useful for
    finalized accepted ledgers where the treatment arm should not silently stop
    at an older campaign cap.
    """
    seen: set[str] = set()
    out: List[EvalRow] = []
    for row in rows:
        if row.problem_id in seen:
            continue
        seen.add(row.problem_id)
        out.append(row)
        if cap > 0 and len(out) >= cap:
            break
    return out
