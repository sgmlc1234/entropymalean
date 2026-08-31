"""Template-only Lean certification for existing CSV problem rows."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import langsmith as ls
from pydantic import BaseModel, Field

from src.utils.lean_interface import LeanChecker
from src.utils.lean_templates import detect_family, generate_lean_template, is_trivial_stub


class CertificationInput(BaseModel):
    """Minimal problem input needed for Lean certification."""

    id: str
    statement: str
    answer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CertificationResult(BaseModel):
    """One JSONL row emitted by the certification workflow."""

    problem_id: str
    release_id: Optional[str] = None
    source_problem_id: Optional[str] = None
    source_run: Optional[str] = None
    source_file: Optional[str] = None
    source_slot: Optional[str] = None
    generation: int = 0
    slot: Optional[int] = None
    operation: Optional[str] = None
    op_type: Optional[str] = None
    operator_variant: Optional[str] = None
    planned_op_type: Optional[str] = None
    planned_operator_variant: Optional[str] = None
    attempted_op_types: List[str] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)
    ancestor_ids: Any = None
    variation_axis: Optional[str] = None
    target_family: Optional[str] = None
    required_params: Dict[str, Any] = Field(default_factory=dict)
    generated_params: Dict[str, Any] = Field(default_factory=dict)
    composition_pattern: Optional[str] = None
    parent_contributions: Dict[str, str] = Field(default_factory=dict)
    avoid_patterns: List[str] = Field(default_factory=list)
    quality_target: Optional[str] = None
    quality_verdict: Optional[str] = None
    quality_flags: List[str] = Field(default_factory=list)
    interestingness_score: float = 0.0
    feedback_for_next_generation: Optional[str] = None
    reasoning_pattern: Optional[str] = None
    solution_skeleton: Dict[str, Any] = Field(default_factory=dict)
    projected_params: Dict[str, Any] = Field(default_factory=dict)
    projection_check: Dict[str, Any] = Field(default_factory=dict)
    semantic_parent_contribution: Dict[str, str] = Field(default_factory=dict)
    interestingness_features: Dict[str, Any] = Field(default_factory=dict)
    quality_evidence: Dict[str, Any] = Field(default_factory=dict)
    solution_verify_passed: Optional[bool] = None
    solution_verify_flags: List[str] = Field(default_factory=list)
    axis_applied: Optional[str] = None
    axis_aligned: bool = False
    planner_source: Optional[str] = None
    parent_eligible: bool = False
    selection_reason: Optional[str] = None
    canonical_signature: Optional[str] = None
    retry_count: int = 0
    retry_reasons: List[str] = Field(default_factory=list)
    quality_retry_count: int = 0
    retry_exhausted: bool = False
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list)
    replan_count: int = 0
    replan_reason: Optional[str] = None
    replan_source: Optional[str] = None
    discarded_operator_card: Dict[str, Any] = Field(default_factory=dict)
    survival_status: Optional[str] = None
    fallback_survivor_duplicate: bool = False
    failure_signature: Optional[str] = None
    source_kind: Optional[str] = None
    problem_style: Optional[str] = None
    target_style: Optional[str] = None
    certification_route: Optional[str] = None
    parent_context_cards: List[Dict[str, Any]] = Field(default_factory=list)
    operator_card: Dict[str, Any] = Field(default_factory=dict)
    slot_outcome: Optional[str] = None
    proof_plan: Optional[str] = None
    proof_obligations: List[str] = Field(default_factory=list)
    proof_verify_summary: Optional[str] = None
    gen0_failure_packet: Dict[str, Any] = Field(default_factory=dict)
    statement: Optional[str] = None
    statement_sha256: Optional[str] = None
    answer: Optional[str] = None
    answer_sha256: Optional[str] = None
    solution: Optional[str] = None
    verification_code: Optional[str] = None
    formal_statement: Optional[str] = None
    formal_statement_sha256: Optional[str] = None
    lean_header: Optional[str] = None
    formal_status: Optional[str] = None
    benchmark: Optional[str] = None
    family: Optional[str] = None
    status: str
    lean_level: int = 0  # legacy scale; frozen — new consumers read certificate
    certificate_level: str = "none"
    certificate: Dict[str, Any] = Field(default_factory=dict)
    lean_code: Optional[str] = None
    anti_stub_passed: bool = False
    aligned: bool = False
    difficulty: Optional[str] = None
    difficulty_label: Optional[str] = None
    generation_notes: Optional[str] = None
    llm_used: bool = False
    llm_model: Optional[str] = None
    lean_available: Optional[bool] = None
    lean_path: Optional[str] = None
    lean_version: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    input_metadata: Dict[str, Any] = Field(default_factory=dict)


@ls.traceable(name="tool.detect_family", run_type="tool")
def _detect_family(statement: str) -> Optional[str]:
    return detect_family(statement)


@ls.traceable(name="tool.generate_template", run_type="tool")
def _generate_template(statement: str, answer: str, family: Optional[str]):
    return generate_lean_template(statement, answer, family=family)


@ls.traceable(name="tool.anti_stub", run_type="tool")
def _anti_stub_passed(lean_code: str) -> bool:
    return not is_trivial_stub(lean_code)


def certify_problem(
    input: CertificationInput,
    checker: Optional[LeanChecker] = None,
    generate_harder: bool = False,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    generator: Optional[Any] = None,
) -> CertificationResult:
    """Synchronous public wrapper for certifying one problem."""
    return asyncio.run(
        certify_problem_async(
            input,
            checker=checker,
            generate_harder=generate_harder,
            generation_model=generation_model,
            generation_temperature=generation_temperature,
            generator=generator,
        )
    )


async def certify_problem_async(
    input: CertificationInput,
    checker: Optional[LeanChecker] = None,
    generate_harder: bool = False,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    generator: Optional[Any] = None,
) -> CertificationResult:
    """Certify one problem through the minimal LangGraph workflow."""
    from src.certification.graph import build_certification_graph

    graph = build_certification_graph(
        checker=checker,
        generate_harder=generate_harder,
        generation_model=generation_model,
        generation_temperature=generation_temperature,
        generator=generator,
    )
    final_state = await graph.ainvoke({"input": input, "errors": []})
    return final_state["result"]


def _row_to_input(row: Dict[str, Any]) -> CertificationInput:
    problem_id = str(row.get("id") or row.get("release_id") or "").strip()
    return CertificationInput(
        id=problem_id,
        statement=str(row.get("statement") or ""),
        answer=str(row.get("answer") or ""),
        metadata={k: v for k, v in row.items() if k not in {"id", "statement", "answer"}},
    )


async def certify_csv_async(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    checker: Optional[LeanChecker] = None,
    generate_harder: bool = False,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    generator: Optional[Any] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[CertificationResult]:
    """Certify rows from a v1-style CSV and write one JSON object per line."""
    from src.certification.graph import build_certification_graph

    checker = checker or LeanChecker()
    graph = build_certification_graph(
        checker=checker,
        generate_harder=generate_harder,
        generation_model=generation_model,
        generation_temperature=generation_temperature,
        generator=generator,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)

    batch_name = run_name or (
        f"generate_and_certify_csv/{input_path.name}"
        if generate_harder
        else f"certify_csv/{input_path.name}"
    )
    batch_tags = ["lean-certification", "csv"] + list(tags or [])
    if generate_harder:
        batch_tags.append("llm-generation")
    batch_metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "limit": limit,
        "workflow": "llm_harder_generation_then_lean_l2_certification"
        if generate_harder
        else "lean_l2_certification",
        "graph": "certification_graph",
        "llm_used": generate_harder,
        "llm_model": generation_model,
        "generation": 1 if generate_harder else 0,
        **dict(metadata or {}),
    }
    project = project_name or os.getenv("LANGSMITH_PROJECT")

    results: List[CertificationResult] = []
    with ls.trace(
        name=batch_name,
        run_type="chain",
        inputs={"input_path": str(input_path), "limit": limit},
        project_name=project,
        tags=batch_tags,
        metadata=batch_metadata,
    ) as batch_run:
        with input_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    break
                cert_input = _row_to_input(row)
                final_state = await graph.ainvoke(
                    {"input": cert_input, "errors": []},
                    config={
                        "run_name": (
                            f"generate+certify/{cert_input.id}"
                            if generate_harder
                            else f"certify/{cert_input.id}"
                        ),
                        "tags": [
                            "certification-row",
                            "llm-generation" if generate_harder else "template-only",
                            f"row:{idx}",
                        ],
                        "metadata": {
                            "problem_id": cert_input.id,
                            "row_index": idx,
                            "input_path": str(input_path),
                            "llm_used": generate_harder,
                            "llm_model": generation_model,
                            "generation": 1 if generate_harder else 0,
                        },
                        "configurable": {
                            "thread_id": f"certify:{input_path.stem}:{idx}:{cert_input.id}",
                        },
                    },
                )
                result = final_state["result"]
                results.append(result)

        counts: Dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        batch_run.end(
            outputs={
                "total": len(results),
                "counts": counts,
                "output_path": str(output_path),
            }
        )

    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")

    return results


def certify_csv(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    checker: Optional[LeanChecker] = None,
    generate_harder: bool = False,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    generator: Optional[Any] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[CertificationResult]:
    """Synchronous public wrapper for CSV certification."""
    return asyncio.run(
        certify_csv_async(
            input_path,
            output_path,
            limit=limit,
            checker=checker,
            generate_harder=generate_harder,
            generation_model=generation_model,
            generation_temperature=generation_temperature,
            generator=generator,
            project_name=project_name,
            run_name=run_name,
            tags=tags,
            metadata=metadata,
        )
    )
