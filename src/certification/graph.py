"""Traceable LangGraph workflow for Lean certification."""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
import langsmith as ls

from src.certification.certifier import (
    CertificationInput,
    CertificationResult,
    _anti_stub_passed,
    _detect_family,
    _generate_template,
)
from src.certification.generation import (
    GeneratedProblem,
    GenerationConfig,
    PlannerContractError,
    default_generation_config,
    generate_harder_problem,
    validate_generated_contract,
)
from src.certification.levels import build_certificate_record
from src.utils.lean_interface import LeanChecker


ProblemGenerator = Callable[[CertificationInput, GenerationConfig], GeneratedProblem]


class CertificationState(TypedDict, total=False):
    """State for one problem certification run."""

    input: CertificationInput
    parent_input: CertificationInput
    start_time: float
    generated_problem: GeneratedProblem
    generation: int
    generation_error: Optional[str]
    difficulty_label: Optional[str]
    family: Optional[str]
    lean_code: Optional[str]
    theorem_name: Optional[str]
    proof_method: Optional[str]
    template_params: Dict[str, Any]
    anti_stub_passed: bool
    aligned: bool
    alignment_note: str
    lean_available: bool
    lean_path: Optional[str]
    lean_version: Optional[str]
    lean_ok: bool
    lean_error: Optional[str]
    llm_used: bool
    llm_model: Optional[str]
    result: CertificationResult
    errors: List[str]
    trace_metadata: Dict[str, Any]


def _elapsed(state: CertificationState) -> float:
    start = state.get("start_time") or time.time()
    return round(time.time() - start, 6)


def _metadata_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def _result(
    state: CertificationState,
    *,
    status: str,
    lean_level: int = 0,
    error: Optional[str] = None,
    certificate_level: str = "none",
    certificate: Optional[Dict[str, Any]] = None,
) -> CertificationResult:
    cert_input = state["input"]
    metadata = cert_input.metadata or {}
    lean_code = state.get("lean_code")
    return CertificationResult(
        certificate_level=certificate_level,
        certificate=dict(certificate or {}),
        problem_id=cert_input.id,
        release_id=metadata.get("release_id"),
        source_problem_id=metadata.get("source_problem_id"),
        source_run=metadata.get("source_run"),
        source_file=metadata.get("source_file"),
        source_slot=metadata.get("source_slot"),
        generation=int(metadata.get("generation") or state.get("generation") or 0),
        slot=metadata.get("slot"),
        operation=metadata.get("op_type") or metadata.get("operation"),
        op_type=metadata.get("op_type"),
        operator_variant=metadata.get("operator_variant"),
        parent_ids=[str(item) for item in _metadata_list(metadata.get("parent_ids"))],
        ancestor_ids=_metadata_list(metadata.get("ancestor_ids")),
        variation_axis=metadata.get("variation_axis"),
        target_family=metadata.get("target_family"),
        required_params=dict(metadata.get("required_params") or {}),
        generated_params=dict(metadata.get("generated_params") or {}),
        composition_pattern=metadata.get("composition_pattern"),
        parent_contributions=dict(metadata.get("parent_contributions") or {}),
        avoid_patterns=list(metadata.get("avoid_patterns") or []),
        quality_target=metadata.get("quality_target"),
        quality_verdict=metadata.get("quality_verdict"),
        quality_flags=list(metadata.get("quality_flags") or []),
        interestingness_score=float(metadata.get("interestingness_score") or 0.0),
        feedback_for_next_generation=metadata.get("feedback_for_next_generation"),
        reasoning_pattern=metadata.get("reasoning_pattern"),
        solution_skeleton=dict(metadata.get("solution_skeleton") or {}),
        projected_params=dict(metadata.get("projected_params") or {}),
        projection_check=dict(metadata.get("projection_check") or {}),
        semantic_parent_contribution=dict(metadata.get("semantic_parent_contribution") or {}),
        interestingness_features=dict(metadata.get("interestingness_features") or {}),
        quality_evidence=dict(metadata.get("quality_evidence") or {}),
        solution_verify_passed=metadata.get("solution_verify_passed"),
        solution_verify_flags=list(metadata.get("solution_verify_flags") or []),
        axis_applied=metadata.get("axis_applied"),
        axis_aligned=bool(metadata.get("axis_aligned", False)),
        planner_source=metadata.get("planner_source"),
        parent_eligible=bool(metadata.get("parent_eligible", False)),
        selection_reason=metadata.get("selection_reason"),
        canonical_signature=metadata.get("canonical_signature"),
        retry_count=int(metadata.get("retry_count") or 0),
        retry_reasons=list(metadata.get("retry_reasons") or []),
        quality_retry_count=int(metadata.get("quality_retry_count") or 0),
        retry_exhausted=bool(metadata.get("retry_exhausted", False)),
        attempt_history=list(metadata.get("attempt_history") or []),
        failure_signature=metadata.get("failure_signature"),
        source_kind=metadata.get("source_kind"),
        problem_style=metadata.get("problem_style"),
        target_style=metadata.get("target_style"),
        certification_route=metadata.get("certification_route"),
        parent_context_cards=list(metadata.get("parent_context_cards") or []),
        operator_card=dict(metadata.get("operator_card") or {}),
        slot_outcome=metadata.get("slot_outcome"),
        proof_plan=metadata.get("proof_plan"),
        proof_obligations=list(metadata.get("proof_obligations") or []),
        proof_verify_summary=metadata.get("proof_verify_summary"),
        statement=cert_input.statement,
        statement_sha256=metadata.get("statement_sha256"),
        answer=cert_input.answer,
        answer_sha256=metadata.get("answer_sha256"),
        solution=metadata.get("solution"),
        verification_code=metadata.get("verification_code"),
        formal_statement=lean_code or metadata.get("formal_statement"),
        formal_statement_sha256=metadata.get("formal_statement_sha256"),
        lean_header=metadata.get("lean_header"),
        formal_status=("certified" if lean_code else metadata.get("formal_status")),
        benchmark=metadata.get("benchmark"),
        family=state.get("family"),
        status=status,
        lean_level=lean_level,
        lean_code=lean_code,
        anti_stub_passed=bool(state.get("anti_stub_passed")),
        aligned=bool(state.get("aligned")),
        difficulty=metadata.get("difficulty"),
        difficulty_label=metadata.get("difficulty_label") or state.get("difficulty_label"),
        generation_notes=metadata.get("generation_notes"),
        llm_used=bool(state.get("llm_used", False)),
        llm_model=state.get("llm_model"),
        lean_available=state.get("lean_available"),
        lean_path=state.get("lean_path"),
        lean_version=state.get("lean_version"),
        error=error,
        elapsed_seconds=_elapsed(state),
        input_metadata=dict(metadata),
    )


def _annotate_current_run(
    state: CertificationState, extra: Optional[Dict[str, Any]] = None
) -> None:
    run_tree = ls.get_current_run_tree()
    if run_tree is None:
        return
    result = state.get("result")
    metadata = {
        "problem_id": state["input"].id,
        "family": state.get("family"),
        "lean_level": getattr(result, "lean_level", None),
        "status": getattr(result, "status", None),
        "llm_used": bool(state.get("llm_used", False)),
        "llm_model": state.get("llm_model"),
        "source_problem_id": state["input"].metadata.get("source_problem_id"),
        "generation": state["input"].metadata.get("generation") or state.get("generation"),
        "slot": state["input"].metadata.get("slot"),
        "op_type": state["input"].metadata.get("op_type"),
        "operator_variant": state["input"].metadata.get("operator_variant"),
        "parent_ids": state["input"].metadata.get("parent_ids"),
        "variation_axis": state["input"].metadata.get("variation_axis"),
        "target_family": state["input"].metadata.get("target_family"),
        "required_params": state["input"].metadata.get("required_params"),
        "generated_params": state["input"].metadata.get("generated_params"),
        "composition_pattern": state["input"].metadata.get("composition_pattern"),
        "parent_contributions": state["input"].metadata.get("parent_contributions"),
        "quality_target": state["input"].metadata.get("quality_target"),
        "quality_verdict": state["input"].metadata.get("quality_verdict"),
        "quality_flags": state["input"].metadata.get("quality_flags"),
        "quality_evidence": state["input"].metadata.get("quality_evidence"),
        "interestingness_score": state["input"].metadata.get("interestingness_score"),
        "reasoning_pattern": state["input"].metadata.get("reasoning_pattern"),
        "projection_check": state["input"].metadata.get("projection_check"),
        "axis_applied": state["input"].metadata.get("axis_applied"),
        "axis_aligned": state["input"].metadata.get("axis_aligned"),
        "planner_source": state["input"].metadata.get("planner_source"),
        "difficulty_label": state["input"].metadata.get("difficulty_label")
        or state.get("difficulty_label"),
        **dict(state.get("trace_metadata") or {}),
    }
    if extra:
        metadata.update(extra)
    run_tree.metadata.update({k: v for k, v in metadata.items() if v is not None})


def build_certification_graph(
    checker: Optional[LeanChecker] = None,
    *,
    generate_harder: bool = False,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    generator: Optional[ProblemGenerator] = None,
):
    """Build the traceable certification graph.

    Flow:
      Optional generate_candidate -> detect_family -> template
      -> anti_stub -> lean_health -> lean_check -> END
    """
    lean_checker = checker or LeanChecker()
    generation_config = default_generation_config(
        model=generation_model,
        temperature=generation_temperature,
    )
    graph = StateGraph(CertificationState)

    async def generate_harder_node(state: CertificationState) -> CertificationState:
        parent = state["input"]
        generated: Optional[GeneratedProblem] = None
        try:
            if generator is None:
                generated = await generate_harder_problem(parent, config=generation_config)
            else:
                maybe_generated = generator(parent, generation_config)
                generated = (
                    await maybe_generated
                    if inspect.isawaitable(maybe_generated)
                    else maybe_generated
                )
            validate_generated_contract(parent, generated)
            generated = generated.model_copy(
                update={
                    "axis_aligned": True,
                    "axis_applied": generated.axis_applied
                    or parent.metadata.get("variation_axis", ""),
                }
            )
            generated_input = CertificationInput(
                id=generated.id,
                statement=generated.statement,
                answer=generated.answer,
                metadata={
                    **parent.metadata,
                    "source_problem_id": parent.id,
                    "source_statement": parent.statement,
                    "source_answer": parent.answer,
                    "generation": 1,
                    "difficulty_label": generated.difficulty_label,
                    "generation_notes": generated.harder_reason,
                    "solution": generated.solution,
                    "generated_params": generated.params,
                    "reasoning_pattern": generated.reasoning_pattern,
                    "solution_skeleton": generated.solution_skeleton,
                    "projected_params": generated.projected_params,
                    "projection_check": generated.projection_check,
                    "axis_applied": generated.axis_applied,
                    "axis_aligned": generated.axis_aligned,
                    "raw_llm_output": generated.raw_llm_output,
                    "operator_variant": parent.metadata.get("operator_variant"),
                },
            )
            next_state = {
                "parent_input": parent,
                "input": generated_input,
                "generated_problem": generated,
                "generation": 1,
                "difficulty_label": generated.difficulty_label,
                "family": generated.family,
                "llm_used": True,
                "llm_model": generation_config.model,
                "start_time": time.time(),
            }
            _annotate_current_run(
                {**state, **next_state},
                {
                    "node_output": "harder_problem_generated",
                    "source_problem_id": parent.id,
                    "generated_problem_id": generated.id,
                    "generated_family": generated.family,
                    "generated_params": generated.params,
                    "reasoning_pattern": generated.reasoning_pattern,
                    "projection_check": generated.projection_check,
                    "axis_aligned": generated.axis_aligned,
                    "llm_used": True,
                    "llm_model": generation_config.model,
                },
            )
            return next_state
        except Exception as exc:
            failed_input = parent
            if isinstance(exc, PlannerContractError) and generated is not None:
                failed_input = CertificationInput(
                    id=parent.id,
                    statement=parent.statement,
                    answer=parent.answer,
                    metadata={
                        **parent.metadata,
                        "generated_params": generated.params,
                        "reasoning_pattern": generated.reasoning_pattern,
                        "solution_skeleton": generated.solution_skeleton,
                        "projected_params": generated.projected_params,
                        "projection_check": generated.projection_check,
                        "axis_applied": generated.axis_applied,
                        "axis_aligned": generated.axis_aligned,
                    },
                )
            next_state = {
                "parent_input": parent,
                "input": failed_input,
                "generation": 1,
                "generation_error": str(exc),
                "llm_used": True,
                "llm_model": generation_config.model,
            }
            status = (
                "planner_axis_mismatch"
                if isinstance(exc, PlannerContractError)
                else "generation_failed"
            )
            result = _result(
                {**state, **next_state},
                status=status,
                error=str(exc)[:500],
            )
            next_state["result"] = result
            _annotate_current_run(
                {**state, **next_state},
                {"status": status, "generation_error": str(exc)[:250]},
            )
            return next_state

    def detect_family_node(state: CertificationState) -> CertificationState:
        family = state.get("family") or _detect_family(state["input"].statement)
        next_state = {
            "start_time": state.get("start_time") or time.time(),
            "family": family,
            "llm_used": bool(state.get("llm_used", False)),
            "llm_model": state.get("llm_model"),
        }
        _annotate_current_run(
            {**state, **next_state},
            {"family": family, "node_output": "family_detected"},
        )
        return next_state

    def generate_template_node(state: CertificationState) -> CertificationState:
        template = _generate_template(
            state["input"].statement,
            state["input"].answer,
            family=state.get("family"),
        )
        if template is None:
            result = _result(
                state,
                status="unsupported",
                error="No supported Lean template for this problem family",
            )
            next_state = {
                "lean_code": None,
                "theorem_name": None,
                "proof_method": None,
                "template_params": {},
                "result": result,
            }
            _annotate_current_run({**state, **next_state}, {"status": "unsupported"})
            return next_state
        next_state = {
            "family": template.family,
            "lean_code": template.lean_code,
            "theorem_name": template.theorem_name,
            "proof_method": template.proof_method,
            "template_params": template.params,
            "aligned": template.aligned,
            "alignment_note": template.alignment_note,
        }
        _annotate_current_run(
            {**state, **next_state},
            {
                "theorem_name": template.theorem_name,
                "proof_method": template.proof_method,
                "aligned": template.aligned,
            },
        )
        return next_state

    def anti_stub_check_node(state: CertificationState) -> CertificationState:
        passed = _anti_stub_passed(state.get("lean_code") or "")
        if not passed:
            result = _result(state, status="failed", error="Trivial Lean stub rejected")
            next_state = {"anti_stub_passed": False, "result": result}
            _annotate_current_run({**state, **next_state}, {"status": "failed"})
            return next_state
        return {"anti_stub_passed": True}

    async def lean_health_check_node(state: CertificationState) -> CertificationState:
        available = await lean_checker.health_check()
        lean_meta = {
            "lean_path": getattr(lean_checker, "lean_path", None),
            "lean_version": getattr(lean_checker, "lean_version", None),
        }
        if not available:
            next_state = {"lean_available": False, **lean_meta}
            result = _result(
                {**state, **next_state},
                status="lean_unavailable",
                error="Lean binary unavailable",
            )
            next_state["result"] = result
            _annotate_current_run({**state, **next_state}, {"status": "lean_unavailable"})
            return next_state
        next_state = {"lean_available": True, **lean_meta}
        _annotate_current_run({**state, **next_state}, {"lean_available": True})
        return next_state

    async def lean_type_check_node(state: CertificationState) -> CertificationState:
        ok, lean_output = await lean_checker.type_check(state.get("lean_code") or "")
        if not ok:
            result = _result(state, status="failed", error=lean_output[:500])
            next_state = {"lean_ok": False, "lean_error": lean_output[:500], "result": result}
            _annotate_current_run({**state, **next_state}, {"status": "failed"})
            return next_state

        lean_level = 3 if state.get("aligned") else 2
        result = _result(
            state,
            status="certified",
            lean_level=lean_level,
            certificate=build_certificate_record(
                statement_checked=True,
                proof_checked=True,
                # The numeric route type-checks the whole file but does not
                # run the strict autoImplicit=false statement probe.
                auto_implicit_false=False,
                proof_method="native_decide",
                faithfulness=(
                    "faithful" if state.get("aligned") else "unaudited"
                ),
                alignment_method=(
                    "template_param_match" if state.get("aligned") else "none"
                ),
                verifier="lean_cli",
            ),
        )
        next_state = {"lean_ok": True, "result": result}
        _annotate_current_run(
            {**state, **next_state},
            {"status": "certified", "lean_level": lean_level},
        )
        return next_state

    def route_after_template(state: CertificationState) -> str:
        return "end" if state.get("result") else "anti_stub_check"

    def route_after_anti_stub(state: CertificationState) -> str:
        return "end" if state.get("result") else "lean_health_check"

    def route_after_health(state: CertificationState) -> str:
        return "end" if state.get("result") else "lean_type_check"

    def route_after_generation(state: CertificationState) -> str:
        return "end" if state.get("result") else "detect_family"

    route_after_template.__name__ = "route_template"
    route_after_anti_stub.__name__ = "route_stub"
    route_after_health.__name__ = "route_lean"
    route_after_generation.__name__ = "route_generation"

    if generate_harder:
        graph.add_node("generate_candidate", generate_harder_node)
    graph.add_node("detect_family", detect_family_node)
    graph.add_node("template", generate_template_node)
    graph.add_node("anti_stub", anti_stub_check_node)
    graph.add_node("lean_health", lean_health_check_node)
    graph.add_node("lean_check", lean_type_check_node)

    if generate_harder:
        graph.set_entry_point("generate_candidate")
        graph.add_conditional_edges(
            "generate_candidate",
            route_after_generation,
            {"detect_family": "detect_family", "end": END},
        )
    else:
        graph.set_entry_point("detect_family")
    graph.add_edge("detect_family", "template")
    graph.add_conditional_edges(
        "template",
        route_after_template,
        {"anti_stub_check": "anti_stub", "end": END},
    )
    graph.add_conditional_edges(
        "anti_stub",
        route_after_anti_stub,
        {"lean_health_check": "lean_health", "end": END},
    )
    graph.add_conditional_edges(
        "lean_health",
        route_after_health,
        {"lean_type_check": "lean_check", "end": END},
    )
    graph.add_edge("lean_check", END)
    return graph.compile()
