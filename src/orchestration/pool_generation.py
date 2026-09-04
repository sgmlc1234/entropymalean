"""Generation-level 5-pool orchestrator built on LangGraph Send fan-out."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import operator
import os
import re
from functools import lru_cache
import subprocess
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict, Set

import langsmith as ls
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, field_validator, model_validator

from src.certification import CertificationInput, CertificationResult
from src.certification.alignment import elaborated_goal_alignment
from src.certification.levels import axiom_audit, build_certificate_record
from src.certification.generation import (
    orchestrator_config,
    verification_config,
    GeneratedProblem,
    GenerationConfig,
    _chat_completion_text_async,
    _chat_completion_text_sync,
    _openai_client,
    _openrouter_extra_body,
    _parse_json_object,
    _schema_response_format,
    _structured_schema_unsupported,
    default_generation_config,
)
from src.no_go_policy import (
    ACCEPTED_PROXY_SEVERE_FLAGS,
    PLANNER_MEMORY_LESSONS,
    QUALITY_RETRYABLE_NO_GO_FLAGS,
    RETRY_PATCH_INSTRUCTIONS,
    build_no_go_policy_pack,
    format_lessons,
    format_no_go_policy_pack,
    lessons_for_slot,
    planner_lessons,
)
from src.orchestration.quality import (
    QualityResult,
    derive_accepted_proxy,
    derive_curation_decision,
    derive_entropy_direction,
    derive_misformalization_taxonomy,
    verify_slot_quality,
)
from src.evaluation.lean_verifier import LeanVerifyResult, verify_lean_proof
from src.retrieval.jixia_memory import jixia_digest
from src.retrieval.leansearch import (
    DEFAULT_LIMIT as DEFAULT_LEANSEARCH_LIMIT,
    build_diagnostic_query,
    build_statement_query,
    format_premise_pack,
    max_queries_per_problem,
    retrieve_premise_pack,
    should_retrieve_for_diagnostics,
)
from src.retrieval.novelty_memory import (
    DEFAULT_ACCEPTED_LEDGER_PATH as DEFAULT_NOVELTY_ACCEPTED_LEDGER_PATH,
    build_memory_delta_contract,
    build_novelty_memory_pack,
    cards_from_rows,
    compact_card,
    evaluate_candidate_novelty,
    format_novelty_memory_pack,
)
from src.utils.codex_cli import call_codex_cli
from src.utils.lean_interface import LeanChecker
from src.utils.lean_templates import SUPPORTED_FAMILIES, detect_family


POOL_SIZE = 5
DEFAULT_CROSSOVER_COUNT = 2
DEFAULT_GEN0_MAX_PARALLEL = 3
GEN0_MAX_PARALLEL_CAP = 5
DEFAULT_PLANNER_MEMORY_LIMIT = 24
DEFAULT_LEANSEARCH_ENABLED = True
DEFAULT_PLANNER_CURATED_MEMORY_DIR = Path("data/curated")
DEFAULT_BENCH_VERSION = "emg2-dynamic-v1"
DEFAULT_TARGET_ACCEPTED_PER_GENERATION = 3
DEFAULT_RESERVE_BUDGET = 3
THEOREM_LINEAGE_CAP = 2
# autoImplicit must stay off for ALL theorem-route verification: with it on,
# an invented identifier (e.g. `quatCircle` in the 2026-07-28 smoke run)
# silently becomes an auto-bound implicit hypothesis and a vacuous statement
# certifies as a theorem.
THEOREM_CANONICAL_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 2000000\n"
    "set_option autoImplicit false"
)
_RUNTIME_RESULT_STORE: Dict[str, Any] = {}
FAMILY_PARAM_KEYS = {
    "gcd": {"a", "b"},
    "gcd_divisor_sum": {"a", "b"},
    "units_digit": {"base", "exp"},
    "divisor_sum": {"n"},
    "divisor_sum_mod": {"n", "a"},
    "stars_and_bars": {"vars", "sum"},
    "arithmetic_series": {"n_terms", "first", "diff"},
    "modular_congruence": {"a", "m"},
}
FAMILY_PARAM_RANGES = {
    "gcd": {"a": (2, 50000), "b": (2, 50000)},
    "gcd_divisor_sum": {"a": (2, 50000), "b": (2, 50000)},
    "units_digit": {"base": (2, 99), "exp": (2, 5000)},
    "divisor_sum": {"n": (2, 2000)},
    "divisor_sum_mod": {"n": (2, 2000), "a": (1, 1_000_000)},
    "stars_and_bars": {"vars": (2, 6), "sum": (1, 30)},
    "arithmetic_series": {"n_terms": (2, 100), "first": (0, 200), "diff": (1, 100)},
    "modular_congruence": {"a": (1, 1_000_000), "m": (2, 10000)},
}
FAMILY_CAP = 2
COMPOSITION_PATTERNS = {
    "mutation": {"parameter_shift", "structure_expansion"},
    "crossover": {"parameter_transfer", "family_bridge"},
}
FUSION_MECHANISMS = {
    "invariant_transplant",
    "goal_form_transplant",
    "obstruction_as_lemma",
    "witness_exchange",
    "parameter_coupling",
    "sequential_composition",
}
CHECKPOINT_IDS = {
    "reasoning_pattern",
    "solution_skeleton",
    "projected_params",
    "two_step_reasoning",
    "semantic_parent_contribution",
    "family_certified",
    "rich_factorization",
    "rich_gcd",
    "nontrivial_modular_reduction",
    "nontrivial_mod_remainder",
    "binomial_formula",
    "arithmetic_sum_formula",
    "numeric_answer_verified",
}
CHECKPOINT_ALIASES = {
    "params_projected": "projected_params",
    "parent_contribution": "semantic_parent_contribution",
    "lean_certified": "family_certified",
    "lean-certifiable family": "family_certified",
    "sigma computed via multiplicative formula": "rich_factorization",
    "sigma formula applied as product over prime powers": "rich_factorization",
    "binomial coefficient computed explicitly": "binomial_formula",
    "answer verified numerically": "numeric_answer_verified",
    "sum formula applied": "arithmetic_sum_formula",
    "closed-form sum formula": "arithmetic_sum_formula",
    "multi-step reduction shown explicitly": "nontrivial_modular_reduction",
}
SUPPORTED_FAMILY_NAMES = sorted(SUPPORTED_FAMILIES)
PROBLEM_STYLES = {"numeric_answer", "theorem_proof"}
CERTIFICATION_ROUTES = {"template_numeric", "theorem_prover"}


class PoolWorkItem(BaseModel):
    """One orchestrator-owned slot dispatch."""

    slot: int
    op_type: str
    operator_variant: str = ""
    parent_ids: List[str]
    parent_refs: List[int] = Field(default_factory=list)
    variation_axis: str = ""
    reasoning_goal: str = ""
    target_family: str = ""
    required_params: Dict[str, Any] = Field(default_factory=dict)
    composition_pattern: str = ""
    parent_contributions: Dict[str, str] = Field(default_factory=dict)
    avoid_patterns: List[str] = Field(default_factory=list)
    quality_target: str = ""
    operator_goal: str = ""
    required_checkpoints: List[str] = Field(default_factory=list)
    avoid_signatures: List[str] = Field(default_factory=list)
    fusion_contract: Dict[str, Any] = Field(default_factory=dict)
    target_style: str = "numeric_answer"
    parent_context_cards: List[Dict[str, Any]] = Field(default_factory=list)
    operator_card: Dict[str, Any] = Field(default_factory=dict)
    goal: str = ""
    constraints: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    fusion_goal: str = ""
    parent_roles: Dict[str, str] = Field(default_factory=dict)
    memory_delta_contract: Dict[str, Any] = Field(default_factory=dict)
    difficulty_label: str = "medium"
    planner_source: str = "deterministic_fallback"


class Gen0SeedWorkItem(BaseModel):
    """Central orchestrator-owned proof-completion assignment for one seed."""

    seed_id: str
    problem_style: str
    formal_statement: str = ""
    lean_header: str = ""
    proof_body_available_before: bool = False
    proof_k: int
    proof_turns: int
    model: str
    lane: int
    requested_parallelism: int
    effective_parallelism: int


class PlannerMemoryCard(BaseModel):
    """Compact cross-run lesson card for the orchestrator planner."""

    case_id: str = ""
    kind: str
    problem_style: str = ""
    certification_route: str = ""
    op_type: str = ""
    operator_variant: str = ""
    target_family: str = ""
    goal: str = ""
    reasoning_signature: str = ""
    status: str = ""
    quality_verdict: str = ""
    quality_flags: List[str] = Field(default_factory=list)
    failure_class: str = ""
    selection_reason: str = ""
    lesson: str = ""
    raw_surface: Dict[str, Any] = Field(default_factory=dict)
    source_run: str = ""
    source_problem_id: str = ""


class TheoremGeneratedProblem(BaseModel):
    """A theorem-style child problem generated from theorem parents."""

    id: str
    source_problem_id: str
    contract_status: str = "generated"
    failure_reason: str = ""
    statement: str
    formal_statement: str
    lean_header: str = THEOREM_CANONICAL_HEADER
    lean_code: str
    statement_chunks: List[str] = Field(default_factory=list)
    selected_parent_checkpoints: List[str] = Field(default_factory=list)
    allowed_statement_delta: str = "immediate_corollary"
    proof_surface: str = "inferred_or_unspecified"
    proof_plan: str = ""
    proof_obligations: List[str] = Field(default_factory=list)
    difficulty_label: str = "medium"
    harder_reason: str = ""
    parent_contribution_evidence: Dict[str, str] = Field(default_factory=dict)
    unified_obligation: str = ""
    why_not_conjunction: str = ""
    patch_target_fields: List[str] = Field(default_factory=list)
    must_not_change_fields: List[str] = Field(default_factory=list)
    patch_applied: str = ""
    #: Tactic blocks proving parent <-> child, required only for
    #: `mutation_silent`. Empty for every other variant, where equivalence to
    #: the parent would mean the child had failed.
    equivalence_forward: str = ""
    equivalence_backward: str = ""
    raw_llm_output: Dict[str, Any] = Field(default_factory=dict)


class TheoremAlignmentResult(BaseModel):
    """Verifier verdict for natural statement vs Lean theorem coverage."""

    aligned: bool
    verdict: str = "fail"
    supported_claims: List[str] = Field(default_factory=list)
    missing_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    field_patch_instructions: List[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _normalize_verdict_consistency(self) -> "TheoremAlignmentResult":
        verdict_text = str(self.verdict or "").strip().lower()
        if self.aligned and verdict_text not in {"pass", "passed", "aligned", "ok"}:
            self.verdict = "pass"
        elif not self.aligned and verdict_text in {"pass", "passed", "aligned", "ok"}:
            self.verdict = "fail"
        return self

    @field_validator(
        "supported_claims",
        "missing_claims",
        "unsupported_claims",
        "field_patch_instructions",
        mode="before",
    )
    @classmethod
    def _normalize_string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, dict):
            items: List[str] = []
            for key, item in value.items():
                if isinstance(item, (list, tuple)):
                    items.extend(str(part) for part in item if str(part).strip())
                elif str(item).strip():
                    items.append(f"{key}: {item}")
            return items
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return cls._normalize_string_list(parsed)
        return [str(value)]


class PoolRunResult(BaseModel):
    """Public result for one pool generation run."""

    run_name: str
    generation_count: int
    pool_size: int
    input_path: str
    output_path: str
    summary_output_path: Optional[str] = None
    planner: Dict[str, Any] = Field(default_factory=dict)
    run_manifest: Dict[str, Any] = Field(default_factory=dict)
    planner_memory: Dict[str, Any] = Field(default_factory=dict)
    planner_case_pack: Dict[str, Any] = Field(default_factory=dict)
    novelty_memory: Dict[str, Any] = Field(default_factory=dict)
    work_items: List[Dict[str, Any]] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    saved_pool: List[Dict[str, Any]] = Field(default_factory=list)
    failed_slots: List[Dict[str, Any]] = Field(default_factory=list)
    passed_results_count: int = 0
    cumulative_passed_results_count: int = 0
    #: rows produced this run but not written because an identical row was already on file
    deduplicated_on_write: int = 0
    generation_zero: Dict[str, Any] = Field(default_factory=dict)
    gen0_enriched_input_path: Optional[str] = None
    generation_feedback: Dict[str, Any] = Field(default_factory=dict)
    generations: List[Dict[str, Any]] = Field(default_factory=list)
    generation_save_status: str = "partial"
    completed_at: Optional[str] = None
    completed_at_compact: Optional[str] = None
    langsmith_trace_hint: Dict[str, Any] = Field(default_factory=dict)
    langsmith_upload: Dict[str, Any] = Field(default_factory=dict)


class PoolState(TypedDict, total=False):
    input_path: str
    output_path: str
    summary_output_path: Optional[str]
    run_name: str
    pool_size: int
    survivor_count: int
    crossover_count: int
    max_generations: int
    max_parallel: int
    max_retries: int
    gen0_proof_k: int
    gen0_proof_turns: int
    gen0_max_seed_seconds: float
    gen0_max_parallel: int
    target_accepted_per_generation: int
    reserve_budget: int
    disable_reserve_slots: bool
    generation_model: Optional[str]
    generation_temperature: Optional[float]
    generation_count: int
    current_generation: List[CertificationInput]
    work_items: List[Dict[str, Any]]
    planner: Dict[str, Any]
    slot_outputs: Annotated[List[Dict[str, Any]], operator.add]
    approved_candidates: List[Dict[str, Any]]
    failed_slots: List[Dict[str, Any]]
    results: List[CertificationResult]
    results_ref: str
    summary: Dict[str, Any]
    run_manifest: Dict[str, Any]
    generation_feedback: Dict[str, Any]
    generation_survival_status: str
    plan_outcome_cards: List[Dict[str, Any]]
    generations: List[Dict[str, Any]]
    all_passed_results: List[Dict[str, Any]]
    all_passed_results_count: int
    generation_zero: Dict[str, Any]
    gen0_enriched_input_path: str
    lean_available: bool
    trace_metadata: Dict[str, Any]
    planner_memory_dir: str
    planner_memory_limit: int
    disable_planner_memory: bool
    leansearch_enabled: bool
    leansearch_limit: int
    planner_memory: Dict[str, Any]
    planner_case_pack: Dict[str, Any]
    novelty_memory: Dict[str, Any]
    reserve_work_items: List[Dict[str, Any]]
    reserve_round_pending: bool
    reserve_round_done: bool
    dispatch_mode: str


PlannerFn = Callable[[List[CertificationInput], Dict[str, Any]], Dict[str, Any]]
SlotGeneratorFn = Callable[[CertificationInput, GenerationConfig], GeneratedProblem]
TheoremGeneratorFn = Callable[[CertificationInput, GenerationConfig], TheoremGeneratedProblem]
TheoremVerifierFn = Callable[..., Any]
TheoremAlignmentVerifierFn = Callable[..., Any]
ReplannerFn = Callable[[Dict[str, Any], CertificationResult, List[Dict[str, Any]], GenerationConfig], Any]
SeedProofCompleterFn = Callable[[CertificationInput, GenerationConfig], Any]
AggregateSelectorFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def row_to_input(row: Dict[str, Any]) -> CertificationInput:
    problem_id = str(row.get("id") or row.get("release_id") or "").strip()
    return CertificationInput(
        id=problem_id,
        statement=str(row.get("statement") or ""),
        answer=str(row.get("answer") or ""),
        metadata={k: v for k, v in row.items() if k not in {"id", "statement", "answer"}},
    )


def _benchmark_from_path(path: Path) -> str:
    name = path.name.lower()
    for benchmark in ("proofnet", "minif2f", "putnambench"):
        if benchmark in name:
            return benchmark
    return ""


@lru_cache(maxsize=1)
def _benchmark_informal_statements() -> Dict[str, str]:
    """The real informal statement for each seed, keyed by its theorem name.

    The seed CSVs the pipeline reads carry `Prove the theorem <id>.` in their
    `statement` column for all 100 seeds — a stand-in, not a description. The
    actual prose lives in the benchmark sheets under `goal`, keyed by a sheet id
    (`proofnet-3`) rather than by the theorem name, which is why the two were
    never joined.

    The cost of the gap is not cosmetic. A worker breeding from a seed was shown
    the parent's Lean and the sentence "Prove the theorem
    Artin_exercise_2_11_3.", so it never saw what a benchmark problem sounds
    like: short, question-shaped, `$-delimited math. It wrote what it could —
    Lean read back as one long English sentence — and 0 of 146 released
    statements carry any mathematical notation while 98 of their parents do.
    """
    out: Dict[str, str] = {}
    root = _repo_root() / "data" / "benchmarks"
    if not root.is_dir():
        return out
    for sheet in sorted(root.glob("*/seeds_50_levels.csv")):
        try:
            with sheet.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    prose = str(row.get("goal") or "").strip()
                    match = re.search(
                        r"\b(?:theorem|lemma)\s+([^\s({\[:]+)",
                        f"{row.get('lean_goal') or ''} {row.get('solution') or ''}",
                    )
                    if match and prose:
                        out[match.group(1)] = prose
        except OSError:
            continue
    return out


def load_seed_inputs(input_path: Path, *, pool_size: int = POOL_SIZE) -> List[CertificationInput]:
    input_path = Path(input_path)
    inferred_benchmark = _benchmark_from_path(input_path)
    with input_path.open(newline="", encoding="utf-8") as fh:
        rows = [row_to_input(row) for row in csv.DictReader(fh)]
    prose = _benchmark_informal_statements()
    recovered = 0
    for row in rows:
        real = prose.get(row.id)
        if real and str(row.statement or "").strip().startswith("Prove the theorem"):
            row.statement = real
            recovered += 1
    if recovered:
        print(f"[seeds] informal statement recovered for {recovered}/{len(rows)} seed(s)")
    if inferred_benchmark:
        for row in rows:
            row.metadata.setdefault("benchmark", inferred_benchmark)
    if len(rows) < pool_size:
        raise ValueError(
            f"Pool generation requires at least {pool_size} seed rows; got {len(rows)}"
        )
    return rows[:pool_size]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text_if_present(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _mathlib_rev(repo_root: Path) -> str:
    try:
        manifest = json.loads((repo_root / "lake-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    packages = manifest.get("packages") if isinstance(manifest, dict) else []
    for package in packages or []:
        if isinstance(package, dict) and package.get("name") == "mathlib":
            return str(package.get("rev") or "")
    return ""


def _git_output(repo_root: Path, args: List[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _build_run_manifest(
    *,
    input_path: Path,
    output_path: Path,
    run_name: str,
    generation_model: Optional[str],
    max_generations: int,
    pool_size: int,
    completed_at: str,
    completed_at_compact: str,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    return {
        "bench_version": DEFAULT_BENCH_VERSION,
        "lean_toolchain": _read_text_if_present(repo_root / "lean-toolchain"),
        "mathlib_rev": _mathlib_rev(repo_root),
        "repo_git_commit": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "repo_git_dirty": bool(_git_output(repo_root, ["status", "--porcelain"])),
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "output_path": str(output_path),
        "run_name": run_name,
        "generation_model": generation_model,
        "max_generations": max_generations,
        "pool_size": pool_size,
        "completed_at": completed_at,
        "completed_at_compact": completed_at_compact,
    }


def _run_manifest_digest(manifest: Dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_langsmith_trace_upload(
    *,
    root_run_name: str,
    project_name: Optional[str],
    since_seconds: int = 600,
) -> Dict[str, Any]:
    """Best-effort server-side check that the LangSmith root trace exists."""
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return {"enabled": False, "verified": False, "reason": "missing_api_key"}
    if not (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or os.getenv("LANGCHAIN_TRACING")
    ):
        return {"enabled": False, "verified": False, "reason": "tracing_env_disabled"}
    try:
        client = ls.Client()
        start_time = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)
        runs = list(
            client.list_runs(
                project_name=project_name,
                is_root=True,
                start_time=start_time,
                limit=50,
                select=["id", "name", "trace_id", "start_time"],
            )
        )
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return {
            "enabled": True,
            "verified": False,
            "root_run_name": root_run_name,
            "project_name": project_name,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    for run in runs:
        if getattr(run, "name", None) == root_run_name:
            return {
                "enabled": True,
                "verified": True,
                "root_run_name": root_run_name,
                "project_name": project_name,
                "run_id": str(getattr(run, "id", "") or ""),
                "trace_id": str(getattr(run, "trace_id", "") or getattr(run, "id", "") or ""),
            }
    return {
        "enabled": True,
        "verified": False,
        "root_run_name": root_run_name,
        "project_name": project_name,
        "checked_recent_root_runs": len(runs),
        "reason": "root_run_not_found",
    }


def _problem_by_id(pool: List[CertificationInput]) -> Dict[str, CertificationInput]:
    return {problem.id: problem for problem in pool}


def _problem_family(problem: CertificationInput) -> str:
    return str(problem.metadata.get("family") or detect_family(problem.statement) or "")


def _problem_style(problem: CertificationInput) -> str:
    metadata = problem.metadata or {}
    explicit = str(metadata.get("problem_style") or metadata.get("target_style") or "")
    if explicit in PROBLEM_STYLES:
        return explicit
    has_formal_artifact = (
        str(metadata.get("formal_statement") or "").strip()
        or str(metadata.get("lean_code") or "").strip()
        or str(metadata.get("lean_header") or "").strip()
    )
    if has_formal_artifact and not metadata.get("generated_params"):
        return "theorem_proof"
    if _problem_family(problem) in SUPPORTED_FAMILIES:
        return "numeric_answer"
    if has_formal_artifact:
        return "theorem_proof"
    return "numeric_answer"


def _certification_route_for_style(problem_style: str) -> str:
    return "theorem_prover" if problem_style == "theorem_proof" else "template_numeric"


def _template_fit(problem: CertificationInput) -> Dict[str, Any]:
    family = _problem_family(problem)
    return {
        "family": family or "unsupported",
        "supported": family in SUPPORTED_FAMILIES,
        "params": _pool_params(problem) if family in SUPPORTED_FAMILIES else {},
    }


def _reusable_atoms(problem: CertificationInput) -> List[str]:
    text = f"{problem.statement} {problem.answer} {problem.metadata.get('formal_statement', '')}".lower()
    atoms = []
    for label, needles in {
        "prime_modulus": ["prime", "mod", "congruent", "quadratic residue"],
        "contradiction": ["irrational", "contradiction", "suppose"],
        "algebraic_structure": ["integral domain", "principal ideal", "gcd"],
        "holomorphic_constant": ["holomorphic", "constant", "imaginary"],
        "finite_counting": ["finset", "card", "count", "subset"],
        "divisibility": ["divisible", "divisor", "dvd"],
    }.items():
        if any(needle in text for needle in needles):
            atoms.append(label)
    return atoms[:5]


def _ints(text: str) -> List[int]:
    import re

    return [int(value) for value in re.findall(r"\d+", text or "")]


def _pool_params(problem: CertificationInput) -> Dict[str, Any]:
    family = _problem_family(problem)
    nums = _ints(problem.statement)
    if family in {"gcd", "gcd_divisor_sum"} and len(nums) >= 2:
        return {"a": nums[0], "b": nums[1]}
    if family == "divisor_sum_mod" and len(nums) >= 2:
        return {"n": nums[0], "a": nums[1]}
    if family == "divisor_sum" and nums:
        return {"n": nums[-1]}
    if family == "units_digit" and len(nums) >= 2:
        return {"base": nums[0], "exp": nums[1]}
    if family == "modular_congruence" and len(nums) >= 2:
        return {"a": nums[0], "m": nums[1]}
    if family == "stars_and_bars" and nums:
        var_count = max(2, len(set(__import__("re").findall(r"x_(\d+)", problem.statement))))
        return {"vars": var_count, "sum": nums[-1]}
    if family == "arithmetic_series" and len(nums) >= 3:
        return {"n_terms": nums[0], "first": nums[1], "diff": nums[2] - nums[1]}
    return {}


def _answer_int(problem: CertificationInput) -> Optional[int]:
    try:
        return int(str(problem.answer).strip())
    except (TypeError, ValueError):
        return None


def _value_in_range(family: str, key: str, value: Any) -> bool:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return False
    low, high = FAMILY_PARAM_RANGES.get(family, {}).get(key, (0, -1))
    return low <= int_value <= high


def _canonical_signature(result: CertificationResult) -> str:
    family = result.family or result.target_family or "unknown"
    params = dict(result.generated_params or result.required_params or {})
    if (
        result.target_style == "theorem_proof"
        or result.certification_route == "theorem_prover"
        or family == "theorem_proof"
    ):
        formal_surface = _canonical_theorem_surface(
            result.formal_statement or result.lean_code or result.statement or ""
        )
        if formal_surface:
            return json.dumps(
                {"family": "theorem_proof", "formal": formal_surface},
                sort_keys=True,
                ensure_ascii=False,
            )
    if not params:
        params = {"statement": result.statement or "", "answer": result.answer or ""}
    return json.dumps({"family": family, "params": params}, sort_keys=True, ensure_ascii=False)


def _canonical_theorem_surface(text: Any) -> str:
    surface = str(text or "")
    surface = surface.split(":= by", 1)[0]
    surface = re.sub(r"\b(theorem|lemma)\s+[A-Za-z0-9_'.]+", r"\1 _", surface)
    surface = re.sub(r"\s+", " ", surface).strip().lower()
    return surface


def _failure_signature(result: CertificationResult) -> str:
    if result.quality_flags:
        return ",".join(sorted(set(result.quality_flags)))
    if result.error:
        return str(result.error)[:120]
    return result.status


def _failure_class(result: CertificationResult) -> str:
    """Stable retry/replan taxonomy shared by theorem, numeric, and quality routes."""
    text = " ".join(
        str(value or "")
        for value in (
            result.status,
            result.error,
            result.proof_verify_summary,
            result.failure_signature,
            (result.quality_evidence or {}).get("contract_status"),
            (result.quality_evidence or {}).get("failure_reason"),
            (result.quality_evidence or {}).get("failure_class"),
        )
    ).lower()
    flags = set(result.quality_flags or [])
    if result.status == "certified" and result.quality_verdict == "weak":
        return "quality_weak"
    if result.status == "alignment_failed" or "statement_lean_alignment" in flags:
        return "alignment_failed"
    if result.status == "statement_failed" or "statement_typecheck_failed" in text:
        return "statement_typecheck_failed"
    if "axiom_audit_failed" in text:
        return "axiom_audit_failed"
    if any(marker in text for marker in ("unterminated string", "json", "parse_json", "expecting value")):
        return "llm_json_parse_error"
    if any(marker in text for marker in ("invalid_formal_shape", "unexpected token", "invalid argument", "syntax error")):
        return "invalid_formal_shape" if "invalid_formal_shape" in text else "lean_syntax_error"
    if "sorry" in text or "proof contains `sorry`" in text:
        return "proof_contains_sorry"
    if "parent_rewrite_without_proof_body" in text:
        return "parent_proof_surface_missing"
    if any(marker in text for marker in ("theorem_too_broad", "too broad", "unsupported generality")):
        return "theorem_too_broad"
    if result.certification_route == "theorem_prover" and result.status == "proof_failed":
        return "proof_failed"
    if result.status in {"unsupported", "planner_axis_mismatch"}:
        return "numeric_template_unsupported"
    if result.status == "generation_failed":
        if any(
            marker in text
            for marker in (
                "rate limit",
                "rate_limit",
                "429",
                "too many requests",
                "connection error",
                "connection reset",
                "timed out",
                "timeout",
                "service unavailable",
                "server overloaded",
                "internal server error",
                "bad gateway",
            )
        ):
            return "llm_transport_error"
        return "generation_failed"
    if result.status == "slot_failed":
        return "slot_exception"
    return result.status or "unknown"


def _generated_surface_summary(result: CertificationResult) -> Dict[str, Any]:
    """Raw-enough artifact digest for retry/replan prompts."""
    return {
        "problem_id": result.problem_id,
        "statement": _prompt_text(result.statement, limit=1200),
        "formal_statement": _prompt_text(result.formal_statement, limit=2000),
        "lean_code": _prompt_text(result.lean_code, limit=3500),
        "proof_surface": (result.quality_evidence or {}).get("proof_surface"),
        "proof_plan": _prompt_text(result.proof_plan, limit=1200),
        "proof_obligations": list(result.proof_obligations or [])[:8],
        "generated_params": dict(result.generated_params or {}),
        "answer": _prompt_text(result.answer, limit=500),
        "solution": _prompt_text(result.solution, limit=1600),
        "reasoning_pattern": result.reasoning_pattern,
    }


def _attempt_history_card(
    *,
    attempt: int,
    item: Dict[str, Any],
    result: CertificationResult,
    retry_feedback: str = "",
    replan_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "attempt": attempt,
        "operator_card": _operator_card(item),
        "target_style": item.get("target_style", result.target_style),
        "generated_surface_summary": _generated_surface_summary(result),
        "status": result.status,
        "error_class": _failure_class(result),
        "lean_error_summary": _prompt_text(result.error or result.proof_verify_summary, limit=700),
        "quality_flags": list(result.quality_flags or [])[:8],
        "failure_signature": _failure_signature(result),
        "retry_feedback": _prompt_text(retry_feedback, limit=700),
        "replan_decision": replan_decision or {},
    }


def _attempt_history_summary(
    attempt_history: List[Dict[str, Any]],
    *,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for card in list(attempt_history or [])[-limit:]:
        operator_card = dict(card.get("operator_card") or {})
        surface = dict(card.get("generated_surface_summary") or {})
        summary.append(
            {
                "attempt": card.get("attempt"),
                "op_type": operator_card.get("op_type"),
                "operator_variant": operator_card.get("operator_variant"),
                "target_style": card.get("target_style") or operator_card.get("target_style"),
                "target_family": operator_card.get("target_family"),
                "goal": _prompt_text(operator_card.get("goal") or operator_card.get("operator_goal"), limit=180),
                "status": card.get("status"),
                "error_class": card.get("error_class"),
                "failure_signature": card.get("failure_signature"),
                "quality_flags": list(card.get("quality_flags") or [])[:6],
                "lean_error_summary": _prompt_text(card.get("lean_error_summary"), limit=350),
                "generated_surface": {
                    "statement": _prompt_text(surface.get("statement"), limit=900),
                    "formal_statement": _prompt_text(surface.get("formal_statement"), limit=1500),
                    "lean_code": _prompt_text(surface.get("lean_code"), limit=2200),
                    "generated_params": dict(surface.get("generated_params") or {}),
                    "answer": _prompt_text(surface.get("answer"), limit=300),
                    "solution": _prompt_text(surface.get("solution"), limit=900),
                    "reasoning_pattern": surface.get("reasoning_pattern"),
                    "proof_plan": _prompt_text(surface.get("proof_plan"), limit=900),
                },
                "retry_feedback": _prompt_text(card.get("retry_feedback"), limit=700),
                "replan_decision": dict(card.get("replan_decision") or {}),
            }
        )
    return summary


def _attempt_history_trace_summary(
    attempt_history: List[Dict[str, Any]],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Small attempt history for LangSmith.

    Full attempt histories are still kept in JSONL rows. Trace payloads only
    need enough context to locate the failing attempt without uploading long
    Lean bodies repeatedly.
    """
    trace_cards: List[Dict[str, Any]] = []
    for card in _attempt_history_summary(attempt_history, limit=limit):
        surface = dict(card.get("generated_surface") or {})
        trace_cards.append(
            {
                "attempt": card.get("attempt"),
                "op_type": card.get("op_type"),
                "operator_variant": card.get("operator_variant"),
                "target_style": card.get("target_style"),
                "target_family": card.get("target_family"),
                "status": card.get("status"),
                "error_class": card.get("error_class"),
                "failure_signature": card.get("failure_signature"),
                "quality_flags": list(card.get("quality_flags") or [])[:6],
                "lean_error_summary": _prompt_text(card.get("lean_error_summary"), limit=240),
                "statement_preview": _prompt_text(surface.get("statement"), limit=240),
                "formal_statement_preview": _prompt_text(surface.get("formal_statement"), limit=320),
                "proof_plan_preview": _prompt_text(surface.get("proof_plan"), limit=240),
                "retry_feedback": _prompt_text(card.get("retry_feedback"), limit=240),
                "replan_decision": {
                    key: value
                    for key, value in dict(card.get("replan_decision") or {}).items()
                    if key in {"replan_source", "replan_reason"}
                },
            }
        )
    return trace_cards


def _operator_card(item: Dict[str, Any]) -> Dict[str, Any]:
    """Compact slot contract consumed by the generator and verifier."""
    parent_cards = list(item.get("parent_context_cards") or [])
    max_retries = int(item.get("max_retries") or os.getenv("POOL_GENERATION_MAX_RETRIES", "3"))
    proof_surfaces = [
        "parent_rewrite",
        "simp_only",
        "constructor_cases",
        "exact_existing",
        "direct_group_calc",
        "direct_proof",
    ]
    theorem_cards = [card for card in parent_cards if card.get("problem_style") == "theorem_proof"]
    if theorem_cards and not any(
        (card.get("proof_context") or {}).get("proof_body_available") for card in theorem_cards
    ):
        proof_surfaces = [
            surface
            for surface in proof_surfaces
            if surface not in {"parent_rewrite", "exact_existing"}
        ]
    if theorem_cards and item.get("operator_variant") not in {"mutation_hard", "crossover_hard"}:
        proof_surfaces = [surface for surface in proof_surfaces if surface != "direct_proof"]
    return {
        "op_type": item.get("op_type", ""),
        "operator_variant": item.get("operator_variant", ""),
        "target_style": item.get("target_style", "numeric_answer"),
        "target_family": item.get("target_family", ""),
        "operator_goal": item.get("operator_goal")
        or item.get("reasoning_goal")
        or item.get("goal")
        or item.get("variation_axis", ""),
        "goal": item.get("goal") or item.get("operator_goal") or item.get("reasoning_goal") or "",
        "constraints": list(item.get("constraints") or []),
        "avoid": list(item.get("avoid") or item.get("avoid_patterns") or []),
        "fusion_goal": item.get("fusion_goal", ""),
        # Carried into the trace because this is the field the run exists to
        # observe. It was dropped by the compaction, so even a populated
        # mechanism showed as empty in LangSmith.
        "fusion_mechanism": (item.get("fusion_contract") or {}).get("fusion_mechanism")
        or item.get("fusion_mechanism", ""),
        "parent_roles": dict(item.get("parent_roles") or {}),
        "memory_delta_contract": dict(item.get("memory_delta_contract") or {}),
        "composition_pattern": item.get("composition_pattern", ""),
        "parent_ids": list(item.get("parent_ids") or []),
        "parent_cards": parent_cards,
        "theorem_decompositions": [
            card.get("theorem_decomposition", {})
            for card in parent_cards
            if card.get("problem_style") == "theorem_proof"
        ],
        "required_checkpoints": list(item.get("required_checkpoints") or []),
        "avoid_signatures": list(item.get("avoid_signatures") or []),
        "fusion_contract": dict(item.get("fusion_contract") or {}),
        "theorem_allowed_statement_deltas": [
            "same_statement",
            "specialize_hypothesis",
            "project_conclusion",
            "immediate_corollary",
        ],
        "theorem_proof_surfaces": proof_surfaces,
        # Style preference only. The pre-2135c41 hard gate on this list
        # rejected 40 Lean-correct proofs; the worker prompt now states the
        # list must never justify a cannot_execute refusal.
        "theorem_proof_surfaces_policy": "advisory_preference_only",
        "retry_constraints": {
            "max_retries": max_retries,
            "keep_target_style": True,
            "keep_target_family": item.get("target_style") != "theorem_proof",
            "patch_target_fields": list(item.get("patch_target_fields") or []),
            "must_not_change_fields": list(item.get("must_not_change_fields") or []),
        },
    }


def _compact_memory_delta_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not contract:
        return {}
    return {
        "similar_card_ids": list(contract.get("similar_card_ids") or [])[:5],
        "must_not_repeat": [
            _prompt_text(item, limit=180)
            for item in list(contract.get("must_not_repeat") or [])[:5]
        ],
        "required_distinguishing_delta": _prompt_text(
            contract.get("required_distinguishing_delta"), limit=360
        ),
        "allowed_overlap": _prompt_text(contract.get("allowed_overlap"), limit=260),
        "novelty_rationale": _prompt_text(contract.get("novelty_rationale"), limit=260),
    }


def _compact_operator_card_for_pool(card: Dict[str, Any]) -> Dict[str, Any]:
    """Prevent recursive parent-card metadata growth across generations."""
    return {
        "op_type": card.get("op_type", ""),
        "operator_variant": card.get("operator_variant", ""),
        "target_style": card.get("target_style", ""),
        "target_family": card.get("target_family", ""),
        "operator_goal": _prompt_text(card.get("operator_goal") or card.get("goal"), limit=600),
        "goal": _prompt_text(card.get("goal") or card.get("operator_goal"), limit=600),
        "constraints": list(card.get("constraints") or [])[:10],
        "avoid": list(card.get("avoid") or [])[:10],
        "fusion_goal": _prompt_text(card.get("fusion_goal"), limit=500),
        "parent_roles": dict(card.get("parent_roles") or {}),
        "memory_delta_contract": _compact_memory_delta_contract(
            dict(card.get("memory_delta_contract") or {})
        ),
        "parent_ids": list(card.get("parent_ids") or [])[:4],
        "required_checkpoints": list(card.get("required_checkpoints") or [])[:10],
        "avoid_signatures": list(card.get("avoid_signatures") or [])[:10],
    }


def _compact_novelty_assessment(assessment: Dict[str, Any]) -> Dict[str, Any]:
    if not assessment:
        return {}
    return {
        "verdict": assessment.get("verdict"),
        "matched_card_id": assessment.get("matched_card_id"),
        "reason": _prompt_text(assessment.get("reason"), limit=260),
        "exact_blockers": list(assessment.get("exact_blockers") or [])[:3],
        "matched_cards": [
            compact_card(card)
            for card in list(assessment.get("matched_cards") or [])[:3]
            if isinstance(card, dict)
        ],
        "gate_cards": [
            compact_card(card)
            for card in list(assessment.get("gate_cards") or assessment.get("matched_cards") or [])[:3]
            if isinstance(card, dict)
        ],
    }


def _compact_quality_evidence_for_pool(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Keep selection-relevant quality evidence without nested raw artifacts."""
    return {
        "checkpoint_coverage": evidence.get("checkpoint_coverage"),
        "missing_checkpoints": list(evidence.get("missing_checkpoints") or [])[:12],
        "reasoning_signature": _prompt_text(evidence.get("reasoning_signature"), limit=500),
        "signature_group": _prompt_text(evidence.get("signature_group"), limit=160),
        "crossover_kind": evidence.get("crossover_kind"),
        "accepted_proxy": dict(evidence.get("accepted_proxy") or {}),
        "curation_decision": dict(evidence.get("curation_decision") or {}),
        "parent_contribution_source": evidence.get("parent_contribution_source"),
        "novelty_flags": list(evidence.get("novelty_flags") or [])[:12],
        "novelty_memory": _compact_novelty_assessment(
            dict(evidence.get("novelty_memory") or {})
        ),
        "feature_delta": dict(evidence.get("feature_delta") or {}),
        "premise_pack": dict(evidence.get("premise_pack") or {}),
    }


def _infer_theorem_proof_surface(lean_code: str) -> str:
    """Best-effort diagnostic label for proof shape.

    This is intentionally not a certification gate. The Lean compiler is the
    source of truth; this label only helps quality evidence and retry feedback.
    """
    code = lean_code or ""
    lowered = code.lower()
    if "exact ⟨" in code or "refine ⟨" in code or "\n  use " in code:
        return "constructor_cases"
    if "simpa" in lowered or "simp" in lowered or "norm_num" in lowered or "decide" in lowered:
        return "simp_only"
    if ".mul_ratcast" in lowered or "exact hx" in lowered or "exact " in lowered:
        return "exact_existing"
    if "calc" in lowered:
        return "direct_group_calc"
    if ":= by" in code:
        return "direct_proof"
    return "inferred_or_unspecified"


def _default_operator_variant(op_type: str, *, target_family: str = "", parent_family: str = "") -> str:
    if op_type == "survivor":
        return "survivor"
    if op_type == "crossover":
        return "crossover_hard"
    if op_type == "mutation":
        return "mutation_hard" if target_family and target_family == parent_family else "mutation_easy"
    return op_type


def _root_lineage_id(problem_id: str) -> str:
    """The first seed an id descends from. Both id forms are understood."""
    roots = _new_roots_of(problem_id)
    if roots:
        return roots[0]
    root = str(problem_id or "")
    for marker in ("__x__", "__theorem_gen", "__fallback"):
        if marker in root:
            root = root.split(marker, 1)[0]
    return root


#: How many accepted descendants one benchmark seed may contribute. Without a
#: cap the search concentrates: in the released corpus, 13 of 18 ProofNet rows
#: with intact lineage descended from a single seed, and generations 3-5 kept
#: re-mining it. A cap spends the budget across the seed set instead, which is
#: what makes the treatment arm a statement about the benchmark rather than
#: about six problems.
SEED_OFFSPRING_CAP = 20

from src.orchestration.problem_ids import child_id, roots_of as _new_roots_of


#: Legacy suffixes, still stripped so rows written before the id change keep
#: resolving to their seeds.
_ROOT_SUFFIX = re.compile(r"(__theorem_gen\d+|__gen\d+_[a-z_]+)+$")


def _root_seeds(problem_id: str) -> Set[str]:
    r"""The benchmark seeds a row descends from, read off its identifier.

    This used to strip `(__theorem_gen\d+)+$`, which never matched: the legacy
    id ends in the statement fingerprint, so the anchor failed and every row
    "rooted" at its own full id. Two things rest on this answer -- the per-seed
    offspring cap and the refusal to cross a seed with its own descendant -- and
    both were reading a root that did not exist, so neither could fire. The
    shared parser understands the current form and the legacy one.
    """
    roots: Set[str] = set(_new_roots_of(problem_id))
    if roots:
        return roots
    for part in str(problem_id or "").split("__x__"):
        while True:
            stripped = _ROOT_SUFFIX.sub("", part)
            if stripped == part:
                break
            part = stripped
        if part:
            roots.add(part)
    return roots


def _exhausted_seeds(accepted: Sequence[Any], cap: int = SEED_OFFSPRING_CAP) -> Set[str]:
    """Seeds that have already contributed `cap` accepted descendants."""
    tally: Counter = Counter()
    for row in accepted or []:
        data = row.model_dump() if hasattr(row, "model_dump") else row
        if not isinstance(data, dict):
            continue
        pid = str(data.get("problem_id") or data.get("id") or "")
        for root in _root_seeds(pid):
            tally[root] += 1
    return {seed for seed, count in tally.items() if count >= cap}


def _drop_exhausted_parents(pool: List[Any], exhausted: Set[str]) -> List[Any]:
    """Remove parents whose lineage is spent, unless that would empty the pool.

    Returning an empty pool would stall the generation, which is worse than
    exceeding a cap, so the filter yields to that: a cap is a budget, not an
    invariant.
    """
    if not exhausted:
        return pool
    kept = []
    for parent in pool:
        data = parent.model_dump() if hasattr(parent, "model_dump") else parent
        pid = str((data or {}).get("problem_id") or (data or {}).get("id") or "")
        if not (_root_seeds(pid) & exhausted):
            kept.append(parent)
    return kept if len(kept) >= 2 else pool


def _same_root_lineage(parent_ids: List[str]) -> bool:
    roots = [_root_lineage_id(parent_id) for parent_id in parent_ids]
    return len(roots) != len(set(roots))


def _reject_same_lineage_crossovers(items: List[PoolWorkItem]) -> None:
    """Downgrade any crossover whose parents share a root, whatever produced it.

    Three code paths promote a slot to crossover, and the normalisation pass
    only guards the one the planner writes. A forced exploration slot reached
    the generator asking it to cross `seed` with `seed__theorem_gen1`; the
    generator refused on its own contract ("uses a parent and its descendant")
    and the slot was lost. Enforcing the rule once, after every producer has
    run, costs nothing and cannot be bypassed by adding a fourth producer.
    """
    for item in items:
        if item.op_type != "crossover" or len(item.parent_ids) < 2:
            continue
        if not _same_root_lineage(item.parent_ids):
            continue
        kept = item.parent_ids[0]
        item.op_type = "mutation"
        item.operator_variant = "mutation_easy"
        item.parent_ids = [kept]
        item.parent_refs = item.parent_refs[:1]
        item.composition_pattern = "parameter_shift"
        item.fusion_contract = {}
        item.parent_contributions = {}
        item.fusion_goal = ""
        item.parent_roles = {}
        item.avoid_patterns = list(
            dict.fromkeys(list(item.avoid_patterns) + ["same_lineage_crossover"])
        )
        item.operator_goal = (
            "downgraded same-lineage crossover to mutation: a parent and its own "
            "descendant cannot be crossed; vary the single parent instead"
        )
        item.reasoning_goal = item.operator_goal
        item.variation_axis = item.operator_goal


def _result_root_lineages(result: CertificationResult) -> List[str]:
    ids = list(result.parent_ids or [])
    if not ids:
        ids = [result.source_problem_id or result.problem_id]
    roots = [_root_lineage_id(item) for item in ids if item]
    return list(dict.fromkeys(root for root in roots if root))


def _family_complexity_rank(family: str) -> int:
    return {
        "units_digit": 1,
        "modular_congruence": 1,
        "gcd": 1,
        "arithmetic_series": 1,
        "stars_and_bars": 1,
        "divisor_sum": 2,
        "gcd_divisor_sum": 3,
        "divisor_sum_mod": 3,
    }.get(str(family or ""), 1)


def _fusion_parent(fusion_contract: Dict[str, Any], label: str) -> Dict[str, Any]:
    value = fusion_contract.get(label) or {}
    return value if isinstance(value, dict) else {}


def _fusion_parent_contributions(item: Dict[str, Any]) -> Dict[str, str]:
    contributions = dict(item.get("parent_contributions") or {})
    fusion_contract = dict(item.get("fusion_contract") or {})
    for label in ("parent_A", "parent_B"):
        parent = _fusion_parent(fusion_contract, label)
        parent_id = str(parent.get("id") or "")
        contribution = str(parent.get("contribution") or "")
        if parent_id and contribution and not contributions.get(parent_id):
            contributions[parent_id] = contribution
    return contributions


def _checkpoint_id(value: str) -> str:
    checkpoint = str(value or "").strip().lower()
    if checkpoint in CHECKPOINT_IDS:
        return checkpoint
    if checkpoint in CHECKPOINT_ALIASES:
        return CHECKPOINT_ALIASES[checkpoint]
    for phrase, checkpoint_id in CHECKPOINT_ALIASES.items():
        if phrase in checkpoint:
            return checkpoint_id
    if "prime factor" in checkpoint or "divisor count" in checkpoint:
        return "rich_factorization"
    if "gcd" in checkpoint and ("prime" in checkpoint or "factor" in checkpoint):
        return "rich_gcd"
    if "mod" in checkpoint and ("reduction" in checkpoint or "dividend" in checkpoint):
        return "nontrivial_modular_reduction"
    if "remainder" in checkpoint:
        return "nontrivial_mod_remainder"
    if "answer" in checkpoint and "verified" in checkpoint:
        return "numeric_answer_verified"
    if "binomial" in checkpoint or "stars" in checkpoint:
        return "binomial_formula"
    if "arithmetic" in checkpoint or "sum formula" in checkpoint:
        return "arithmetic_sum_formula"
    if "certified" in checkpoint or "lean" in checkpoint:
        return "family_certified"
    return ""


def _normalize_required_checkpoints(values: List[Any]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        checkpoint_id = _checkpoint_id(str(value))
        if checkpoint_id and checkpoint_id not in normalized:
            normalized.append(checkpoint_id)
    return normalized[:4]


def _fold_fusion_contract(item: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the planner's flat fusion fields into the nested contract it is validated against.

    The planner is asked for a flat shape -- `fusion_mechanism`, `parent_roles`
    keyed by parent index -- and validation demands a nested one with
    `parent_A.id` and `parent_B.id`. Nothing bridged them, so a plan naming
    `parameter_coupling` was rejected outright with `invalid parent_A.id`, the
    deterministic fallback replaced the whole plan, and the fallback's contract
    supplied the mechanism instead. That is the last of the six layers between
    the planner choosing a strategy and the worker being told which one.

    Building the nested form here rather than demanding it in the prompt keeps
    the planner's contract small: it names the mechanism and which parent plays
    which role, and the ids it already put in `parent_ids` fill the rest.
    """
    contract = dict(item.get("fusion_contract") or {})
    for key in ("fusion_mechanism", "why_not_concatenation", "shared_lean_surface"):
        value = str(item.get(key) or "").strip()
        if value and not str(contract.get(key) or "").strip():
            contract[key] = value
    roles = dict(item.get("parent_roles") or {})
    if roles and not contract.get("parent_roles"):
        contract["parent_roles"] = dict(roles)
    if str(item.get("op_type") or "") != "crossover":
        return contract

    parent_ids = [str(pid) for pid in (item.get("parent_ids") or []) if str(pid)]
    if len(parent_ids) < 2:
        return contract

    def _role_for(index: int, parent_id: str) -> str:
        # `parent_roles` arrives keyed by position ("0", "1") or by id.
        for key in (parent_id, str(index)):
            value = str(roles.get(key) or "").strip()
            if value:
                return value
        return "object_domain" if index == 0 else "goal_form"

    goal = str(item.get("fusion_goal") or item.get("goal") or "").strip()
    for label, index in (("parent_A", 0), ("parent_B", 1)):
        existing = dict(contract.get(label) or {})
        if str(existing.get("id") or "") in parent_ids:
            continue
        contract[label] = {
            "id": parent_ids[index],
            "semantic_role": _role_for(index, parent_ids[index]),
            "contribution": str(existing.get("contribution") or "").strip()
            or goal
            or "planned crossover contribution",
        }
    return contract


def _default_fusion_contract(
    *,
    first: CertificationInput,
    second: CertificationInput,
    mechanism: str,
    parent_a_role: str,
    parent_b_role: str,
    parent_a_contribution: str,
    parent_b_contribution: str,
    new_problem_core: str,
    why_not_concatenation: str = "",
    risk: str = "Template surface may reduce this to a sequential composite.",
) -> Dict[str, Any]:
    # `mechanism` is left empty by every deterministic caller on purpose. All
    # five used to pass "sequential_composition", which meant a fallback contract
    # asserted a fusion strategy nobody had chosen -- and since the planner's own
    # choice was being dropped further upstream, this was where 150 of 186
    # crossovers got their mechanism. An unset field is the truthful record of a
    # fallback that made no such decision, and it leaves room for the planner's
    # answer when there is one.
    return {
        "parent_A": {
            "id": first.id,
            "semantic_role": parent_a_role,
            "contribution": parent_a_contribution,
        },
        "parent_B": {
            "id": second.id,
            "semantic_role": parent_b_role,
            "contribution": parent_b_contribution,
        },
        "fusion_mechanism": mechanism,
        "why_not_concatenation": why_not_concatenation,
        "new_problem_core": new_problem_core,
        "expected_lean_footprint": ["Nat", "List.range", "Nat.gcd", "native_decide"],
        "risk": risk,
    }


def _selection_priority(result: CertificationResult) -> tuple[int, int]:
    """Prefer generated strong/composite candidates before carrying old anchors.

    The judge leads. Ordering is not a gate in name, but a pool holds five and a
    priority of 9 puts a row behind every seed copy, so the ranking decides who
    becomes a parent as surely as an eligibility flag does. The ladder below is
    built from surface heuristics -- crossover kind, signature group, family --
    and they stay, as tie-breakers within a tier; what they no longer do is
    outrank a reader who has looked at the proof.
    """
    evidence = dict(result.quality_evidence or {})
    judge = dict(evidence.get("judge") or {})
    if result.status not in {"certified", "survivor"}:
        return (9, int(result.slot or 0))
    if judge.get("ran") and judge.get("verdict") == "reject":
        return (9, int(result.slot or 0))
    # `near_duplicate` is a similarity heuristic, not an identity one -- exact
    # repeats are caught by the hash gate and never reach here -- so it is a
    # reason to rank a row lower, not to bury it behind the seeds.
    if "near_duplicate" in set(result.quality_flags or []) and not judge.get("ran"):
        return (9, int(result.slot or 0))
    if not judge.get("ran") and result.quality_verdict == "weak" and (
        result.retry_exhausted or not _entropy_increase(result)
    ):
        return (9, int(result.slot or 0))
    if judge.get("ran") and judge.get("verdict") == "keep":
        quality = str(judge.get("quality") or "")
        if quality == "strong":
            return (0, int(result.slot or 0))
        if quality == "acceptable":
            return (1, int(result.slot or 0))
    signature_group = str(evidence.get("signature_group") or "")
    crossover_kind = str(evidence.get("crossover_kind") or "")
    family = result.family or result.target_family or detect_family(result.statement or "")
    if _is_generated_result(result) and _accepted_proxy_pass(result):
        return (0, int(result.slot or 0))
    if result.quality_verdict == "strong" and result.op_type == "crossover" and crossover_kind == "true_fusion":
        return (1, int(result.slot or 0))
    if result.op_type == "crossover" and crossover_kind == "lemma_bundle_master":
        return (2, int(result.slot or 0))
    if signature_group in {"modular", "counting"} and result.op_type not in {"survivor", "fallback_survivor"}:
        return (3, int(result.slot or 0))
    if family in {"gcd_divisor_sum", "divisor_sum_mod"} and result.op_type != "survivor":
        return (4, int(result.slot or 0))
    if _is_generated_result(result) and _entropy_increase(result):
        return (5, int(result.slot or 0))
    if result.op_type not in {"survivor", "fallback_survivor"}:
        return (6, int(result.slot or 0))
    return (7, int(result.slot or 0))


def _retryable_generation_failure(result: CertificationResult) -> bool:
    if result.status == "planner_axis_mismatch":
        return True
    if _failure_class(result) in {
        "llm_json_parse_error",
        "llm_transport_error",
        "invalid_formal_shape",
        "parent_proof_surface_missing",
        "theorem_too_broad",
    }:
        return True
    if result.status != "generation_failed":
        return False
    error = str(result.error or "").lower()
    return any(
        marker in error
        for marker in [
            "expecting",
            "json",
            "delimiter",
            "outside",
            "contract",
            "required",
            "missing params",
            "missing from params",
            "axis_failed",
            "unsupported generated family",
        ]
    )


def _retryable_lean_repair_failure(result: CertificationResult, op_type: str) -> bool:
    """Allow one compiler-feedback repair for generated slots only."""
    return (
        op_type != "survivor"
        and result.status == "failed"
        and bool(result.lean_code)
        and bool(result.anti_stub_passed)
        and result.error is not None
    )


def _retryable_theorem_failure(result: CertificationResult, op_type: str) -> bool:
    return (
        op_type != "survivor"
        and result.certification_route == "theorem_prover"
        and result.status in {
            "proof_failed",
            "alignment_failed",
            "statement_failed",
            "vacuous",
            "judge_rejected",
            # Retryable because the failure is usually the equivalence proof,
            # not the encoding: writing these five by hand, two needed a real
            # argument -- `Int.ModEq` unfolded to divisibility before `omega`
            # could see it, and the squares-mod-3 case needed the residue split
            # supplied explicitly -- and the first attempt at each was wrong.
            "silent_not_equivalent",
            # Retryable: the slot can still produce a different theorem from the
            # same parents, and the retry brief names the statement it collided
            # with.
            "duplicate_statement",
        }
        and result.error is not None
    )


def _is_plan_level_failure(
    result: CertificationResult,
    *,
    op_type: str,
    failure_history: List[str],
    attempt_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Decide whether retrying the same OperatorCard is likely anchoring on a bad plan."""
    if op_type == "survivor":
        return False
    # The judge is the only reader that has seen both parents and the child, so
    # when it says the pairing itself is the problem it is answering this
    # question directly. It matters most for crossover: `parallel`,
    # `constant_supplier`, and `arithmetic_on_answers` are all verdicts that two
    # parents had no natural point of contact, and no rewrite from the same two
    # parents fixes that — only a different pairing does.
    if str(((result.quality_evidence or {}).get("judge") or {}).get("fix_scope") or "") == "replan":
        return True
    # The coupling gate answers the same question from the proof rather than
    # from a reading of it, and the paragraph above is its argument too: a
    # crossover whose parents were traced into the proof graph and never met
    # below the closing term has no natural point of contact, and a rewrite
    # from the same pair does not create one. Without this the gate's verdict
    # sent the slot back to the same OperatorCard, which is the retry this
    # function exists to prevent.
    #
    # Gated on `measurable` for the reason the gate itself is: an empty
    # attribution makes a depth of 0 mean "not measured", and replanning on
    # that would discard pairings no evidence is against.
    coupling = (result.quality_evidence or {}).get("structural_novelty") or {}
    if (op_type == "crossover"
            and coupling.get("flag") == "parallel_crossover"
            and coupling.get("measurable") is True
            and coupling.get("coupling_depth") == 0):
        return True
    failure_cls = _failure_class(result)
    if failure_cls in {"theorem_too_broad", "invalid_formal_shape"}:
        return True
    if failure_cls in {"lean_syntax_error", "llm_json_parse_error"}:
        recent = _attempt_history_summary(attempt_history or [], limit=3)
        return sum(1 for card in recent if card.get("error_class") == failure_cls) >= 2
    text = " ".join(
        str(value or "")
        for value in (
            result.status,
            result.error,
            result.proof_verify_summary,
            result.failure_signature,
            (result.quality_evidence or {}).get("contract_status"),
            (result.quality_evidence or {}).get("failure_reason"),
        )
    ).lower()
    if any(
        marker in text
        for marker in (
            "contract_failed",
            "theorem_contract_failed",
            "lacks shared lean surface",
            "lacks a unified non-conjunction obligation",
            "parent theorem surface mismatch",
        )
    ):
        return True
    if result.certification_route == "theorem_prover" and result.status in {
        "proof_failed",
        "alignment_failed",
    }:
        signature = _failure_signature(result)
        repeated_signature = len(failure_history) >= 2 and failure_history[-1] == signature
        if repeated_signature:
            return True
        recent = _attempt_history_summary(attempt_history or [], limit=3)
        return sum(1 for card in recent if card.get("failure_signature") == signature) >= 2
    return False


def _choose_replan_parent_card(parent_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not parent_cards:
        return {}
    proof_body_cards = [
        card
        for card in parent_cards
        if (card.get("proof_context") or {}).get("proof_body_available")
    ]
    return dict((proof_body_cards or parent_cards)[0])


def _op_type_locked() -> bool:
    """Whether a slot's operator may change during replanning.

    Off by default, because the mixed pipeline should still rescue a failed
    crossover as a mutation. `OP_TYPE_LOCK=1` pins it, which is what a
    single-operator ablation arm needs.
    """
    return os.getenv("OP_TYPE_LOCK", "0") == "1"


def _deterministic_replan_operator_card(
    item: Dict[str, Any],
    result: CertificationResult,
) -> Dict[str, Any]:
    """Replace a failing plan with the smallest safer OperatorCard for the same slot."""
    replanned = dict(item)
    discarded_card = _operator_card(item)
    parent_cards = list(item.get("parent_context_cards") or [])
    chosen_card = _choose_replan_parent_card(parent_cards)
    chosen_parent_id = str(
        chosen_card.get("id")
        or (item.get("parent_ids") or [""])[0]
    )
    target_style = str(item.get("target_style") or "numeric_answer")
    previous_variant = str(item.get("operator_variant") or "")

    # An ablation that compares crossover-only, mutation-only and mixed modes can
    # only do so if the mode survives the run. It did not: of 25 slots planned as
    # crossover in a crossover-only group, 14 came back as mutations, because a
    # crossover that fails is rescued by retrying it as a mutation. That rescue
    # is right for a mixed run, where the slot is worth filling however it fills,
    # and wrong for a controlled one, where the arm is the thing being measured.
    #
    # Locked, the ladder stays inside the operator: crossover_hard ->
    # crossover_easy -> give up. A slot that gives up is recorded as a failure of
    # that arm, which is the honest reading, rather than being quietly counted as
    # the other arm's output.
    if _op_type_locked() and item.get("op_type") == "crossover":
        previous = str(item.get("operator_variant") or "")
        replanned["operator_variant"] = (
            "crossover_easy" if previous != "crossover_easy" else "crossover_easy"
        )
        replanned["op_type"] = "crossover"
        replanned["variation_axis"] = (
            "retry the crossover at lower intensity; the pair must still meet, "
            "and this slot may not become a mutation"
        )
        replanned["reasoning_goal"] = replanned["variation_axis"]
        replanned["operator_goal"] = replanned["variation_axis"]
        replanned["required_checkpoints"] = ["family_certified"]
        replanned["op_type_locked"] = True
    elif item.get("op_type") == "crossover":
        replanned["op_type"] = "mutation"
        replanned["operator_variant"] = "mutation_easy"
        replanned["parent_ids"] = [chosen_parent_id]
        replanned["parent_refs"] = []
        replanned["parent_context_cards"] = [chosen_card] if chosen_card else []
        replanned["fusion_contract"] = {}
        replanned["parent_contributions"] = {}
        replanned["composition_pattern"] = "structure_expansion"
        replanned["variation_axis"] = (
            "rescue failed crossover by generating one theorem-style mutation from "
            f"parent {chosen_parent_id}"
        )
        replanned["reasoning_goal"] = (
            "use one available parent checkpoint and produce a smaller executable child"
        )
        replanned["operator_goal"] = replanned["reasoning_goal"]
        replanned["quality_target"] = (
            "certified theorem-style child that preserves parent style after crossover giveup"
        )
        replanned["required_checkpoints"] = ["family_certified"]
    elif previous_variant == "mutation_hard":
        replanned["operator_variant"] = "mutation_easy"
        replanned["variation_axis"] = (
            "rescue hard mutation by using a smaller immediate theorem-style corollary"
        )
        replanned["reasoning_goal"] = "select one executable parent checkpoint"
        replanned["operator_goal"] = replanned["reasoning_goal"]
        replanned["required_checkpoints"] = ["family_certified"]
    elif previous_variant == "mutation_easy":
        # The floor used to be `mutation_easy`, so a slot that failed there was
        # retried at the same intensity and usually failed the same way. A silent
        # mutation is the one rung below: it re-encodes the parent in different
        # Lean and changes no mathematics, so it asks less of the generator than
        # any variation does while still producing a row that is not the parent's
        # surface. It is not a cheaper variation — it is a different claim about
        # the row, and it is recorded as one.
        replanned["operator_variant"] = "mutation_silent"
        replanned["variation_axis"] = (
            "re-encode the parent theorem in different Lean without changing the "
            "mathematics; prove the equivalence in both directions"
        )
        replanned["reasoning_goal"] = replanned["variation_axis"]
        replanned["operator_goal"] = replanned["variation_axis"]
        replanned["required_checkpoints"] = ["family_certified", "silent_equivalence"]
    else:
        replanned["operator_variant"] = "mutation_easy" if item.get("op_type") == "mutation" else previous_variant

    if target_style == "theorem_proof":
        replanned["target_style"] = "theorem_proof"
        replanned["target_family"] = "theorem_proof"
        replanned["required_params"] = {}
        replanned["avoid_signatures"] = list(
            dict.fromkeys(
                list(item.get("avoid_signatures") or [])
                + ["parent_rewrite_without_proof_body", _failure_signature(result)]
            )
        )
    replanned["planner_source"] = "slot_replan"
    replanned["replan_source"] = "deterministic_fallback"
    replanned["replan_reason"] = _failure_signature(result)
    replanned["discarded_operator_card"] = discarded_card
    replanned["operator_card"] = _operator_card(replanned)
    return replanned


def _replan_response_format() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "replan_reason",
            "op_type",
            "operator_variant",
            "parent_ids",
            "operator_goal",
            "required_checkpoints",
            "avoid_signatures",
        ],
        "properties": {
            "decision": {"type": "string", "enum": ["replan", "giveup"]},
            "replan_reason": {"type": "string"},
            "op_type": {"type": "string", "enum": ["mutation", "crossover"]},
            "operator_variant": {
                "type": "string",
                "enum": [
                    "mutation_easy",
                    "mutation_hard",
                    "mutation_silent",
                    "crossover_easy",
                    "crossover_hard",
                ],
            },
            "parent_ids": {"type": "array", "items": {"type": "string"}},
            "operator_goal": {"type": "string"},
            "required_checkpoints": {"type": "array", "items": {"type": "string"}},
            "avoid_signatures": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact no-go flags or failure signatures the next worker must avoid. Prefer flags from Replan NoGoPolicyPack.",
            },
        },
    }
    return _schema_response_format("slot_replan_operator_card", schema)


def _build_replan_messages(
    item: Dict[str, Any],
    result: CertificationResult,
    attempt_history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = (
        "You are a slot-local replanner. You do not generate a math problem. "
        "You only revise the OperatorCard so the next worker attempt has an executable contract. "
        "Return JSON only."
    )
    target_style = item.get("target_style", "numeric_answer")
    recent_flags = list(result.quality_flags or [])
    for attempt in attempt_history[-4:]:
        if isinstance(attempt, dict):
            recent_flags.extend(str(flag) for flag in attempt.get("quality_flags") or [])
    no_go_pack = build_no_go_policy_pack(
        op_type=str(item.get("op_type") or result.op_type or "mutation"),
        target_style=str(target_style),
        target_family=str(item.get("target_family") or result.target_family or ""),
        operator_variant=str(item.get("operator_variant") or ""),
        recent_failure_flags=recent_flags,
        limit=8,
    )
    user = f"""
Rules:
- Keep target_style={target_style}. If target_style=theorem_proof, never downgrade to numeric_answer.
- Prefer smaller executable plans over broader claims.
- If parent proof bodies are unavailable, avoid parent_rewrite and exact_existing style goals.
- Replan at most by changing op_type, operator_variant, parent_ids, operator_goal,
  required_checkpoints, and avoid_signatures.
- If no safe executable plan exists, return decision=giveup.
- Good downgrades: crossover_hard -> mutation_easy, crossover_easy -> mutation_easy,
  mutation_hard -> mutation_easy, mutation_easy -> mutation_silent.
- OP_TYPE_LOCK: when the current OperatorCard says op_type_locked, you may not
  change op_type. A failing crossover is retried as crossover_easy or given up;
  it does not become a mutation. This run is measuring that operator.
- mutation_silent is the last rung and means something different from the others:
  the child re-expresses the parent in different Lean and changes no mathematics.
  Choose it only when no variation has worked for this slot, never as a first
  plan, and never for crossover.

{format_no_go_policy_pack(no_go_pack, title="Replan NoGoPolicyPack")}

Current OperatorCard:
{json.dumps(_operator_card(item), ensure_ascii=False, indent=2)[:5000]}

ParentContextCards:
{json.dumps(item.get("parent_context_cards") or [], ensure_ascii=False, indent=2)[:7000]}

Recent attempt history:
{json.dumps(_attempt_history_summary(attempt_history), ensure_ascii=False, indent=2)[:8000]}

Last failure:
- status: {result.status}
- error_class: {_failure_class(result)}
- failure_signature: {_failure_signature(result)}
- error: {_prompt_text(result.error or result.proof_verify_summary, limit=900)}
{_judge_block(result)}{_coupling_block(result)}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _coupling_block(result: CertificationResult) -> str:
    """Where in the proof the two parents failed to meet.

    The coupling gate can now trigger a replan on its own, and it arrives
    carrying `parallel_crossover` in the failure signature and nothing else --
    the same blindness `_judge_block` was written to fix, from the other
    source. The gate knows more than the flag: it traced each `have` back to a
    parent and can name the two nodes that were discharged separately. That is
    what a replanner needs, because the fix is either a different pairing or a
    target that forces one node to be built from the other.
    """
    evidence = (result.quality_evidence or {}).get("structural_novelty") or {}
    if evidence.get("flag") != "parallel_crossover" or evidence.get("measurable") is not True:
        return ""
    attribution = evidence.get("attribution") or {}
    lines = ["\nCoupling gate rejected the last child, from its proof graph:"]
    if attribution:
        by_parent: Dict[str, List[str]] = {}
        for node, parent in attribution.items():
            by_parent.setdefault(str(parent), []).append(str(node))
        for parent, nodes in by_parent.items():
            lines.append(f"- {parent} reached only: {', '.join(nodes[:4])}")
    lines.append(
        "- no `have` took input from both parents; they met only at the closing term."
    )
    lines.append(
        "- a rewrite from the same pair does not fix this. Either pick a different "
        "parent whose conclusion the other parent's argument must consume, or set an "
        "operator_goal whose obligation cannot be discharged without feeding one "
        "parent's result into the other's derivation."
    )
    return "\n".join(lines)


def _judge_block(result: CertificationResult) -> str:
    """What the quality judge found, for a replan it triggered.

    A `judge_rejected` row reaches the replanner as a status and a signature,
    and `result.error` carries the Lean summary — which is empty here, because
    the proof compiled. The replanner was therefore being asked to choose
    different parents while blind to the reason the last pairing failed, and
    the reason is exactly what should drive the choice: `constant_supplier`
    means one parent contributed a numeral, so pick a parent whose content the
    target cannot avoid.
    """
    judge = (result.quality_evidence or {}).get("judge") or {}
    if not judge.get("ran") or judge.get("verdict") != "reject":
        return ""
    lines = ["\nQuality judge rejected the last child:", f"- failure: {judge.get('failure') or 'unnamed'}"]
    if judge.get("reason"):
        lines.append(f"- finding: {_prompt_text(judge.get('reason'), limit=400)}")
    if judge.get("retry_plan"):
        lines.append(f"- proposed change: {_prompt_text(judge.get('retry_plan'), limit=400)}")
    lines.append(
        "- act on this finding when choosing parent_ids and operator_goal. If the "
        "judge found that a parent contributed only a constant or was avoidable, "
        "either give that parent a target its content is needed for, or replace it."
    )
    return "\n".join(lines) + "\n"


async def _llm_replan_operator_card(
    item: Dict[str, Any],
    result: CertificationResult,
    attempt_history: List[Dict[str, Any]],
    config: GenerationConfig,
) -> Optional[Dict[str, Any]]:
    content = await _chat_completion_text_async(
        model=orchestrator_config(config).model,
        messages=_build_replan_messages(item, result, attempt_history),
        temperature=0,
        response_format=_replan_response_format(),
    )
    raw = _parse_json_object(content)
    if raw.get("decision") == "giveup":
        return None

    replanned = _deterministic_replan_operator_card(item, result)
    target_style = str(item.get("target_style") or "numeric_answer")
    parent_ids = [str(parent_id) for parent_id in raw.get("parent_ids") or []]
    known_parent_ids = {
        str(card.get("id"))
        for card in (item.get("parent_context_cards") or [])
        if str(card.get("id") or "").strip()
    }
    if parent_ids and set(parent_ids).issubset(known_parent_ids):
        replanned["parent_ids"] = parent_ids
        replanned["parent_context_cards"] = [
            card for card in (item.get("parent_context_cards") or []) if str(card.get("id")) in set(parent_ids)
        ]
    replanned["op_type"] = str(raw.get("op_type") or replanned.get("op_type") or "mutation")
    if target_style == "theorem_proof" and replanned["op_type"] not in {"mutation", "crossover"}:
        replanned["op_type"] = "mutation"
    # The lock outranks the replanner's own judgement. It is told the operator is
    # fixed, and this is what makes that true rather than advisory.
    if _op_type_locked():
        replanned["op_type"] = str(item.get("op_type") or replanned["op_type"])
        replanned["op_type_locked"] = True
    replanned["operator_variant"] = str(raw.get("operator_variant") or replanned.get("operator_variant") or "mutation_easy")
    replanned["operator_goal"] = str(raw.get("operator_goal") or replanned.get("operator_goal") or "")
    replanned["reasoning_goal"] = replanned["operator_goal"]
    replanned["required_checkpoints"] = list(raw.get("required_checkpoints") or replanned.get("required_checkpoints") or [])
    replanned["avoid_signatures"] = list(
        dict.fromkeys(
            list(item.get("avoid_signatures") or [])
            + list(raw.get("avoid_signatures") or [])
            + [_failure_signature(result)]
        )
    )
    replanned["target_style"] = target_style
    if target_style == "theorem_proof":
        replanned["target_family"] = "theorem_proof"
        replanned["required_params"] = {}
    replanned["planner_source"] = "slot_replan"
    replanned["replan_source"] = "orchestrator_llm"
    replanned["replan_reason"] = str(raw.get("replan_reason") or _failure_signature(result))
    replanned["discarded_operator_card"] = _operator_card(item)
    replanned["operator_card"] = _operator_card(replanned)
    return replanned


async def _replan_operator_card(
    item: Dict[str, Any],
    result: CertificationResult,
    attempt_history: List[Dict[str, Any]],
    config: GenerationConfig,
    replanner: Optional[ReplannerFn] = None,
) -> Dict[str, Any]:
    try:
        raw = (
            replanner(item, result, attempt_history, config)
            if replanner is not None
            else _llm_replan_operator_card(item, result, attempt_history, config)
        )
        replanned = await _maybe_await(raw)
        if replanned is not None:
            return dict(replanned)
    except Exception:
        pass
    return _deterministic_replan_operator_card(item, result)


BASE_QUALITY_RETRYABLE_FLAGS = {
    "missing_parent_contribution",
    "indirect_parent_contribution",
    "missing_quality_checkpoints",
    "trivial_mod_remainder",
    "claimed_harder_but_metric_not_increased",
    "no_required_param_delta",
    "repeated_reasoning_signature",
    "canonical_answer_mismatch",
    "solution_answer_mismatch",
    "solution_skeleton_answer_mismatch",
    "solution_modulus_mismatch",
    "solution_gcd_mismatch",
    "repair_not_harder",
    "informal_statement_internal_terms",
    # Structural novelty, judged from the Lean proof (see quality._structural_novelty).
    # Both are retryable: the generator has the parent in hand and can be told
    # precisely which move failed, which is the case a second attempt can fix.
    "decorative_mutation",
    "parallel_crossover",
    "vacuous_hypotheses",
    "judge_reject",
}
QUALITY_RETRYABLE_FLAGS = BASE_QUALITY_RETRYABLE_FLAGS | set(QUALITY_RETRYABLE_NO_GO_FLAGS)


def _quality_retry_reasons(result: CertificationResult) -> List[str]:
    if result.status != "certified" or result.quality_verdict != "weak":
        return []
    if "certification_not_successful" in set(result.quality_flags or []):
        return []
    flags = set(result.quality_flags or [])
    if (result.quality_evidence or {}).get("crossover_kind") == "mutation_like":
        flags.add("mutation_like_crossover")
    retryable = sorted(flags & QUALITY_RETRYABLE_FLAGS)
    return [f"quality:{flag}" for flag in retryable]


def _quality_retry_feedback(result: CertificationResult, attempt: int) -> str:
    if not _quality_retry_reasons(result):
        return ""
    evidence = dict(result.quality_evidence or {})
    missing_checkpoints = list(evidence.get("missing_checkpoints") or [])[:5]
    retryable_flags = [
        reason.split(":", 1)[1] for reason in _quality_retry_reasons(result)
    ]
    patch_instructions = {
        "missing_parent_contribution": (
            "For each missing parent id, update parent_contribution_evidence and "
            "solution_skeleton.parent_contributions with the exact field affected; "
            "if target_style=theorem_proof, update formal_statement or proof_obligations instead."
        ),
        "indirect_parent_contribution": (
            "Move parent contribution from prose/inspiration into params, target_computation, "
            "verification_steps, formal_statement, or proof_obligations."
        ),
        "solution_modulus_mismatch": (
            "Repair solution text so the explicit modulus claim matches generated_params.modulus; "
            "do not write ambiguous phrases that make 'mod m = answer' look like a definition of m."
        ),
        "repeated_reasoning_signature": (
            "Change the reasoning_pattern and skeleton structure, not only the numbers."
        ),
        "judge_reject": (
            "A quality review judged this child solvable by recalling its parent "
            "rather than by reasoning. Its own finding is carried in the brief "
            "below; act on that specific point, not on the flag name."
        ),
        "vacuous_hypotheses": (
            "The hypotheses cannot all hold at once: Lean derived False from them "
            "alone, so the theorem is true for every conclusion and measures "
            "nothing. This usually means two parent constraints were combined "
            "into an unsatisfiable system. Relax or replace one constraint so a "
            "witness exists, and state the witness in proof_plan."
        ),
        "missing_quality_checkpoints": (
            "Add visible evidence for each missing checkpoint in the generated structured fields."
        ),
        # These two carry their own brief, assembled from the measurement that
        # rejected the row (which tactics were shared, where the parents met).
        # The generic text here is only the fallback when that evidence is
        # missing, because a retry told merely that it failed retries at random.
        "decorative_mutation": (
            "The child keeps the parent's entire proof skeleton and adds only closing "
            "steps, so the new notation carries no mathematical content. Change what "
            "must be proved -- the conclusion's shape, the modulus, the exponent, or a "
            "constant generalised into a second variable -- not the size of a coefficient."
        ),
        "parallel_crossover": (
            "The two parents are discharged independently and combined only at the "
            "closing step. Make one parent's result an input to the other's derivation: "
            "let it supply a bound, a modulus, an index, or a case distinction that the "
            "second parent's argument consumes."
        ),
        "same_formal_statement_as_parent": (
            "Change formal_statement and lean_code to prove a small new theorem, not the exact parent surface; "
            "valid patches are hypothesis_specialization, conclusion_projection, or immediate_corollary."
        ),
        "repair_not_harder": (
            "Do not submit a same-statement repair. Change operator_goal away from same_statement_repair "
            "by adding one explicit new proof obligation that is still locally provable."
        ),
        "same_lineage_crossover": (
            "Downgrade to mutation or choose a parent with a different root lineage; do not crossover a parent "
            "with its own descendant."
        ),
        "computational_crossover_only": (
            "Do not rely only on native_decide/computation. Add a theorem-style proof obligation whose statement "
            "uses both parent roles, or downgrade to mutation."
        ),
        "parameter_shift_only_theorem": (
            "Do not only change numerals in the parent theorem. Add one nonnumeric proof obligation, "
            "a derived local lemma, or a conclusion_projection that changes the theorem shape."
        ),
        "auxiliary_conjunct_only_theorem": (
            "Do not keep the parent conclusion as one conjunct and append a side fact. "
            "Use conclusion_projection, hypothesis_specialization, or make the new obligation replace the old goal form."
        ),
        "fin_one_vacuity_theorem": (
            "Do not collapse a decomposition theorem into a Fin 1 vacuity lemma. "
            "Use a non-vacuous index set or a parent checkpoint that still carries mathematical content."
        ),
        "concrete_native_decide_projection": (
            "Do not turn a theorem parent into a one-number native_decide computation. "
            "Keep a symbolic hypothesis/conclusion or a reusable proof checkpoint from the parent."
        ),
        "tautological_checkpoint_theorem": (
            "Do not prove divisibility only from a hypothesis that already states the exact required value. "
            "Replace the tautological checkpoint with a derived local lemma or a less direct hypothesis."
        ),
        "piecewise_branch_only_theorem": (
            "Do not only select an easy branch of a piecewise solution function. "
            "Add a theorem-level reason for why the branch applies or project a nontrivial parent checkpoint."
        ),
        "trivial_negation_chain": (
            "Do not repeat negation wrappers such as -(-u) or triple negation. "
            "Change formal_statement/proof_obligations to add a semantic lemma, stronger hypothesis interaction, "
            "or nontrivial conclusion projection."
        ),
        "trivial_add_zero_padding": (
            "Do not use + 0 or add_zero as the only change. Replace the padding with a real theorem-level "
            "obligation that changes the conclusion or a required proof checkpoint."
        ),
        "typeclass_narrowing_only": (
            "Do not only strengthen a typeclass such as Ring to CommRing while keeping the same conclusion. "
            "Either use the stronger structure in the conclusion/proof obligation or choose a smaller semantic mutation."
        ),
        "syntactic_wrapper_only": (
            "Do not submit a wrapper/simpa-only theorem. Change one of statement, formal_statement, or "
            "proof_obligations so the child has a new mathematical obligation."
        ),
        "side_by_side_conjunction": (
            "Do not prove parent A and parent B side by side. Convert the crossover into a pipeline: "
            "one parent supplies a checkpoint/object/hypothesis that appears inside the other's target."
        ),
        "parent_checkpoint_not_consumed": (
            "Your parent_usage text is not enough. Put each parent's checkpoint into formal_statement or lean_code "
            "so at least two parent-derived Lean atoms are observable in the child proof surface."
        ),
        "proof_infrastructure_only": (
            "Do not make an exact helper fact such as finset/card/prod the final result. "
            "Patch statement/formal_statement/proof_plan so that helper appears as a hypothesis or intermediate lemma "
            "feeding a final theorem target."
        ),
        "aggregate_helper_only": (
            "Do not stop at a standalone aggregate fact. Use the aggregate inside the final conclusion or proof_plan."
        ),
        "direct_parent_corollary_only": (
            "Do not return a direct projection/subset/index corollary of the parent. "
            "Patch goal and formal_statement to add a new proof obligation, latent parameter target, "
            "or nontrivial hypothesis specialization."
        ),
        "linear_equation_shift_corollary_only": (
            "Do not solve the same linear equation and change only the final arithmetic corollary such as y+k=c or t≤c. "
            "Patch to a parameter characterization, uniqueness theorem, or a theorem that consumes a second checkpoint."
        ),
        "affine_index_drift_only": (
            "Do not only change the index map such as u(p+k), u(2p), or a window length. "
            "Patch fusion_goal/proof_plan/formal_statement so a second aggregate or checkpoint is consumed."
        ),
        "cardinality_only_window": (
            "Cardinality alone is not accepted-grade. Add another consumed domain fact such as sum/product/membership, "
            "or switch to a different theorem target."
        ),
        "lineage_complexity_without_new_role": (
            "Simplify the generated surface and name one new mathematical role. "
            "Do not add a longer expression or lineage unless it changes the final proof obligation."
        ),
        "ap_index_only_theorem": (
            "Do not generate another single-index AP evaluation such as m = affine(p,q)+c or a m = constant. "
            "Patch to a closed-form, uniqueness, parameter-characterization, or extremal theorem."
        ),
        "ap_shifted_local_corollary_only": (
            "Do not generate local shifted AP corollaries such as a (m+k), local gaps, or midpoint sums. "
            "Patch to a closed-form, uniqueness, parameter-characterization, or theorem using a different final role."
        ),
        "ap_bound_padding_only": (
            "Do not wrap the solved AP value a+20*d=135 in an arbitrary upper-bound hypothesis such as 135≤B, "
            "135+t≤B, or C≤B. Patch to a closed-form, uniqueness, parameter-characterization, or a final theorem "
            "where the AP checkpoint is consumed by another mathematical object."
        ),
        "mod_inverse_same_conclusion_paraphrase": (
            "Do not keep the same modulo-inverse conclusion n=57 while only rewriting the hypothesis. "
            "Use the inverse as an input to another theorem or change the final goal type."
        ),
        "solved_parameter_quotient_corollary_only": (
            "Do not expose only arithmetic consequences of the solved inverse n=57 such as n/19=3, "
            "n=19*q, or 2^(n/19)=8. Patch so the inverse-derived parameter is consumed by a different "
            "theorem target, preferably through crossover or a symbolic condition."
        ),
        "residue_finset_cardinality_restatement": (
            "Do not restate the fixed residue-set cardinality or 3^n mod 8 fact. "
            "Make the residue/cardinality checkpoint feed another theorem target."
        ),
        "fixed_finite_aggregate_computation": (
            "Do not submit a fixed finite-set sum/product/card expression closed by native_decide. "
            "Introduce a symbolic condition, characterization, or pipeline input that is necessary for the final theorem."
        ),
        "cardinality_arithmetic_pipeline_only": (
            "Do not combine fixed cardinalities only as arithmetic in a modular-power expression. "
            "Make one cardinality checkpoint drive a symbolic condition, classification, or nontrivial final theorem role."
        ),
        "native_decide_fixed_domain_computation": (
            "Avoid fixed-domain native_decide computations as final generated problems. "
            "Move the computation into an intermediate checkpoint and prove a theorem-style final obligation."
        ),
        "artificial_bridge_to_existing_pipeline": (
            "Do not add an artificial bridge hypothesis just to feed an already accepted pipeline. "
            "Patch the crossover so the new parent changes the final theorem role, not only an input bound."
        ),
        "numeric_bound_fitting_crossover": (
            "Do not fuse parents by fitting constants into an inequality such as 135 ≤ 132 + n/19. "
            "Patch to a natural theorem target where both parent checkpoints define a reusable object or condition."
        ),
        "informal_statement_internal_terms": (
            "Rewrite only the informal statement in mathematical theorem style. Remove workflow terms such as "
            "checkpoint, parent, certified, generated, mutation, crossover, pipeline, operator, proof obligation, "
            "Lean, and formal. Put process evidence in proof_plan or parent_usage; do not change formal_statement "
            "or lean_code unless statement/formal alignment requires it."
        ),
    }
    patch_instructions.update(RETRY_PATCH_INSTRUCTIONS)
    # The structural-novelty checks measured *why* this row failed — which
    # tactics it shared, where its parents met — so they carry a brief written
    # against this row rather than against the flag in general. Prefer it: a
    # retry told only the flag name differs from the first attempt by luck.
    measured_brief = str(
        (evidence.get("structural_novelty") or {}).get("retry_brief") or ""
    )
    if measured_brief:
        for flag in ("decorative_mutation", "parallel_crossover"):
            if flag in retryable_flags:
                patch_instructions[flag] = measured_brief
    # The judge read this row and said why it failed. That sentence is worth
    # more to a retry than any flag-level advice, because it names the
    # mathematics rather than the category.
    judge_brief_text = str((evidence.get("judge") or {}).get("retry_brief") or "")
    if judge_brief_text and "judge_reject" in retryable_flags:
        patch_instructions["judge_reject"] = judge_brief_text
    selected_patches = [
        f"{flag}: {patch_instructions[flag]}"
        for flag in retryable_flags
        if flag in patch_instructions
    ]
    parts = [
        f"Previous attempt certified but quality_verdict=weak on attempt {attempt}.",
        f"quality_flags={retryable_flags or list(result.quality_flags or [])}.",
        f"missing_checkpoints={missing_checkpoints}.",
        f"crossover_kind={evidence.get('crossover_kind', '')}.",
        f"reasoning_signature={evidence.get('reasoning_signature', '')}.",
        f"solution_verification={evidence.get('solution_verification', {})}.",
        f"field_patch_instructions={selected_patches}.",
        f"feedback={str(result.feedback_for_next_generation or '')[:240]}.",
        "Keep target_family fixed; revise params, reasoning_pattern, solution_skeleton, "
        "parent_contribution_evidence, and solution to remove these weak-quality reasons.",
    ]
    return " ".join(part for part in parts if part)


def _lean_error_line_context(result: CertificationResult, *, radius: int = 2) -> List[str]:
    code = result.lean_code or ""
    if not code:
        return []
    line_numbers = {
        int(match.group(1))
        for match in re.finditer(r"(?:line|@ line)\s+(\d+)", str(result.error or result.proof_verify_summary or ""))
    }
    if not line_numbers:
        return []
    lines = code.splitlines()
    snippets: List[str] = []
    seen: set[int] = set()
    for line_no in sorted(line_numbers)[:3]:
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        for idx in range(start, end + 1):
            if idx in seen:
                continue
            seen.add(idx)
            snippets.append(f"{idx}: {lines[idx - 1]}")
    return snippets[:18]


def _theorem_patch_instructions(result: CertificationResult) -> List[str]:
    error = str(result.error or result.proof_verify_summary or "").lower()
    instructions: List[str] = []
    if "unknown identifier" in error or "unknown constant" in error:
        instructions.append(
            "Remove invented lemma names; use only parent-visible names or direct tactic proof."
        )
    if "did not find an occurrence" in error or "rewrite" in error:
        instructions.append(
            "Patch the rw/rewrite target to an expression that appears in the local goal, or replace the rewrite with a smaller direct proof."
        )
    if "simp made no progress" in error:
        instructions.append(
            "Do not rely on simp alone; rewrite lean_code using an explicit witness, a parent-visible theorem application, or shrink the statement."
        )
    if "no goals" in error:
        instructions.append(
            "A tactic ran after its goal was already closed: the script has more "
            "steps or branches than open goals. Keep formal_statement unchanged; "
            "delete the tactics after the closing step and make focus bullets "
            "(`·`) and `<;>` branch counts match the actual number of goals."
        )
    if "unsolved goals" in error:
        instructions.append(
            "The diagnostics list the exact remaining goals. Keep formal_statement "
            "unchanged first: add the missing case branches or finishing tactics "
            "that close each listed goal. Only shrink formal_statement to one "
            "selected_parent_checkpoint if a remaining goal is genuinely unprovable."
        )
    if "application type mismatch" in error or "type mismatch" in error:
        instructions.append(
            "Keep the theorem intent but repair binders, coercions, and proof body types before changing the statement."
        )
    if "linarith failed" in error:
        instructions.append(
            "Remove linarith unless the assumptions contain the exact linear contradiction; prefer a smaller projected conclusion."
        )
    if "sorry" in str(result.lean_code or "").lower():
        instructions.append("Remove sorry; regenerate only the proof body or shrink the theorem.")
    if not instructions:
        instructions.append(
            "First try a smaller formal_statement and a simpler proof body before broadening the theorem."
        )
    return instructions[:4]


def _retry_feedback_for_result(
    result: CertificationResult,
    op_type: str,
    attempt: int,
    attempt_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    history_prefix = ""
    if attempt_history:
        history_prefix = (
            "Prior attempt history summary="
            + json.dumps(_attempt_history_summary(attempt_history), ensure_ascii=False)[:1800]
            + ". "
        )
    if result.status == "planner_axis_mismatch":
        return history_prefix + (
            f"Previous attempt failed with status={result.status}; "
            f"failure_signature={_failure_signature(result)}; "
            f"previous_error={str(result.error or '')[:500]}; "
            "fix the structured generation contract. For crossover, populate "
            "parent_contribution_evidence and solution_skeleton.parent_contributions "
            "with every parent id exactly; do not put parent evidence only in axis_applied."
        )
    if _retryable_generation_failure(result):
        return history_prefix + (
            f"Previous attempt failed with status={result.status}; "
            f"failure_class={_failure_class(result)}; "
            f"failure_signature={_failure_signature(result)}; "
            f"previous_error={str(result.error or '')[:500]}; "
            "field_patch_instruction: return status=generated with a smaller statement/formal_statement/lean_code, "
            "or return status=cannot_execute with reason if the OperatorCard is not executable. "
            "Do not output proof_surface; the verifier infers proof shape from lean_code. "
            "patch_target_fields=[status,reason,statement,formal_statement,lean_code,proof_plan,parent_usage]; "
            "must_not_change_fields=[target_style,target_family,parent_ids]."
        )
    if _retryable_lean_repair_failure(result, op_type):
        return history_prefix + (
            f"Previous attempt failed with status={result.status}; "
            f"failure_signature={_failure_signature(result)}; "
            f"previous_error={str(result.error or '')[:500]}; "
            "change only params, answer, and solution_skeleton inside the same target_family; "
            "consult ParentProofContext for proof-style hints without copying unsupported proof code."
        )
    if _retryable_theorem_failure(result, op_type):
        if result.status == "statement_failed":
            line_context = _lean_error_line_context(result)
            return history_prefix + (
                "Previous theorem attempt failed statement-level type checking: "
                "the formal_statement alone (closed with sorry, under "
                "`set_option autoImplicit false`) does not elaborate. "
                f"previous_error={str(result.error or result.proof_verify_summary or '')[:700]}; "
                f"line_patch_context={line_context}; "
                "Fix the statement itself before any proof: correct binder types, "
                "use only Mathlib-visible names (no invented identifiers), and "
                "repair notation. Every variable must be explicitly bound. "
                "If the statement refers to a helper def/lemma you authored in "
                "lean_code, inline that object into the statement with explicit "
                "binders (e.g. `∃ f : ℝ → Quaternion ℝ, (∀ t, ...) ∧ ...`) — the "
                "statement must elaborate on its own against Mathlib only. "
                "patch_target_fields=[formal_statement,lean_code,proof_plan]; "
                "must_not_change_fields=[target_style,target_family,parent_ids,statement]."
            )
        if result.status == "alignment_failed":
            alignment = dict((result.quality_evidence or {}).get("theorem_alignment") or {})
            return history_prefix + (
                "Previous theorem attempt failed statement/formal alignment before Lean verification; "
                f"unsupported_claims={alignment.get('unsupported_claims', [])}; "
                f"missing_claims={alignment.get('missing_claims', [])}; "
                f"field_patch_instructions={alignment.get('field_patch_instructions', [])}; "
                "revise statement_chunks, statement, formal_statement, and lean_code so every "
                "natural-language claim is represented formally. Remove prose-only claims."
            )
        line_context = _lean_error_line_context(result)
        patch_instructions = _theorem_patch_instructions(result)
        return history_prefix + (
            f"Previous theorem attempt failed Lean verification; "
            f"failure_signature={_failure_signature(result)}; "
            f"previous_error={str(result.error or result.proof_verify_summary or '')[:700]}; "
            f"line_patch_context={line_context}; "
            f"field_patch_instructions={patch_instructions}; "
            "patch_target_fields=[formal_statement,lean_code,proof_plan,parent_usage]; "
            "Do not output proof_surface; rewrite lean_code so the proof is visibly an explicit witness, "
            "parent theorem application, simp computation, constructor case proof, or other complete Lean proof. "
            "must_not_change_fields=[target_style,target_family,parent_ids,statement] unless "
            "the error requires shrinking formal_statement. "
            "Keep target_style=theorem_proof and parent style."
        )
    quality_feedback = _quality_retry_feedback(result, attempt)
    if quality_feedback:
        return history_prefix + quality_feedback
    return history_prefix + (
        f"Previous attempt failed with status={result.status}; "
        f"failure_signature={_failure_signature(result)}; "
        f"previous_error={str(result.error or '')[:500]}; revise the generated child."
    )


def _signature_group_limit(signature_group: str) -> int:
    if signature_group == "theorem_proof":
        return POOL_SIZE
    if signature_group == "gcd_sigma":
        return 1
    return 2


def _plan_outcome_summary(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    recent = list(cards or [])[-15:]
    op_attempts: Counter[str] = Counter()
    op_saved: Counter[str] = Counter()
    op_weak: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    axes_not_selected: List[str] = []
    elite_skeleton_cards: List[Dict[str, Any]] = []
    success_case_cards: List[Dict[str, Any]] = []
    failure_case_cards: List[Dict[str, Any]] = []
    weak_signature_summary: Counter[str] = Counter()
    signature_groups: Counter[str] = Counter()
    for card in recent:
        op_type = card.get("op_type") or "unknown"
        if op_type in {"mutation", "crossover"}:
            op_attempts[op_type] += 1
            if card.get("selected_for_next_pool"):
                op_saved[op_type] += 1
            if card.get("quality_verdict") == "weak":
                op_weak[op_type] += 1
        if card.get("failure_signature"):
            failures[card["failure_signature"]] += 1
        evidence = dict(card.get("quality_evidence") or {})
        reasoning_signature = str(evidence.get("reasoning_signature") or "")
        signature_group = str(evidence.get("signature_group") or "")
        if card.get("selected_for_next_pool") and signature_group:
            signature_groups[signature_group] += 1
        if card.get("selected_for_next_pool") and reasoning_signature:
            operator_card = dict(card.get("operator_card") or {})
            success_card = {
                "slot": card.get("slot"),
                "op_type": card.get("op_type"),
                "operator_variant": operator_card.get("operator_variant"),
                "target_family": card.get("target_family"),
                "goal": _prompt_text(operator_card.get("goal") or operator_card.get("operator_goal"), limit=180),
                "reasoning_signature": reasoning_signature,
                "checkpoint_coverage": evidence.get("checkpoint_coverage"),
                "lesson": _planner_memory_lesson(
                    kind="success",
                    row={
                        **card,
                        "operator_card": operator_card,
                        "quality_evidence": evidence,
                        "target_family": card.get("target_family"),
                    },
                    quality_flags=list(card.get("quality_flags") or []),
                    failure_class=str(card.get("failure_class") or ""),
                    reasoning_signature=reasoning_signature,
                ),
                "raw_surface": {
                    "statement": _prompt_text(card.get("statement"), limit=4000),
                    "formal_statement": _prompt_text(card.get("formal_statement"), limit=9000),
                    "lean_code": _prompt_text(card.get("lean_code"), limit=14000),
                    "proof_plan": _prompt_text(card.get("proof_plan"), limit=4000),
                    "proof_obligations": list(card.get("proof_obligations") or [])[:12],
                    "answer": _prompt_text(card.get("answer"), limit=1000),
                    "solution": _prompt_text(card.get("solution"), limit=7000),
                    "attempt_history": _planner_attempt_history_surface(card.get("attempt_history")),
                },
            }
            elite_skeleton_cards.append(
                {
                    "slot": success_card["slot"],
                    "op_type": success_card["op_type"],
                    "target_family": success_card["target_family"],
                    "reasoning_signature": success_card["reasoning_signature"],
                    "checkpoint_coverage": success_card["checkpoint_coverage"],
                }
            )
            success_case_cards.append(success_card)
        if card.get("quality_verdict") == "weak" and reasoning_signature:
            weak_signature_summary[reasoning_signature] += 1
        if card.get("op_type") != "survivor" and not card.get("selected_for_next_pool"):
            axis = str(card.get("planned_variation_axis") or "")[:160]
            if axis and axis not in axes_not_selected:
                axes_not_selected.append(axis)
            flags = list(card.get("quality_flags") or [])[:6]
            failure_class = str(card.get("failure_class") or "")
            failure_case_cards.append(
                {
                    "slot": card.get("slot"),
                    "op_type": card.get("op_type"),
                    "target_family": card.get("target_family"),
                    "status": card.get("status"),
                    "quality_verdict": card.get("quality_verdict"),
                    "quality_flags": flags,
                    "failure_class": failure_class,
                    "failure_signature": card.get("failure_signature"),
                    "reasoning_signature": reasoning_signature,
                    "lesson": _planner_memory_lesson(
                        kind="failure",
                        row=card,
                        quality_flags=flags,
                        failure_class=failure_class,
                        reasoning_signature=reasoning_signature,
                    ),
                    "raw_surface": {
                        "statement": _prompt_text(card.get("statement"), limit=4000),
                        "formal_statement": _prompt_text(card.get("formal_statement"), limit=9000),
                        "lean_code": _prompt_text(card.get("lean_code"), limit=14000),
                        "proof_plan": _prompt_text(card.get("proof_plan"), limit=4000),
                        "proof_obligations": list(card.get("proof_obligations") or [])[:12],
                        "answer": _prompt_text(card.get("answer"), limit=1000),
                        "solution": _prompt_text(card.get("solution"), limit=7000),
                        "error": _prompt_text(card.get("error") or card.get("proof_verify_summary"), limit=5000),
                        "attempt_history": _planner_attempt_history_surface(card.get("attempt_history")),
                    },
                }
            )
    return {
        "recent_cards": [
            {
                "slot": card.get("slot"),
                "op_type": card.get("op_type"),
                "planned_op_type": card.get("planned_op_type"),
                "target_family": card.get("target_family"),
                "status": card.get("status"),
                "quality_verdict": card.get("quality_verdict"),
                "quality_flags": list(card.get("quality_flags") or [])[:6],
                "failure_signature": card.get("failure_signature"),
                "selected_for_next_pool": bool(card.get("selected_for_next_pool")),
                "selection_reason": card.get("selection_reason"),
                "reasoning_signature": (
                    dict(card.get("quality_evidence") or {}).get("reasoning_signature")
                ),
            }
            for card in recent
        ],
        "op_type_stats": {
            op_type: {
                "attempts": op_attempts[op_type],
                "selected": op_saved[op_type],
                "weak": op_weak[op_type],
                "selected_rate": round(op_saved[op_type] / op_attempts[op_type], 3)
                if op_attempts[op_type]
                else 0.0,
                "weak_rate": round(op_weak[op_type] / op_attempts[op_type], 3)
                if op_attempts[op_type]
                else 0.0,
            }
            for op_type in ("mutation", "crossover")
            if op_attempts[op_type]
        },
        "recurrent_failure_signatures": [
            sig for sig, count in failures.most_common(5) if count >= 2
        ],
        "axes_planned_but_not_selected": axes_not_selected[:6],
        "elite_skeleton_cards": elite_skeleton_cards[-6:],
        "success_case_cards": success_case_cards[-5:],
        "failure_case_cards": failure_case_cards[-5:],
        "weak_signature_summary": dict(weak_signature_summary.most_common(6)),
        "dominant_signature_groups": {
            group: count for group, count in signature_groups.items() if count >= 2
        },
    }


def _op_type_allocation_hint(
    plan_outcome_cards: List[Dict[str, Any]],
    *,
    pool_size: int,
    survivor_count: int,
    crossover_count: int,
) -> Dict[str, Any]:
    non_survivor = pool_size - survivor_count
    summary = _plan_outcome_summary(plan_outcome_cards)
    stats = summary.get("op_type_stats", {})
    mutation = stats.get("mutation", {})
    crossover = stats.get("crossover", {})
    if not mutation or not crossover:
        return {
            "mutation": non_survivor - crossover_count,
            "crossover": crossover_count,
            "confidence": "default",
            "rationale": "Insufficient history; keep requested default split.",
            "plan_outcome_summary": summary,
        }
    crossover_bad = float(crossover.get("weak_rate", 0.0)) >= 0.5 or float(crossover.get("selected_rate", 0.0)) < 0.5
    if crossover_bad:
        crossover_slots = max(1, crossover_count - 1)
        return {
            "mutation": non_survivor - crossover_slots,
            "crossover": crossover_slots,
            "confidence": "data_driven",
            "min_crossover_exploration": 1 if crossover_count else 0,
            "rationale": (
                "Recent crossover slots were often weak or unselected; keep one "
                "crossover_easy exploration slot and put remaining pressure on mutation."
            ),
            "plan_outcome_summary": summary,
        }
    return {
        "mutation": non_survivor - crossover_count,
        "crossover": crossover_count,
        "confidence": "data_driven",
        "min_crossover_exploration": 1 if crossover_count else 0,
        "rationale": "Recent crossover quality is acceptable; keep requested split.",
        "plan_outcome_summary": summary,
    }


#: What the planner reads from a previous-generation case's raw surface. The
#: statement is the surface it is told to prefer or avoid; the proof body, the
#: solution text, the attempt history and the operator card are not, and they
#: are what makes a case 12,910 characters of which 12,465 is `raw_surface`.
#: Six such cases filled 49,745 characters, 53% of the planner's whole prompt,
#: to convey six statements and six lessons.
_FEEDBACK_SURFACE_FIELDS = ("source_kind", "formal_statement", "statement", "error")


def _trim_feedback_case(case: Any) -> Dict[str, Any]:
    """One previous-generation case with its proof artifacts dropped."""
    row = dict(case or {})
    raw = dict(row.get("raw_surface") or {})
    if raw:
        row["raw_surface"] = {
            key: _prompt_text(raw.get(key), limit=900)
            for key in _FEEDBACK_SURFACE_FIELDS
            if str(raw.get(key) or "").strip()
        }
    return row


def _planner_feedback_payload(generation_feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Raw-enough previous-generation cases for planner strategy selection."""
    summary = dict(generation_feedback.get("plan_outcome_summary") or {})
    return {
        "success_cases": [_trim_feedback_case(c) for c in summary.get("success_case_cards", [])],
        "failure_cases": [_trim_feedback_case(c) for c in summary.get("failure_case_cards", [])],
        "elite_skeleton_cards": summary.get("elite_skeleton_cards", []),
        "weak_signature_summary": summary.get("weak_signature_summary", {}),
        "dominant_signature_groups": summary.get("dominant_signature_groups", {}),
        "recurrent_failure_signatures": summary.get("recurrent_failure_signatures", []),
        "axes_planned_but_not_selected": summary.get("axes_planned_but_not_selected", []),
    }


def _reserve_failure_profile(results: List[CertificationResult]) -> Dict[str, Any]:
    generated = [result for result in results if _is_generated_result(result)]
    proxy_flags = Counter(
        flag
        for result in generated
        for flag in list(_accepted_proxy(result).get("flags") or [])
    )
    quality_flags = Counter(flag for result in generated for flag in list(result.quality_flags or []))
    selection_reasons = Counter(
        str(result.selection_reason or "unset") for result in generated if not result.parent_eligible
    )
    avoid_signatures = [
        str((result.quality_evidence or {}).get("reasoning_signature") or "")
        for result in generated
        if (result.quality_evidence or {}).get("reasoning_signature")
    ]
    return {
        "accepted_proxy_flags": dict(proxy_flags),
        "quality_flags": dict(quality_flags),
        "selection_reasons": dict(selection_reasons),
        "avoid_signatures": list(dict.fromkeys(avoid_signatures))[:10],
    }


def _reserve_goal_from_profile(profile: Dict[str, Any], *, op_type: str) -> tuple[str, List[str]]:
    flags = set(profile.get("accepted_proxy_flags") or {}) | set(profile.get("quality_flags") or {})
    reasons = set(profile.get("selection_reasons") or {})
    avoid = [
        "reserve_slot",
        "do_not_repeat_prior_failed_surface",
        "do_not_emit_projection_only_or_same_statement",
    ]
    if flags & {"ap_index_only_theorem", "ap_shifted_local_corollary_only"}:
        return (
            "closed_form_or_characterization: AP single-index and shifted-local corollaries are forbidden; prove a closed-form, uniqueness, parameter-characterization, or extremal theorem",
            avoid + [
                "ap_index_only_theorem",
                "ap_shifted_local_corollary_only",
                "single_index_evaluation",
                "shifted_local_corollary",
                "hidden_parameter_index_only",
                "required_new_role:closed_form_or_characterization",
            ],
        )
    if flags & {"mod_inverse_same_conclusion_paraphrase"}:
        return (
            "inverse_as_input: do not keep final goal n=57; use the modular inverse as an input/checkpoint in a different theorem target",
            avoid + [
                "mod_inverse_same_conclusion_paraphrase",
                "same_conclusion_n_equals_57",
                "modulo_quotient_paraphrase",
                "required_new_role:pipeline_input_or_classification",
            ],
        )
    if flags & {"residue_finset_cardinality_restatement"}:
        return (
            "residue_checkpoint_pipeline: do not restate S.card or 3^n mod 8; make the residue/cardinality checkpoint feed another theorem target",
            avoid + [
                "residue_finset_cardinality_restatement",
                "fixed_residue_cardinality_only",
                "required_new_role:pipeline_input",
            ],
        )
    if flags & {
        "fixed_finite_aggregate_computation",
        "native_decide_fixed_domain_computation",
        "cardinality_arithmetic_pipeline_only",
        "artificial_bridge_to_existing_pipeline",
        "numeric_bound_fitting_crossover",
    }:
        return (
            "symbolic_or_pipeline_target: fixed finite-set/cardinality arithmetic, artificial bridges, and fitted numeric bounds are forbidden as final targets; introduce a symbolic condition or consume the computation inside a larger theorem",
            avoid + [
                "fixed_finite_aggregate_computation",
                "native_decide_fixed_domain_computation",
                "cardinality_arithmetic_pipeline_only",
                "artificial_bridge_to_existing_pipeline",
                "numeric_bound_fitting_crossover",
                "fixed_domain_native_decide_final_goal",
                "cardinality_arithmetic_wrapper",
                "artificial_bridge_hypothesis",
                "constant_fitted_bound",
                "required_new_role:symbolic_condition_or_master_theorem",
            ],
        )
    if flags & {"projection_only_theorem", "same_formal_statement_as_parent", "formal_surface_not_changed"}:
        return (
            "reserve repair: change the conclusion/proof obligation shape, not just project or restate the parent",
            avoid + ["projection_only_theorem", "same_formal_statement_as_parent"],
        )
    if flags & {"proof_infrastructure_only", "aggregate_helper_only"}:
        return (
            "domain_pipeline_sum: helper facts such as exact finset/card/prod are not enough; make the helper feed a final theorem target",
            avoid + ["proof_infrastructure_only", "helper_only", "aggregate_helper_only", "target_playbook:domain_pipeline_sum"],
        )
    if flags & {
        "direct_parent_corollary_only",
        "linear_equation_shift_corollary_only",
        "ap_bound_padding_only",
        "solved_parameter_quotient_corollary_only",
        "same_target_role_already_accepted",
        "witness_packaging_only",
        "coefficient_engineering_only",
    }:
        return (
            "new_final_theorem_role: avoid direct corollaries, same target roles, witness packaging, and coefficient engineering; add a distinct theorem obligation",
            avoid + [
                "direct_parent_corollary_only",
                "linear_equation_shift_corollary_only",
                "ap_bound_padding_only",
                "solved_parameter_quotient_corollary_only",
                "same_target_role_already_accepted",
                "witness_packaging_only",
                "coefficient_engineering_only",
                "direct_parent_corollary",
                "linear_shift_corollary",
                "ap_bound_padding",
                "solved_parameter_quotient_corollary",
                "same_target_role_variant",
                "explicit_witness_packaging",
                "rational_coefficient_engineering",
                "target_playbook:latent_parameter_solve",
            ],
        )
    if flags & {"affine_index_drift_only", "cardinality_only_window"}:
        return (
            "card_and_sum_pipeline: do not only change the affine index or window length; consume a second aggregate/checkpoint in the final theorem",
            avoid + ["affine_index_drift_only", "cardinality_only_window", "same_domain_affine_drift", "target_playbook:card_and_sum_pipeline"],
        )
    if flags & {"lineage_complexity_without_new_role"}:
        return (
            "nontrivial_hypothesis_specialization: reduce lineage complexity and add one clear new mathematical role instead of a longer expression",
            avoid + ["lineage_complexity_without_new_role", "id_explosion", "target_playbook:nontrivial_hypothesis_specialization"],
        )
    if flags & {"parent_checkpoint_not_consumed", "unused_checkpoint", "crossover_parent_usage_not_observable"}:
        return (
            "reserve crossover: make the parent checkpoint appear in formal_statement, proof_plan, or lean_code",
            avoid + ["unused_parent_checkpoint", "prose_only_parent_usage"],
        )
    if "repeated_reasoning_signature" in reasons or "lineage_cap" in reasons:
        return (
            "reserve diversification: use an underused parent/root and create a different reasoning signature",
            avoid + ["repeated_reasoning_signature", "same_lineage_crossover"],
        )
    if op_type == "crossover":
        return (
            "lemma_bundle_master or domain_pipeline_sum: build a pipeline_composite with observable parent usage",
            avoid + ["side_by_side_conjunction", "target_playbook:lemma_bundle_master"],
        )
    return (
        "reserve mutation: create an accepted-level semantic change with complete Lean proof",
        avoid + ["trivial_local_corollary"],
    )


def _build_reserve_work_items(
    pool: List[CertificationInput],
    results: List[CertificationResult],
    *,
    target_accepted: int,
    reserve_budget: int,
    crossover_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    generated = [result for result in results if _is_generated_result(result)]
    accepted_count = sum(1 for result in generated if _accepted_grade_proxy_pass(result))
    needed = max(0, int(target_accepted) - accepted_count)
    reserve_count = min(int(reserve_budget), needed + 1 if needed else 0)
    if reserve_count <= 0:
        return []
    profile = _reserve_failure_profile(results)
    parent_map = _problem_by_id(pool)
    lineage_counts = Counter(
        root for result in results for root in _result_root_lineages(result)
    )
    parents = sorted(
        pool,
        key=lambda parent: (lineage_counts[_root_lineage_id(parent.id)], parent.id),
    )
    work_items: List[Dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for index in range(reserve_count):
        slot = POOL_SIZE + index
        theorem_pool = all(_problem_style(parent) == "theorem_proof" for parent in pool)
        # The lock has to reach here too. The planner's plan is validated
        # against `crossover_count`, and under the lock a plan that misses it is
        # rejected outright — but reserve slots are built after that check, from
        # the failure profile rather than from the plan, and this branch chose
        # crossover on its own. In a mutation-only arm with crossover_count=0 it
        # produced a crossover in every generation, which is the one thing the
        # arm exists to exclude: the failures of one operator were being counted
        # as the output of the other.
        # Under the lock the reserve slot follows the arm, and the arm is named
        # by its budget: `crossover_count=0` is the mutation arm and must not
        # produce one, `crossover_count>0` is the crossover arm and should.
        # Unlocked, the reserve slot keeps its own judgement.
        locked_to_mutation = _op_type_locked() and (crossover_count or 0) == 0
        use_crossover = (
            theorem_pool and len(pool) >= 2 and index == 0 and not locked_to_mutation
        )
        if use_crossover:
            first = parents[index % len(parents)]
            second = next(
                (
                    parent
                    for parent in parents
                    if parent.id != first.id
                    and not _same_root_lineage([first.id, parent.id])
                    and tuple(sorted([first.id, parent.id])) not in used_pairs
                ),
                parents[(index + 1) % len(parents)],
            )
            used_pairs.add(tuple(sorted([first.id, second.id])))
            goal, avoid = _reserve_goal_from_profile(profile, op_type="crossover")
            item = PoolWorkItem(
                slot=slot,
                op_type="crossover",
                operator_variant="crossover_easy",
                parent_ids=[first.id, second.id],
                parent_refs=[pool.index(first), pool.index(second)],
                target_style="theorem_proof",
                target_family="theorem_proof",
                variation_axis=goal,
                reasoning_goal=goal,
                operator_goal=goal,
                composition_pattern="family_bridge",
                parent_contributions={
                    first.id: "reserve parent checkpoint must be consumed in the child proof surface",
                    second.id: "reserve target theorem must consume the other parent checkpoint",
                },
                avoid_patterns=avoid,
                avoid=avoid,
                required_checkpoints=[
                    "theorem_style_preserved",
                    "lean_proof_complete",
                    "semantic_parent_contribution",
                ],
                avoid_signatures=list(profile.get("avoid_signatures") or []),
                fusion_goal="pipeline_composite or lemma_bundle_master with field-observable parent usage",
                parent_roles={
                    first.id: "checkpoint source",
                    second.id: "goal/proof target source",
                },
                fusion_contract=_default_fusion_contract(
                    first=first,
                    second=second,
                    mechanism="",
                    parent_a_role="proof_skeleton",
                    parent_b_role="goal_form",
                    parent_a_contribution="checkpoint/object/hypothesis is consumed by the child",
                    parent_b_contribution="goal form/proof target consumes the checkpoint",
                    new_problem_core="reserve pipeline/master theorem with observable parent usage",
                    why_not_concatenation="One parent checkpoint must appear in formal_statement, proof_plan, or lean_code.",
                ),
                quality_target="accepted-proxy theorem crossover",
                planner_source="reserve_deterministic",
            )
            payload = item.model_dump()
            payload["parent_context_cards"] = _parent_context_cards([first, second])
        else:
            parent = parents[index % len(parents)]
            parent_style = _problem_style(parent)
            family = _problem_family(parent) or ("theorem_proof" if parent_style == "theorem_proof" else "")
            target_family = "theorem_proof" if parent_style == "theorem_proof" else family
            goal, avoid = _reserve_goal_from_profile(profile, op_type="mutation")
            item = PoolWorkItem(
                slot=slot,
                op_type="mutation",
                operator_variant="mutation_hard" if parent_style == "theorem_proof" else "mutation_easy",
                parent_ids=[parent.id],
                parent_refs=[pool.index(parent)],
                target_style=parent_style,
                target_family=target_family,
                variation_axis=goal,
                reasoning_goal=goal,
                operator_goal=goal,
                composition_pattern="structure_expansion" if parent_style == "theorem_proof" else "parameter_shift",
                parent_contributions={parent.id: "reserve mutation must add a semantic proof obligation"},
                avoid_patterns=avoid,
                avoid=avoid,
                required_checkpoints=(
                    ["theorem_style_preserved", "lean_proof_complete", "harder_proof_obligation"]
                    if parent_style == "theorem_proof"
                    else ["reasoning_pattern", "solution_skeleton", "projected_params"]
                ),
                avoid_signatures=list(profile.get("avoid_signatures") or []),
                quality_target="accepted-proxy mutation",
                planner_source="reserve_deterministic",
            )
            payload = item.model_dump()
            payload["parent_context_cards"] = _parent_context_cards([parent])
        payload["source_kind"] = "reserve_generated"
        payload["operator_card"] = _operator_card(payload)
        work_items.append(payload)
    return work_items


def _json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def _row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _planner_memory_problem_style(row: Dict[str, Any]) -> str:
    explicit = str(row.get("problem_style") or row.get("target_style") or "").strip()
    if explicit in PROBLEM_STYLES:
        return explicit
    route = str(row.get("certification_route") or "")
    if route == "theorem_prover" or row.get("formal_statement") or row.get("lean_code"):
        return "theorem_proof"
    return "numeric_answer"


def _planner_memory_failure_class(row: Dict[str, Any]) -> str:
    explicit = str(row.get("failure_class") or "").strip()
    if explicit:
        return explicit
    status = str(row.get("status") or "")
    if status == "generation_failed":
        return "generation_failed"
    if status == "proof_failed":
        return "proof_failed"
    if status in {"alignment_failed", "planner_axis_mismatch"}:
        return status
    if str(row.get("aligned") or "").lower() == "false":
        return "alignment_failed"
    if row.get("error"):
        return status or "failed"
    return ""


def _planner_memory_lesson(
    *,
    kind: str,
    row: Dict[str, Any],
    quality_flags: List[str],
    failure_class: str,
    reasoning_signature: str,
) -> str:
    if kind == "success":
        variant = _json_dict(row.get("operator_card")).get("operator_variant") or row.get("operator_variant")
        goal = _json_dict(row.get("operator_card")).get("goal") or row.get("operator_goal") or row.get("goal")
        surface = str(reasoning_signature or row.get("target_family") or row.get("family") or "this surface")
        return f"Reuse this pattern when compatible: {variant or row.get('op_type')} produced parent-eligible {surface}; goal={str(goal or '')[:160]}"
    flag_lessons = {
        "same_formal_statement_as_parent": "Avoid same_statement_repair unless the goal is Gen0 proof completion; generated children must change the formal theorem surface.",
        "trivial_negation_chain": "Avoid theorem mutations that only add repeated negation wrappers such as -(-u) or triple negation.",
        "trivial_add_zero_padding": "Avoid theorem mutations that only add + 0 or add_zero padding; require a new mathematical proof obligation.",
        "typeclass_narrowing_only": "Avoid making a theorem harder only by strengthening Ring to CommRing or similar typeclass narrowing with the same conclusion.",
        "syntactic_wrapper_only": "Avoid syntactic wrapper repairs that keep the same theorem content; require a semantic change in statement or proof obligation.",
        "side_by_side_conjunction": "Do not plan side-by-side conjunction crossover; one parent result must feed into the other's goal or proof obligation.",
        "repair_not_harder": "Avoid repair-only theorem mutations as generated children; require a visible theorem-style improvement.",
        "parameter_shift_only_theorem": "Do not plan theorem mutation as a plain numeral shift with mechanical proof.",
        "auxiliary_conjunct_only_theorem": "Avoid appending a side conjunct to the parent conclusion; prefer projection or hypothesis specialization.",
        "same_lineage_crossover": "Do not crossover a seed with its descendant; use mutation on one lineage instead.",
        "computational_crossover_only": "Do not treat sequential arithmetic pipelines as strong fusion.",
        "fin_one_vacuity_theorem": "Avoid Fin 1/vacuity theorem children; require a non-vacuous proof obligation.",
        "concrete_native_decide_projection": "Avoid one-number native_decide theorem children from theorem parents.",
        "tautological_checkpoint_theorem": "Avoid checkpoint hypotheses that make the conclusion immediate by restatement.",
        "piecewise_branch_only_theorem": "Avoid only selecting an easy branch of a piecewise solution function.",
        "certification_not_successful": "Choose a smaller executable target when certification failed.",
        "missing_formal_statement": "For theorem slots, require formal_statement and lean_code artifacts.",
        "parent_checkpoint_not_consumed": "Do not plan crossover from prose-only parent usage; require parent checkpoints to appear in the child Lean/formal surface.",
        "unused_checkpoint": "Do not pass a parent checkpoint through as an unused hypothesis; it must affect the final goal or a necessary proof step.",
        "accepted_proxy_failed": "Certified but not accepted-grade; plan a child with a real semantic change and observable parent usage.",
        "curation_scaffold": "Treat this as a scaffold/intermediate case, not as a paper-grade success; make the next plan consume it into a new final theorem role.",
        "curation_reject": "Avoid this final theorem pattern; it passed certification but failed curation as a usable generated problem.",
        "formal_surface_not_changed": "Avoid theorem children whose formal surface is unchanged from the parent.",
        "crossover_parent_usage_not_observable": "For crossover, make both parent contributions visible in statement, formal_statement, proof_plan, or lean_code.",
        "manual_qa_reject": "Treat manually rejected certified rows as failure examples for future planning.",
        "orderof_orientation_projection": "Do not keep generating orderOf orientation variants after one unit-witness/order theorem is already represented; add a new object/value/proof obligation.",
        "linear_factor_paraphrase_only": "Do not keep generating X - C 1 versus C 1 - X linear-factor paraphrases after a stronger rational-root obstruction exists.",
        "too_narrow_root_instance": "Avoid a single root-at-one theorem when the available parent supports a broader no-rational-root obstruction.",
        "axiom_backed_seed_or_child": "Do not treat axiom-backed Lean artifacts as successful proof bodies.",
        "proof_infrastructure_only": "Do not treat helper facts such as exact finset/card/prod as accepted problems; make the helper feed a final theorem target.",
        "aggregate_helper_only": "Avoid aggregate helper-only rows; use card/sum/product data inside a final theorem rather than as the final result.",
        "direct_parent_corollary_only": "Avoid direct parent corollaries such as odd/even AP sums unless a new proof obligation or latent parameter target is added.",
        "linear_equation_shift_corollary_only": "Avoid solving the same linear equation and only changing the final corollary y+k=c or t≤c; require a new theorem role.",
        "affine_index_drift_only": "Avoid repeating the same domain pipeline by only changing u(p+k), u(2p), or a window length.",
        "cardinality_only_window": "Do not use only a cardinality-controlled range window; consume another domain aggregate or checkpoint in the final theorem.",
        "lineage_complexity_without_new_role": "Avoid long lineage/id-exploded expressions unless a new mathematical role is clearly introduced.",
        "ap_index_only_theorem": "Avoid AP single-index/hidden-parameter evaluations; plan a closed-form, uniqueness, or parameter-characterization theorem instead.",
        "ap_shifted_local_corollary_only": "Avoid AP shifted local corollaries such as a(m+k), local gaps, or midpoint sums; plan a closed-form or parameter-characterization theorem instead.",
        "ap_bound_padding_only": "Avoid AP bound-padding corollaries that only turn a+20*d=135 into a+20*d≤B or a shifted two-step bound; use the AP checkpoint as an input to a real final theorem.",
        "mod_inverse_same_conclusion_paraphrase": "Avoid modulo-398 inverse paraphrases that keep n=57 as the final goal; use the inverse as an input to another theorem.",
        "solved_parameter_quotient_corollary_only": "Avoid quotient/power corollaries that only expose arithmetic after n=57; make the inverse-derived parameter feed a separate theorem target.",
        "residue_finset_cardinality_restatement": "Avoid restating the fixed residue finset cardinality or 3^n mod 8 fact; make it feed a different target.",
        "fixed_finite_aggregate_computation": "Avoid fixed finite-set aggregate expressions closed by native_decide as final generated problems.",
        "cardinality_arithmetic_pipeline_only": "Avoid combining fixed cardinalities only as arithmetic in a modular-power expression; require a symbolic condition or classification role.",
        "native_decide_fixed_domain_computation": "Avoid fixed-domain native_decide computations; require a symbolic or pipeline theorem role.",
        "artificial_bridge_to_existing_pipeline": "Avoid bridge hypotheses that only route a new parent into an already accepted pipeline; require the new parent to change the final theorem role.",
        "numeric_bound_fitting_crossover": "Avoid constant-fitted numeric inequalities as crossover; require a natural object, condition, or reusable theorem role.",
        "order_equality_selector_only": "Avoid using orderOf equality only as an if/indicator selector inside an unrelated rational factor; make it drive a group-theoretic target.",
        "same_target_role_already_accepted": "Avoid another same-target-role orderOf/unit variant after one accepted theorem already covers that role; change the final theorem obligation.",
        "witness_packaging_only": "Avoid explicit witness packaging as the final theorem; make the witness feed a new theorem target or characterization.",
        "coefficient_engineering_only": "Avoid irrationality variants that only change a nonzero rational coefficient; require a new algebraic object, quantifier, or theorem role.",
    }
    flag_lessons.update(PLANNER_MEMORY_LESSONS)
    for flag in quality_flags:
        if flag in flag_lessons:
            return flag_lessons[flag]
    failure_lessons = {
        "generation_failed": "Simplify the OperatorCard; avoid broad or underspecified generation surfaces.",
        "proof_failed": "Plan a smaller theorem child tied to visible parent checkpoints.",
        "alignment_failed": "Keep natural statement and Lean theorem exactly aligned.",
        "llm_json_parse_error": "Use a simpler worker contract likely to return valid JSON.",
        "low_quality_syntactic_mutation": "Treat certified syntactic closures as failures; require a semantic theorem change before planning this pattern again.",
        "accepted_proxy_failed": "Treat certified rows that fail accepted-proxy as failures; require accepted-grade novelty before repeating the pattern.",
        "curation_scaffold": "Use scaffold rows only as intermediate material; require a new final theorem role before treating the pattern as successful.",
        "curation_reject": "Do not repeat this curated-reject pattern as a final generated theorem.",
    }
    return failure_lessons.get(failure_class, "Avoid repeating this failed planning pattern.")


def _memory_theorem_surface_without_name(text: Any) -> str:
    surface = str(text or "")
    surface = surface.split(":= by", 1)[0]
    surface = re.sub(r"\b(theorem|lemma)\s+[A-Za-z0-9_'.]+", r"\1 _", surface)
    return re.sub(r"\s+", " ", surface).strip().lower()


def _memory_theorem_conclusion(text: Any) -> str:
    surface = _memory_theorem_surface_without_name(text)
    return surface.rsplit(":", 1)[-1].strip() if ":" in surface else surface


def _planner_memory_low_quality_flags(row: Dict[str, Any]) -> List[str]:
    """Reclassify historical certified-but-low-quality theorem rows without rewriting JSONL."""
    problem_style = _planner_memory_problem_style(row)
    if problem_style != "theorem_proof":
        return []
    status = str(row.get("status") or "")
    if status != "certified":
        return []
    flags = {str(flag) for flag in _json_list(row.get("quality_flags")) if str(flag)}
    evidence = _json_dict(row.get("quality_evidence"))
    raw = _planner_raw_surface_from_row(row)
    text = " ".join(
        [
            str(raw.get("statement") or ""),
            str(raw.get("formal_statement") or ""),
            str(raw.get("lean_code") or ""),
            str(raw.get("proof_plan") or ""),
            str(raw.get("solution") or ""),
            json.dumps(evidence, ensure_ascii=False),
        ]
    )
    compact = re.sub(r"\s+", "", text).lower()
    lower = text.lower()
    formal_lower = str(raw.get("formal_statement") or "").lower()
    formal_compact = re.sub(r"\s+", "", str(raw.get("formal_statement") or "")).lower()
    if re.search(r"^\s*axiom\s+", str(raw.get("lean_code") or ""), flags=re.MULTILINE):
        flags.add("axiom_backed_seed_or_child")
    if "-(-" in compact or "neg_neg" in lower or "double negation" in lower or "triple negation" in lower:
        flags.add("trivial_negation_chain")
    if "+0" in compact or "0+" in compact or "add_zero" in lower or "zero padding" in lower:
        flags.add("trivial_add_zero_padding")
    if any(marker in lower for marker in ("syntactic wrapper", "wrapper only", "simpa only", "immediate by simpa")):
        flags.add("syntactic_wrapper_only")
    if any(
        marker in lower
        for marker in (
            "project the",
            "projected from",
            "first component",
            "second component",
            "exact poddprime.1",
            "exact poddprime.2",
        )
    ) and any(marker in lower for marker in ("∧", "odd p", "p.prime", "nat.prime")):
        flags.add("projection_only_theorem")
    if (
        "orderof" in lower
        and "hu.neg.unit" in lower
        and "∃" not in text
        and "exists" not in lower
        and any(marker in lower for marker in ("orderof ((hu.neg.unit)⁻¹", "orderof (hu.neg.unit) = orderof", "orderof (hu.neg.unit) = orderof ((hu.neg.unit)⁻¹"))
    ):
        flags.add("orderof_orientation_projection")
    if (
        "polynomial ℚ" in lower
        and "x^3-3*x-1" in compact
        and "orderof" in lower
        and "hu.neg.unit" in lower
        and any(marker in lower for marker in ("hpoly_checkpoint", "order_goal_from_checkpoint", "from_cubic_checkpoint", "cubic checkpoint"))
    ):
        flags.add("unused_checkpoint")
    if (
        "polynomial ℚ" in lower
        and "x^3-3*x-1" in compact
        and "∣" in text
        and any(marker in compact for marker in ("x-c(1:ℚ)", "c(1:ℚ)-x"))
        and "∀x" not in compact
        and "polynomial.isroot" not in formal_lower
        and "exists_neg_one_unit_bridge" not in lower
    ):
        flags.add("linear_factor_paraphrase_only")
    if (
        "x^3-3*x-1" in formal_compact
        and "∣" in formal_compact
        and any(marker in formal_compact for marker in ("x-c(1:ℚ)", "c(1:ℚ)-x"))
        and "exists_neg_one_unit_bridge" not in lower
    ):
        flags.add("linear_factor_paraphrase_only")
    if (
        "polynomial.isroot" in formal_lower
        and "x^3-3*x-1" in formal_compact
        and any(marker in formal_compact for marker in ("(1:ℚ)", "1:ℚ"))
        and "∀x" not in formal_compact
    ):
        flags.add("too_narrow_root_instance")
    prime_domain_surface = (
        "prime divisors" in lower
        or "nat.prime" in lower
        or "filter(funx" in compact
    )
    divisor_sum_500_surface = (
        "positive divisors of 500" in lower
        or "nat.divisors500" in compact
        or "sigma_1(500)" in lower
        or "sigma 1) 500" in lower
    )
    ap_surface = (
        "arithmetic progression" in lower
        or "finset.range98" in compact
        or "u(n+1)" in lower
        or "u(n + 1)" in lower
        or "a(n+2)-a(n+1)" in compact
        or "3*p-q" in compact
    )
    if (
        str(row.get("op_type") or "") == "mutation"
        and ap_surface
        and any(marker in compact for marker in ("m+5=2014", "hm:m+", "a(m+"))
        and any(marker in compact for marker in ("a(m+", "-a(m+", "+a(m+", "2*a(m+"))
        and not any(marker in lower for marker in ("closed form", "all indices", "every term"))
    ):
        flags.add("ap_shifted_local_corollary_only")
    if (
        str(row.get("op_type") or "") == "mutation"
        and "y+6" in compact
        and ("2*12" in compact or "12-(y+6)=y-12" in compact)
        and ("y=9" in compact or "y = 9" in lower)
        and any(marker in compact for marker in ("y+1=10", "y+3=12", "y+4=13", "t≤12"))
    ):
        flags.add("linear_equation_shift_corollary_only")
    if (
        str(row.get("op_type") or "") in {"mutation", "crossover"}
        and any(
            marker in compact
            for marker in (
                "nat.divisors(30^4)",
                "((nat.divisors(30^4)).erase1).erase(30^4)",
            )
        )
        and ("finset.icc17" in compact or "nat.gcdx8=1" in compact)
        and any(marker in compact for marker in ("3^", "%8", ".card+"))
        and not any(marker in lower for marker in ("exists", "unique", "least", "greatest", "iff", "classification", "characterization"))
    ):
        flags.add("cardinality_arithmetic_pipeline_only")
    if str(row.get("op_type") or "") == "mutation":
        if prime_domain_surface and divisor_sum_500_surface and any(
            marker in lower
            for marker in (
                "prime_divisor_finset",
                "finset of prime divisors",
                "has cardinality",
                ".card =",
                "product of the prime divisors",
                ".prod id",
                "sum of the prime divisors",
            )
        ):
            flags.add("proof_infrastructure_only")
        if ap_surface and not prime_domain_surface and any(
            marker in lower
            for marker in (
                "odd-indexed",
                "even-indexed",
                "odd indexed",
                "even indexed",
                "first 98 terms",
            )
        ):
            flags.add("direct_parent_corollary_only")
    if str(row.get("op_type") or "") == "crossover" and ap_surface and prime_domain_surface:
        has_domain_sum = any(
            marker in lower
            for marker in (
                "h_prime_sum",
                "finite sum of its elements",
                "rational finite sum",
            )
        )
        has_domain_card = "h_card" in lower or "cardinality" in lower or ".card=4" in compact
        shifted_or_scaled_index = any(
            marker in compact
            for marker in (
                "u(p+1)",
                "u(p+2)",
                "u(p+3)",
                "u(2*p)",
                ".card)*p+1",
            )
        )
        if shifted_or_scaled_index and not (has_domain_card and has_domain_sum):
            flags.add("affine_index_drift_only")
        if has_domain_card and not has_domain_sum and "finset.range" in compact:
            flags.add("cardinality_only_window")
    if len(str(row.get("problem_id") or "").split("__theorem_gen")) >= 6 and not any(
        marker in lower
        for marker in ("master theorem", "pipeline", "finite sum of its elements")
    ):
        flags.add("lineage_complexity_without_new_role")
    if (
        ("24 divides" in lower or "24 ∣" in lower or "(24 : ℤ) ∣" in text)
        and any(marker in lower for marker in ("3 divides", "4 divides", "8 divides", "(3 : ℤ) ∣", "(4 : ℤ) ∣", "(8 : ℤ) ∣"))
        and any(marker in lower for marker in ("unpack 24", "quotient witnessing", "exhibit 8*k", "exhibit 3*k", "exhibit 6*k"))
    ):
        flags.add("divisibility_weaken_only_theorem")
    if "fin 1" in lower and "matrix.det" in lower and any(
        marker in lower
        for marker in (
            "one-by-one",
            "digits 1 through",
            "digits 6 through",
            "normalize each fin 1 determinant",
            "constant integer entries",
        )
    ):
        flags.add("fin_one_concrete_arithmetic_theorem")
    child_surface = str(raw.get("formal_statement") or raw.get("lean_code") or "")
    child_conclusion = _memory_theorem_conclusion(child_surface)
    parent_cards = _json_list(row.get("parent_context_cards"))
    parent_cards.extend(_json_list(_json_dict(row.get("operator_card")).get("parent_cards")))
    for card in parent_cards:
        if not isinstance(card, dict):
            continue
        parent_surface = (
            card.get("formal_statement")
            or card.get("lean_code")
            or _json_dict(card.get("proof_context")).get("lean_code")
            or ""
        )
        parent_conclusion = _memory_theorem_conclusion(parent_surface)
        if (
            child_conclusion
            and parent_conclusion == child_conclusion
            and "[commring" in _memory_theorem_surface_without_name(child_surface)
            and "[ring" in _memory_theorem_surface_without_name(parent_surface)
        ):
            flags.add("typeclass_narrowing_only")
    if "commring" in lower and any(marker in lower for marker in ("typeclass narrowing", "ring to commring", "commring specialization")):
        flags.add("typeclass_narrowing_only")
    memory_only_flags = {
        "orderof_orientation_projection",
        "linear_factor_paraphrase_only",
        "too_narrow_root_instance",
        "axiom_backed_seed_or_child",
    }
    return sorted(flags & (set(ACCEPTED_PROXY_SEVERE_FLAGS) | memory_only_flags))


def _planner_attempt_history_surface(value: Any) -> List[Dict[str, Any]]:
    entries = _json_list(value)[-4:]
    surface: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        generated = dict(entry.get("generated_surface_summary") or entry.get("generated_surface") or {})
        surface.append(
            {
                "attempt": entry.get("attempt"),
                "status": entry.get("status"),
                "error_class": entry.get("error_class"),
                "failure_signature": entry.get("failure_signature"),
                "quality_flags": list(entry.get("quality_flags") or [])[:8],
                "lean_error_summary": _prompt_text(entry.get("lean_error_summary"), limit=2000),
                "retry_feedback": _prompt_text(entry.get("retry_feedback"), limit=2000),
                "replan_decision": dict(entry.get("replan_decision") or {}),
                "generated_surface": {
                    "statement": _prompt_text(generated.get("statement"), limit=3000),
                    "formal_statement": _prompt_text(generated.get("formal_statement"), limit=6000),
                    "lean_code": _prompt_text(generated.get("lean_code") or generated.get("lean_code_head"), limit=9000),
                    "proof_plan": _prompt_text(generated.get("proof_plan"), limit=3000),
                    "solution": _prompt_text(generated.get("solution"), limit=4000),
                    "answer": _prompt_text(generated.get("answer"), limit=800),
                    "generated_params": dict(generated.get("generated_params") or {}),
                    "reasoning_pattern": generated.get("reasoning_pattern"),
                },
            }
        )
    return surface


def _planner_raw_surface_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    operator_card = _json_dict(row.get("operator_card"))
    evidence = _json_dict(row.get("quality_evidence"))
    return {
        "source_kind": _prompt_text(row.get("source_kind"), limit=120),
        "statement": _prompt_text(row.get("statement"), limit=4000),
        "formal_statement": _prompt_text(row.get("formal_statement"), limit=9000),
        "lean_code": _prompt_text(row.get("lean_code"), limit=14000),
        "proof_plan": _prompt_text(row.get("proof_plan"), limit=4000),
        "proof_obligations": _json_list(row.get("proof_obligations"))[:12],
        "answer": _prompt_text(row.get("answer"), limit=1000),
        "solution": _prompt_text(row.get("solution"), limit=7000),
        "error": _prompt_text(row.get("error") or row.get("proof_verify_summary"), limit=5000),
        "attempt_history": _planner_attempt_history_surface(row.get("attempt_history")),
        "operator_card": operator_card,
        "quality_evidence": evidence,
        "jixia_analysis": jixia_digest(row.get("jixia_analysis")),
    }


#: Fields the planner actually reads from a memory case. Everything else in a
#: card exists for other consumers.
_PLANNER_CASE_FIELDS = (
    "case_id",
    "kind",
    "op_type",
    "operator_variant",
    "goal",
    "lesson",
    "reasoning_signature",
    "quality_flags",
    "failure_class",
    "source_problem_id",
    "target_family",
)


def _planner_cases_for_prompt(cases: Sequence[Any]) -> List[Dict[str, Any]]:
    """The memory pack as the planner should receive it: lessons, not artifacts.

    Two things were wrong with sending the cards whole. The first is size: each
    card carries a `raw_surface` holding the full statement, Lean proof, proof
    plan, solution and error text of the row it came from, and one of them ran
    to 77,730 characters. Twenty-four cards came to 511 KB, which was 87% of the
    planner's entire prompt, to deliver about 12 KB of lesson. The planner never
    reads the Lean — it chooses parents and operators — so the artifact is pure
    ballast that buries the part it does read.

    The second is repetition. Of sixteen failure cards in one real pack, eleven
    carried the same `lesson` string verbatim ("Treat this as a scaffold ..."),
    so the same sentence was paid for eleven times and said nothing more the
    eleventh time than the first. Cases are deduplicated on the lesson, keeping
    the first occurrence, which is the highest-ranked one.
    """
    out: List[Dict[str, Any]] = []
    seen_lessons: set[str] = set()
    for case in cases:
        row = dict(case)
        lesson = " ".join(str(row.get("lesson") or "").split())
        if lesson and lesson in seen_lessons:
            continue
        if lesson:
            seen_lessons.add(lesson)
        out.append({
            key: row[key]
            for key in _PLANNER_CASE_FIELDS
            if row.get(key) not in (None, "", [], {})
        })
    return out


ENTROPY_DECREASE_MEMORY_FLAGS = {
    "same_formal_statement_as_parent",
    "same_statement_rephrase",
    "formal_surface_not_changed",
    "syntactic_wrapper_only",
    "projection_only_theorem",
    "trivial_negation_chain",
    "trivial_add_zero_padding",
    "typeclass_narrowing_only",
    "side_by_side_conjunction",
    "mutation_like_crossover",
    "weak_inspiration_only_crossover",
    "parent_checkpoint_not_consumed",
    "unused_checkpoint",
    "fin_one_vacuity_theorem",
    "fin_one_concrete_arithmetic_theorem",
    "tautological_checkpoint_theorem",
    "statement_lean_alignment_failed",
    "alignment_failed",
    "axiom_backed_seed_or_child",
    "orderof_orientation_projection",
    "linear_factor_paraphrase_only",
    "too_narrow_root_instance",
    "parent_theorem_assumption_smuggling",
}


def _planner_row_entropy_direction(row: Dict[str, Any], flags: Iterable[str]) -> str:
    evidence = _json_dict(row.get("quality_evidence"))
    entropy = evidence.get("entropy_direction")
    if isinstance(entropy, dict):
        direction = str(entropy.get("direction") or "").strip()
        if direction:
            return direction
    elif isinstance(entropy, str) and entropy.strip():
        return entropy.strip()
    flag_set = {str(flag) for flag in flags if str(flag)}
    if str(row.get("status") or "") != "certified":
        return "decrease"
    if flag_set & ENTROPY_DECREASE_MEMORY_FLAGS:
        return "decrease"
    return "increase"


def _planner_memory_card_from_row(row: Dict[str, Any], *, source_file: Path) -> Optional[PlannerMemoryCard]:
    op_type = str(row.get("op_type") or "")
    if op_type in {"survivor", "fallback_survivor", "seed_proof_completion"}:
        return None
    status = str(row.get("status") or "")
    quality_verdict = str(row.get("quality_verdict") or "")
    quality_flags = [str(flag) for flag in _json_list(row.get("quality_flags")) if str(flag)]
    evidence = _json_dict(row.get("quality_evidence"))
    operator_card = _json_dict(row.get("operator_card"))
    reasoning_signature = str(evidence.get("reasoning_signature") or row.get("reasoning_signature") or "")
    parent_eligible = _row_bool(row.get("parent_eligible"))
    low_quality_memory_flags = _planner_memory_low_quality_flags(row)
    accepted_proxy = dict(evidence.get("accepted_proxy") or {})
    accepted_proxy_flags = [
        str(flag) for flag in list(accepted_proxy.get("flags") or []) if str(flag).strip()
    ]
    curation = _json_dict(evidence.get("curation_decision"))
    curation_class = str(curation.get("curation_class") or "").strip()
    curation_flags = [
        str(flag) for flag in _json_list(curation.get("flags")) if str(flag).strip()
    ]
    if curation_class in {"scaffold", "reject"}:
        accepted_proxy_flags.append(f"curation_{curation_class}")
        accepted_proxy_flags.extend(curation_flags)
    manual_reject = str(
        row.get("_manual_qa_decision") or row.get("_manual_qa_reason") or ""
    ).lower()
    if manual_reject and "accept" not in manual_reject:
        accepted_proxy_flags.append("manual_qa_reject")
    all_quality_flags = sorted(set(quality_flags) | set(low_quality_memory_flags) | set(accepted_proxy_flags))
    entropy_direction = _planner_row_entropy_direction(row, all_quality_flags)
    entropy_increase = entropy_direction == "increase"
    if low_quality_memory_flags or accepted_proxy_flags:
        quality_flags = sorted(set(quality_flags) | set(low_quality_memory_flags) | set(accepted_proxy_flags))
    is_success = (
        status == "certified"
        and entropy_increase
        and (parent_eligible or accepted_proxy.get("pass", True) is not False)
        and curation_class not in {"scaffold", "reject"}
        and not low_quality_memory_flags
        and not accepted_proxy_flags
    )
    failure_class = _planner_memory_failure_class(row)
    if curation_class in {"scaffold", "reject"}:
        is_success = False
        failure_class = f"curation_{curation_class}"
    if (
        not is_success
        and low_quality_memory_flags
        and (not failure_class or failure_class in {"accepted_proxy_failed", "curation_scaffold"})
    ):
        failure_class = "low_quality_syntactic_mutation"
    elif not is_success and accepted_proxy_flags and not failure_class:
        failure_class = "accepted_proxy_failed"
    is_failure = (
        status in {"generation_failed", "proof_failed", "alignment_failed", "planner_axis_mismatch"}
        or quality_verdict == "weak"
        or bool(failure_class and quality_flags)
        or bool(low_quality_memory_flags)
        or bool(accepted_proxy_flags)
        or curation_class in {"scaffold", "reject"}
    )
    if not is_success and not is_failure:
        return None
    kind = "success" if is_success else "failure"
    target_family = str(
        row.get("target_family")
        or row.get("family")
        or operator_card.get("target_family")
        or ""
    )
    card = PlannerMemoryCard(
        case_id=f"{source_file.stem}:{str(row.get('problem_id') or row.get('id') or '')[:120]}",
        kind=kind,
        problem_style=_planner_memory_problem_style(row),
        certification_route=str(row.get("certification_route") or ""),
        op_type=op_type,
        operator_variant=str(operator_card.get("operator_variant") or row.get("operator_variant") or ""),
        target_family=target_family,
        goal=str(operator_card.get("goal") or row.get("operator_goal") or row.get("goal") or "")[:240],
        reasoning_signature=reasoning_signature[:240],
        status=status,
        quality_verdict=quality_verdict,
        quality_flags=quality_flags[:6],
        failure_class=failure_class,
        selection_reason=str(row.get("selection_reason") or "")[:160],
        lesson=_planner_memory_lesson(
            kind=kind,
            row=row,
            quality_flags=quality_flags,
            failure_class=failure_class,
            reasoning_signature=reasoning_signature,
        )[:260],
        raw_surface=_planner_raw_surface_from_row(row),
        source_run=source_file.stem,
        source_problem_id=str(row.get("problem_id") or row.get("id") or "")[:120],
    )
    if curation_class in {"scaffold", "reject"}:
        card.raw_surface["memory_reclassification"] = {
            "memory_kind": curation_class,
            "planner_case_kind": "failure",
            "reason": curation.get("reason") or f"curation_{curation_class}",
            "flags": sorted(set(low_quality_memory_flags) | set(accepted_proxy_flags)),
        }
    elif low_quality_memory_flags or accepted_proxy_flags:
        card.raw_surface["memory_reclassification"] = {
            "memory_kind": "failure",
            "reason": "certified_low_quality_or_accepted_proxy_failed",
            "flags": sorted(set(low_quality_memory_flags) | set(accepted_proxy_flags)),
        }
    return card


def _planner_memory_manifest(
    *,
    enabled: bool,
    cards: List[PlannerMemoryCard],
    source_files: List[str],
) -> Dict[str, Any]:
    curated_count = sum(
        1
        for card in cards
        if "curated" in card.source_run.lower()
        or str(dict(card.raw_surface or {}).get("source_kind") or "").startswith("curated")
    )
    return {
        "enabled": enabled,
        "card_count": len(cards),
        "case_count": len(cards),
        "success_count": sum(1 for card in cards if card.kind == "success"),
        "failure_count": sum(1 for card in cards if card.kind == "failure"),
        "curated_case_count": curated_count,
        "reclassified_low_quality_count": sum(
            1
            for card in cards
            if dict(card.raw_surface or {}).get("memory_reclassification")
        ),
        "reclassified_low_quality_flags": dict(
            Counter(
                flag
                for card in cards
                for flag in dict(dict(card.raw_surface or {}).get("memory_reclassification") or {}).get("flags", [])
            )
        ),
        "source_files": source_files,
        "cards": [card.model_dump() for card in cards],
        "cases": [card.model_dump() for card in cards],
    }


def _planner_memory_trace_manifest(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Small LangSmith-safe manifest; raw cases stay local/in prompt only."""
    cases = list(memory.get("cases") or memory.get("cards") or [])
    return {
        "enabled": bool(memory.get("enabled")),
        "card_count": int(memory.get("card_count") or len(cases)),
        "case_count": int(memory.get("case_count") or len(cases)),
        "success_count": int(memory.get("success_count") or 0),
        "failure_count": int(memory.get("failure_count") or 0),
        "curated_case_count": int(memory.get("curated_case_count") or 0),
        "reclassified_low_quality_count": int(memory.get("reclassified_low_quality_count") or 0),
        "reclassified_low_quality_flags": dict(memory.get("reclassified_low_quality_flags") or {}),
        "source_files": list(memory.get("source_files") or [])[:8],
        "case_index": [
            {
                "case_id": case.get("case_id"),
                "kind": case.get("kind"),
                "problem_style": case.get("problem_style"),
                "op_type": case.get("op_type"),
                "operator_variant": case.get("operator_variant"),
                "target_family": case.get("target_family"),
                "status": case.get("status"),
                "quality_verdict": case.get("quality_verdict"),
                "quality_flags": list(case.get("quality_flags") or [])[:4],
                "failure_class": case.get("failure_class"),
                "source_run": case.get("source_run"),
                "source_problem_id": case.get("source_problem_id"),
            }
            for case in cases[:24]
            if isinstance(case, dict)
        ],
    }


def _read_jsonl_dict_rows(path: Path, *, limit: int = 2000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


_THEOREM_NAME = re.compile(r"\b(theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_'.]*")
_BINDER_NAME = re.compile(r"\b[hH][A-Za-z0-9_']*\s*:")


def _dedup_surface(formal: Any) -> str:
    """A theorem statement reduced to what makes it that theorem.

    Delegates to the dedup module so the write-time check and the corpus gate
    cannot drift apart: two normalisations that disagree would let a row pass
    one and fail the other, which is worse than either alone.
    """
    from src.certification.dedup import dedup_surface

    return dedup_surface(formal)


_CORPUS_INDEX: Any = None


def _corpus_index() -> Any:
    """Fingerprints of every certified statement the pipeline has produced.

    Built once per process. `DEDUP_CORPUS_ROOTS` overrides the directories;
    `DEDUP_GATE=0` disables the gate, which is only for reproducing an old run
    whose duplicates are part of what is being reproduced.
    """
    global _CORPUS_INDEX
    if _CORPUS_INDEX is None:
        from src.certification.dedup import build_index

        roots = [
            Path(part.strip())
            for part in os.getenv("DEDUP_CORPUS_ROOTS", "data/certified").split(",")
            if part.strip()
        ]
        _CORPUS_INDEX = build_index(roots)
        print(f"[dedup] corpus index: {len(_CORPUS_INDEX)} statements", flush=True)
    return _CORPUS_INDEX


def _statement_fingerprint(formal_statement: str) -> str:
    """Eight hex characters of the statement's normalised surface.

    Short enough to keep ids readable and long enough that a collision between
    two genuinely different theorems in one campaign is not a practical
    concern. Derived from the mathematics, so it is stable across re-runs.
    """
    return hashlib.sha256(
        _dedup_surface(formal_statement).encode("utf-8")
    ).hexdigest()[:8]


def _novelty_row_key(row: Any) -> str:
    if hasattr(row, "model_dump"):
        data = dict(row.model_dump())
    elif isinstance(row, dict):
        data = dict(row)
    else:
        return ""
    metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
    # Identity is the theorem, not the slot that produced it. Keying on
    # `problem_id` first let the same statement through repeatedly: a five-seed,
    # five-generation run accepted one theorem three times and another twice,
    # because each came from a different slot and so carried a different id.
    # The surface is what a reader would call the same problem, so the surface
    # decides.
    formal = _dedup_surface(
        data.get("formal_statement")
        or data.get("lean_code")
        or metadata.get("formal_statement")
        or metadata.get("lean_code")
        or ""
    )
    if formal:
        return "surface:" + hashlib.sha256(formal.encode("utf-8")).hexdigest()
    statement = " ".join(str(data.get("statement") or metadata.get("statement") or "").split())
    if statement:
        return "prose:" + hashlib.sha256(statement.encode("utf-8")).hexdigest()
    problem_id = str(
        data.get("problem_id")
        or data.get("id")
        or data.get("source_problem_id")
        or metadata.get("problem_id")
        or ""
    )
    return f"id:{problem_id}" if problem_id else ""


def _run_local_novelty_rows(state: Dict[str, Any]) -> List[Any]:
    """Rows from this run before the current planner call."""
    rows: List[Any] = []
    output_path = str(state.get("output_path") or "")
    if output_path:
        rows.extend(_read_jsonl_dict_rows(Path(output_path)))
    rows.extend(list(state.get("all_passed_results") or []))
    rows.extend(list(state.get("plan_outcome_cards") or []))
    rows.extend(list(state.get("approved_candidates") or []))

    deduped: List[Any] = []
    seen: set[str] = set()
    for row in rows:
        key = _novelty_row_key(row)
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _novelty_accepted_ledger_path(state: Dict[str, Any]) -> Path:
    configured = (
        state.get("novelty_accepted_ledger_path")
        or os.getenv("NOVELTY_ACCEPTED_LEDGER_PATH")
        or DEFAULT_NOVELTY_ACCEPTED_LEDGER_PATH
    )
    path = Path(str(configured))
    return path if path.is_absolute() else _repo_root() / path


def _build_state_novelty_memory(
    state: Dict[str, Any],
    pool: List[CertificationInput],
) -> Dict[str, Any]:
    if bool(state.get("disable_novelty_memory")) or os.getenv("DISABLE_NOVELTY_MEMORY", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return {"enabled": False, "cards": [], "planner_view": {}}
    return build_novelty_memory_pack(
        pool,
        accepted_ledger_path=_novelty_accepted_ledger_path(state),
        run_rows=_run_local_novelty_rows(state),
    )


def _novelty_memory_trace_manifest(memory: Dict[str, Any]) -> Dict[str, Any]:
    view = dict(memory.get("planner_view") or {})
    soft_neighbors = dict(view.get("soft_neighbors") or {})
    return {
        "enabled": bool(memory.get("enabled")),
        "accepted_ledger_path": memory.get("accepted_ledger_path"),
        "accepted_card_count": int(memory.get("accepted_card_count") or 0),
        "run_local_card_count": int(memory.get("run_local_card_count") or 0),
        "planner_view": {
            "exact_blockers": dict(view.get("exact_blockers") or {"accepted": [], "run_local": []}),
            "soft_neighbors": {
                "accepted": [
                    compact_card(card)
                    for card in list(soft_neighbors.get("accepted", view.get("accepted_neighbors") or []))[:4]
                    if isinstance(card, dict)
                ],
                "run_local": [
                    compact_card(card)
                    for card in list(soft_neighbors.get("run_local", view.get("run_local_neighbors") or []))[:3]
                    if isinstance(card, dict)
                ],
            },
            "instructions": list(view.get("instructions") or [])[:4],
        },
    }


def _attach_novelty_contracts(
    work_items: List[Dict[str, Any]],
    novelty_memory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cards = list(novelty_memory.get("cards") or []) if novelty_memory.get("enabled") else []
    updated: List[Dict[str, Any]] = []
    for item in work_items:
        next_item = dict(item)
        if next_item.get("op_type") != "survivor":
            generated = build_memory_delta_contract(next_item, cards, k=3)
            provided = (
                dict(next_item.get("memory_delta_contract"))
                if isinstance(next_item.get("memory_delta_contract"), dict)
                else {}
            )
            contract = dict(generated)
            for key, value in provided.items():
                if value not in (None, "", [], {}):
                    contract[key] = value
            next_item["memory_delta_contract"] = contract
        next_item["operator_card"] = _operator_card(next_item)
        updated.append(next_item)
    return updated


def _extend_novelty_memory_with_rows(
    novelty_memory: Dict[str, Any],
    rows: List[Any],
    *,
    source_kind: str,
) -> Dict[str, Any]:
    if not novelty_memory.get("enabled") or not rows:
        return novelty_memory
    existing = list(novelty_memory.get("cards") or [])
    existing_ids = {str(card.get("problem_id") or "") for card in existing if isinstance(card, dict)}
    new_cards = [
        card
        for card in cards_from_rows(rows, source_kind=source_kind)
        if str(card.get("problem_id") or "") not in existing_ids
    ]
    if not new_cards:
        return novelty_memory
    extended = dict(novelty_memory)
    extended["cards"] = existing + new_cards
    extended["run_local_card_count"] = int(extended.get("run_local_card_count") or 0) + len(new_cards)
    return extended


def _structural_overlap_curation_flags(result: CertificationResult) -> List[str]:
    """Map structural novelty overlap to manual-QA style scaffold flags."""
    text = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''} {result.solution or ''}"
    ).lower()
    compact = re.sub(r"\s+", "", text)
    flags: List[str] = []
    if any(
        marker in text
        for marker in (
            "arithmetic progression",
            "a + 6d",
            "a+6d",
            "lcm 3720",
            "gcd(31",
            "greatest common divisor",
        )
    ):
        flags.append("ap_shifted_local_corollary_only")
    if any(
        marker in text
        for marker in (
            "modulo 398",
            "mod 398",
            "remainder 1 modulo 398",
            "multiplicative inverse of 7",
        )
    ):
        flags.append("finite_residue_bookkeeping_only")
    if any(marker in text for marker in ("proper divisors", "units digit", "quotient-weighted")) or "finset.sum" in compact:
        flags.append("fixed_finite_aggregate_computation")
    if (
        any(marker in text for marker in ("orderof", "commutes", "cyclic", "conjugation", "conjugate"))
        and any(
            marker in text
            for marker in (
                "group",
                "a*b*c",
                "dabc",
                "c a b",
                "a b c",
                "four-factor",
                "three-factor",
            )
        )
    ):
        flags.append("cyclic_transport_same_target_role")
    if any(
        marker in text
        for marker in (
            "sqrt",
            "square-root",
            "irrational",
            "cubic",
            "rational affine",
            "linear combination",
            "q^3",
        )
    ):
        flags.append("theorem_local_corollary_dominated")
    if (
        any(
            marker in text
            for marker in (
                "compact",
                "locally compact",
                "neighborhood",
                "closure",
                "complete as a uniform space",
            )
        )
        and any(marker in text for marker in ("rational", "rationals", "ℚ"))
    ):
        flags.append("topology_direct_consequence_only")
    return sorted(set(flags))


def _merge_novelty_memory_quality(
    result: CertificationResult,
    quality: QualityResult,
    novelty_memory: Dict[str, Any],
) -> QualityResult:
    if (
        not novelty_memory.get("enabled")
        or not _is_generated_result(result)
        or result.status != "certified"
    ):
        return quality

    assessment = evaluate_candidate_novelty(result.model_dump(), novelty_memory.get("cards") or [], k=3)
    evidence = dict(quality.quality_evidence or {})
    # The judge's verdict survives this function. Novelty memory and curation are
    # surface heuristics -- "the top match shares family or surface terms" -- and
    # they were still forcing `weak` after the verdict moved to the judge, which
    # put five certified rows back where they started: all five judged
    # keep/strong, all five downgraded here, none of them able to become a
    # parent. Their flags and their accepted_proxy signal are kept, because the
    # orchestrator weighs those as selection risks; what stops is the override.
    judge_kept = bool(
        (evidence.get("judge") or {}).get("ran")
        and (evidence.get("judge") or {}).get("verdict") == "keep"
    )
    evidence["novelty_memory"] = assessment
    novelty_flags = set(str(flag) for flag in evidence.get("novelty_flags") or [])
    quality_flags = list(quality.quality_flags or [])
    verdict = str(assessment.get("verdict") or "")
    quality_verdict = quality.quality_verdict
    feedback = quality.feedback_for_next_generation or ""

    if verdict == "near_duplicate":
        if "near_duplicate" not in quality_flags:
            quality_flags.append("near_duplicate")
        novelty_flags.add("near_duplicate")
        if assessment.get("exact_blockers"):
            if "exact_duplicate_memory" not in quality_flags:
                quality_flags.append("exact_duplicate_memory")
            novelty_flags.add("exact_duplicate_memory")
        quality_verdict = "weak"
        proxy = dict(evidence.get("accepted_proxy") or {})
        proxy["pass"] = False
        proxy_flags = set(str(flag) for flag in proxy.get("flags") or [])
        proxy_flags.add("near_duplicate")
        if assessment.get("exact_blockers"):
            proxy_flags.add("exact_duplicate_memory")
        proxy["flags"] = sorted(proxy_flags)
        proxy["reason"] = assessment.get("reason") or "near_duplicate"
        evidence["accepted_proxy"] = proxy
        if judge_kept:
            quality_verdict = quality.quality_verdict
        feedback = (
            (feedback + " ") if feedback else ""
        ) + "NoveltyMemory verdict is near_duplicate; regenerate with the required distinguishing delta."
    elif verdict == "structural_overlap":
        novelty_flags.add("structural_overlap")
        for flag in _structural_overlap_curation_flags(result):
            if flag not in quality_flags:
                quality_flags.append(flag)
            novelty_flags.add(flag)
    elif verdict:
        novelty_flags.add(verdict)

    evidence["novelty_flags"] = sorted(flag for flag in novelty_flags if flag)
    evidence["accepted_proxy"] = derive_accepted_proxy(result, quality_flags, evidence)
    evidence["entropy_direction"] = derive_entropy_direction(result, quality_flags, evidence)
    evidence["curation_decision"] = derive_curation_decision(result, quality_flags, evidence)
    evidence["misformalization"] = derive_misformalization_taxonomy(result, quality_flags, evidence)
    curation = dict(evidence.get("curation_decision") or {})
    if curation.get("curation_class") != "paper":
        if curation.get("curation_class") == "reject" and not judge_kept:
            quality_verdict = "weak"
        feedback = (
            (feedback + " ") if feedback else ""
        ) + (
            f"CurationDecision is {curation.get('curation_class')}; "
            "use as scaffold or regenerate with a new final theorem role."
        )
    return quality.model_copy(
        update={
            "quality_verdict": quality_verdict,
            "quality_flags": quality_flags,
            "feedback_for_next_generation": feedback,
            "quality_evidence": evidence,
        }
    )


def _result_trace_manifest(row: Dict[str, Any]) -> Dict[str, Any]:
    """Small per-result manifest for summaries and LangSmith state.

    Full theorem text and Lean code stay in JSONL; carrying them through
    LangGraph state makes trace outputs exceed LangSmith field limits.
    """
    evidence = _json_dict(row.get("quality_evidence"))
    return {
        "problem_id": row.get("problem_id") or row.get("id"),
        "generation": row.get("generation"),
        "slot": row.get("slot"),
        "op_type": row.get("op_type"),
        "operator_variant": row.get("operator_variant"),
        "parent_ids": _json_list(row.get("parent_ids")),
        "status": row.get("status"),
        "quality_verdict": row.get("quality_verdict"),
        "quality_flags": _json_list(row.get("quality_flags")),
        "quality_evidence": {
            "reasoning_signature": evidence.get("reasoning_signature"),
            "signature_group": evidence.get("signature_group"),
            "crossover_kind": evidence.get("crossover_kind"),
            "accepted_proxy": evidence.get("accepted_proxy"),
            "misformalization": evidence.get("misformalization"),
            "missing_checkpoints": list(evidence.get("missing_checkpoints") or [])[:8],
            "novelty_flags": list(evidence.get("novelty_flags") or [])[:8],
            "novelty_memory": _compact_novelty_assessment(
                dict(evidence.get("novelty_memory") or {})
            ),
        },
        "feedback_for_next_generation": _prompt_text(row.get("feedback_for_next_generation"), limit=500),
        "parent_eligible": row.get("parent_eligible"),
        "selection_reason": row.get("selection_reason"),
        "accepted_proxy": evidence.get("accepted_proxy"),
        "crossover_kind": evidence.get("crossover_kind"),
        "reasoning_signature": evidence.get("reasoning_signature"),
        "signature_group": evidence.get("signature_group"),
        "reasoning_pattern": row.get("reasoning_pattern"),
        "failure_class": row.get("failure_class") or _planner_memory_failure_class(row),
        "failure_signature": row.get("failure_signature"),
        "statement_preview": _prompt_text(row.get("statement"), limit=240),
        "formal_statement_preview": _prompt_text(row.get("formal_statement"), limit=240),
        "error_preview": _prompt_text(row.get("error") or row.get("proof_verify_summary"), limit=240),
    }


def _results_trace_manifest(rows: List[Dict[str, Any]], *, limit: int = 12) -> List[Dict[str, Any]]:
    return [_result_trace_manifest(row) for row in rows[:limit]]


def _runtime_store_put(kind: str, payload: Any) -> str:
    key = f"{kind}:{uuid.uuid4().hex}"
    _RUNTIME_RESULT_STORE[key] = payload
    return key


def _runtime_store_pop(key: Any, default: Any = None) -> Any:
    if not key:
        return default
    return _RUNTIME_RESULT_STORE.pop(str(key), default)


def _runtime_store_get(key: Any, default: Any = None) -> Any:
    if not key:
        return default
    return _RUNTIME_RESULT_STORE.get(str(key), default)


def _certification_result_manifest(result: CertificationResult) -> Dict[str, Any]:
    return _result_trace_manifest(result.model_dump())


def _accepted_proxy(result: CertificationResult) -> Dict[str, Any]:
    evidence = dict(result.quality_evidence or {})
    proxy = evidence.get("accepted_proxy")
    return dict(proxy) if isinstance(proxy, dict) else {"pass": False, "flags": ["missing_accepted_proxy"], "reason": "missing_accepted_proxy"}


def _accepted_proxy_pass(result: CertificationResult) -> bool:
    return bool(_accepted_proxy(result).get("pass"))


def _accepted_grade_proxy_pass(result: CertificationResult) -> bool:
    proxy = _accepted_proxy(result)
    return bool(proxy.get("pass")) and bool(proxy.get("accepted_grade_pass"))


def _entropy_direction(result: CertificationResult) -> str:
    evidence = dict(result.quality_evidence or {})
    entropy = evidence.get("entropy_direction")
    if isinstance(entropy, dict):
        return str(entropy.get("direction") or "")
    return str(entropy or "")


def _entropy_increase(result: CertificationResult) -> bool:
    return _entropy_direction(result) == "increase"


def _is_generated_result(result: CertificationResult) -> bool:
    return result.op_type not in {"survivor", "fallback_survivor", "seed_proof_completion"}


def _yield_funnel(results: List[CertificationResult]) -> Dict[str, Any]:
    generated = [result for result in results if _is_generated_result(result)]
    certified = [result for result in generated if result.status == "certified"]
    non_weak = [result for result in certified if result.quality_verdict != "weak"]
    parent_eligible = [result for result in certified if result.parent_eligible]
    accepted_proxy = [result for result in generated if _accepted_proxy_pass(result)]
    accepted_grade_proxy = [result for result in generated if _accepted_grade_proxy_pass(result)]
    entropy_increase = [result for result in certified if _entropy_increase(result)]
    proxy_flags = Counter(
        flag
        for result in generated
        for flag in list(_accepted_proxy(result).get("flags") or [])
    )
    selection_reasons = Counter(
        str(result.selection_reason or "unset") for result in generated if not result.parent_eligible
    )
    return {
        "generated": len(generated),
        "certified": len(certified),
        "non_weak": len(non_weak),
        "parent_eligible": len(parent_eligible),
        "accepted_proxy": len(accepted_proxy),
        "accepted_grade_proxy": len(accepted_grade_proxy),
        "entropy_increase": len(entropy_increase),
        "ledger_accepted": 0,
        "accepted_proxy_flags": dict(proxy_flags),
        "selection_reasons": dict(selection_reasons),
    }


def _slot_output_ref(result: CertificationResult, *, failed: bool) -> Dict[str, Any]:
    """Pass large slot results by reference through LangGraph state.

    LangSmith traces LangGraph node outputs. Full theorem rows can contain long
    Lean proofs, parent contexts, and attempt histories, which exceed LangSmith
    field limits during long runs. The JSONL writer still receives the full
    result via this in-process store; only the graph state/trace gets a compact
    manifest.
    """
    ref = _runtime_store_put("slot_result", result.model_dump())
    return {
        "slot": result.slot,
        "op_type": result.op_type,
        "result_ref": ref,
        "result_manifest": _certification_result_manifest(result),
        "failed": failed,
    }


def _slot_output_result(output: Dict[str, Any]) -> CertificationResult:
    if output.get("result_ref"):
        payload = _runtime_store_get(output.get("result_ref"))
        if payload is None:
            raise RuntimeError(f"missing slot result ref: {output.get('result_ref')}")
        return CertificationResult.model_validate(payload)
    return CertificationResult.model_validate(output["result"])


def _results_state_ref(results: List[CertificationResult]) -> str:
    return _runtime_store_put("generation_results", [result.model_dump() for result in results])


def _state_results(state: Dict[str, Any]) -> List[CertificationResult]:
    if state.get("results_ref"):
        payload = _runtime_store_get(state.get("results_ref"), [])
        return [CertificationResult.model_validate(item) for item in payload]
    return [
        item if isinstance(item, CertificationResult) else CertificationResult.model_validate(item)
        for item in list(state.get("results", []) or [])
    ]


def _generation_feedback_trace_manifest(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Keep feedback useful for planning/debugging without embedding full Lean artifacts."""
    plan_summary = dict(feedback.get("plan_outcome_summary") or {})
    return {
        "op_type_outcomes": feedback.get("op_type_outcomes", {}),
        "planned_op_type_outcomes": feedback.get("planned_op_type_outcomes", {}),
        "quality_flags": feedback.get("quality_flags", {}),
        "dominant_signature_groups": feedback.get("dominant_signature_groups", {}),
        "planner_source": feedback.get("planner_source"),
        "planner_warnings": list(feedback.get("planner_warnings") or [])[:8],
        "repeated_weak_patterns": list(feedback.get("repeated_weak_patterns") or [])[:12],
        "judge_failures_by_parents": list(feedback.get("judge_failures_by_parents") or [])[:12],
        "quality_retry_count": feedback.get("quality_retry_count", 0),
        "retry_exhausted_count": feedback.get("retry_exhausted_count", 0),
        "replan_count": feedback.get("replan_count", 0),
        "replanned_slots": list(feedback.get("replanned_slots") or [])[:10],
        "giveup_slots": list(feedback.get("giveup_slots") or [])[:10],
        "backfilled_slots": list(feedback.get("backfilled_slots") or [])[:10],
        "backfill_events": list(feedback.get("backfill_events") or [])[:10],
        "accepted_proxy_count": feedback.get("accepted_proxy_count", 0),
        "accepted_grade_proxy_count": feedback.get("accepted_grade_proxy_count", 0),
        "target_accepted_per_generation": feedback.get("target_accepted_per_generation"),
        "reserve_slots_run": feedback.get("reserve_slots_run", 0),
        "reserve_slots_selected": feedback.get("reserve_slots_selected", 0),
        "yield_funnel": feedback.get("yield_funnel", {}),
        "generation_survival_status": feedback.get("generation_survival_status"),
        "weak_slots": list(feedback.get("weak_slots") or [])[:10],
        "failed_slots": _results_trace_manifest(list(feedback.get("failed_slots") or []), limit=10),
        "rejected_slots": _results_trace_manifest(list(feedback.get("rejected_slots") or []), limit=10),
        "plan_outcome_summary": {
            "op_type_stats": plan_summary.get("op_type_stats", {}),
            "weak_signature_summary": plan_summary.get("weak_signature_summary", {}),
            "dominant_signature_groups": plan_summary.get("dominant_signature_groups", {}),
            "recurrent_failure_signatures": list(plan_summary.get("recurrent_failure_signatures") or [])[:10],
            "axes_planned_but_not_selected": list(plan_summary.get("axes_planned_but_not_selected") or [])[:10],
            "success_case_count": len(plan_summary.get("success_case_cards") or []),
            "failure_case_count": len(plan_summary.get("failure_case_cards") or []),
        },
    }


def _semantic_terms(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", str(text or "").lower())
        if token
        not in {
            "theorem",
            "lemma",
            "import",
            "mathlib",
            "true",
            "false",
            "with",
            "from",
            "have",
            "show",
            "proof",
        }
    }


def _pool_semantic_terms(pool: List[CertificationInput]) -> set[str]:
    terms: set[str] = set()
    for problem in pool:
        card = _parent_context_card(problem)
        metadata = dict(problem.metadata or {})
        terms.update(_semantic_terms(problem.statement))
        terms.update(_semantic_terms(problem.answer))
        terms.update(_semantic_terms(metadata.get("formal_statement")))
        terms.update(_semantic_terms(metadata.get("lean_code")))
        terms.update(str(atom).lower() for atom in card.get("reusable_atoms") or [])
        decomposition = dict(card.get("theorem_decomposition") or {})
        terms.update(str(atom).lower() for atom in decomposition.get("reusable_lean_atoms") or [])
        for checkpoint in decomposition.get("proof_checkpoints") or []:
            terms.update(_semantic_terms(checkpoint))
    return terms


def _planner_case_terms(card: PlannerMemoryCard) -> set[str]:
    raw = dict(card.raw_surface or {})
    terms = _semantic_terms(
        " ".join(
            str(raw.get(field) or "")
            for field in ("statement", "formal_statement", "lean_code", "proof_plan", "solution", "error")
        )
    )
    terms.update(_semantic_terms(card.reasoning_signature))
    terms.update(_semantic_terms(card.target_family))
    terms.update(_semantic_terms(card.operator_variant))
    terms.update(_semantic_terms(" ".join(card.quality_flags or [])))
    evidence = dict(raw.get("quality_evidence") or {})
    terms.update(_semantic_terms(evidence.get("reasoning_signature")))
    terms.update(_semantic_terms(evidence.get("signature_group")))
    jixia = dict(raw.get("jixia_analysis") or {})
    terms.update(_semantic_terms(" ".join(map(str, jixia.get("typeReferences") or []))))
    terms.update(_semantic_terms(" ".join(map(str, jixia.get("valueReferences") or []))))
    return terms


def _pool_dataset_hints(pool: List[CertificationInput]) -> set[str]:
    text = " ".join(
        [
            *(problem.id for problem in pool),
            *(str((problem.metadata or {}).get("benchmark") or "") for problem in pool),
            *(str((problem.metadata or {}).get("source_file") or "") for problem in pool),
        ]
    ).lower()
    hints: set[str] = set()
    for name in ("putnam", "proofnet", "minif2f", "mini_f2f", "mathd"):
        if name in text:
            hints.add("minif2f" if name in {"mini_f2f", "mathd"} else name)
    return hints


def _planner_case_dataset_hint(card: PlannerMemoryCard) -> str:
    text = " ".join(
        [
            card.case_id,
            card.source_run,
            card.source_problem_id,
            card.problem_style,
            card.certification_route,
        ]
    ).lower()
    if "putnam" in text:
        return "putnam"
    if "proofnet" in text:
        return "proofnet"
    if "minif2f" in text or "mini_f2f" in text or "mathd" in text:
        return "minif2f"
    return ""


def _raw_surface_key(card: PlannerMemoryCard) -> tuple[str, str, str]:
    raw = dict(card.raw_surface or {})
    return (
        str(raw.get("statement") or "")[:220],
        str(raw.get("formal_statement") or "")[:260],
        str(raw.get("lean_code") or "")[:260],
    )


def _planner_memory_roots(memory_root: Path) -> List[Path]:
    """Read run memory plus curated planner cases without adding a new DB layer."""
    roots: List[Path] = []
    if memory_root.exists():
        roots.append(memory_root)
    curated_root = memory_root.parent / DEFAULT_PLANNER_CURATED_MEMORY_DIR.name
    if curated_root.exists() and curated_root.resolve() not in {root.resolve() for root in roots}:
        roots.append(curated_root)
    return roots


def _is_curated_memory_row(path: Path, row: Dict[str, Any]) -> bool:
    source_kind = str(row.get("source_kind") or "").lower()
    return "curated" in path.parts or source_kind.startswith("curated")


def select_planner_memory_cards(
    pool: List[CertificationInput],
    *,
    memory_dir: Optional[Path],
    limit: int = DEFAULT_PLANNER_MEMORY_LIMIT,
    enabled: bool = True,
    exclude_paths: Optional[List[Path]] = None,
    slot_op_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not enabled or limit <= 0 or memory_dir is None:
        return _planner_memory_manifest(enabled=False, cards=[], source_files=[])
    memory_root = Path(memory_dir)
    roots = _planner_memory_roots(memory_root)
    if not roots:
        return _planner_memory_manifest(enabled=True, cards=[], source_files=[])
    exclude = {path.resolve() for path in (exclude_paths or []) if path}
    pool_styles = {_problem_style(problem) for problem in pool}
    pool_routes = {_certification_route_for_style(style) for style in pool_styles}
    pool_families = {_problem_family(problem) for problem in pool if _problem_family(problem)}
    pool_terms = _pool_semantic_terms(pool)
    #: Which operators this generation will dispatch, so memory from the other
    #: kind is ranked down rather than competing on equal terms.
    wanted_ops = {str(op) for op in (slot_op_types or []) if op}
    pool_dataset_hints = _pool_dataset_hints(pool)
    candidates: List[tuple[int, float, PlannerMemoryCard, str]] = []
    files: List[Path] = []
    for root in roots:
        # Recursive, because campaign outputs live in per-run subdirectories
        # (`data/certified/run-a/…`). A non-recursive glob on the configured
        # root still found the 151 legacy files sitting directly in it, so the
        # memory was never empty — it was simply frozen before this campaign,
        # and none of the failures this run learned about could reach it.
        files.extend(
            sorted(root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[:40]
        )
    for file_index, path in enumerate(files):
        if path.resolve() in exclude:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            card = _planner_memory_card_from_row(row, source_file=path)
            if card is None:
                continue
            score = 0
            is_curated = _is_curated_memory_row(path, row)
            if is_curated:
                score += 16
                evidence = _json_dict(row.get("quality_evidence"))
                novelty_flags = {str(flag) for flag in _json_list(evidence.get("novelty_flags"))}
                if (
                    "isunit_objectification_bridge" in novelty_flags
                    and {"isunit", "orderof", "units"} & pool_terms
                ):
                    score += 10
            # Operator match. A crossover slot learns nothing from a mutation
            # case and vice versa: the two fail in disjoint ways — of 2,800
            # generated rows, `parallel_crossover` appeared only under crossover
            # and `decorative_mutation` only under mutation. Scored rather than
            # filtered, so a thin pool still surfaces something rather than
            # nothing.
            if card.op_type and wanted_ops:
                score += 8 if card.op_type in wanted_ops else -6
            if card.problem_style in pool_styles:
                score += 6
            elif "theorem_proof" in pool_styles and card.problem_style == "numeric_answer":
                score -= 4
            if card.certification_route and card.certification_route in pool_routes:
                score += 4
            if card.target_family and card.target_family in pool_families:
                score += 2
            case_dataset = _planner_case_dataset_hint(card)
            if case_dataset and case_dataset in pool_dataset_hints:
                score += 10
            elif case_dataset and pool_dataset_hints:
                score -= 3
            term_overlap = len(pool_terms & _planner_case_terms(card))
            # Subject overlap, uncapped. It used to be `min(8, term_overlap)`,
            # which meant a card sharing twenty mathematical objects with the
            # pool scored the same as one sharing eight, while the boosts that
            # ignore subject entirely — dataset hint +10, op_type +8, style +6,
            # route +4, curated crossover +10 — summed to 38. A curated card
            # about `IsUnit` could therefore outrank a card about the very
            # theorem being planned. Measured on a real ProofNet group whose
            # pool was five linear-algebra seeds: none of the twenty-four cards
            # selected mentioned a vector space, though the corpus held 275 rows
            # that did.
            score += 2 * term_overlap
            if card.kind == "success":
                score += 2
                if is_curated and card.op_type == "crossover":
                    score += 8
            if card.kind == "failure":
                score += 3
                if is_curated and card.op_type == "crossover":
                    score += 4
            if card.failure_class == "low_quality_syntactic_mutation":
                score += 5
                if term_overlap:
                    score += 5
            candidates.append((score, -float(file_index), card, str(path)))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: List[PlannerMemoryCard] = []
    seen_failure_keys: set[tuple[str, ...]] = set()
    seen_success_keys: set[tuple[str, ...]] = set()
    # Embedding + MMR decide the final order; the hand-weighted score above is
    # kept as a prior rather than discarded, because operator and style matter
    # and the failure being fixed is monoculture, not the presence of metadata.
    # Measured before this change: three pools drawn from algebra, number theory
    # and induction received 6 of the same 8 cards, all from one file.
    if candidates:
        try:
            from src.retrieval.memory_search import build_query, search_memory

            query_text = build_query(
                [dict(problem.metadata or {}, statement=problem.statement) for problem in pool]
            )
            if query_text.strip():
                span = max(
                    1e-6,
                    max(c[0] for c in candidates) - min(c[0] for c in candidates),
                )
                low = min(c[0] for c in candidates)
                prior = [(c[0] - low) / span for c in candidates]
                ordered = search_memory(
                    query_text,
                    [c[2] for c in candidates],
                    limit=len(candidates),
                    prior=prior,
                )
                # A reserved head, chosen on similarity alone. `prior` is the
                # hand-weighted score above, and while it encodes things worth
                # keeping — operator match, style, curation — it also decides
                # which cards are near the front before any embedding is
                # consulted, so re-ranking within it cannot rescue a candidate
                # set that was assembled on non-subject grounds. Reserving a
                # fraction of the head for the closest cards by query similarity
                # guarantees the planner sees the pool's own subject even when
                # every structural signal points elsewhere.
                reserved = max(2, limit // 3)
                by_similarity = search_memory(
                    query_text,
                    [c[2] for c in candidates],
                    limit=min(reserved, len(candidates)),
                )
                head = {id(card) for card in by_similarity}
                rank = {id(card): index for index, card in enumerate(ordered)}
                candidates = sorted(
                    candidates,
                    key=lambda c: (
                        0 if id(c[2]) in head else 1,
                        rank.get(id(c[2]), len(ordered)),
                    ),
                )
                candidates = [
                    (len(candidates) - index, recency, card, source)
                    for index, (_score, recency, card, source) in enumerate(candidates)
                ]
        except Exception:
            pass

    total_success = sum(1 for _score, _recency, card, _source_file in candidates if card.kind == "success")
    total_failure = sum(1 for _score, _recency, card, _source_file in candidates if card.kind == "failure")
    success_limit = limit if total_failure == 0 else min(total_success, max(1, limit // 3))
    failure_limit = limit if total_success == 0 else max(1, limit - success_limit)
    counts = Counter()
    source_files: List[str] = []
    for _score, _recency, card, source_file in candidates:
        if card.kind == "success":
            if counts["success"] >= success_limit:
                continue
            key = (card.problem_style, card.op_type, card.target_family, card.reasoning_signature, *_raw_surface_key(card))
            if key in seen_success_keys:
                continue
            seen_success_keys.add(key)
            counts["success"] += 1
        else:
            if counts["failure"] >= failure_limit:
                continue
            primary_flag = card.quality_flags[0] if card.quality_flags else card.failure_class
            key = (card.problem_style, primary_flag, card.reasoning_signature, *_raw_surface_key(card))
            if key in seen_failure_keys:
                continue
            seen_failure_keys.add(key)
            counts["failure"] += 1
        selected.append(card)
        if source_file not in source_files:
            source_files.append(source_file)
        if len(selected) >= limit:
            break
    return _planner_memory_manifest(enabled=True, cards=selected, source_files=source_files[:8])


def deterministic_fallback_plan(
    pool: List[CertificationInput],
    *,
    pool_size: int = POOL_SIZE,
    survivor_count: int = 1,
    crossover_count: int = DEFAULT_CROSSOVER_COUNT,
) -> Dict[str, Any]:
    """Create a valid central plan without calling an LLM."""
    if len(pool) < pool_size:
        raise ValueError(f"Need {pool_size} pool items, got {len(pool)}")
    if survivor_count not in {0, 1}:
        raise ValueError("MVP supports survivor_count 0 or 1")
    if crossover_count < 0:
        raise ValueError("crossover_count must be non-negative")
    generated_slots = pool_size - survivor_count
    if crossover_count > generated_slots:
        raise ValueError("crossover_count cannot exceed generated slot count")
    if crossover_count and len(pool) < 2:
        raise ValueError("crossover requires at least two parent problems")

    items: List[Dict[str, Any]] = []
    slot = 0
    if survivor_count == 1:
        items.append(
            PoolWorkItem(
                slot=slot,
                op_type="survivor",
                operator_variant="survivor",
                parent_ids=[pool[0].id],
                variation_axis="preserve certified elite unchanged",
                composition_pattern="survivor",
                quality_target="carry forward the strongest certified seed unchanged",
                operator_goal="carry forward certified elite unchanged",
                planner_source="deterministic_fallback",
            ).model_dump()
        )
        slot += 1

    warnings: List[str] = []
    crossover_items: List[Dict[str, Any]] = []

    def _try_crossover(first: CertificationInput, second: CertificationInput) -> Optional[PoolWorkItem]:
        first_family = _problem_family(first)
        second_family = _problem_family(second)
        first_params = _pool_params(first)
        second_params = _pool_params(second)
        if (
            first_family in {"gcd", "gcd_divisor_sum"}
            and second_family in {"divisor_sum", "gcd_divisor_sum"}
            and {"a", "b"} <= set(first_params)
        ):
            fusion_contract = _default_fusion_contract(
                first=first,
                second=second,
                mechanism="",
                parent_a_role="object_domain",
                parent_b_role="computation_target",
                parent_a_contribution="GCD inputs define n",
                parent_b_contribution="sum divisors operation is applied to n",
                new_problem_core="compute sigma(gcd(a,b))",
                why_not_concatenation="",
            )
            return PoolWorkItem(
                slot=slot,
                op_type="crossover",
                operator_variant="crossover_hard",
                parent_ids=[first.id, second.id],
                variation_axis="use the gcd parent as the derived integer and the divisor-sum parent as the target operation",
                operator_goal="build a derived-object chain: gcd inputs define n, then divisor-sum computes sigma(n)",
                reasoning_goal="build a two-step gcd_then_sigma solution skeleton",
                target_family="gcd_divisor_sum",
                composition_pattern="family_bridge",
                parent_contributions={
                    first.id: "GCD inputs define n",
                    second.id: "sum divisors operation is applied to n",
                },
                avoid_patterns=["do not drop the divisor-sum operation", "do not use prose-only crossover"],
                required_checkpoints=["reasoning_pattern", "solution_skeleton", "two_step_reasoning", "semantic_parent_contribution"],
                fusion_contract=fusion_contract,
                quality_target="statement and params must couple gcd with divisor sum",
                planner_source="deterministic_fallback",
            )
        if first_family == "divisor_sum_mod" and second_family in {"divisor_sum", "gcd_divisor_sum"}:
            return _try_crossover(second, first)
        if first_family == "gcd_divisor_sum" and second_family == "gcd":
            return _try_crossover(second, first)
        if first_family == "divisor_sum" and second_family in {"modular_congruence", "divisor_sum_mod"} and "n" in first_params:
            a_value = second_params.get("a") or _answer_int(second)
            if a_value is not None and _value_in_range("divisor_sum_mod", "a", a_value):
                fusion_contract = _default_fusion_contract(
                    first=first,
                    second=second,
                    mechanism="",
                    parent_a_role="object_domain",
                    parent_b_role="computation_target",
                    parent_a_contribution="sum divisors of n defines the modulus",
                    parent_b_contribution=f"mod dividend {int(a_value)} is reused exactly",
                    new_problem_core="compute a mod sigma(n)",
                    why_not_concatenation="",
                )
                return PoolWorkItem(
                    slot=slot,
                    op_type="crossover",
                    operator_variant="crossover_hard",
                    parent_ids=[first.id, second.id],
                    variation_axis="use the divisor-sum answer as the modulus for a modular computation",
                    operator_goal="build a derived-object chain: divisor sum defines m, then reduce a modulo m",
                    reasoning_goal="build a two-step sigma_then_mod solution skeleton",
                    target_family="divisor_sum_mod",
                    composition_pattern="family_bridge",
                    parent_contributions={
                        first.id: "sum divisors of n defines the modulus",
                        second.id: f"mod dividend {int(a_value)} is reused exactly",
                    },
                    avoid_patterns=["do not replace the derived modulus by a free modulus"],
                    required_checkpoints=["reasoning_pattern", "solution_skeleton", "two_step_reasoning", "semantic_parent_contribution"],
                    fusion_contract=fusion_contract,
                    quality_target="statement and params must couple divisor sum with modular reduction",
                    planner_source="deterministic_fallback",
                )
        if _problem_style(first) == "theorem_proof" and _problem_style(second) == "theorem_proof":
            fusion_contract = _default_fusion_contract(
                first=first,
                second=second,
                mechanism="",
                parent_a_role="proof_skeleton",
                parent_b_role="goal_form",
                parent_a_contribution="selected checkpoint/object/hypothesis supplies the pipeline input",
                parent_b_contribution="target theorem goal consumes that pipeline input",
                new_problem_core="one theorem-level pipeline composite, not theorem A and theorem B side by side",
                why_not_concatenation=(
                    "The child must use one parent checkpoint/object/hypothesis inside the other "
                    "parent's formal_statement, proof_obligations, or proof_plan."
                ),
            )
            return PoolWorkItem(
                slot=slot,
                op_type="crossover",
                operator_variant="crossover_easy",
                parent_ids=[first.id, second.id],
                target_style="theorem_proof",
                variation_axis="attempt a theorem-level pipeline or lemma-bundle master crossover",
                operator_goal=(
                    "lemma_bundle_master or pipeline_composite: consume parent checkpoints as "
                    "different subgoals inside one master theorem"
                ),
                reasoning_goal=(
                    "build one Lean theorem where parent checkpoints are consumed in a single "
                    "master proof, not theorem A and theorem B side by side"
                ),
                target_family="theorem_proof",
                composition_pattern="family_bridge",
                parent_contributions={
                    first.id: "pipeline input checkpoint/object/hypothesis",
                    second.id: "goal form or proof target that consumes the input",
                },
                avoid_patterns=[
                    "do not create theorem A and theorem B side-by-side",
                    "do not use independent conjunction",
                    "must_be_pipeline_composite",
                    "lemma_bundle_master_allowed",
                ],
                required_checkpoints=[
                    "theorem_style_preserved",
                    "lean_proof_complete",
                    "semantic_parent_contribution",
                    "multiple_parent_checkpoints",
                ],
                fusion_contract=fusion_contract,
                fusion_goal=(
                    "Bundle→Master: parent A and parent B supply certified checkpoints that "
                    "appear as distinct proof obligations or intermediate lemmas inside one final theorem"
                ),
                parent_roles={
                    first.id: "checkpoint/subgoal source",
                    second.id: "checkpoint/subgoal source",
                },
                quality_target="Lean proof complete and master theorem consumes both parent checkpoints",
                planner_source="deterministic_fallback",
            )
        return None

    used_crossover_parents: set[str] = set()
    for _ in range(crossover_count):
        item: Optional[PoolWorkItem] = None
        pair_label = ""
        for first in pool:
            for second in pool:
                if first.id == second.id:
                    continue
                if first.id in used_crossover_parents or second.id in used_crossover_parents:
                    continue
                pair_label = f"{first.id},{second.id}"
                item = _try_crossover(first, second)
                if item is not None:
                    break
            if item is not None:
                break
        if item is not None:
            crossover_items.append(item.model_dump())
            used_crossover_parents.update(item.parent_ids)
        else:
            warnings.append(
                f"crossover_downgraded_to_mutation: slot {slot} pair {pair_label or 'none'}"
            )
            crossover_count -= 1
            continue
        slot += 1

    items.extend(crossover_items)

    mutation_needed = pool_size - len(items)
    for idx in range(mutation_needed):
        parent = pool[(slot + idx) % len(pool)]
        parent_style = _problem_style(parent)
        family = _problem_family(parent) or ("theorem proof" if parent_style == "theorem_proof" else "supported template")
        target_family = "theorem_proof" if parent_style == "theorem_proof" else (family if family != "supported template" else "")
        operator_variant = _default_operator_variant(
            "mutation",
            target_family=target_family,
            parent_family=_problem_family(parent),
        )
        if idx == 0 and target_family:
            operator_variant = "mutation_hard"
        items.append(
            PoolWorkItem(
                slot=slot,
                op_type="mutation",
                operator_variant=operator_variant,
                parent_ids=[parent.id],
                target_style=parent_style,
                variation_axis=f"increase difficulty while preserving the {family} style",
                operator_goal=f"build a harder {family} child that preserves parent style",
                reasoning_goal=(
                    "strengthen the theorem statement/proof obligation while preserving theorem style"
                    if parent_style == "theorem_proof"
                    else f"choose a richer {family} reasoning skeleton, then project it to canonical params"
                ),
                target_family=target_family,
                composition_pattern="parameter_shift",
                avoid_patterns=[
                    "do not turn theorem/proof parents into numeric-answer children"
                    if parent_style == "theorem_proof"
                    else "do not keep the same canonical parameters as the parent"
                ],
                required_checkpoints=(
                    ["theorem_style_preserved", "lean_proof_complete", "harder_proof_obligation"]
                    if parent_style == "theorem_proof"
                    else ["reasoning_pattern", "solution_skeleton", "projected_params"]
                ),
                quality_target=(
                    "Lean proof complete and theorem style preserved"
                    if parent_style == "theorem_proof"
                    else "strictly increase the family-specific difficulty metric"
                ),
                planner_source="deterministic_fallback",
            ).model_dump()
        )
        slot += 1

    return {
        "planner_source": "deterministic_fallback",
        "plan_rationale": "Deterministic 5-pool fallback: fixed survivor plus generated slots.",
        "warnings": warnings,
        "work_items": items,
    }


def validate_pool_plan(
    plan: Dict[str, Any],
    pool: List[CertificationInput],
    *,
    pool_size: int = POOL_SIZE,
    survivor_count: int = 1,
    crossover_count: int = DEFAULT_CROSSOVER_COUNT,
) -> List[Dict[str, Any]]:
    items = [dict(item) for item in (plan.get("work_items") or plan.get("dispatch_items") or [])]
    if len(items) != pool_size:
        raise ValueError(f"planner must emit exactly {pool_size} work_items")

    parent_map = _problem_by_id(pool)
    parent_ids = set(parent_map)
    pool_ids = [problem.id for problem in pool]

    def _force_theorem_crossover_exploration(items_: List[PoolWorkItem]) -> None:
        if crossover_count <= 0:
            return
        if not all(_problem_style(problem) == "theorem_proof" for problem in pool):
            return
        current_crossover_count = sum(1 for item in items_ if item.op_type == "crossover")
        target_crossover_count = min(
            crossover_count,
            sum(1 for item in items_ if item.op_type != "survivor"),
        )
        if current_crossover_count >= target_crossover_count:
            return

        used_pairs = {
            tuple(sorted(item.parent_ids))
            for item in items_
            if item.op_type == "crossover" and len(item.parent_ids) == 2
        }
        mutation_items = [item for item in items_ if item.op_type == "mutation"]
        for target in mutation_items:
            if current_crossover_count >= target_crossover_count:
                break
            first_id = target.parent_ids[0] if target.parent_ids else pool_ids[0]
            second_id = next(
                (
                    problem.id
                    for problem in pool
                    if problem.id != first_id
                    and not _same_root_lineage([first_id, problem.id])
                    and tuple(sorted([first_id, problem.id])) not in used_pairs
                ),
                "",
            )
            if not second_id:
                continue
            first = parent_map[first_id]
            second = parent_map[second_id]
            target.op_type = "crossover"
            target.operator_variant = "crossover_easy"
            target.parent_ids = [first_id, second_id]
            target.parent_refs = [pool_ids.index(first_id), pool_ids.index(second_id)]
            target.target_style = "theorem_proof"
            target.target_family = "theorem_proof"
            target.composition_pattern = "family_bridge"
            target.variation_axis = "forced theorem pipeline crossover exploration"
            target.operator_goal = (
                "pipeline_composite: use one parent checkpoint/object/hypothesis as an input "
                "to the other parent's theorem goal"
            )
            target.reasoning_goal = target.operator_goal
            target.fusion_goal = (
                "parent A supplies a checkpoint/object/hypothesis that is visible in "
                "parent B's formal_statement, proof_obligations, or proof_plan"
            )
            target.parent_roles = {
                first_id: "pipeline input source",
                second_id: "goal/proof target source",
            }
            target.parent_contributions = {
                first_id: "pipeline input checkpoint/object/hypothesis",
                second_id: "goal form or proof target that consumes the input",
            }
            target.fusion_contract = _default_fusion_contract(
                first=first,
                second=second,
                mechanism="",
                parent_a_role="proof_skeleton",
                parent_b_role="goal_form",
                parent_a_contribution="selected checkpoint/object/hypothesis supplies the pipeline input",
                parent_b_contribution="target theorem goal consumes that pipeline input",
                new_problem_core="one theorem-level pipeline composite, not theorem A and theorem B side by side",
                why_not_concatenation=(
                    "The child must use one parent checkpoint/object/hypothesis inside the other "
                    "parent's formal_statement, proof_obligations, or proof_plan."
                ),
            )
            target.avoid_patterns = list(
                dict.fromkeys(
                    list(target.avoid_patterns)
                    + [
                        "forced_crossover_exploration",
                        "must_be_pipeline_composite",
                        "side_by_side_conjunction_forbidden",
                    ]
                )
            )
            target.required_checkpoints = list(
                dict.fromkeys(
                    list(target.required_checkpoints)
                    + ["theorem_style_preserved", "lean_proof_complete", "semantic_parent_contribution"]
                )
            )
            target.quality_target = "Lean proof complete and pipeline parent usage observable"
            target.parent_context_cards = _parent_context_cards([first, second])
            target.operator_card = _operator_card(target.model_dump())
            used_pairs.add(tuple(sorted(target.parent_ids)))
            current_crossover_count += 1

    def _force_bounded_generalization_slot(items_: List[PoolWorkItem]) -> None:
        generated = [item for item in items_ if item.op_type != "survivor"]
        if not generated:
            return
        if any(
            "bounded_generalization" in " ".join(
                [
                    str(item.operator_goal or ""),
                    str(item.reasoning_goal or ""),
                    str(item.variation_axis or ""),
                    " ".join(str(value) for value in item.constraints),
                    " ".join(str(value) for value in item.required_checkpoints),
                ]
            )
            for item in generated
        ):
            return
        target = next((item for item in generated if item.op_type == "mutation"), generated[0])
        if target.target_style == "theorem_proof":
            goal = (
                "bounded_generalization: generalize one constant, index, hypothesis, "
                "domain, or conclusion parameter while keeping parent proof checkpoints visible"
            )
        else:
            goal = (
                "bounded_generalization: generalize one numeric parameter or derived object "
                "inside the supported family while keeping canonical params executable"
            )
        target.operator_goal = f"{goal}; prior_goal={target.operator_goal or target.reasoning_goal}"
        target.reasoning_goal = target.operator_goal
        target.variation_axis = target.operator_goal
        target.constraints = list(
            dict.fromkeys(
                list(target.constraints)
                + [
                    "bounded_generalization",
                    "change_exactly_one_generalization_axis",
                    "parent_checkpoint_visible",
                ]
            )
        )
        target.required_checkpoints = list(
            dict.fromkeys(list(target.required_checkpoints) + ["bounded_generalization"])
        )
        target.avoid_patterns = list(
            dict.fromkeys(
                list(target.avoid_patterns)
                + ["arbitrary_broad_generalization", "same_statement_rephrase"]
            )
        )

    seen_slots = set()
    normalized: List[PoolWorkItem] = []
    for item in items:
        goal = str(
            item.get("goal")
            or item.get("operator_goal")
            or item.get("reasoning_goal")
            or item.get("variation_axis")
            or item.get("op_type")
            or ""
        ).strip()
        constraints = item.get("constraints") or []
        if isinstance(constraints, str):
            constraints = [constraints]
        avoid = item.get("avoid") or item.get("avoid_patterns") or []
        if isinstance(avoid, str):
            avoid = [avoid]
        parent_refs = item.get("parent_refs") or []
        if (not item.get("parent_ids")) and parent_refs:
            try:
                item["parent_ids"] = [pool_ids[int(ref)] for ref in parent_refs]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"slot {item.get('slot')} has invalid parent_refs") from exc
        if item.get("fusion_contract") and parent_refs:
            fusion_contract = dict(item.get("fusion_contract") or {})
            for label in ("parent_A", "parent_B"):
                parent = dict(fusion_contract.get(label) or {})
                raw_ref = parent.get("ref")
                if raw_ref is not None and not parent.get("id"):
                    try:
                        parent["id"] = pool_ids[int(raw_ref)]
                    except (IndexError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"slot {item.get('slot')} has invalid fusion_contract.{label}.ref"
                        ) from exc
                fusion_contract[label] = parent
            item["fusion_contract"] = fusion_contract
        work_item = PoolWorkItem.model_validate(
            {
                **item,
                "variation_axis": item.get("variation_axis") or goal,
                "reasoning_goal": item.get("reasoning_goal") or goal,
                "target_family": item.get("target_family") or "",
                "required_params": item.get("required_params") or {},
                "composition_pattern": item.get("composition_pattern")
                or (
                    "survivor"
                    if item.get("op_type") == "survivor"
                    else (
                        "parameter_shift"
                        if item.get("op_type") == "mutation"
                        else "family_bridge"
                    )
                ),
                "parent_contributions": item.get("parent_contributions") or {},
                "avoid_patterns": avoid,
                "quality_target": item.get("quality_target") or "",
                "operator_goal": item.get("operator_goal") or goal,
                "required_checkpoints": item.get("required_checkpoints") or [],
                "avoid_signatures": item.get("avoid_signatures") or [],
                # The planner answers with flat fields because that is what the
                # schema asks for; the pipeline reads them from a nested
                # `fusion_contract`. Without this fold the mechanism the planner
                # chose is present in its reply and invisible to everything
                # downstream.
                "fusion_contract": _fold_fusion_contract(item),
                "target_style": item.get("target_style") or "numeric_answer",
                "parent_context_cards": item.get("parent_context_cards") or [],
                "operator_card": item.get("operator_card") or {},
                "operator_variant": item.get("operator_variant") or "",
                "parent_refs": parent_refs,
                "goal": goal,
                "constraints": constraints,
                "avoid": avoid,
                "fusion_goal": item.get("fusion_goal") or "",
                "parent_roles": item.get("parent_roles") or {},
                "memory_delta_contract": item.get("memory_delta_contract") or {},
                "difficulty_label": item.get("difficulty_label") or "medium",
                "planner_source": plan.get("planner_source")
                or item.get("planner_source")
                or "orchestrator_llm",
            }
        )
        if work_item.slot in seen_slots:
            raise ValueError(f"duplicate slot {work_item.slot}")
        seen_slots.add(work_item.slot)
        if work_item.slot < 0 or work_item.slot >= pool_size:
            raise ValueError(f"slot {work_item.slot} outside pool size {pool_size}")
        if work_item.op_type not in {"survivor", "mutation", "crossover"}:
            raise ValueError(f"unsupported op_type {work_item.op_type!r}")
        if any(parent_id not in parent_ids for parent_id in work_item.parent_ids):
            raise ValueError(f"slot {work_item.slot} references unknown parent ids")
        if (
            work_item.op_type == "crossover"
            and len(work_item.parent_ids) == 2
            and len(set(work_item.parent_ids)) == 2
            and _same_root_lineage(work_item.parent_ids)
        ):
            kept_parent_id = work_item.parent_ids[0]
            work_item.op_type = "mutation"
            work_item.operator_variant = "mutation_easy"
            work_item.parent_ids = [kept_parent_id]
            work_item.parent_refs = work_item.parent_refs[:1]
            work_item.composition_pattern = "parameter_shift"
            work_item.fusion_contract = {}
            work_item.parent_contributions = {}
            work_item.fusion_goal = ""
            work_item.parent_roles = {}
            work_item.avoid_patterns = list(
                dict.fromkeys(list(work_item.avoid_patterns) + ["same_lineage_crossover"])
            )
            work_item.avoid_signatures = list(
                dict.fromkeys(list(work_item.avoid_signatures) + ["same_lineage_crossover"])
            )
            work_item.operator_goal = (
                "downgraded same-lineage crossover to mutation; create a goal-form projection "
                "from one parent instead of crossing a parent with its descendant"
            )
            work_item.reasoning_goal = work_item.operator_goal
            work_item.variation_axis = work_item.operator_goal
        slot_parents = [parent_map[parent_id] for parent_id in work_item.parent_ids]
        has_theorem_parent = any(_problem_style(parent) == "theorem_proof" for parent in slot_parents)
        inferred_target_style = _target_style_for_item(work_item.model_dump(), slot_parents)
        if has_theorem_parent:
            if work_item.op_type != "survivor" and inferred_target_style != "theorem_proof":
                raise ValueError(
                    f"slot {work_item.slot} cannot project theorem parent to numeric_answer"
                )
        if work_item.op_type == "survivor" and slot_parents:
            inferred_target_style = _problem_style(slot_parents[0])
        work_item.target_style = inferred_target_style
        if has_theorem_parent and work_item.op_type != "survivor":
            work_item.target_family = "theorem_proof"
        work_item.parent_context_cards = _parent_context_cards(slot_parents)
        if work_item.op_type == "survivor" and len(work_item.parent_ids) != 1:
            raise ValueError(f"survivor slot {work_item.slot} must have one parent")
        if work_item.op_type == "mutation" and len(work_item.parent_ids) != 1:
            raise ValueError(f"mutation slot {work_item.slot} must have one parent")
        if work_item.op_type == "crossover":
            if len(work_item.parent_ids) != 2 or len(set(work_item.parent_ids)) != 2:
                raise ValueError(f"crossover slot {work_item.slot} must have two distinct parents")
            fusion_contract = dict(work_item.fusion_contract or {})
            if fusion_contract:
                parent_a = _fusion_parent(fusion_contract, "parent_A")
                parent_b = _fusion_parent(fusion_contract, "parent_B")
                for label, parent in (("parent_A", parent_a), ("parent_B", parent_b)):
                    parent_id = str(parent.get("id") or "")
                    contribution = str(parent.get("contribution") or "")
                    if not parent_id or parent_id not in work_item.parent_ids:
                        raise ValueError(f"crossover slot {work_item.slot} invalid {label}.id")
                    if not contribution.strip():
                        parent["contribution"] = (
                            work_item.fusion_goal
                            or work_item.operator_goal
                            or "planned crossover contribution"
                        )
                        fusion_contract[label] = parent
                mechanism = str(fusion_contract.get("fusion_mechanism") or "")
                if mechanism and mechanism not in FUSION_MECHANISMS:
                    raise ValueError(
                        f"crossover slot {work_item.slot} unsupported fusion_mechanism"
                    )
                work_item.parent_contributions = _fusion_parent_contributions(
                    work_item.model_dump()
                )
            elif work_item.parent_roles or work_item.fusion_goal:
                role_text = dict(work_item.parent_roles or {})
                work_item.parent_contributions = {
                    parent_id: str(role_text.get(parent_id) or work_item.fusion_goal or work_item.operator_goal)
                    for parent_id in work_item.parent_ids
                }
            else:
                work_item.parent_contributions = {
                    parent_id: work_item.operator_goal or "planned crossover contribution"
                    for parent_id in work_item.parent_ids
                }
            if work_item.target_style == "theorem_proof":
                parent_cards = list(work_item.parent_context_cards or [])
                atom_sets = [
                    set((card.get("theorem_decomposition") or {}).get("reusable_lean_atoms") or [])
                    for card in parent_cards
                    if card.get("problem_style") == "theorem_proof"
                ]
                shared_atoms = set.intersection(*atom_sets) if len(atom_sets) >= 2 else set()
                shared_surface = str(fusion_contract.get("shared_lean_surface") or "").strip()
                mechanism = str(fusion_contract.get("fusion_mechanism") or "")
                why_not_concatenation = str(fusion_contract.get("why_not_concatenation") or "")
                if mechanism == "sequential_composition" or not why_not_concatenation.strip():
                    work_item.avoid_patterns.append("crossover_may_be_pipeline_composite")
                if not shared_atoms and shared_surface.lower() in {"", "none", "not_available"}:
                    work_item.avoid_patterns.append("crossover_lacks_shared_lean_surface")
                    work_item.avoid_patterns.append("must_be_pipeline_composite")
                    if work_item.operator_variant == "crossover_hard":
                        work_item.operator_variant = "crossover_easy"
                    if not work_item.fusion_goal:
                        work_item.fusion_goal = (
                            "pipeline_composite: use parent A's checkpoint/object/hypothesis "
                            "as an input to parent B's theorem goal; do not create a side-by-side conjunction"
                        )
                    if mechanism not in FUSION_MECHANISMS:
                        # Recorded, not silently rewritten. This line used to
                        # default an unrecognised mechanism to
                        # `sequential_composition`, and since that was also the
                        # only mechanism the prompt explained, 150 of 186
                        # crossovers ended up there and the other five were never
                        # once attempted. A default that quietly collapses a
                        # design space is worse than an empty field, because the
                        # field at least shows up as missing.
                        fusion_contract["fusion_mechanism"] = ""
                        fusion_contract["mechanism_rejected"] = mechanism
                    if not str(fusion_contract.get("why_not_concatenation") or "").strip():
                        fusion_contract["why_not_concatenation"] = (
                            "This must be a pipeline handoff: one parent changes the formal_statement, "
                            "proof_obligations, or proof_plan of the other, not theorem A ∧ theorem B."
                        )
                    if fusion_contract:
                        work_item.fusion_contract = fusion_contract
        if work_item.op_type != "survivor" and len(work_item.variation_axis.strip()) < 4:
            work_item.variation_axis = work_item.operator_goal or work_item.op_type
        if work_item.op_type != "survivor" and len(work_item.reasoning_goal.strip()) < 4:
            work_item.reasoning_goal = work_item.variation_axis
        if work_item.op_type != "survivor" and len(work_item.quality_target.strip()) < 4:
            work_item.quality_target = work_item.operator_goal or work_item.reasoning_goal
        if work_item.op_type != "survivor":
            allowed_patterns = COMPOSITION_PATTERNS[work_item.op_type]
            if work_item.composition_pattern not in allowed_patterns:
                raise ValueError(
                    f"slot {work_item.slot} has invalid composition_pattern "
                    f"{work_item.composition_pattern!r}"
                )
        if work_item.op_type != "survivor" and work_item.target_style == "theorem_proof":
            work_item.target_family = work_item.target_family or "theorem_proof"
        if (
            work_item.op_type != "survivor"
            and work_item.target_style != "theorem_proof"
            and not work_item.target_family
        ):
            first_parent = _problem_by_id(pool)[work_item.parent_ids[0]]
            work_item.target_family = _problem_family(first_parent)
        if (
            work_item.op_type != "survivor"
            and work_item.target_style != "theorem_proof"
            and work_item.target_family not in SUPPORTED_FAMILIES
        ):
            raise ValueError(f"slot {work_item.slot} has unsupported target_family")
        if not work_item.operator_variant:
            first_parent = _problem_by_id(pool)[work_item.parent_ids[0]]
            work_item.operator_variant = _default_operator_variant(
                work_item.op_type,
                target_family=work_item.target_family,
                parent_family=_problem_family(first_parent),
            )
        allowed_variants = {
            "survivor",
            "mutation_easy",
            "mutation_hard",
            # Offered to the planner in the prompt's shape example and rejected
            # here, which meant a plan that used it was thrown out whole and
            # replaced by the deterministic fallback -- the same shape of failure
            # as the crossover contract, where the prompt asked for one form and
            # validation demanded another.
            "mutation_silent",
            "crossover_easy",
            "crossover_hard",
        }
        if work_item.operator_variant not in allowed_variants:
            raise ValueError(
                f"slot {work_item.slot} has invalid operator_variant {work_item.operator_variant!r}"
            )
        if work_item.op_type == "mutation" and not work_item.operator_variant.startswith("mutation_"):
            raise ValueError(f"mutation slot {work_item.slot} has incompatible operator_variant")
        if work_item.op_type == "crossover" and not work_item.operator_variant.startswith("crossover_"):
            raise ValueError(f"crossover slot {work_item.slot} has incompatible operator_variant")
        if work_item.op_type == "survivor":
            work_item.operator_variant = "survivor"
        if work_item.operator_variant == "mutation_hard":
            first_parent = _problem_by_id(pool)[work_item.parent_ids[0]]
            parent_family = _problem_family(first_parent)
            if (
                work_item.target_style != "theorem_proof"
                and parent_family in SUPPORTED_FAMILIES
                and _family_complexity_rank(
                work_item.target_family
                ) < _family_complexity_rank(parent_family)
            ):
                raise ValueError(
                    f"mutation_hard slot {work_item.slot} simplifies parent family "
                    f"{parent_family!r} to {work_item.target_family!r}"
                )
        if work_item.op_type != "survivor":
            work_item.required_checkpoints = _normalize_required_checkpoints(
                work_item.required_checkpoints
            )
        if work_item.required_params and not isinstance(work_item.required_params, dict):
            raise ValueError(f"slot {work_item.slot} required_params must be an object")
        allowed_params = FAMILY_PARAM_KEYS.get(work_item.target_family, set())
        unknown_params = set(work_item.required_params) - allowed_params
        if work_item.target_style != "theorem_proof" and unknown_params:
            raise ValueError(
                f"slot {work_item.slot} has invalid required_params keys: "
                f"{sorted(unknown_params)}"
            )
        for key, value in work_item.required_params.items():
            if work_item.target_style == "theorem_proof":
                break
            low, high = FAMILY_PARAM_RANGES[work_item.target_family][key]
            try:
                int_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"slot {work_item.slot} required_params[{key}] must be an integer"
                ) from exc
            if int_value < low or int_value > high:
                raise ValueError(
                    f"slot {work_item.slot} required_params[{key}]={int_value} "
                    f"outside [{low}, {high}]"
                )
        if work_item.op_type == "crossover":
            missing_contrib = [
                parent_id
                for parent_id in work_item.parent_ids
                if not str(work_item.parent_contributions.get(parent_id, "")).strip()
            ]
            for parent_id in missing_contrib:
                work_item.parent_contributions[parent_id] = (
                    work_item.fusion_goal or work_item.operator_goal or "planned crossover contribution"
                )
        work_item.operator_card = _operator_card(work_item.model_dump())
        normalized.append(work_item)

    _force_theorem_crossover_exploration(normalized)
    _force_bounded_generalization_slot(normalized)
    _reject_same_lineage_crossovers(normalized)
    for item in normalized:
        item.operator_card = _operator_card(item.model_dump())

    if sorted(seen_slots) != list(range(pool_size)):
        raise ValueError("slots must be contiguous from 0 to pool_size-1")
    if sum(1 for item in normalized if item.op_type == "survivor") != survivor_count:
        raise ValueError(f"plan must contain survivor_count={survivor_count}")
    actual_crossover = sum(1 for item in normalized if item.op_type == "crossover")
    if actual_crossover > crossover_count:
        raise ValueError(f"plan must contain at most crossover_count={crossover_count}")
    # Also when `crossover_count` is 0. The mutation arm planned one
    # `crossover_easy` slot anyway: the lock stops replanning from changing an
    # operator, and this stops the planner from introducing the other one in the
    # first place. Without both, "mutation-only" is not a condition that holds.
    if _op_type_locked() and actual_crossover != crossover_count:
        raise ValueError(
            f"op_type is locked: plan must contain exactly crossover_count="
            f"{crossover_count}, got {actual_crossover}"
        )
    return [item.model_dump() for item in sorted(normalized, key=lambda item: item.slot)]


def _planner_messages(
    pool: List[CertificationInput],
    *,
    pool_size: int,
    survivor_count: int,
    crossover_count: int,
    op_type_allocation_hint: Optional[Dict[str, Any]] = None,
    generation_feedback: Optional[Dict[str, Any]] = None,
    planner_memory: Optional[Dict[str, Any]] = None,
    novelty_memory: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    pool_lines = []
    for idx, problem in enumerate(pool):
        card = _parent_context_card(problem)
        pool_lines.append(
            f"[{idx}] ParentContextCard="
            f"{json.dumps(_planner_parent_card(card), ensure_ascii=False)[:1800]}"
        )
    system = (
        "You are the central orchestrator for a Lean-certifiable math problem pool. "
        "Your job is to synthesize high-level slot plans for independent workers. "
        "Return JSON only."
    )
    feedback_block = ""
    if generation_feedback:
        raw_feedback = _planner_feedback_payload(generation_feedback)
        feedback_block = (
            "\nPrevious-generation raw cases:\n"
            "- These are verifier-derived cases from earlier generations of this same run.\n"
            "- Prefer successful raw surfaces when compatible; avoid repeating failed surfaces and failure signatures.\n"
            f"{json.dumps(raw_feedback, ensure_ascii=False)}\n"
        )
    allocation_block = ""
    if op_type_allocation_hint:
        allocation_block = (
            "\nOp-type allocation hint:\n"
            f"{json.dumps(op_type_allocation_hint, ensure_ascii=False)[:1800]}\n"
        )
    memory_block = ""
    if planner_memory and planner_memory.get("enabled"):
        cases = planner_memory.get("cases") or planner_memory.get("cards") or []
        cases = _planner_cases_for_prompt(cases)
        memory_payload = {
            "success_cases": [case for case in cases if dict(case).get("kind") == "success"],
            "failure_cases": [case for case in cases if dict(case).get("kind") == "failure"],
        }
        memory_block = (
            "\nCross-run raw semantic case pack:\n"
            "- These are prior verifier-derived success/failure raw cases, not current parent IDs.\n"
            "- Use them only as strategy evidence for choosing safer OperatorCards.\n"
            "- Never reference case_id or source_problem_id as a parent; only parent_refs from the current pool are valid.\n"
            f"{json.dumps(memory_payload, ensure_ascii=False)}\n"
        )
    novelty_block = ""
    if novelty_memory and novelty_memory.get("enabled"):
        novelty_block = (
            "\n"
            + format_novelty_memory_pack(novelty_memory)
            + "\n"
            "- Treat exact_blockers as surfaces/signatures that generated slots must not recreate.\n"
            "- Treat soft_neighbors as evidence for delta planning; they are not parents and are not automatic failures.\n"
            "- For each generated work_item, include memory_delta_contract with: "
            "similar_card_ids, must_not_repeat, required_distinguishing_delta, "
            "allowed_overlap, novelty_rationale.\n"
            "- Closest accepted/run-local analogues are evidence, not parents.\n"
            "- Do not recreate the matched target semantics.\n"
        )
    pool_target_style = (
        "theorem_proof"
        if any(_problem_style(problem) == "theorem_proof" for problem in pool)
        else "numeric_answer"
    )
    planner_no_go_pack = build_no_go_policy_pack(
        op_type="crossover" if crossover_count > 0 else "mutation",
        target_style=pool_target_style,
        target_family="theorem_proof" if pool_target_style == "theorem_proof" else "",
        operator_variant="crossover_easy" if crossover_count > 0 else "mutation_easy",
        recent_failure_flags=list((generation_feedback or {}).get("quality_flags") or [])
        + list((generation_feedback or {}).get("weak_signature_summary") or []),
        limit=10,
    )
    # Distilled lessons replace the 58-rule pack here. The pack listed rules by
    # flag, and 46 of its 58 never fired on 1,431 rows, so most of what it
    # carried was advice about failures this pipeline does not make while the
    # ones it does make arrived as flag names.
    policy_block = (
        "\n"
        + format_lessons(
            planner_lessons(
                list((generation_feedback or {}).get("quality_flags") or [])
            ),
            title="Planning lessons from this pool's own failures",
        )
        + "\n"
    )
    theorem_only_block = ""
    min_crossover = 1 if crossover_count > 0 and len(pool) >= 2 else 0
    # "at most N" let the planner answer zero and still comply, which is what it
    # did: asked for four crossovers it returned four mutations, and the pipeline
    # then promoted mutations into crossovers after the fact and stamped a
    # mechanism on them. A ceiling is right for the mixed pipeline, where the
    # slot is worth filling whichever way fills it. An ablation arm needs the
    # count to be the thing measured, so under the lock it becomes exact.
    _locked = _op_type_locked()
    # `at most N` is a ceiling with no floor, so a plan with zero crossovers
    # broke no rule. Measured standalone, five of six plans had none at
    # crossover_count=2, and production delivered 354 where the budget implied
    # roughly 512. The locked arms still say `exactly`; the mixed pipeline now
    # asks for the budget and requires a reason for spending less, rather than
    # permitting silence.
    crossover_requirement = (
        f"exactly {crossover_count}" if _locked else f"{crossover_count}"
    )
    crossover_budget_line = (
        (
            f"crossover_count={crossover_count} is a requirement, not a budget. This run "
            "measures the crossover operator, so a plan with fewer crossover slots is "
            "invalid. If a pair looks unpromising, still plan the crossover and say so in "
            "plan_rationale; do not substitute a mutation."
        )
        if _locked
        else (
            f"crossover_count={crossover_count} is the requested crossover budget and the "
            "expected number of crossover slots. Planning fewer is allowed only when no pair "
            "in the pool can be fused, and then plan_rationale must name the pairs you "
            "rejected and why. Choose crossover_easy or crossover_hard per slot on the merits "
            "of the pair; neither is the default."
        )
    )
    if all(_problem_style(problem) == "theorem_proof" for problem in pool):
        # This block was written when the goal was to get anything at all to
        # certify, and it steered three choices it should only have informed.
        # It named crossover_easy as the shape of every crossover slot, which is
        # why the planner chose crossover_hard zero times in 1,281 slots. It said
        # "prefer mutation_easy", which is why the tiers never balanced. And it
        # offered pipeline_composite as the fallback whenever fusion looked hard,
        # which is the `parallel` failure -- two parents proved separately and
        # joined at the end -- that the judge rejected thirty times in one
        # release. The constraints that remain are the ones about what a
        # theorem-only pool may target, not about which operator is safest.
        theorem_only_block = (
            "\nCurrent pool style: theorem_proof only.\n"
            "- Treat numeric-family rules below as irrelevant unless explicitly requested by a numeric parent.\n"
            "- Every generated non-survivor slot should target target_style=theorem_proof and target_family=theorem_proof.\n"
            "- All six mutation and crossover variants are available. Pick per slot from the parents in front of you; do not fall back to one variant because it is likelier to certify.\n"
            "- A shared Lean atom is helpful for crossover but not required; a plausible theorem-level handoff is enough.\n"
            "- sequential_composition is one of six fusion mechanisms and the weakest. Do not reach for it because fusion looks hard: a child whose parents are proved separately and joined at the end is rejected, so plan a different mechanism or plan a mutation instead.\n"
            "- Treat TFAE/characterization as pilot vocabulary only unless two certified iff/implication parents make it obviously executable.\n"
            "- Do not use side-by-side conjunction; if no genuine interaction between the parents can be named, plan a mutation for that slot.\n"
        )
    user = f"""
Plan the next generation for a fixed pool of {pool_size} problems.
{theorem_only_block}

Your job:
- Produce minimal high-level OperatorCards for independent slot workers.
- Stay at the operator/reasoning level. Do not solve detailed numeric params unless a parent-derived value is mandatory.
- Optimize for next-generation parent quality, not just immediate generation count.
- Use only supported single families and supported composite families.

Hard rules:
- Emit exactly {pool_size} work_items.
- Emit exactly {survivor_count} survivor slots.
- Emit {crossover_requirement} crossover slots.
- Emit at least one generated non-survivor slot whose goal starts with bounded_generalization when possible.
- The six variants exist to be used. If every generated mutation slot in your plan has the same operator_variant, plan_rationale must say why the pool admits no other.
- bounded_generalization is valid for numeric_answer and theorem_proof routes.
- {crossover_budget_line}
- Emit at least {min_crossover} crossover slot(s) when possible.
- Use the op-type allocation hint when present, but do not collapse to mutation-only unless no executable parent pair exists.
- If using fewer crossovers than the allocation hint recommends or fewer than the minimum above, include the reason in plan_rationale.
- Remaining non-survivor slots must be mutation slots.
- survivor: one parent_id, no generation.
- mutation: one parent_ref, target_style, target_family, operator_variant, goal.
- crossover: two distinct parent_refs, target_style, target_family, operator_variant, goal, optional fusion_goal/parent_roles.
- target_style must be numeric_answer for numeric parents and theorem_proof for theorem/proof parents.
- If any selected parent card has problem_style=theorem_proof, the work item must use target_style=theorem_proof and target_family=theorem_proof.
- Never project a theorem/proof parent into a numeric-answer child.
- Prefer parent_refs over parent_ids. parent_refs are zero-based indices from the Seed pool; the system projects them to exact IDs.
- operator_variant must be survivor, mutation_easy, mutation_hard, mutation_silent, crossover_easy, or crossover_hard.
- Use mutation_easy for unsupported/abstract parents that need a stable Lean-template bridge.
- Use mutation_hard only when the parent already has a supported/certified surface and can absorb a stricter reasoning checkpoint.
- mutation_hard must not simplify a supported parent into an easier single-step family.
- Use mutation_silent when the parent's mathematics should be held fixed and only its Lean surface restated. See the mutation variant rules below.
- crossover_easy and crossover_hard are both first choices, not a default and an exception. Use crossover_hard when the two parents contribute different kinds of thing and the child needs an argument neither parent's proof contains; use crossover_easy when one parent supplies the setting and the other supplies a step that fits into it.
- Never crossover two parents with the same root lineage, e.g. a seed and its `__theorem_gen` descendant. Use mutation on one parent instead.
- Do not invent parent IDs.
- Do not try to solve all exact numeric params at the planner level; the slot generator owns detailed params.
- Do not emit proof obligations, checkpoint lists, quality evidence, or detailed params. The worker and verifier own those.
- Put any mandatory constraints into short constraints/avoid lists.
- Crossover may include a short fusion_goal and parent_roles. Avoid nested schemas unless truly helpful.
- NoveltyMemoryPack is not a parent source. Use it only to avoid accepted/run-local near-duplicates.
- Every generated non-survivor slot should include memory_delta_contract. Name what must change, not only what is forbidden.

Accepted-grade hard target:
- Plan each generated slot as either a paper_frontier attempt or an explicit scaffold-to-frontier step.
- A scaffold/local corollary is useful only when the goal explains how it will be consumed by a later or bundled final theorem.
- Do not present direct corollary, local bookkeeping, same-role cyclic transport, topology direct consequence, or fixed finite aggregate surfaces as paper_frontier goals.
- At least one generated non-survivor slot should target an accepted-grade playbook mode when possible:
  latent_parameter_solve, domain_pipeline_sum, card_and_sum_pipeline, lemma_bundle_master,
  nontrivial_hypothesis_specialization, or bounded_generalization.
- Put the playbook mode at the start of goal, put the concrete success condition in constraints,
  and put any known reject pattern in avoid.
- Do not plan helper-only theorem rows as final targets; helpers must feed a final theorem.

Entropy direction target:
- Prefer entropy-increase children over merely harder-looking children.
- Entropy increase includes a new proof obligation, a new formal surface, a parent checkpoint feeding a new target, or a bounded generalization.
- Entropy decrease includes same-statement repairs, syntactic wrappers, vacuity, unused parents, and side-by-side theorem conjunctions.
- Certified corollaries/checkpoints are not automatically failures, but they must be planned as inputs to a new theorem role or bounded generalization.
- Numeric bounded_generalization means changing one canonical family parameter or derived object while staying inside supported ranges and keeping the answer deterministic.

Execution surface:
- Workers first output a solution skeleton / reasoning pattern, then project it to family + params.
- The system rebuilds statement and answer from those params.
- Free-form hidden wording is ignored.
- Supported composite families are the only valid way to create derived-object crossover statements.
- For theorem_proof target_style, workers output statement, formal_statement, lean_header, proof_plan, proof_obligations, and lean_code instead of numeric params.
- For theorem_proof target_style, plan over theorem_decomposition.proof_checkpoints and main_conclusion; do not invent a larger theorem than those chunks can support.
- For theorem_proof target_style, operator_goal must fit one accepted-grade mode when possible:
  latent_parameter_solve, domain_pipeline_sum, card_and_sum_pipeline, lemma_bundle_master,
  nontrivial_hypothesis_specialization, bounded_generalization, hypothesis_specialization, or conclusion_projection.
- Use immediate_corollary only as a risky fallback, and only when it creates a new proof obligation rather than a direct parent corollary.
- For theorem_proof mutation, prefer small same-domain changes: preserve the parent theorem shape and alter one proof checkpoint, proof style, hypothesis, or immediate corollary.
- For theorem_proof mutation, do not plan a plain adjacent numeral shift. The worker needs a nonnumeric proof change: derived local lemma, changed conclusion shape, extra conjunct, or explicit hypothesis specialization.
- For theorem_proof mutation, prefer goal-form projection or hypothesis specialization over `parent conclusion ∧ extra fact`; auxiliary-conjunct-only strengthening will be rejected as weak.
- Bounded generalization is allowed: generalize one constant, index, hypothesis, domain, or conclusion parameter while keeping the parent proof checkpoints visible.
- Do not generalize to arbitrary prime/order/topological/algebraic classes unless the parent card exposes the required reusable schema and the goal names the bounded proof obligation.
- For theorem_proof crossover, use crossover only when parent cards share a Lean domain/reusable atom or the fusion_contract can name one unified obligation. If the domains are disjoint, emit mutation instead.
- For theorem_proof crossover, a shared Lean atom is helpful but not required.
- For theorem_proof crossover, avoid broad paired claims like "infinite and invariant", "normal and topological", or side-by-side theorem conjunctions.
- For theorem_proof crossover, true fusion is best, but pipeline_composite and lemma_bundle_master are allowed.
- lemma_bundle_master means parent A/B certified lemmas or proof checkpoints become different subgoals/intermediate lemmas inside one final theorem; it is not theorem A ∧ theorem B.
- tfae_characterization is allowed only as a pilot vocabulary when existing certified parents already expose iff/implication surfaces; otherwise prefer lemma_bundle_master or mutation.
- The theorem route is accepted only when local Lean verification reports a complete proof.

Supported composite families:
- gcd_divisor_sum: a,b integers 2..50000. Meaning: n = GCD(a,b), then compute divisor_sum(n).
- divisor_sum_mod: n integer 2..2000, a integer 1..1000000. Meaning: m = divisor_sum(n), then compute a mod m.

Mutation variant rules:
- mutation_easy   One parent, a change that keeps the family certifiable: a
      parameter becomes a variable, an instance becomes the law, a bound moves.
      Use when the parent has an obvious axis to push on.
- mutation_hard   One parent, a change that forces a new reasoning step: the
      conclusion becomes a different kind of claim, a modulus or exponent moves
      enough to need a new case split, a hypothesis is weakened and the argument
      must survive it. Use when the parent's proof has a step that would break.
- mutation_silent The same mathematics in different Lean. Notation swaps
      (`a % n = r` <-> `a ≡ r [MOD n]`, `n ∣ a` <-> `a ≡ 0 [ZMOD n]`), a
      definition unfolded, an inline expression named as a hypothesis, a
      dichotomy restated as an exclusion. The child must be provably equivalent
      to its parent in BOTH directions, and that equivalence is checked.
      This is a first choice, not a demotion, and it is the only variant that
      measures something the other two cannot. easy and hard both change the
      difficulty and the surface at once, so a solver that drops on one of them
      may have dropped for either reason. A silent child holds the mathematics
      fixed by construction, so a drop can only be the surface: it isolates
      memorisation from reasoning.
      Plan one when a parent is worth that measurement -- a clean, well-known
      statement whose Lean surface has an obvious alternative -- and plan one
      when a parent has resisted several easy and hard attempts, since a
      restatement can succeed where a difficulty change did not.
      Do not use it as a cheap way to fill a slot; a silent child that cannot
      prove its equivalence is discarded, not flagged.

Fusion contract rules for crossover:
- You are not generating the final problem. You are assigning semantic roles for a slot worker.
- parent_A and parent_B must use existing parent ids exactly.
- semantic_role must be one of: invariant, object_domain, obstruction, goal_form, proof_skeleton, computation_target, parameter_family.
- fusion_mechanism must be one of the six below. Five of them had never been chosen
  across 186 crossovers because only the sixth was ever explained; 73% of those
  crossovers then measured as `mutation_like` and only 5 as true fusion. Pick the
  mechanism that fits the pair, and prefer any of the first five over the sixth.

  invariant_transplant  Parent A proves something invariant -- a modular property,
      a monotonicity, a bound preserved by an operation. Carry that invariant onto
      the object parent B is about, and state the conclusion in B's terms.
      Fails when the invariant holds for every object of that type, not just B's.

  goal_form_transplant  Parent A's conclusion has a shape -- `∃!`, `↔`, `IsLeast`,
      a classification -- and parent B has content stated in a weaker shape. Restate
      B's content in A's shape, so the proof must supply what the shape demands
      (uniqueness, both directions, minimality).
      Fails when the hypotheses already force the witness, which makes the shape
      decoration. Ask what would have to be ruled out; if nothing, do not use this.

  obstruction_as_lemma  Parent A proves something impossible -- no rational root,
      no solution mod n, an empty intersection. Use that impossibility to eliminate
      a case that parent B's argument would otherwise have to treat.
      Fails when the eliminated case was already impossible for a cheaper reason, or
      when no case is actually removed. The obstruction must bite on something that
      would otherwise be there.

  witness_exchange  Parent A constructs or determines a specific object -- a value,
      an index, a set element. Parent B asserts that something with a property
      exists. Use A's object as B's witness.
      Fails when B's existential can be closed by a trivial witness (0, 1, the empty
      set) without A. The exchange must be the only reachable witness.

  parameter_coupling  Both parents are parameterised. Bind their parameters with one
      equation so neither is free, and state a conclusion that holds only on that
      locus.
      Fails when the equation determines nothing, or leaves one parameter free. The
      coupled system must have fewer degrees of freedom than the two apart.

  sequential_composition  Parent A's output is fed to parent B's input. This is the
      weakest of the six and the one the pipeline defaults to. It is acceptable when
      the handoff is real, but it will not be judged strong, and it is the mechanism
      that produces `A solved, B solved, results combined`.
- Do not use sequential_composition for side-by-side conjunction.
- Whichever you choose: a parent whose statement is true with no hypotheses at all
  cannot be the parent that contributes. It is a fact Mathlib already supplies, and
  applying it to something the other parent built costs nothing. Check this before
  assigning roles.
- why_not_concatenation must explain why this is not just theorem A ∧ theorem B.
- For theorem_proof crossover, include fusion_goal or parent_roles that identify the pipeline_goal: which parent checkpoint/object/hypothesis appears in the other parent's formal_statement, proof_obligations, or proof_plan.
- For lemma_bundle_master, include fusion_goal or parent_roles that identify which parent checkpoint supports which subgoal of the master theorem.
- For theorem_proof crossover, side-by-side conjunction is invalid and will be stored as weak, not parent-eligible.

Quality rules:
- Prefer family_bridge when parent B changes the solving method or a major parameter. Use
  parameter_transfer only when the transferred value still makes the child harder than the
  target-family parent.
- For cross-family family_bridge, the non-target parent contribution may be semantic when it
  changes the reasoning pattern, especially for supported composite families.
- quality_target is a backward-compatible summary; operator_goal and required_checkpoints are the hard worker contract.
- Accepted-grade playbook: prefer latent_parameter_solve, domain_pipeline_sum,
  card_and_sum_pipeline, lemma_bundle_master, or nontrivial_hypothesis_specialization.
- Avoid certified-but-low-value patterns: proof_infrastructure_only helpers,
  direct_parent_corollary_only, affine_index_drift_only, cardinality_only_window,
  and lineage_complexity_without_new_role.
- For theorem crossover, a helper fact must feed a final theorem target; exact
  finset/card/prod facts alone are not accepted-grade.
- For AP/domain pipelines, changing only u(p+k), u(2p), or a window length is
  not enough; consume another domain aggregate or checkpoint.
- Allowed required_checkpoints IDs:
  reasoning_pattern, solution_skeleton, projected_params, two_step_reasoning,
  semantic_parent_contribution, family_certified, rich_factorization, rich_gcd,
  nontrivial_modular_reduction, nontrivial_mod_remainder, binomial_formula,
  arithmetic_sum_formula, numeric_answer_verified.
- Do not write checkpoints like "answer exceeds 30000", "n_terms >= 60", or
  "final divisor_sum answer verified > 1200"; those are brittle quality targets, not verifier contracts.
- Avoid trivial answers 0, 1, m-1 for modular_congruence unless there is a clear reason.
- required_params must use only these canonical keys and ranges:
  gcd: a,b integers 2..50000
  gcd_divisor_sum: a,b integers 2..50000
  units_digit: base integer 2..99, exp integer 2..5000
  divisor_sum: n integer 2..2000
  divisor_sum_mod: n integer 2..2000, a integer 1..1000000
  stars_and_bars: vars integer 2..6, sum integer 1..30
  arithmetic_series: n_terms integer 2..100, first integer 0..200, diff integer 1..100
  modular_congruence: a integer 1..1000000, m integer 2..10000

	Seed pool:
	{chr(10).join(pool_lines)}
	{policy_block}
	{novelty_block}
	{memory_block}
	{feedback_block}
	{allocation_block}

Return this compact JSON shape:
{{
  "plan_rationale": "include actual op split and any deviation from allocation hint",
  "work_items": [
    {{
      "slot": 0,
      "op_type": "survivor|mutation|crossover",
      "operator_variant": "survivor|mutation_easy|mutation_hard|mutation_silent|crossover_easy|crossover_hard",
      "parent_refs": [0],
      "target_style": "numeric_answer|theorem_proof",
      "target_family": "supported family for generated slots",
      "goal": "compact high-level worker goal",
      "constraints": ["short hard constraint"],
      "avoid": ["short anti-pattern"],
      "fusion_goal": "optional crossover fusion goal",
      "fusion_mechanism": "crossover slots only: invariant_transplant|goal_form_transplant|obstruction_as_lemma|witness_exchange|parameter_coupling|sequential_composition",
      "why_not_concatenation": "crossover slots only: one sentence on why this is not theorem A and theorem B stated together",
      "parent_roles": {{"0": "object/domain source", "1": "proof/goal source"}},
      "memory_delta_contract": {{
        "similar_card_ids": ["accepted_or_run_local_id"],
        "must_not_repeat": ["matched target semantics to avoid"],
        "required_distinguishing_delta": "specific new object, target role, proof obligation, or consumed checkpoint",
        "allowed_overlap": "same family/domain allowed only if a new role is explicit",
        "novelty_rationale": "why this slot should not be a near-duplicate"
      }}
    }}
  ]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _planner_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["plan_rationale", "work_items"],
        "properties": {
            "plan_rationale": {
                "type": "string",
                "description": "Short rationale for the slot allocation.",
            },
            "work_items": {
                "type": "array",
                "description": "Exactly one minimal OperatorCard per pool slot.",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": [
                        "slot",
                        "op_type",
                        "parent_refs",
                        "target_style",
                        "operator_variant",
                        "goal",
                    ],
                    "properties": {
                        "slot": {"type": "integer", "description": "Contiguous pool slot index."},
                        "op_type": {
                            "type": "string",
                            "enum": ["survivor", "mutation", "crossover"],
                            "description": "survivor preserves one parent; mutation uses one parent; crossover uses two distinct parents.",
                        },
                        "operator_variant": {
                            "type": "string",
                            "enum": [
                                "survivor",
                                "mutation_easy",
                                "mutation_hard",
                                "crossover_easy",
                                "crossover_hard",
                            ],
                            "description": "Small execution mode for the slot worker.",
                        },
                        "parent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional exact parent IDs. Prefer parent_refs.",
                        },
                        "parent_refs": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Zero-based pool indices.",
                        },
                        "target_style": {
                            "type": "string",
                            "enum": ["numeric_answer", "theorem_proof"],
                            "description": "Preserve parent style.",
                        },
                        "target_family": {
                            "type": "string",
                            "enum": SUPPORTED_FAMILY_NAMES + ["theorem_proof", ""],
                            "description": "Supported numeric family, theorem_proof, or empty for survivor.",
                        },
                        "goal": {
                            "type": "string",
                            "description": (
                                "Compact high-level execution goal. Prefix theorem goals with paper_frontier or "
                                "scaffold_to_frontier. Start theorem generated goals with an "
                                "accepted-grade playbook mode when possible: latent_parameter_solve, "
                                "domain_pipeline_sum, card_and_sum_pipeline, lemma_bundle_master, or "
                                "nontrivial_hypothesis_specialization. Do not include detailed params or proof obligations. "
                                "A scaffold/local corollary must name the downstream final theorem role it feeds. "
                                "For theorem crossover, prefer a short mode prefix: pipeline_composite, "
                                "lemma_bundle_master, or tfae_characterization pilot."
                            ),
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Short hard constraints the worker must obey. Include the accepted-grade success condition, "
                                "for example helper_fact_feeds_final_theorem, consume_card_and_sum, "
                                "change_final_proof_obligation, parent_checkpoint_visible_in_formal_statement, "
                                "or scaffold_consumed_by_final_theorem."
                            ),
                        },
                        "avoid": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Short anti-patterns. Prefer exact flag names from Planner NoGoPolicyPack when relevant. "
                                "Use verifier flag names when known: proof_infrastructure_only, direct_parent_corollary_only, "
                                "affine_index_drift_only, cardinality_only_window, lineage_complexity_without_new_role, "
                                "side_by_side_conjunction, artificial_bridge_to_existing_pipeline, "
                                "numeric_bound_fitting_crossover, or unused_checkpoint."
                            ),
                        },
                        "fusion_goal": {
                            "type": "string",
                            "description": (
                                "Optional crossover-level fusion goal. For accepted-grade crossover, state exactly which "
                                "parent checkpoint/object/hypothesis is consumed by the final theorem target or proof_plan. "
                                "For lemma_bundle_master, name how parent checkpoints become distinct subgoals or "
                                "intermediate lemmas inside one final theorem. For TFAE, use only when parents expose "
                                "certified iff/implication surfaces."
                            ),
                        },
                        # The field the prompt spends forty lines describing and
                        # the schema never had. The planner could not name a
                        # mechanism however clearly it had chosen one:
                        # `parent_roles` already showed it pairing `invariant`
                        # with `proof_skeleton`, which is `invariant_transplant`
                        # in all but name. Every contract therefore arrived with
                        # the field empty and was defaulted to
                        # `sequential_composition` -- 150 of 186 crossovers, with
                        # the other five mechanisms never once attempted. The
                        # default was the symptom; this absence was the cause.
                        "fusion_mechanism": {
                            "type": "string",
                            "enum": sorted(FUSION_MECHANISMS) + [""],
                            "description": (
                                "How the two parents meet. Choose the one that fits the pair; prefer "
                                "anything over sequential_composition, which is the weakest and the one "
                                "the pipeline collapses to. Empty only when no mechanism applies, in "
                                "which case the slot should probably not be a crossover."
                            ),
                        },
                        "why_not_concatenation": {
                            "type": "string",
                            "description": (
                                "Why this is not theorem A and theorem B stated together. One sentence."
                            ),
                        },
                        "parent_roles": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": (
                                "Optional parent id/ref to semantic role, e.g. checkpoint_provider, "
                                "intermediate_lemma_source, goal_wrapper, subgoal_source, or implication_edge_source. "
                                "Avoid vague roles such as inspiration."
                            ),
                        },
                        "memory_delta_contract": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": (
                                "Compact novelty contract for generated slots. Include similar_card_ids, "
                                "must_not_repeat, required_distinguishing_delta, allowed_overlap, and "
                                "novelty_rationale. Use NoveltyMemoryPack analogues as evidence, not parents."
                            ),
                            "properties": {
                                "similar_card_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Problem ids from the closest accepted or run-local analogues. These ids are "
                                        "evidence for novelty planning only; do not treat them as parents or sources."
                                    ),
                                },
                                "must_not_repeat": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Specific target semantics, formal surfaces, parameter families, or shallow "
                                        "proof-plan moves from similar_card_ids that this slot must avoid."
                                    ),
                                },
                                "required_distinguishing_delta": {
                                    "type": "string",
                                    "description": (
                                        "Concrete delta that makes this candidate materially new, such as a new "
                                        "defining object, target quantity, theorem role, consumed checkpoint, or "
                                        "proof obligation. Avoid vague phrasing like make it different."
                                    ),
                                },
                                "allowed_overlap": {
                                    "type": "string",
                                    "description": (
                                        "What may remain shared with the analogues, for example the broad family or "
                                        "domain, and why that overlap is not enough to make the slot a near-duplicate."
                                    ),
                                },
                                "novelty_rationale": {
                                    "type": "string",
                                    "description": (
                                        "Brief explanation of why the required delta avoids same object, same target "
                                        "semantics, cosmetic paraphrase, helper-only theorem, or parameter-only drift."
                                    ),
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _planner_response_format() -> Dict[str, Any]:
    return _schema_response_format("pool_generation_plan", _planner_response_schema())


def _prompt_text(value: Any, *, limit: int = 4000) -> str:
    if value is None:
        return "not_available"
    text = str(value).strip()
    if not text:
        return "not_available"
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _verification_code_kind(value: Any) -> str:
    text = _prompt_text(value, limit=1000)
    if text == "not_available":
        return "not_available"
    lowered = text.lower()
    if (
        "import mathlib" in lowered
        or ":= by" in lowered
        or lowered.startswith("theorem ")
        or lowered.startswith("lemma ")
    ):
        return "lean"
    if "def verify" in lowered or lowered.startswith("def ") or "return " in lowered:
        return "python"
    return "unknown"


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _metadata_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _lean_has_complete_by_body(source: Any) -> bool:
    """Return true only when `:= by` has a non-empty proof body after it."""
    text = str(source or "")
    if re.search(r"^\s*axiom\s+", text, flags=re.MULTILINE):
        return False
    match = re.search(r":=\s*by\b", text)
    if not match:
        return False
    body = text[match.end() :].strip()
    if not body:
        return False
    meaningful_lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    if not meaningful_lines:
        return False
    body_text = "\n".join(meaningful_lines).lower()
    if any(marker in body_text for marker in ("sorry", "admit", "placeholder", "not provided")):
        return False
    return True


def _parent_proof_context(problem: CertificationInput) -> Dict[str, Any]:
    metadata = problem.metadata or {}
    verification_code = metadata.get("verification_code")
    lean_code = metadata.get("lean_code")
    if not lean_code and (metadata.get("lean_header") or metadata.get("formal_statement")):
        lean_code = "\n".join(
            str(part).strip()
            for part in (metadata.get("lean_header"), metadata.get("formal_statement"))
            if str(part or "").strip()
        )
    lean_text = _prompt_text(lean_code)
    proof_body_available = lean_text != "not_available" and _lean_has_complete_by_body(lean_text)
    return {
        "solution": _prompt_text(metadata.get("solution")),
        "verification_code": {
            "kind": _verification_code_kind(verification_code),
            "content": _prompt_text(verification_code),
        },
        "lean_code": lean_text,
        "proof_body_available": proof_body_available,
        "lean_statement_only": lean_text != "not_available" and not proof_body_available,
        "usable_proof_atoms": _reusable_atoms(problem) if proof_body_available else [],
        "solution_skeleton": _dict_or_empty(metadata.get("solution_skeleton")),
        "quality_evidence": _dict_or_empty(metadata.get("quality_evidence")),
    }


def _theorem_decomposition_card(problem: CertificationInput) -> Dict[str, Any]:
    """Small theorem summary for planning; not a proof checker."""
    metadata = problem.metadata or {}
    proof_context = _parent_proof_context(problem)
    lean_text = str(proof_context.get("lean_code") or "")
    formal_text = str(metadata.get("formal_statement") or lean_text or "").replace("\n", " ")
    theorem_head = formal_text.split(":=", 1)[0]
    depth = 0
    conclusion_start: Optional[int] = None
    for idx, char in enumerate(theorem_head):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0 and conclusion_start is None:
            conclusion_start = idx + 1
    conclusion = theorem_head[conclusion_start:].strip() if conclusion_start else ""
    binder_matches = __import__("re").findall(
        r"[\(\{\[]([^()\{\}\[\]]{1,140})[\)\}\]]",
        theorem_head,
    )
    reusable_atoms = _reusable_atoms(problem)
    checkpoints = list(reusable_atoms)
    if conclusion:
        checkpoints.insert(0, f"prove conclusion: {_prompt_text(conclusion, limit=180)}")
    return {
        "hypotheses": [_prompt_text(item, limit=180) for item in binder_matches[:6]],
        "main_conclusion": _prompt_text(conclusion, limit=300),
        "proof_checkpoints": checkpoints[:6],
        "proof_body_available": bool(proof_context.get("proof_body_available")),
        "lean_statement_only": bool(proof_context.get("lean_statement_only")),
        "reusable_lean_atoms": [
            atom
            for atom in ["Group", "Polynomial", "IsClosed", "Finite", "Subgroup", "Ideal"]
            if atom.lower() in lean_text.lower()
        ][:6],
        "fragile_symbols": [
            symbol
            for symbol in ["rootSet", "Sylow", "Quaternion", "IsCoprime", "derivative"]
            if symbol.lower() in f"{problem.statement} {lean_text}".lower()
        ][:6],
        "allowed_strengthenings": [
            "add one local lemma/checkpoint that preserves the parent theorem style",
            "strengthen the proof obligation only when the Lean statement also contains it",
        ],
        "forbidden_claims": [
            "do not add prose-only claims absent from formal_statement/lean_code",
            "do not switch theorem/proof parents into numeric-answer exercises",
        ],
    }


def _parent_proof_skeleton(problem: Any, *, limit: int = 12) -> Dict[str, Any]:
    """The parent's proof as claims and dependencies, not as tactic text.

    The generator was given each parent's Lean in full and told that the verifier
    would parse it into a dependency graph -- the scoring rule, without the
    structure the rule scores. It then had to rebuild that structure from raw
    text for both parents before it could see where they might meet, and the
    cheapest move when you cannot see a meeting point is to finish one parent and
    start the other. 150 of 186 crossovers were sequential composition.

    What is carried is each `have`'s claim and what it rests on. The tactic body
    is dropped: to notice that one parent establishes `n % 3 = 0 ∨ ...` and the
    other needs exactly that, the statement is the useful part and its proof is
    bulk.
    """
    try:
        from src.certification.novelty import proof_graph
    except Exception:  # pragma: no cover
        return {}
    lean = str((_parent_proof_context(problem) or {}).get("lean_code") or "")
    if not lean.strip():
        return {}
    try:
        nodes, root = proof_graph(lean)
    except Exception:  # pragma: no cover
        return {}
    if not nodes:
        return {}
    return {
        "steps": [
            {
                "name": str(node.get("name") or ""),
                "claim": _prompt_text(node.get("type"), limit=220),
                "uses": list(node.get("uses") or [])[:6],
            }
            for node in nodes[:limit]
        ],
        "step_count": len(nodes),
        "closing_term": _prompt_text(root, limit=220),
        "note": (
            "Named intermediate claims of this parent's proof and what each rests "
            "on. A crossover meets where one parent's claim is what the other "
            "needs; look here before settling for a pipeline."
        ),
    }


def _parent_context_card(problem: CertificationInput) -> Dict[str, Any]:
    problem_style = _problem_style(problem)
    route = _certification_route_for_style(problem_style)
    allowed_target_styles = (
        ["theorem_proof"] if problem_style == "theorem_proof" else ["numeric_answer"]
    )
    return {
        "id": problem.id,
        "problem_style": problem_style,
        "certification_route": route,
        "family": _problem_family(problem) or "unsupported",
        "statement_preview": _prompt_text(problem.statement, limit=500),
        "answer_preview": _prompt_text(problem.answer, limit=250),
        # The parent's Lean verbatim, not a preview. A child's relation to its
        # parent is decidable in Lean — prove the implication either way — but
        # only if the parent's statement survives in a form Lean can read. The
        # released corpus recorded previews instead, so 286 of 330 parent
        # references came back as fragments like `b = 9` and the comparison
        # could not be attempted at all.
        "formal_statement": str((problem.metadata or {}).get("formal_statement") or ""),
        "lean_header": str((problem.metadata or {}).get("lean_header") or ""),
        "proof_context": _parent_proof_context(problem),
        "theorem_decomposition": (
            _theorem_decomposition_card(problem) if problem_style == "theorem_proof" else {}
        ),
        "proof_skeleton": (
            _parent_proof_skeleton(problem) if problem_style == "theorem_proof" else {}
        ),
        "reusable_atoms": _reusable_atoms(problem),
        "allowed_target_styles": allowed_target_styles,
        "allowed_target_families": SUPPORTED_FAMILY_NAMES if problem_style == "numeric_answer" else [],
        "forbidden_transfers": (
            [
                "do not turn a theorem/proof parent into a numeric-answer child",
                "do not use theorem content as decorative prose only",
            ]
            if problem_style == "theorem_proof"
            else ["do not claim theorem-level proof obligations for template_numeric children"]
        ),
        "template_fit": _template_fit(problem),
    }


def _parent_context_cards(parents: List[CertificationInput]) -> List[Dict[str, Any]]:
    return [_parent_context_card(parent) for parent in parents]


def _planner_parent_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Compact parent view for the orchestrator planner."""
    proof_context = dict(card.get("proof_context") or {})
    return {
        "id": card.get("id"),
        "problem_style": card.get("problem_style"),
        "certification_route": card.get("certification_route"),
        "family": card.get("family"),
        "statement_preview": card.get("statement_preview"),
        "answer_preview": card.get("answer_preview"),
        "theorem_decomposition": card.get("theorem_decomposition", {}),
        "reusable_atoms": card.get("reusable_atoms", []),
        "allowed_target_styles": card.get("allowed_target_styles", []),
        "allowed_target_families": card.get("allowed_target_families", []),
        "proof_artifacts_available": {
            "solution": proof_context.get("solution") != "not_available",
            "verification_code_kind": (proof_context.get("verification_code") or {}).get("kind"),
            "lean_code": proof_context.get("lean_code") != "not_available",
            "proof_body_available": bool(proof_context.get("proof_body_available")),
            "lean_statement_only": bool(proof_context.get("lean_statement_only")),
        },
        "template_fit": card.get("template_fit", {}),
    }


def _target_style_for_item(item: Dict[str, Any], parents: List[CertificationInput]) -> str:
    if any(_problem_style(parent) == "theorem_proof" for parent in parents):
        return "theorem_proof"
    explicit = str(item.get("target_style") or "")
    if explicit in PROBLEM_STYLES:
        return explicit
    return "numeric_answer"


def llm_plan_generation(
    pool: List[CertificationInput],
    *,
    pool_size: int,
    survivor_count: int,
    crossover_count: int,
    generation_model: Optional[str],
    generation_temperature: Optional[float],
    generation_feedback: Optional[Dict[str, Any]] = None,
    op_type_allocation_hint: Optional[Dict[str, Any]] = None,
    planner_memory: Optional[Dict[str, Any]] = None,
    novelty_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = default_generation_config(model=generation_model, temperature=generation_temperature)
    messages = _planner_messages(
        pool,
        pool_size=pool_size,
        survivor_count=survivor_count,
        crossover_count=crossover_count,
        generation_feedback=generation_feedback,
        op_type_allocation_hint=op_type_allocation_hint,
        planner_memory=planner_memory,
        novelty_memory=novelty_memory,
    )
    content = _chat_completion_text_sync(
        model=orchestrator_config(config).model,
        messages=messages,
        temperature=0.1,
        response_format=_planner_response_format(),
    )
    raw = _parse_json_object(content)
    return {"planner_source": "orchestrator_llm", **raw}


def _theorem_response_format() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "status",
            "statement",
            "formal_statement",
            "lean_code",
            "proof_plan",
            "parent_usage",
            "reason",
        ],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["generated", "cannot_execute", "contract_failed"],
                "description": "generated only when a complete Lean artifact is returned.",
            },
            "contract_status": {
                "type": "string",
                "enum": ["generated", "cannot_execute", "contract_failed"],
                "description": "Backward-compatible alias for status.",
            },
            "failure_reason": {
                "type": "string",
                "description": "Backward-compatible failure reason.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation of the artifact or why it cannot be executed.",
            },
            "statement": {
                "type": "string",
                "description": "Natural-language theorem statement covered by the Lean theorem.",
            },
            "statement_chunks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional verifier hints. The verifier, not the worker, is authoritative.",
            },
            "selected_parent_checkpoints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional verifier hints.",
            },
            "allowed_statement_delta": {
                "type": "string",
                "enum": [
                    "same_statement",
                    "specialize_hypothesis",
                    "project_conclusion",
                    "immediate_corollary",
                ],
                "description": (
                    "Only allowed theorem-level change from the selected parent checkpoints. "
                    "If none fits, return contract_status=contract_failed."
                ),
            },
            "formal_statement": {
                "type": "string",
                "description": "Exact Lean theorem/lemma statement covering every public statement_chunk.",
            },
            "lean_header": {
                "type": "string",
                "description": "Use the canonical Mathlib header only; do not add specific Mathlib module imports.",
            },
            "lean_code": {
                "type": "string",
                "description": "Full Lean code with complete proof. No sorry.",
            },
            "proof_surface": {
                "type": "string",
                "enum": [
                    "parent_rewrite",
                    "simp_only",
                    "constructor_cases",
                    "exact_existing",
                    "direct_group_calc",
                    "direct_proof",
                ],
                "description": (
                    "Optional legacy diagnostic only. The verifier infers proof shape from "
                    "lean_code and does not require this field."
                ),
            },
            "proof_plan": {
                "type": "string",
                "description": (
                    "Short proof plan matching the Lean proof body. For lemma_bundle_master, name the "
                    "parent-derived subgoals or intermediate lemmas in proof order without adding new fields."
                ),
            },
            "proof_obligations": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional legacy hints. If provided, list only obligations directly discharged in lean_code, "
                    "not aspirational prose or hoped-for library lemmas."
                ),
            },
            "difficulty_label": {"type": "string", "enum": ["easy", "medium", "hard", "superhard"]},
            "harder_reason": {
                "type": "string",
                "description": "Explain the small theorem-style improvement without claiming unsupported generality.",
            },
            "parent_contribution_evidence": {
                "type": "object",
                "description": (
                    "Backward-compatible parent usage map. For lemma_bundle_master, each note should say "
                    "which subgoal or intermediate lemma the parent supports."
                ),
                "additionalProperties": {"type": "string"},
            },
            "parent_usage": {
                "type": "object",
                "description": (
                    "Optional parent id to short usage note. For crossover, ground each note in formal_statement, "
                    "proof_plan, or lean_code; prose-only inspiration is insufficient."
                ),
                "additionalProperties": {"type": "string"},
            },
            "unified_obligation": {
                "type": "string",
                "description": (
                    "For crossover, one formal proof obligation affected by both parents. "
                    "For mutation, short selected checkpoint. Empty only when contract_status is not generated."
                ),
            },
            "why_not_conjunction": {
                "type": "string",
                "description": (
                    "For crossover, explain why this is not parent A theorem AND parent B theorem side by side. "
                    "If no such explanation exists, return contract_status=contract_failed."
                ),
            },
            "patch_target_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "On retry, fields intentionally changed to satisfy retry_feedback.",
            },
            "must_not_change_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "On retry, fields preserved from the prior valid contract.",
            },
            "patch_applied": {
                "type": "string",
                "description": "Short field-level patch summary; empty on first generated attempt.",
            },
        },
    }
    return _schema_response_format("theorem_generated_problem", schema)


# Deprecated Mathlib big-operator binder syntax (`∑ x in s, …`) was replaced
# by membership notation (`∑ x ∈ s, …`); models trained on older corpora still
# emit the former, which fails to parse on the pinned toolchain (observed as
# "unexpected token 'in'" in 11 certification failures). Rewrite it
# deterministically before any Lean invocation.
_DEPRECATED_BIGOP_IN_RE = re.compile(
    r"([∑∏⋃⋂⨆⨅])(\s*)([^,\n∈]*?)(\s)in\s"
)


def _prelint_lean_syntax(text: str) -> str:
    """Deterministic fixes for known-deprecated Lean/Mathlib surface syntax."""
    if not text:
        return text
    return _DEPRECATED_BIGOP_IN_RE.sub(r"\1\2\3\4∈ ", text)


def _normalize_theorem_lean_code(
    *,
    lean_code: str,
    lean_header: str,
    formal_statement: str,
) -> tuple[str, str]:
    """Use one project-wide Mathlib header and keep model output as theorem body.

    Generated theorem rows have occasionally emitted semicolon-separated headers
    or specific Mathlib module imports that are not present in the local project
    cache. The verifier runs inside a Lake project, so `import Mathlib` is the
    stable surface; generated code should not choose cache-sensitive imports.
    """
    source = _prelint_lean_syntax((lean_code or "").strip())
    if not source and formal_statement:
        source = f"{lean_header}\n\n{formal_statement}".strip()
    source = re.sub(r";\s*(?=(import|set_option|open|theorem|lemma)\b)", "\n", source)
    body_lines: List[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("set_option "):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip() or formal_statement.strip()
    return THEOREM_CANONICAL_HEADER, f"{THEOREM_CANONICAL_HEADER}\n\n{body}".rstrip()


def _theorem_alignment_response_format() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "aligned",
            "verdict",
            "supported_claims",
            "missing_claims",
            "unsupported_claims",
            "field_patch_instructions",
            "rationale",
        ],
        "properties": {
            "aligned": {"type": "boolean"},
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "supported_claims": {"type": "array", "items": {"type": "string"}},
            "missing_claims": {"type": "array", "items": {"type": "string"}},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "field_patch_instructions": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
    }
    return _schema_response_format("theorem_alignment_verdict", schema)


def _build_theorem_alignment_messages(
    generated: TheoremGeneratedProblem,
    item: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are an adversarial verifier for theorem problem generation. "
        "Your only job is to decide whether the natural-language statement is fully covered "
        "by the formal_statement and lean_code. You run only after Lean verification is complete. "
        "Return JSON only."
    )
    user = f"""
Static contract:
- PASS only if every mathematical claim in statement and statement_chunks is represented in formal_statement/lean_code.
- FAIL if statement contains stronger prose-only claims, examples, equivalences, topology/algebra properties, or parent references not present in Lean.
- For crossover OperatorCards, FAIL independent conjunctions that merely prove parent A's theorem and parent B's theorem side by side; the formal theorem must have one unified obligation.
- Do not judge whether the Lean proof compiles; this verifier only runs after local Lean completed.
- Be strict but small: report missing/unsupported claims and field-level patch instructions.

Dynamic candidate:
statement:
{generated.statement[:3000]}

statement_chunks:
{json.dumps(generated.statement_chunks, ensure_ascii=False)[:2000]}

formal_statement:
{generated.formal_statement[:3000]}

lean_code:
{generated.lean_code[:5000]}

operator_card:
{json.dumps(_operator_card(item), ensure_ascii=False)[:2500]}

Return JSON with aligned=false when the statement says more than the Lean theorem proves.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def verify_theorem_alignment(
    generated: TheoremGeneratedProblem,
    *,
    item: Dict[str, Any],
    config: GenerationConfig,
) -> TheoremAlignmentResult:
    messages = _build_theorem_alignment_messages(generated, item)
    verifier_config = verification_config(config)
    content = await _chat_completion_text_async(
        model=verifier_config.model,
        messages=messages,
        temperature=0,
        response_format=_theorem_alignment_response_format(),
    )
    raw = _parse_json_object(content)
    return TheoremAlignmentResult.model_validate(raw)


async def _theorem_premise_pack_payload(
    *,
    statement: str,
    formal_statement: str,
    diagnostics: str = "",
    leansearch_enabled: bool = True,
    leansearch_limit: int = DEFAULT_LEANSEARCH_LIMIT,
    phase: str = "initial",
    query_counter: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Return compact prompt/digest payload for theorem LeanSearch hints."""
    if not leansearch_enabled:
        return {"prompt_block": "", "digest": {"source": "disabled"}}
    if phase == "reflection" and not should_retrieve_for_diagnostics(diagnostics):
        return {"prompt_block": "", "digest": {"source": "skipped", "reason": "diagnostics_not_retrieval_trigger"}}
    counter = query_counter if query_counter is not None else {"count": 0}
    max_queries = max_queries_per_problem()
    if max_queries <= 0 or counter.get("count", 0) >= max_queries:
        return {"prompt_block": "", "digest": {"source": "skipped", "reason": "query_budget_exhausted"}}
    counter["count"] = int(counter.get("count", 0)) + 1
    query = (
        build_diagnostic_query(statement, formal_statement, diagnostics)
        if phase == "reflection"
        else build_statement_query(statement, formal_statement)
    )
    pack = await retrieve_premise_pack(
        query,
        limit=leansearch_limit,
        repo_root=Path.cwd(),
        disabled=not leansearch_enabled,
        phase=phase,
    )
    return {"prompt_block": format_premise_pack(pack), "digest": pack.digest()}


def _build_theorem_generation_messages(
    parent: CertificationInput,
    *,
    premise_pack_block: str = "",
) -> List[Dict[str, str]]:
    metadata = parent.metadata or {}
    operator_card = dict(metadata.get("operator_card") or {})
    memory_delta_contract = (
        dict(operator_card.get("memory_delta_contract"))
        if isinstance(operator_card.get("memory_delta_contract"), dict)
        else (
            dict(metadata.get("memory_delta_contract"))
            if isinstance(metadata.get("memory_delta_contract"), dict)
            else {}
        )
    )
    parent_cards = list(metadata.get("parent_context_cards") or metadata.get("parents") or [])
    retry_feedback = str(metadata.get("retry_feedback") or "")
    attempt_history = list(metadata.get("attempt_history") or [])
    recent_flags: List[str] = []
    for attempt in attempt_history[-4:]:
        if isinstance(attempt, dict):
            recent_flags.extend(str(flag) for flag in attempt.get("quality_flags") or [])
    recent_flags.extend(str(flag) for flag in operator_card.get("avoid") or [])
    recent_flags.extend(str(flag) for flag in operator_card.get("avoid_signatures") or [])
    # Only this slot's layer: a mutation cannot fail at fusing two parents, and
    # carrying crossover advice to it is how a memory pool turns into noise.
    slot_lessons = lessons_for_slot(
        str(operator_card.get("op_type") or metadata.get("op_type") or "mutation"),
        observed_flags=recent_flags,
    )
    system = (
        "You are a theorem-style Lean problem worker. Execute one OperatorCard against "
        "the provided ParentContextCards. Return JSON only."
    )
    # A silent mutation is the one variant that must not obey the rest of the
    # mutation contract. Those rules exist to stop a child restating its parent
    # — "preserve the domain but change the final proof obligation", "do not end
    # with the same named lemma the parent's proof ends with" — and a silent
    # mutation restates the parent on purpose, in different Lean, with the
    # mathematics held fixed. Left in force, the two instructions contradict and
    # the worker satisfies neither.
    silent_block = ""
    # The rules below ask for the two equivalence proofs, and the "minimal
    # artifact only" line enumerates the fields to return and used to omit them.
    # The worker obeyed the enumeration: both came back empty on every silent
    # slot, and a row whose equivalence the tactic ladder could not close was
    # discarded even though the worker could have proved it. The enumeration has
    # to name them, the same way the planner's variant list had to name
    # mutation_silent before the planner would ever choose it.
    silent_fields = ""
    silent_story_block = ""
    if str(operator_card.get("operator_variant") or metadata.get("operator_variant") or "") == "mutation_silent":
        silent_fields = ", equivalence_forward, equivalence_backward"
        silent_story_block = """
  - THIS SLOT IS SILENT, so the statement is where most of the re-encoding
    should happen. A concrete setting -- ages, coins, a shop's stock, distances,
    a sequence of measurements -- is the purest surface change there is: it
    leaves the mathematics untouched and moves everything a memorising solver
    keys on. `3 ∣ 2^(2n+1)+1` becomes a question about a lamp that toggles on
    odd steps; a gcd/lcm pair becomes two gear wheels meeting again. Reach for
    one whenever the mathematics genuinely fits.
  - Invent freely. There is exactly one story problem among the hundred seeds,
    so there is no house catalogue to copy -- do not settle on ages and coins
    because they were named here.
  - The story must not add or drop a single condition. If the setting needs a
    quantity to be positive and the theorem does not say so, the story is wrong,
    not the theorem.
  - A story alone is not enough for this slot: the Lean must move too. Renaming
    binders after the story's characters is invisible downstream -- the corpus
    compares statements with binder names normalised away, so `(father son : ℕ)`
    and `(a b : ℕ)` are the same statement and the row is dropped as a
    duplicate. Restructure: fold a hypothesis into the goal, name an
    intermediate quantity, swap a notation, eliminate a variable the story makes
    redundant."""

        from src.certification.silent import worker_rules

        silent_block = (
            "\nTHIS SLOT IS A SILENT MUTATION. The following rules replace the "
            "mutation rules above wherever they conflict:\n"
            "- Ignore \"change the final proof obligation\", \"avoid direct parent "
            "corollaries\", and the MUTATION proof-shape contract. Closing on the "
            "parent's lemma is expected here.\n"
            "- Ignore the slot lessons below about restatements, paraphrases and "
            "wrappers. They are written against variation, and this slot is not "
            "varying anything.\n"
            + worker_rules()
        )
    user = f"""
Hard rules:{silent_block}
- Execute the OperatorCard as a theorem-style Lean worker. Do not create a numeric-answer exercise from theorem/proof parents.
- Return the minimal artifact only: status, statement, formal_statement, lean_code, proof_plan, parent_usage, reason{silent_fields}.
- If the OperatorCard cannot be executed, return status="cannot_execute" and explain the blocked field in reason.
- theorem_proof_surfaces is an advisory style preference only. NEVER return status="cannot_execute" because your intended proof shape is missing from that list; write the correct complete proof in whatever shape actually closes the goal.
- formal_statement must be SELF-CONTAINED: it may reference only Mathlib/Aesop names. Never make the statement depend on helper def/lemma declarations you author in lean_code — the statement alone is the released, trusted object and must elaborate by itself. Encode constructed objects inside the statement with explicit binders (e.g. `∃ f : ℝ → Quaternion ℝ, ...`) instead of naming a private def. Helper lemmas are still fine inside the PROOF body.
- lean_code must include the canonical header and a complete theorem/lemma proof body.
- `statement` is how a human meets this problem, and it is the field a reader
  judges the corpus by. Write it the way the source benchmarks write theirs, not
  as a transcription of the Lean:
  - Never put a Lean name in it. `Int.gcd b x` is "the greatest common divisor
    of $b$ and $x$"; `Nat.choose n k` is "$\\binom{{n}}{{k}}$". A released
    statement mentioning `Set.univ`, `IsExtrOn` or a hypothesis label has been
    transcribed rather than written.
  - Mathematical notation in `$...$`, as the benchmarks do -- `$5^{{30}}$`,
    `$\\log_3 27$`, `$(x+3)$`.
  - Short. One or two sentences, and often a question: "What is the remainder
    when 2003 is divided by 11? Show that it is 1." A long universally
    quantified English clause is not the house style.
  - Follow the PARENT's register. If the parent is a terse competition question,
    write a terse competition question; if it is a textbook exercise, write one.
    The child should read as though it came from the same source as its parent.{silent_story_block}
- Use this exact Lean header, with one command per line:
{THEOREM_CANONICAL_HEADER}
- Do not add specific Mathlib module imports; the local verifier runs in a Lake project with `import Mathlib`.
- Do not use sorry.
- Do not cite Mathlib lemma names unless they appear in ParentContextCards or are standard tactics. Prefer direct simp/rw/constructor/exact/group/ring/linarith only when the assumptions visibly support them.
- LeanSearch PremisePack candidates are validated local names but still hints. Use them only when they fit the fixed theorem.
- Parent proof artifacts are hints, not authority. If copied, adapt names and verify consistency.
- If retry_feedback is present, fix exactly the named fields first.
- Do not add prose-only claims to statement. The statement must be covered by formal_statement and lean_code.
- Public statement hygiene: statement must be a mathematical theorem statement only, not an agent workflow note.
- Do not put workflow or lineage words in statement: checkpoint, parent, certified, generated, mutation, crossover, pipeline, operator, proof obligation, Lean, formal.
- Put parent usage, proof process, and generation rationale only in proof_plan, parent_usage, or reason.
- For mutation, preserve the parent theorem's domain and conclusion shape unless OperatorCard explicitly asks for an immediate corollary or bounded_generalization.
- Bounded generalization means changing one constant, index, hypothesis, domain, or conclusion parameter while keeping the parent proof checkpoints visible in formal_statement/proof_plan/lean_code.
- Do not jump to arbitrary broad classes. If generalization is requested, keep it local to reusable atoms exposed by ParentContextCards.
- For crossover, true fusion is best. A pipeline composite is acceptable when one parent supplies a checkpoint/object/hypothesis that appears inside the other parent's formal_statement, proof_plan, or proof obligations.
- For crossover, lemma_bundle_master is acceptable: use multiple parent checkpoints as distinct subgoals/intermediate lemmas inside one final theorem.
- For lemma_bundle_master, keep the same minimal JSON shape. Put bundle evidence in proof_plan and parent_usage; do not invent extra schema fields.
- For lemma_bundle_master, the final theorem must not be `theoremA ∧ theoremB`; parent_usage must name which parent supports which subgoal.
- Accepted-grade theorem outputs should be final theorem targets, not helper-only facts. Avoid exact finset/card/prod lemmas as the final result unless they feed a larger theorem.
- If the OperatorCard is only a local corollary, same-role cyclic transport, topology direct consequence, or fixed finite aggregate computation, return status="cannot_execute" unless the card explicitly says how that scaffold is consumed by a new final theorem role.
- If the OperatorCard says paper_frontier, do not output a scaffold-only theorem. If it says scaffold_to_frontier, the final statement must already include the frontier role, not merely the intermediate lemma.
- Avoid direct parent corollaries and same-domain affine drift. For AP/domain pipelines, changing only u(p+k), u(2p), or a window length is not enough unless another aggregate/checkpoint is consumed in the proof.
- Preserve the parent domain when useful, but change the final proof obligation. A valid child should be closer to latent_parameter_solve, domain_pipeline_sum, card_and_sum_pipeline, lemma_bundle_master, or nontrivial_hypothesis_specialization than to a direct corollary.
- TFAE/characterization is pilot-only: use it only if the OperatorCard explicitly requests it and parent cards expose iff/implication surfaces.
- For crossover, do not join independent parent theorems with `and`/conjunction. Side-by-side conjunction is stored as weak and will not become a parent.
- parent_usage should briefly name where each parent affected the artifact. The verifier will judge whether that evidence is sufficient.
Proof-shape contract (checked mechanically against lean_code, not against your description):
- The verifier parses your proof into a dependency graph: each `have` is a node, and an edge runs from one node to any later node whose tactic text names it. The closing term is the root.
- CROSSOVER: at least one `have` must take input from BOTH parents -- transitively, through other `have`s. A proof where each parent's material reaches only the closing term is scored as parallel, not fused, and is rejected for any slot above `easy`. Discharging one parent's obligation and using it merely to show a coefficient is nonzero, a set is nonempty, or an `if` condition holds does NOT count: that is the closing term meeting them, not a fusion.
- MUTATION: the proof must not end with the same named lemma the parent's proof ends with. If the parent closes with `exact h.foo_bar hx` and you close with `exact h'.foo_bar hy`, you have restated the parent with more notation, however much you added above that line.
- Write the load-bearing steps as named `have`s so the dependency is visible. A proof closed in one line is scored as having no coupling at all.

OperatorCard:
{json.dumps(operator_card, ensure_ascii=False, indent=2)[:6000]}

{format_lessons(slot_lessons, title="Lessons for this slot type")}

MemoryDeltaContract:
{json.dumps(memory_delta_contract, ensure_ascii=False, indent=2) if memory_delta_contract else "not_available"}
- This is compact novelty context, not a raw memory case pack and not a parent source.
- Change the final theorem obligation or proof role from matched surfaces; avoid helper-only theorem, AP affine drift, and Lean-surface paraphrase.
- If the required_distinguishing_delta cannot be satisfied, return status="cannot_execute" rather than a near-copy.

ParentContextCards:
{json.dumps(parent_cards, ensure_ascii=False, indent=2)[:10000]}

LeanSearch PremisePack:
{premise_pack_block.strip() or "not_available"}

retry_feedback:
{retry_feedback[:3000] or "not_available"}

AttemptHistoryCard:
{json.dumps(_attempt_history_summary(attempt_history), ensure_ascii=False, indent=2)[:5000] if attempt_history else "not_available"}

Return JSON object:
{{
  "status": "generated|cannot_execute",
  "statement": "public mathematical theorem statement with no workflow/internal terms",
  "formal_statement": "Lean theorem/lemma statement",
  "lean_code": "full Lean file with complete proof",
  "proof_plan": "short proof plan",
  "parent_usage": {{"parent_id": "short usage note"}},
  "reason": "short explanation"
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _coerce_theorem_string_list(value: Any) -> List[str]:
    items = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    coerced: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            preferred = (
                item.get("text")
                or item.get("checkpoint")
                or item.get("obligation")
                or item.get("statement")
                or item.get("claim")
                or item.get("description")
            )
            text = str(preferred).strip() if preferred else json.dumps(item, ensure_ascii=False)
        else:
            text = str(item).strip()
        if text:
            coerced.append(text)
    return coerced


def _coerce_string_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        for key, value in value.items()
        if str(key).strip()
    }


def _normalize_theorem_generation_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(raw)
    if not normalized.get("contract_status") and normalized.get("status"):
        normalized["contract_status"] = normalized.get("status")
    if not normalized.get("failure_reason") and normalized.get("reason"):
        normalized["failure_reason"] = normalized.get("reason")
    if not normalized.get("parent_contribution_evidence") and normalized.get("parent_usage"):
        normalized["parent_contribution_evidence"] = normalized.get("parent_usage")
    status = str(normalized.get("contract_status") or "").strip().lower()
    if status in {"", "ok", "success", "complete", "completed", "certified"}:
        normalized["contract_status"] = "generated"
    if status in {"cannot_execute", "cannot-execute", "not_executable"}:
        normalized["contract_status"] = "contract_failed"
    for field in (
        "statement_chunks",
        "selected_parent_checkpoints",
        "proof_obligations",
        "patch_target_fields",
        "must_not_change_fields",
    ):
        normalized[field] = _coerce_theorem_string_list(normalized.get(field))
    normalized["parent_contribution_evidence"] = _coerce_string_dict(
        normalized.get("parent_contribution_evidence")
    )
    if not normalized.get("proof_plan") and isinstance(normalized.get("reason"), str):
        normalized["proof_plan"] = normalized["reason"]
    if not normalized.get("harder_reason") and isinstance(normalized.get("reason"), str):
        normalized["harder_reason"] = normalized["reason"]
    return normalized


async def generate_theorem_problem(
    parent: CertificationInput,
    config: Optional[GenerationConfig] = None,
) -> TheoremGeneratedProblem:
    config = config or default_generation_config()
    metadata = dict(parent.metadata or {})
    retry_feedback = str(metadata.get("retry_feedback") or "")
    formal_statement = str(metadata.get("formal_statement") or metadata.get("lean_code") or "").strip()
    premise_payload = await _theorem_premise_pack_payload(
        statement=parent.statement,
        formal_statement=formal_statement,
        diagnostics=retry_feedback,
        leansearch_enabled=_metadata_bool(metadata.get("leansearch_enabled", DEFAULT_LEANSEARCH_ENABLED)),
        leansearch_limit=int(metadata.get("leansearch_limit") or DEFAULT_LEANSEARCH_LIMIT),
        phase="reflection" if retry_feedback else "initial",
    )
    messages = _build_theorem_generation_messages(
        parent,
        premise_pack_block=str(premise_payload.get("prompt_block") or ""),
    )
    content = await _chat_completion_text_async(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        response_format=_theorem_response_format(),
    )
    parsed = _parse_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError(
            "llm_json_parse_error: top-level JSON is "
            f"{type(parsed).__name__}, not an object"
        )
    raw = _normalize_theorem_generation_raw(parsed)
    raw["premise_pack"] = dict(premise_payload.get("digest") or {})
    contract_status = str(raw.get("contract_status") or "generated").strip() or "generated"
    raw_lean_header = str(raw.get("lean_header") or THEOREM_CANONICAL_HEADER).strip()
    formal_statement = _prelint_lean_syntax(str(raw.get("formal_statement") or "").strip())
    lean_header, lean_code = _normalize_theorem_lean_code(
        lean_code=str(raw.get("lean_code") or ""),
        lean_header=raw_lean_header,
        formal_statement=formal_statement,
    )
    # The child inherits whatever its parent had to open. Normalization returns
    # the bare canonical header, which is right for imports and wrong for
    # notation the parent's statement already depended on.
    opens = _inherited_opens(parent)
    if opens:
        lean_header = _header_with_opens(lean_header, opens)
        lean_code = _header_with_opens(lean_code, opens) if lean_code.strip() else lean_code
    # The discriminator is a hash of the statement rather than the slot number
    # on purpose: two slots that produced the same theorem then collide, which
    # is what lets the duplicate be dropped, whereas a slot number would give
    # the same problem two names and hide the duplication.
    #
    # The rest of the id used to be `__theorem_gen1` per generation, with the
    # `1` a literal rather than a counter: across one release the segment
    # appeared 209 times and read `gen1` every one of them, so a fourth-
    # generation descendant carried 134 characters that named neither its depth
    # nor any of the operators that produced it. It now carries the operator
    # chain, whose length is the depth and whose codes say what was applied.
    operator_card = _operator_card(metadata)
    child_identifier = child_id(
        [parent.id],
        op_type=str(operator_card.get("op_type") or metadata.get("op_type") or "mutation"),
        operator_variant=str(
            operator_card.get("operator_variant") or metadata.get("operator_variant") or ""
        ),
        fingerprint=_statement_fingerprint(formal_statement),
    )
    return TheoremGeneratedProblem(
        id=child_identifier,
        source_problem_id=parent.id,
        contract_status=contract_status,
        failure_reason=str(raw.get("failure_reason") or "").strip(),
        statement=str(raw.get("statement") or parent.statement).strip(),
        formal_statement=formal_statement,
        lean_header=lean_header,
        lean_code=lean_code,
        equivalence_forward=str(raw.get("equivalence_forward") or "").strip(),
        equivalence_backward=str(raw.get("equivalence_backward") or "").strip(),
        statement_chunks=list(raw.get("statement_chunks") or []),
        selected_parent_checkpoints=list(raw.get("selected_parent_checkpoints") or []),
        allowed_statement_delta=str(raw.get("allowed_statement_delta") or "immediate_corollary"),
        proof_surface=str(raw.get("proof_surface") or _infer_theorem_proof_surface(lean_code)),
        proof_plan=str(raw.get("proof_plan") or ""),
        proof_obligations=list(raw.get("proof_obligations") or []),
        difficulty_label=str(raw.get("difficulty_label") or "medium"),
        harder_reason=str(raw.get("harder_reason") or ""),
        parent_contribution_evidence=dict(raw.get("parent_contribution_evidence") or {}),
        unified_obligation=str(raw.get("unified_obligation") or ""),
        why_not_conjunction=str(raw.get("why_not_conjunction") or ""),
        patch_target_fields=list(raw.get("patch_target_fields") or []),
        must_not_change_fields=list(raw.get("must_not_change_fields") or []),
        patch_applied=str(raw.get("patch_applied") or ""),
        raw_llm_output=raw,
    )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _theorem_candidate_preflight(generated: TheoremGeneratedProblem) -> Dict[str, Any]:
    code = generated.lean_code or ""
    formal = generated.formal_statement or ""
    lowered = code.lower()
    if not re.search(r"(?m)^\s*(theorem|lemma)\s+\w+", code):
        return {
            "passed": False,
            "failure_class": "invalid_formal_shape",
            "summary": "invalid_formal_shape: lean_code must contain a theorem or lemma command",
        }
    if ":=" not in code or code.rstrip().endswith(":=") or formal.rstrip().endswith(":="):
        return {
            "passed": False,
            "failure_class": "invalid_formal_shape",
            "summary": "invalid_formal_shape: theorem/lemma command must include a complete proof body",
        }
    if re.search(r"(?<![A-Za-z_])(sorry|admit)(?![A-Za-z_])", lowered):
        return {
            "passed": False,
            "failure_class": "proof_contains_sorry",
            "summary": "proof_contains_sorry: generated Lean proof contains sorry/admit",
        }
    if generated.statement and generated.statement_chunks:
        chunk_text = " ".join(generated.statement_chunks).strip().lower()
        if len(generated.statement) > max(600, len(chunk_text) * 5):
            return {
                "passed": False,
                "failure_class": "theorem_too_broad",
                "summary": "theorem_too_broad: natural statement is much broader than selected statement chunks",
            }
    return {"passed": True, "failure_class": "", "summary": "preflight passed"}


# ---------------------------------------------------------------------------
# Statement-first certification (see docs/lean_generation_failure_analysis.md
# §B). The worker's single call still returns statement+formal_statement+
# lean_code, but the gates run as: (1) the statement alone must type-check
# with a sorry body, (2) NL↔Lean alignment is judged before any proof effort,
# (3) the proof is verified and, on failure, repaired against the now-frozen
# statement instead of regenerating the whole slot.
# ---------------------------------------------------------------------------


_AXIOM_LINE_RE = re.compile(
    r"'(?P<decl>[^']+)'\s+depends on axioms:\s*\[(?P<axioms>[^\]]*)\]"
)
_NO_AXIOM_RE = re.compile(r"'(?P<decl>[^']+)'\s+does not depend on any axioms")


def theorem_name_of(formal_statement: str) -> Optional[str]:
    match = re.search(r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)", formal_statement or "")
    return match.group(1) if match else None


def parse_axiom_closure(raw_output: str, decl: str) -> Optional[List[str]]:
    """Parse `#print axioms <decl>` output. None = the audit did not report.

    The REPL returns raw JSON, so newlines inside a message body arrive as the
    two-character sequence ``\n``; a long closure wraps across those. Fold
    both real and escaped newlines to spaces before matching.
    """
    raw_output = (raw_output or "").replace("\\n", " ").replace("\n", " ")
    for match in _NO_AXIOM_RE.finditer(raw_output or ""):
        if match.group("decl").endswith(decl):
            return []
    for match in _AXIOM_LINE_RE.finditer(raw_output or ""):
        if match.group("decl").endswith(decl):
            body = match.group("axioms").strip()
            return [item.strip() for item in body.split(",") if item.strip()]
    return None


def _statement_first_enabled() -> bool:
    return os.getenv("POOL_STATEMENT_FIRST", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _proof_repair_turns() -> int:
    try:
        return max(0, int(os.getenv("POOL_PROOF_REPAIR_TURNS", "2")))
    except ValueError:
        return 2


def _statement_surface(text: str) -> str:
    """Whitespace-normalized statement prefix (proof body stripped)."""
    surface = re.sub(r"\s+", " ", str(text or "")).strip()
    match = re.search(r":=", surface)
    if match:
        surface = surface[: match.start()].rstrip()
    return surface


_OPEN_LINE = re.compile(r"^\s*open\b.*$", re.M)


def _inherited_opens(parent: Any) -> List[str]:
    """The `open` lines the parent's own header needed, for the child to reuse.

    ProofNet seeds arrive with headers like `open Filter Set TopologicalSpace`
    and `open scoped Topology`, because their statements are written in that
    notation. A child of such a parent writes `𝓝` and `Tendsto` too — it is
    restating the parent's mathematics — but the child's header was replaced by
    the bare canonical one, which opens nothing. The statement-level check then
    builds `canonical header + statement + sorry` and Lean reports `Unknown
    identifier 𝓝`: a notation error attributed to the model, in a statement the
    model wrote correctly for its parent's context.

    One ProofNet group lost 11 of its 35 rows to exactly this, all five of its
    seeds being topology or analysis. Groups of linear-algebra seeds never saw
    it, because `LinearMap` and `Submodule` need no `open` at all.

    Inheriting rather than widening the canonical header is deliberate: opening
    `Real` and `Complex` together makes `abs`, `exp` and `log` ambiguous, so a
    fixed union of every namespace ProofNet uses would break the groups that
    currently work. What the parent compiled with is known-good for this
    lineage.
    """
    header = str((getattr(parent, "metadata", None) or {}).get("lean_header") or "")
    return [line.strip() for line in _OPEN_LINE.findall(header) if line.strip()]


def _header_with_opens(text: str, opens: Sequence[str]) -> str:
    """Insert the parent's `open` lines into a header or a whole file.

    Placed after the last preamble line rather than appended, because the same
    function is used on `lean_code`, which is `header + blank + theorem`:
    appending would put the `open` after the declaration that needs it, where
    Lean would reject it.
    """
    if not opens:
        return text
    existing = {line.strip() for line in _OPEN_LINE.findall(text)}
    missing = [line for line in opens if line not in existing]
    if not missing:
        return text
    lines = text.splitlines()
    cut = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "set_option ", "open ")):
            cut = index + 1
        elif stripped:
            break
    return "\n".join(lines[:cut] + missing + lines[cut:])


def _statement_sorry_code(lean_header: str, formal_statement: str) -> str:
    """Close the bare statement with sorry for a cheap statement-level check.

    ``set_option autoImplicit false`` is forced so an unresolved identifier in
    the statement fails loudly instead of becoming a silently quantified
    implicit variable — the same discipline the evaluation pipeline applies.
    """
    header = (lean_header or THEOREM_CANONICAL_HEADER).rstrip()
    if "set_option autoImplicit false" not in header:
        header = f"{header}\nset_option autoImplicit false"
    statement = str(formal_statement or "").strip()
    match = re.search(r":=", statement)
    if match:
        statement = statement[: match.start()].rstrip()
    return f"{header}\n\n{statement} := by\n  sorry"


def _proof_repair_response_format() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": True,
        "required": ["lean_code", "proof_plan", "reason"],
        "properties": {
            "lean_code": {
                "type": "string",
                "description": (
                    "Complete corrected Lean file proving the EXACT immutable "
                    "formal_statement. No sorry/admit."
                ),
            },
            "proof_plan": {"type": "string"},
            "reason": {
                "type": "string",
                "description": "One-sentence description of the repair applied.",
            },
        },
    }
    return _schema_response_format("theorem_proof_repair", schema)


def _build_proof_repair_messages(
    generated: TheoremGeneratedProblem,
    *,
    diagnostics: str,
    line_context: List[str],
    patch_instructions: List[str],
    premise_block: str,
    turn: int,
) -> List[Dict[str, str]]:
    system = (
        "You repair Lean 4 (Mathlib) proofs. The theorem statement is frozen: "
        "it already type-checks and matches the natural-language problem. "
        "Return only a corrected complete proof of that exact statement."
    )
    user = f"""Repair the Lean proof below. This is repair turn {turn}.

IMMUTABLE formal_statement (do not change any binder, hypothesis, name, or conclusion):
{generated.formal_statement}

Previous lean_code (failing):
{generated.lean_code}

Lean diagnostics:
{diagnostics[:1600]}

Failing-line context:
{json.dumps(line_context, ensure_ascii=False)}

Patch instructions:
{json.dumps(patch_instructions, ensure_ascii=False)}
{premise_block}
Rules:
- Keep the theorem name and statement character-for-character equivalent.
- Return the full corrected Lean file in lean_code (header lines are re-imposed by the pipeline).
- No sorry, no admit, no new axioms.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def _repair_theorem_proof_candidate(
    generated: TheoremGeneratedProblem,
    *,
    item: Dict[str, Any],
    config: GenerationConfig,
    diagnostics: str,
    turn: int,
) -> Optional[str]:
    """One LLM repair turn. Returns normalized candidate lean_code or None."""
    synthetic = CertificationResult(
        problem_id=generated.id,
        status="proof_failed",
        error=diagnostics[:700],
        proof_verify_summary=diagnostics[:700],
        lean_code=generated.lean_code,
    )
    premise_payload = await _theorem_premise_pack_payload(
        statement=generated.statement,
        formal_statement=generated.formal_statement,
        diagnostics=diagnostics,
        leansearch_enabled=bool(item.get("leansearch_enabled", DEFAULT_LEANSEARCH_ENABLED)),
        leansearch_limit=int(item.get("leansearch_limit", DEFAULT_LEANSEARCH_LIMIT)),
        phase="reflection",
    )
    messages = _build_proof_repair_messages(
        generated,
        diagnostics=diagnostics,
        line_context=_lean_error_line_context(synthetic),
        patch_instructions=_theorem_patch_instructions(synthetic),
        premise_block=premise_payload.get("prompt_block") or "",
        turn=turn,
    )
    content = await _chat_completion_text_async(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        response_format=_proof_repair_response_format(),
    )
    raw = _parse_json_object(content)
    _, candidate = _normalize_theorem_lean_code(
        lean_code=str(raw.get("lean_code") or ""),
        lean_header=generated.lean_header or THEOREM_CANONICAL_HEADER,
        formal_statement=generated.formal_statement,
    )
    if re.search(r"(?<![A-Za-z_])(sorry|admit)(?![A-Za-z_])", candidate.lower()):
        return None
    statement_surface = _statement_surface(generated.formal_statement)
    if statement_surface and statement_surface not in re.sub(r"\s+", " ", candidate):
        # The model rewrote the statement; that breaks the statement-first
        # invariant, so the turn does not count as a candidate.
        return None
    return candidate


#: Children kept so far in this process, for the judge to compare against.
#:
#: The judge is shown a child and its parents. Two children of one pair, built
#: the same way, are therefore invisible to it: `coupled_recurrences_inconsistent`
#: and `incompatible_terminal_coupling` shared a parent pair and fifteen lines of
#: proof, differed by one coupling hypothesis, and were both kept as `strong`.
#: Neither the hash gate (the statements differ) nor the redundancy probe (both
#: parents are conditional) can see it either.
_RUN_SIBLINGS: List[Dict[str, Any]] = []


def note_kept_row(problem_id: str, parent_ids: Sequence[str], statement: str, quality: str = "") -> None:
    """Record a kept child so later judgements in this run can see it."""
    if not str(statement or "").strip():
        return
    _RUN_SIBLINGS.append(
        {
            "problem_id": str(problem_id or ""),
            "parents": {str(p) for p in (parent_ids or [])},
            "statement": str(statement),
            "quality": str(quality or ""),
        }
    )


def siblings_for(parent_ids: Sequence[str], *, limit: int = 5) -> List[Dict[str, Any]]:
    """Kept children whose parents overlap these, closest pairing first."""
    wanted = {str(p) for p in (parent_ids or [])}
    if not wanted:
        return []
    scored = []
    for row in _RUN_SIBLINGS:
        shared = len(wanted & row["parents"])
        if shared:
            scored.append((shared == len(wanted) and len(wanted) > 1, shared, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        {"statement": row["statement"], "quality": row["quality"]}
        for _same_pair, _shared, row in scored[:limit]
    ]


async def _review_problem_quality(
    generated: Any,
    parents: Sequence[Any],
    work_item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask a model whether this child is worth keeping, and record why.

    Disabled unless `PROBLEM_JUDGE=1`, because the deterministic gates it
    replaces were wired in and trusted before they were validated, and three of
    the five turned out to measure the wrong thing. This one stays behind a
    switch until its verdicts have been checked against hand-read rows.

    A judge that errors, times out, or answers unparseably returns `keep`. The
    corpus loses more from silently discarding good problems than from admitting
    mediocre ones, and a gate that fails closed on its own malfunction is the
    hardest kind of corruption to notice afterwards.
    """
    if os.getenv("PROBLEM_JUDGE", "0") != "1":
        return {"ran": False, "why": "PROBLEM_JUDGE not enabled"}
    # Silent mutations used to be excluded here, on the grounds that their gate
    # was the two-way equivalence proof. That gate has since been removed —
    # measured against the cases it existed to catch, it passed a child that had
    # dropped a conjunct and a child that was unrelated — so the exclusion left
    # silent rows with no gate at all: four of them certified in one run with no
    # verdict from anything. The rubric now tells the judge what to ask of this
    # tier instead of assuming it would answer `recall` every time.
    from src.certification.judges import (
        JUDGE_SYSTEM,
        crossover_prompt,
        judge_brief,
        mutation_prompt,
        parse_verdict,
        silent_prompt,
    )
    from src.utils.codex_cli import call_codex_cli, is_usage_limit

    child_statement = str(getattr(generated, "formal_statement", "") or "")
    child_proof = str(getattr(generated, "lean_code", "") or "")
    if not child_statement or not child_proof:
        return {"ran": False, "why": "no statement or proof"}

    pack = [
        {
            "name": str(getattr(p, "id", "") or ""),
            "statement": str((getattr(p, "metadata", None) or {}).get("formal_statement") or ""),
            "proof": str((getattr(p, "metadata", None) or {}).get("lean_code") or ""),
        }
        for p in (parents or [])
    ]
    pack = [p for p in pack if p["statement"]]
    if not pack:
        return {"ran": False, "why": "no parent statement"}

    evidence = dict((work_item or {}).get("novelty_evidence") or {})
    # Lean's redundancy finding travels with the other measurements. It is the
    # one that settles the case the judge got wrong twice: a shared variable
    # between two conjuncts read as interaction, when the second conjunct was
    # true for every value that variable could take.
    redundancy = dict((work_item or {}).get("redundancy_evidence") or {})
    if redundancy.get("measured"):
        evidence["redundancy"] = {
            key: redundancy.get(key)
            for key in (
                "universal_parents",
                "free_hypotheses",
                "redundant",
            )
            if redundancy.get(key) is not None
        }
    # The equivalence probe, for a silent slot, described as what it is. It used
    # to decide the row; it was measured against cases it should have caught and
    # caught none of them, so it now travels as a note rather than a verdict.
    # The rubric tells the judge how much to read into it.
    # What the round-trip found, in the judge's terms. It is the only evidence
    # about the *statement* rather than the theorem, and it was recorded on the
    # row and shown to nobody -- the mirror of the redundancy gap, where the
    # reader existed and the writer did not.
    alignment = dict((work_item or {}).get("alignment_evidence") or {})
    if alignment.get("status"):
        evidence["goal_roundtrip"] = {
            "prose_matches_the_elaborated_goal": alignment.get("equivalent"),
            "read_back_from_the_lean_alone": str(alignment.get("informalized_statement") or "")[:600],
            "why": str(alignment.get("rationale") or "")[:400],
        }

    silent_probe = dict((work_item or {}).get("silent_evidence") or {})
    op_type = str(getattr(generated, "op_type", "") or (work_item or {}).get("op_type") or "")

    # Precedents are drawn from this operator's judgments only. A mutation
    # verdict turns on whether the proof still ends on the parent's lemma; a
    # crossover verdict on whether two parents met. Neither reasoning transfers,
    # so unlike the planner's lessons — which share most of their catalogue —
    # these pools stay apart. Empty until the judge has run, and silent then.
    precedents = ""
    try:
        from src.certification.judge_memory import format_precedents, similar_judgments
        from src.retrieval.memory_search import build_query

        # The query asks about the *relationship*, not the child alone. A
        # judgment turns on how the child sits against its parents, so a lookup
        # keyed only on the child retrieves rows that share a topic — two ZMod
        # problems match whether or not either was a constant-supplier — while
        # the stored entries are indexed on child statement plus failure plus
        # reason. Putting the parents and the measured evidence into the query
        # makes the two sides ask the same question.
        query_rows = [
            {"formal_statement": child_statement, "statement_nl": str(getattr(generated, "statement", "") or "")}
        ]
        query_rows.extend({"formal_statement": p["statement"]} for p in pack[:2])
        query = build_query(query_rows)
        signals = [
            f"{name} {evidence.get(name)}"
            for name in ("coupling_depth", "closing_lemma_match", "skeleton_distance")
            if evidence.get(name) is not None
        ]
        if signals:
            query += " Measured: " + ", ".join(signals) + "."
        precedents = format_precedents(similar_judgments(query, op_type, limit=3))
    except Exception:
        precedents = ""

    # The plan is what makes `fix_scope` answerable: a child that did exactly
    # what it was told is not the generator's failure, and that distinction
    # cannot be drawn from the parents and the child alone.
    plan = _operator_card(work_item or {})
    kin = siblings_for([p["name"] for p in pack])
    if op_type == "crossover" and len(pack) >= 2:
        user = crossover_prompt(
            pack, child_statement, child_proof, evidence,
            precedents=precedents, plan=plan, siblings=kin,
        )
    elif str((work_item or {}).get("operator_variant") or "") == "mutation_silent":
        # Its question is the inverse of the mutation rubric's, and asked through
        # that rubric it came back backwards: a child that dropped all three of
        # its parent's bounds passed with the drop named in the judge's own
        # reasoning, and a child that only removed an alias was called materially
        # different from a statement it is provably equivalent to.
        user = silent_prompt(
            pack[0]["statement"], pack[0]["proof"], child_statement, child_proof, evidence,
            precedents=precedents, plan=plan, siblings=kin,
        )
    else:
        user = mutation_prompt(
            pack[0]["statement"], pack[0]["proof"], child_statement, child_proof, evidence,
            precedents=precedents, plan=plan, siblings=kin,
        )

    judge_model = os.getenv("PROBLEM_JUDGE_MODEL", "gpt-5.6-terra")
    # Traced like the planner and the worker are. This call goes straight to
    # `call_codex_cli` rather than through `_chat_completion_text_*`, which is
    # where the other two get their span, so without this the judge is the one
    # agent of the three whose prompt and answer never reach LangSmith — and it
    # is the one whose input is hardest to reconstruct afterwards, because the
    # prompt carries retrieved precedents and measured evidence that are not
    # stored anywhere else.
    with ls.trace(
        name="problem_quality_judge",
        run_type="llm",
        inputs={
            "model": judge_model,
            "op_type": op_type,
            "problem_id": str(getattr(generated, "id", "") or ""),
            "parent_ids": [p["name"] for p in pack],
            "system": JUDGE_SYSTEM,
            "prompt": user,
            "evidence": evidence,
            "plan": plan,
            "precedents_used": bool(precedents),
        },
        tags=["pool-generation", "judge", f"judge:{op_type}"],
    ) as judge_run:
        reply = await call_codex_cli(
            model=judge_model,
            system=JUDGE_SYSTEM,
            user=user,
            timeout_seconds=float(os.getenv("PROBLEM_JUDGE_TIMEOUT", "420")),
        )
        if reply.error:
            limited = is_usage_limit(str(reply.error))
            judge_run.end(outputs={"ran": False, "error": str(reply.error)[:300]})
            return {
                "ran": False,
                "why": ("usage_limit_reached" if limited else str(reply.error)[:140]),
                "verdict": "keep",
            }
        verdict = parse_verdict(reply.raw_text)
        verdict["ran"] = True
        verdict["op_type"] = op_type
        verdict["judge_model"] = judge_model
        verdict["retry_brief"] = judge_brief(verdict)
        judge_run.end(
            outputs={
                "raw": reply.raw_text[:4000],
                "verdict": verdict.get("verdict"),
                "quality": verdict.get("quality"),
                "failure": verdict.get("failure"),
                "reason": verdict.get("reason"),
                "retry_plan": verdict.get("retry_plan"),
                "fix_scope": verdict.get("fix_scope"),
                "judge_error": verdict.get("judge_error"),
                "elapsed_seconds": reply.elapsed_seconds,
            }
        )
    try:
        from src.certification.judge_memory import record_judgment

        record_judgment(
            problem_id=str(getattr(generated, "id", "") or ""),
            op_type=op_type,
            verdict=verdict,
            parent_statement=pack[0]["statement"],
            child_statement=child_statement,
            judge_model=judge_model,
            generator_model=str(getattr(generated, "llm_model", "") or ""),
        )
    except Exception:
        pass
    return verdict


async def _certify_theorem_child(
    *,
    parent_input: CertificationInput,
    item: Dict[str, Any],
    # The real parent rows, which `parent_input` does not carry: for crossover
    # it is a synthetic prompt object whose `parents` metadata holds statements
    # but no Lean. The quality judge needs the parent proofs — its whole
    # question is whether the child's proof does something theirs does not.
    parents: Optional[List[CertificationInput]] = None,
    generation_count: int,
    config: GenerationConfig,
    theorem_generator: Optional[TheoremGeneratorFn],
    theorem_verifier: Optional[TheoremVerifierFn],
    theorem_alignment_verifier: Optional[TheoremAlignmentVerifierFn] = None,
    theorem_proof_repairer: Optional[Callable[..., Any]] = None,
) -> CertificationResult:
    started = time.time()
    try:
        generated = await _maybe_await(
            theorem_generator(parent_input, config)
            if theorem_generator is not None
            else generate_theorem_problem(parent_input, config=config)
        )
        if not isinstance(generated, TheoremGeneratedProblem):
            generated = TheoremGeneratedProblem.model_validate(generated)
        raw_surface = (
            generated.raw_llm_output.get("proof_surface")
            if isinstance(generated.raw_llm_output, dict)
            else None
        )
        generated.proof_surface = str(
            raw_surface or _infer_theorem_proof_surface(generated.lean_code)
        )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        if exc.__traceback__ is not None:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            error_text += (
                f" @ {str(frame.filename).rsplit('/', 1)[-1]}:{frame.lineno}"
            )
        lowered = error_text.lower()
        if any(
            marker in lowered
            for marker in ("json", "unterminated string", "expecting value", "delimiter")
        ):
            failure_cls = "llm_json_parse_error"
        elif any(
            marker in lowered
            for marker in (
                "rate limit",
                "rate_limit",
                "429",
                "too many requests",
                "connection error",
                "connection reset",
                "timed out",
                "timeout",
                "service unavailable",
                "server overloaded",
                "internal server error",
                "bad gateway",
            )
        ):
            failure_cls = "llm_transport_error"
        else:
            failure_cls = "generation_failed"
        quality_evidence = {
            "checkpoint_coverage": 0.0,
            "missing_checkpoints": ["theorem_generation"],
            "reasoning_signature": failure_cls,
            "signature_group": "theorem_generation_failed",
            "parent_contribution": {},
            "feature_delta": {},
            "novelty_flags": [failure_cls],
            "failure_class": failure_cls,
            "proof_verify_summary": error_text[:500],
            "premise_pack": dict(parent_input.metadata.get("premise_pack") or {}),
            # Infrastructure failures must be separable from mathematical or
            # contract failures in funnel statistics.
            "system_error": failure_cls == "llm_transport_error",
        }
        quality_evidence["misformalization"] = derive_misformalization_taxonomy(
            CertificationResult(
                problem_id=f"{parent_input.id}__theorem_generation_failed",
                status="generation_failed",
                error=error_text[:500],
            ),
            [failure_cls],
            quality_evidence,
        )
        return CertificationResult(
            problem_id=f"{parent_input.id}__theorem_generation_failed",
            source_problem_id=parent_input.id,
            generation=generation_count,
            slot=int(item["slot"]),
            operation=item.get("op_type"),
            op_type=item.get("op_type"),
            operator_variant=item.get("operator_variant"),
            parent_ids=list(item.get("parent_ids") or []),
            target_family="theorem_proof",
            quality_verdict="weak",
            quality_flags=[failure_cls],
            interestingness_score=0.0,
            feedback_for_next_generation="Repair theorem generation surface before retrying.",
            reasoning_pattern="theorem_generation_failed",
            quality_evidence=quality_evidence,
            planner_source=item.get("planner_source", ""),
            problem_style="theorem_proof",
            target_style="theorem_proof",
            certification_route="theorem_prover",
            parent_context_cards=list(item.get("parent_context_cards") or []),
            operator_card=_operator_card(item),
            slot_outcome="generation_failed",
            statement="",
            answer="",
            formal_statement="",
            lean_header="",
            formal_status="generation_failed",
            family="theorem_proof",
            status="generation_failed",
            lean_level=0,
            lean_code=None,
            anti_stub_passed=False,
            aligned=False,
            llm_used=True,
            llm_model=config.model,
            lean_available=True,
            error=error_text[:500],
            elapsed_seconds=round(time.time() - started, 6),
            proof_verify_summary=error_text[:500],
            input_metadata=dict(parent_input.metadata or {}),
        )

    proof_repair_evidence: Dict[str, Any] = {}
    # Gate outcomes observed by the staged pipeline; consumed by build_result
    # to emit the row's certificate (see src/certification/tiers.py).
    gate_state: Dict[str, Any] = {
        "statement_probe_ran": False,
        "statement_checked": False,
    }

    def build_result(
        *,
        status: str,
        proof_summary: str,
        alignment: Optional[TheoremAlignmentResult] = None,
        vacuity_evidence: Optional[Dict[str, Any]] = None,
        judge_evidence: Optional[Dict[str, Any]] = None,
        silent_evidence: Optional[Dict[str, Any]] = None,
        dedup_evidence: Optional[Dict[str, Any]] = None,
        redundancy_evidence: Optional[Dict[str, Any]] = None,
        hypothesis_evidence: Optional[Dict[str, Any]] = None,
    ) -> CertificationResult:
        operator_card = _operator_card(item)
        allowed_surfaces = set(operator_card.get("theorem_proof_surfaces") or [])
        raw_surface_present = bool(
            isinstance(generated.raw_llm_output, dict)
            and generated.raw_llm_output.get("proof_surface")
        )
        alignment_summary = ""
        if alignment is not None and not alignment.aligned:
            alignment_summary = (
                "alignment failed: "
                + "; ".join(alignment.unsupported_claims or alignment.missing_claims or [alignment.rationale])
            )[:500]
        vacuity = dict(vacuity_evidence or {"measured": False, "why": "not probed"})
        if vacuity.get("vacuous") and status == "certified":
            status = "vacuous"
        judge = dict(judge_evidence or {"ran": False})
        if judge.get("verdict") == "reject" and status == "certified":
            status = "judge_rejected"
        elif status == "certified":
            # Recorded only once the row survives, so the judge compares against
            # children that were actually kept rather than against everything
            # attempted.
            note_kept_row(
                generated.id,
                [str(p) for p in (item.get("parent_ids") or [])],
                generated.formal_statement or "",
                str(judge.get("quality") or ""),
            )
        # The equivalence probe is measurement, not a verdict. It was written as
        # a gate on the belief that Lean settles "is this the same mathematics",
        # and it does not: `silent_backward (h : child) : parent` asks Lean to
        # prove the parent, which is a theorem, so the tactic block closes it
        # with or without `h`. Tested directly, the probe called a child that
        # dropped a conjunct equivalent to its parent, and called an unrelated
        # true statement equivalent too. What it actually reports is whether the
        # ladder could close both directions, which is a fact about the ladder.
        #
        # So it stops deciding and starts informing. Whether a silent child is
        # the same mathematics in different Lean is a question of degree, which
        # is the judge's, and the probe goes to the judge described as what it
        # is.
        silent = dict(silent_evidence or {"checked": False})
        dedup = dict(dedup_evidence or {"checked": False})
        if dedup.get("duplicate") and status == "certified":
            status = "duplicate_statement"
        # No status of its own: a pruned row is a fixed row, not a failed one,
        # and the removal is already verified by the recompile that authorised
        # it. The evidence records what went so a reader can see the statement
        # was edited and why.
        hypotheses = dict(hypothesis_evidence or {"measured": False, "removed": []})
        is_certified = status == "certified"
        quality_flags = []
        missing_checkpoints = []
        novelty_flags = []
        if status == "statement_failed":
            quality_flags = ["statement_typecheck_failed"]
            missing_checkpoints = ["formal_statement_typechecks"]
            novelty_flags = ["statement_typecheck_failed"]
        elif status == "alignment_failed":
            quality_flags = ["statement_lean_alignment_failed"]
            missing_checkpoints = ["statement_lean_alignment"]
            novelty_flags = ["statement_lean_alignment_failed"]
        elif status == "generation_failed":
            quality_flags = ["theorem_contract_failed"]
            missing_checkpoints = ["theorem_worker_contract"]
            novelty_flags = ["theorem_contract_failed"]
        elif status == "vacuous":
            quality_flags = ["vacuous_hypotheses"]
            missing_checkpoints = ["hypotheses_satisfiable"]
            novelty_flags = ["vacuous_hypotheses"]
        elif status == "judge_rejected":
            quality_flags = ["judge_reject"]
            missing_checkpoints = ["problem_quality_review"]
            novelty_flags = ["judge_reject"]
        elif status == "silent_not_equivalent":
            quality_flags = ["silent_not_equivalent"]
            missing_checkpoints = ["silent_equivalence"]
            novelty_flags = ["silent_not_equivalent"]
        elif status == "duplicate_statement":
            quality_flags = ["duplicate_statement"]
            missing_checkpoints = ["statement_not_already_in_corpus"]
            novelty_flags = ["duplicate_statement"]
        elif not is_certified:
            quality_flags = ["proof_failed"]
            missing_checkpoints = ["lean_proof_complete"]
            novelty_flags = ["proof_failed"]
        summary = alignment_summary or proof_summary
        failure_cls = _failure_class(
            CertificationResult(
                problem_id=generated.id,
                status=status,
                error=summary,
                proof_verify_summary=summary,
            )
        )
        quality_evidence = {
            "checkpoint_coverage": 1.0 if is_certified else 0.0,
            "missing_checkpoints": missing_checkpoints,
            "reasoning_signature": "theorem_proof",
            "signature_group": "theorem_proof",
            "parent_contribution": dict(generated.parent_contribution_evidence or {}),
            "feature_delta": {},
            "novelty_flags": novelty_flags,
            "failure_class": failure_cls,
            "proof_verify_summary": summary,
            "statement_chunks": list(generated.statement_chunks or []),
            "selected_parent_checkpoints": list(generated.selected_parent_checkpoints or []),
            "allowed_statement_delta": generated.allowed_statement_delta,
            "proof_surface": generated.proof_surface,
            "proof_surface_source": (
                "worker_legacy_field" if raw_surface_present else "inferred_from_lean_code"
            ),
            "allowed_proof_surfaces": sorted(allowed_surfaces),
            "vacuity": dict(vacuity),
            "judge": dict(judge),
            "silent": dict(silent),
            "dedup": dict(dedup),
            "redundancy": dict(redundancy_evidence or {}),
            "dead_hypotheses": dict(hypotheses),
            "proof_repair": dict(proof_repair_evidence),
            "alignment_evidence": dict(gate_state.get("alignment_evidence") or {}),
            "proof_surface_allowed_by_operator": (
                not allowed_surfaces or generated.proof_surface in allowed_surfaces
            ),
            "contract_status": generated.contract_status,
            "failure_reason": generated.failure_reason,
            "unified_obligation": generated.unified_obligation,
            "why_not_conjunction": generated.why_not_conjunction,
            "patch_target_fields": list(generated.patch_target_fields or []),
            "must_not_change_fields": list(generated.must_not_change_fields or []),
            "patch_applied": generated.patch_applied,
            "theorem_alignment": alignment.model_dump() if alignment is not None else {},
            "premise_pack": dict(
                (
                    generated.raw_llm_output.get("premise_pack")
                    if isinstance(generated.raw_llm_output, dict)
                    else {}
                )
                or {}
            ),
        }
        quality_evidence["misformalization"] = derive_misformalization_taxonomy(
            CertificationResult(
                problem_id=generated.id,
                status=status,
                error=summary,
                proof_verify_summary=summary,
            ),
            quality_flags,
            quality_evidence,
        )
        if status == "generation_failed":
            result_statement = ""
            result_formal_statement = ""
            result_lean_header = ""
            result_lean_code = None
            result_proof_plan = ""
        else:
            result_statement = generated.statement
            result_formal_statement = generated.formal_statement
            result_lean_header = generated.lean_header
            result_lean_code = generated.lean_code
            result_proof_plan = generated.proof_plan
        return CertificationResult(
            problem_id=generated.id,
            source_problem_id=generated.source_problem_id,
            generation=generation_count,
            slot=int(item["slot"]),
            operation=item.get("op_type"),
            op_type=item.get("op_type"),
            operator_variant=item.get("operator_variant"),
            parent_ids=list(item.get("parent_ids") or []),
            variation_axis=item.get("variation_axis", ""),
            target_family="theorem_proof",
            composition_pattern=item.get("composition_pattern", ""),
            parent_contributions=dict(generated.parent_contribution_evidence or item.get("parent_contributions") or {}),
            avoid_patterns=list(item.get("avoid_patterns") or []),
            quality_target=item.get("quality_target", ""),
            quality_verdict="acceptable" if is_certified else "weak",
            quality_flags=quality_flags,
            interestingness_score=0.7 if is_certified else 0.0,
            feedback_for_next_generation=(
                "Theorem proof complete; preserve theorem style."
                if is_certified
                else (
                    "Repair statement/formal alignment after Lean verification."
                    if status == "alignment_failed"
                    else "Repair Lean proof before reusing this theorem pattern."
                )
            ),
            reasoning_pattern="theorem_proof",
            semantic_parent_contribution=dict(generated.parent_contribution_evidence or {}),
            quality_evidence=quality_evidence,
            axis_applied=item.get("variation_axis", ""),
            axis_aligned=is_certified,
            planner_source=item.get("planner_source", ""),
            problem_style="theorem_proof",
            target_style="theorem_proof",
            certification_route="theorem_prover",
            parent_context_cards=list(item.get("parent_context_cards") or []),
            operator_card=operator_card,
            slot_outcome=status,
            statement=result_statement,
            answer="",
            solution=result_proof_plan,
            formal_statement=result_formal_statement,
            lean_header=result_lean_header,
            formal_status=("certified" if is_certified else status),
            family="theorem_proof",
            status=status,
            lean_level=3 if is_certified else 0,
            certificate=build_certificate_record(
                statement_checked=(
                    bool(gate_state["statement_checked"]) or is_certified
                ),
                proof_checked=is_certified,
                axiom_closure=gate_state.get("axiom_closure"),
                auto_implicit_false=bool(gate_state["statement_probe_ran"]),
                proof_method="tactic_proof" if is_certified else None,
                faithfulness=(
                    ("faithful" if alignment.aligned else "incomparable")
                    if alignment is not None
                    else "unaudited"
                ),
                alignment_method="llm_judge" if alignment is not None else "none",
                verifier="lake_env_lean",
            ),
            lean_code=result_lean_code,
            anti_stub_passed=is_certified and "sorry" not in (result_lean_code or ""),
            aligned=is_certified,
            difficulty_label=generated.difficulty_label,
            generation_notes=generated.harder_reason,
            llm_used=True,
            llm_model=config.model,
            lean_available=status != "lean_unavailable",
            error=None if is_certified else summary[:500],
            elapsed_seconds=round(time.time() - started, 6),
            proof_plan=result_proof_plan,
            proof_obligations=[] if status == "generation_failed" else list(generated.proof_obligations or []),
            proof_verify_summary=summary,
            input_metadata=dict(parent_input.metadata or {}),
        )

    if generated.contract_status != "generated":
        reason = generated.failure_reason or generated.contract_status
        return build_result(
            status="generation_failed",
            proof_summary=f"{generated.contract_status}: {reason}",
        )
    preflight = _theorem_candidate_preflight(generated)
    if not preflight.get("passed"):
        return build_result(
            status="generation_failed",
            proof_summary=str(preflight.get("summary") or "invalid theorem candidate"),
        )
    verify_fn = theorem_verifier or verify_lean_proof
    align_fn = theorem_alignment_verifier or verify_theorem_alignment
    attempt = int(item.get("retry_count") or 0)
    statement_first = _statement_first_enabled()

    async def _run_verify(code: str) -> LeanVerifyResult:
        try:
            verdict = await _maybe_await(verify_fn(code, timeout=300.0))
        except TypeError:
            verdict = await _maybe_await(verify_fn(code))
        if not isinstance(verdict, LeanVerifyResult):
            verdict = LeanVerifyResult(**dict(verdict))
        return verdict

    async def _run_axiom_audit(code: str) -> Optional[List[str]]:
        """Kernel axiom closure of the target declaration.

        An elaborator-accepted proof can still rest on a smuggled `axiom`,
        on `sorryAx`, or on `Lean.ofReduceBool` (`native_decide`); only the
        closure reported by `#print axioms` sees through that.
        """
        decl = theorem_name_of(generated.formal_statement)
        if not decl:
            return None
        # Deliberately the FILE verifier, never the shared REPL: the REPL keeps
        # one environment across commands, so a name declared by an earlier
        # candidate can shadow this one and `#print axioms` then reports the
        # wrong closure (observed as a spurious sorryAx on 2026-07-29).
        probe = await _maybe_await(
            verify_lean_proof(f"{code.rstrip()}\n\n#print axioms {decl}", timeout=300.0)
        )
        if not isinstance(probe, LeanVerifyResult):
            probe = LeanVerifyResult(**dict(probe))
        return parse_axiom_closure(
            "\n".join(
                part for part in (probe.raw_stdout, probe.raw_stderr) if part
            ),
            decl,
        )

    async def _run_alignment() -> TheoremAlignmentResult:
        try:
            verdict = await _maybe_await(align_fn(generated, item=item, config=config))
        except TypeError:
            verdict = await _maybe_await(align_fn(generated))
        if not isinstance(verdict, TheoremAlignmentResult):
            verdict = TheoremAlignmentResult.model_validate(verdict)
        return verdict

    alignment: Optional[TheoremAlignmentResult] = None
    if statement_first:
        # Stage 1 — the statement alone must elaborate (with a sorry body and
        # autoImplicit disabled) before any proof or alignment budget is spent.
        statement_code = _statement_sorry_code(
            generated.lean_header, generated.formal_statement
        )
        with ls.trace(
            name=f"theorem_statement_verify.slot_{item['slot']}.attempt_{attempt}",
            run_type="tool",
            inputs={
                "slot": int(item["slot"]),
                "attempt": attempt,
                "problem_id": generated.id,
                "statement_code_chars": len(statement_code),
            },
            tags=["pool-generation", "theorem-statement-verify", f"slot:{item['slot']}"],
        ) as statement_run:
            statement_result = await _run_verify(statement_code)
            statement_run.end(
                outputs={
                    "ok": statement_result.ok,
                    "summary": statement_result.summary(),
                    "verify_time": statement_result.verify_time,
                    "system_error": statement_result.system_error,
                }
            )
        if statement_result.system_error and "neither `lake` nor `lean`" in statement_result.system_error:
            return build_result(
                status="lean_unavailable", proof_summary=statement_result.summary()
            )
        gate_state["statement_probe_ran"] = True
        if not statement_result.ok:
            return build_result(
                status="statement_failed",
                proof_summary=f"statement_typecheck_failed: {statement_result.summary()}",
            )
        gate_state["statement_checked"] = True

        # Stage 2 — judge NL↔Lean alignment on the type-correct statement
        # before spending any proof effort on a misaligned candidate.
        alignment = await _run_alignment()
        if not alignment.aligned:
            return build_result(
                status="alignment_failed",
                proof_summary="alignment checked before proof verification",
                alignment=alignment,
            )
        # The elaborated-goal round-trip. On by default since 2026-08-11: it is
        # the only check that can catch a row which type-checks, proves, and
        # means something else, and run over the released corpus it found 13 of
        # 146 whose prose does not describe the goal Lean built. Left off, it
        # had covered 5 of those 146.
        #
        # Still annotation rather than a gate — the verdict is the judge's, and
        # the evidence now reaches it.
        if os.getenv("POOL_ALIGNMENT_GOAL_AUDIT", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                # Published to the work item as well, so the judge prompt can
                # read it; `gate_state` alone never reaches the judge.
                gate_state["alignment_evidence"] = await elaborated_goal_alignment(
                    statement_nl=generated.statement,
                    formal_statement=generated.formal_statement,
                    lean_header=generated.lean_header or THEOREM_CANONICAL_HEADER,
                    config=config,
                    verifier=_run_verify,
                )
            except Exception as exc:
                gate_state["alignment_evidence"] = {
                    "source": "elaborated_goal_informalization",
                    "status": "signal_error",
                    "equivalent": None,
                    "rationale": f"{type(exc).__name__}: {exc}"[:300],
                }
            if item is not None:
                item["alignment_evidence"] = gate_state["alignment_evidence"]

    # Stage 3 — verify the full proof; under statement-first, a failing proof
    # is repaired against the frozen statement instead of discarding the slot.
    with ls.trace(
        name=f"theorem_lean_verify.slot_{item['slot']}.attempt_{attempt}",
        run_type="tool",
        inputs={
            "slot": int(item["slot"]),
            "attempt": attempt,
            "op_type": item.get("op_type"),
            "problem_id": generated.id,
            "lean_code_chars": len(generated.lean_code or ""),
            "lean_header": generated.lean_header,
        },
        tags=["pool-generation", "theorem-lean-verify", f"slot:{item['slot']}"],
    ) as lean_run:
        verify_result = await _run_verify(generated.lean_code)
        lean_run.end(
            outputs={
                "ok": verify_result.ok,
                "complete": verify_result.complete,
                "summary": verify_result.summary(),
                "verify_time": verify_result.verify_time,
                "system_error": verify_result.system_error,
            }
        )

    status = "certified" if verify_result.complete else "proof_failed"
    if verify_result.system_error and "neither `lake` nor `lean`" in verify_result.system_error:
        status = "lean_unavailable"
    proof_summary = verify_result.summary()

    if status == "proof_failed" and statement_first:
        repair_fn = theorem_proof_repairer or _repair_theorem_proof_candidate
        repair_budget = _proof_repair_turns()
        repair_attempts: List[Dict[str, Any]] = []
        for turn in range(1, repair_budget + 1):
            diagnostics = verify_result.summary()
            with ls.trace(
                name=f"theorem_proof_repair.slot_{item['slot']}.turn_{turn}",
                run_type="tool",
                inputs={
                    "slot": int(item["slot"]),
                    "attempt": attempt,
                    "turn": turn,
                    "problem_id": generated.id,
                    "diagnostics": diagnostics[:600],
                },
                tags=["pool-generation", "theorem-proof-repair", f"slot:{item['slot']}"],
            ) as repair_run:
                try:
                    candidate = await _maybe_await(
                        repair_fn(
                            generated,
                            item=item,
                            config=config,
                            diagnostics=diagnostics,
                            turn=turn,
                        )
                    )
                except Exception as exc:
                    candidate = None
                    repair_attempts.append(
                        {"turn": turn, "outcome": f"repair_call_error:{type(exc).__name__}"}
                    )
                    repair_run.end(outputs={"outcome": "repair_call_error"})
                    continue
                if not candidate:
                    repair_attempts.append(
                        {"turn": turn, "outcome": "no_valid_candidate"}
                    )
                    repair_run.end(outputs={"outcome": "no_valid_candidate"})
                    continue
                verify_result = await _run_verify(candidate)
                proof_summary = verify_result.summary()
                outcome = "proved" if verify_result.complete else "still_failing"
                repair_attempts.append({"turn": turn, "outcome": outcome})
                repair_run.end(
                    outputs={
                        "outcome": outcome,
                        "summary": proof_summary,
                        "verify_time": verify_result.verify_time,
                    }
                )
            if verify_result.complete:
                generated.lean_code = candidate
                status = "certified"
                break
        proof_repair_evidence.update(
            {
                "enabled": True,
                "budget": repair_budget,
                "turns_used": len(repair_attempts),
                "attempts": repair_attempts,
                "repaired": status == "certified" and bool(repair_attempts),
            }
        )

    if status == "certified":
        # proof_checked requires the axiom closure, not just elaborator
        # acceptance (see src/certification/levels.py).
        closure = await _run_axiom_audit(generated.lean_code)
        gate_state["axiom_closure"] = closure
        audit = axiom_audit(closure)
        if audit["passed"] is not True:
            detail = (
                f"disallowed axioms {audit['disallowed']}"
                if audit["ran"]
                else "axiom closure could not be determined"
            )
            status = "proof_failed"
            proof_summary = f"axiom_audit_failed: {detail}"

    if status != "certified":
        return build_result(status=status, proof_summary=proof_summary, alignment=alignment)

    if alignment is None:
        # Legacy order (POOL_STATEMENT_FIRST off): alignment runs after proof.
        alignment = await _run_alignment()
        if not alignment.aligned:
            return build_result(
                status="alignment_failed", proof_summary=proof_summary, alignment=alignment
            )
    # A compiling proof is not yet a usable problem: if the hypotheses cannot all
    # hold, every conclusion follows and the row measures nothing. Crossover
    # makes this likely rather than exotic — stacking one parent's constraints
    # onto another's is how a satisfiable system stops being one. Probing only
    # rows that already certified costs one Lean call per accepted row and none
    # per rejected one.
    from src.certification.vacuity import is_vacuous
    from src.certification.hypotheses import enabled as hypotheses_enabled, prune_dead

    # The inhabitation probe used to run here, asking whether a set a `∀` ranges
    # over could be empty. It was removed on 2026-08-11 after being measured: it
    # fired on 0 of 6,392 rows, and the reason is that its pattern matches
    # `∀ x ∈ S,` with `S` a bare identifier, which covers 99 of the corpus's
    # 5,669 statements while 661 contain both `∀` and `∈`. Missing 85% of the
    # shapes it exists for means "it did not fire" carries no information, and a
    # check that cannot inform is worse than absent -- it reads on the evidence
    # table as a question that was asked and answered.
    #
    # Vacuity keeps the ground it actually covers: it asks Lean whether the
    # hypotheses entail `False`, with no pattern in the way.
    vacuity_evidence = await is_vacuous(
        verify_lean_proof,
        generated.lean_header or THEOREM_CANONICAL_HEADER,
        generated.formal_statement or "",
        timeout=float(os.getenv("VACUITY_PROBE_TIMEOUT", "120")),
    )
    # A hypothesis the proof never needs makes the problem look harder than it
    # is and points a prover somewhere that leads nowhere. The judge can read a
    # hypothesis but cannot test one, so this is Lean's to settle: drop the
    # binder, recompile, and keep the removal only if the proof still closes.
    #
    # It runs before dedup because it changes the statement, and a pruned
    # statement is the one that should be fingerprinted and judged. Fourteen
    # rows of one release carried twenty-two such hypotheses, found only after
    # the corpus was assembled.
    hypothesis_evidence: Dict[str, Any] = {"measured": False, "removed": []}
    if hypotheses_enabled():
        try:
            hypothesis_evidence = await prune_dead(
                verify_lean_proof,
                generated.formal_statement or "",
                generated.lean_code or "",
                timeout=float(os.getenv("HYPOTHESIS_PRUNE_TIMEOUT", "300")),
            )
        except Exception as error:  # pragma: no cover - transport failures
            hypothesis_evidence = {"measured": False, "removed": [], "why": str(error)[:120]}
        if hypothesis_evidence.get("removed"):
            generated.formal_statement = hypothesis_evidence["formal_statement"]
            generated.lean_code = hypothesis_evidence["lean_code"]
        # The pruned text is already carried on `generated`; keeping a second
        # copy in the evidence would double the row's size for nothing.
        hypothesis_evidence.pop("formal_statement", None)
        hypothesis_evidence.pop("lean_code", None)

    # Identity is the one question left to a hash. Every other gate became the
    # judge's, because every other gate asked a question of degree; whether two
    # rows state the same theorem is not one. The check spans the whole corpus,
    # not this run's output file: the dedup that existed before caught a repeat
    # between generations of one seed group and missed the same statement
    # produced by another group or an earlier campaign, and eight such pairs are
    # already in the released ProofNet rows.
    #
    # It runs before the judge so a duplicate costs no model call.
    dedup_evidence: Dict[str, Any] = {"checked": False}
    if os.getenv("DEDUP_GATE", "1") == "1" and (generated.formal_statement or "").strip():
        index = _corpus_index()
        holder = index.holder(generated.formal_statement)
        # A row is not a duplicate of itself. The identifier is derived from the
        # statement, so a retry that lands on the same theorem gets the same id
        # and finds the entry its own first attempt registered: in one four-
        # generation run four rows were discarded that way, one of them the best
        # silent mutation the run produced.
        if holder and holder == generated.id:
            holder = None
        dedup_evidence = {
            "checked": True,
            "duplicate": bool(holder),
            "duplicate_of": holder or "",
            "corpus_size": len(index),
        }
        if not holder:
            index.add(generated.formal_statement, generated.id)

    # Redundancy evidence for the judge. Inline this runs the syntactic filter
    # and the tactic ladder only -- no prover -- because a parent that assumes
    # nothing and falls to `omega` is the case worth flagging in-flight, and the
    # prover retries that catch the rest belong in the offline scan where a
    # fifteen-minute worst case costs nothing. It never changes status: the
    # check has been rebuilt six times and its findings go to the judge as
    # measurement, not as a verdict.
    redundancy_evidence: Dict[str, Any] = {"measured": False}
    _op = str(getattr(generated, "op_type", "") or (item or {}).get("op_type") or "")
    if _op == "mutation" and parents:
        # Mutation had no Lean check at all: `judge_mutation` compares tactic
        # names in the proof text and never calls the verifier. This asks the
        # question that matters -- does the parent's statement already imply the
        # child -- and hands the answer to the judge as measurement.
        try:
            from src.certification.redundancy import check_mutation

            redundancy_evidence = await check_mutation(
                verify_lean_proof,
                generated.lean_header or THEOREM_CANONICAL_HEADER,
                generated.formal_statement or "",
                str((getattr(parents[0], "metadata", None) or {}).get("formal_statement") or ""),
                variant=str((item or {}).get("operator_variant") or ""),
                timeout=float(os.getenv("REDUNDANCY_TIMEOUT", "120")),
            )
            (item or {})["redundancy_evidence"] = redundancy_evidence
        except Exception as error:  # pragma: no cover
            redundancy_evidence = {"measured": False, "why": str(error)[:120]}
    if _op == "crossover":
        try:
            from src.certification.redundancy import check_parents

            pack = [
                {
                    "name": str(getattr(p, "id", "") or ""),
                    "statement": str((getattr(p, "metadata", None) or {}).get("formal_statement") or ""),
                }
                for p in (parents or [])
            ]
            if len([p for p in pack if p["statement"]]) >= 2:
                redundancy_evidence = await check_parents(
                    verify_lean_proof,
                    generated.lean_header or THEOREM_CANONICAL_HEADER,
                    generated.formal_statement or "",
                    pack,
                    mechanism=str(
                        ((item or {}).get("fusion_contract") or {}).get("fusion_mechanism") or ""
                    ),
                    timeout=float(os.getenv("REDUNDANCY_TIMEOUT", "120")),
                )
                (item or {})["redundancy_evidence"] = redundancy_evidence
        except Exception as error:  # pragma: no cover
            redundancy_evidence = {"measured": False, "why": str(error)[:120]}

    # A silent mutation is gated on equivalence instead of on novelty. Every
    # other variant is asked to be different enough from its parent, which is a
    # question no check can settle and is why it ended up with a model judge.
    # This one inverts it into a question Lean can settle: is the child exactly
    # equivalent, in both directions? One direction alone would admit a
    # weakening, which is a different and easier theorem, and difficulty
    # invariance is the only reason this operator is worth having.
    #
    # An unproven equivalence is not a weak row to be flagged. It is an
    # unsupported claim, and the row is discarded.
    # The two-way equivalence probe used to run here. It was demoted from a gate
    # to evidence on 2026-08-10 after it passed a child that had dropped a
    # conjunct and an unrelated true statement, and removed on 2026-08-11 after
    # its record was read: across all 14 silent rows ever produced it answered
    # "not equivalent" 14 times and "equivalent" never, while the judge kept 5
    # of those rows and argued equivalence in each one. What it measures is
    # whether the tactic ladder can close both directions, and the ladder mostly
    # cannot -- so it was a signal pointing the wrong way, which the judge had to
    # overrule every time.
    #
    # What settles this tier now: `hypothesis_preservation`, which is parsing,
    # and the silent judge, which is reading.
    silent_evidence: Dict[str, Any] = {"checked": False}
    if str(getattr(generated, "operator_variant", "") or (item or {}).get("operator_variant") or "") == "mutation_silent":
        parent_statement = str(
            (getattr(parent_input, "metadata", None) or {}).get("formal_statement") or ""
        )
        try:
            from src.certification.hypothesis_preservation import compare as compare_hypotheses

            silent_evidence = {
                "checked": True,
                "hypothesis_preservation": compare_hypotheses(
                    parent_statement, generated.formal_statement or ""
                ),
            }
        except Exception as error:  # pragma: no cover
            silent_evidence = {"checked": True,
                               "hypothesis_preservation": {"measured": False, "why": str(error)[:120]}}
        if item is not None:
            item["silent_evidence"] = silent_evidence

    # Identity is the one question left to a hash. Every other gate became the
    # judge's, because every other gate asked a question of degree; whether two
    # rows state the same theorem is not one. The check spans the whole corpus,
    # not this run's output file: the dedup that existed before caught a repeat
    # between generations of one seed group and missed the same statement
    # produced by another group or an earlier campaign, and eight such pairs are
    # already in the released ProofNet rows.
    #
    # It runs before the judge so a duplicate costs no model call.
    dedup_evidence: Dict[str, Any] = {"checked": False}
    if os.getenv("DEDUP_GATE", "1") == "1" and (generated.formal_statement or "").strip():
        index = _corpus_index()
        holder = index.holder(generated.formal_statement)
        # A row is not a duplicate of itself. The identifier is derived from the
        # statement, so a retry that lands on the same theorem gets the same id
        # and finds the entry its own first attempt registered: in one four-
        # generation run four rows were discarded that way, one of them the best
        # silent mutation the run produced.
        if holder and holder == generated.id:
            holder = None
        dedup_evidence = {
            "checked": True,
            "duplicate": bool(holder),
            "duplicate_of": holder or "",
            "corpus_size": len(index),
        }
        if not holder:
            index.add(generated.formal_statement, generated.id)

    # Redundancy evidence for the judge. Inline this runs the syntactic filter
    # and the tactic ladder only -- no prover -- because a parent that assumes
    # nothing and falls to `omega` is the case worth flagging in-flight, and the
    # prover retries that catch the rest belong in the offline scan where a
    # fifteen-minute worst case costs nothing. It never changes status: the
    # check has been rebuilt six times and its findings go to the judge as
    # measurement, not as a verdict.
    redundancy_evidence: Dict[str, Any] = {"measured": False}
    _op = str(getattr(generated, "op_type", "") or (item or {}).get("op_type") or "")
    if _op == "mutation" and parents:
        # Mutation had no Lean check at all: `judge_mutation` compares tactic
        # names in the proof text and never calls the verifier. This asks the
        # question that matters -- does the parent's statement already imply the
        # child -- and hands the answer to the judge as measurement.
        try:
            from src.certification.redundancy import check_mutation

            redundancy_evidence = await check_mutation(
                verify_lean_proof,
                generated.lean_header or THEOREM_CANONICAL_HEADER,
                generated.formal_statement or "",
                str((getattr(parents[0], "metadata", None) or {}).get("formal_statement") or ""),
                variant=str((item or {}).get("operator_variant") or ""),
                timeout=float(os.getenv("REDUNDANCY_TIMEOUT", "120")),
            )
            (item or {})["redundancy_evidence"] = redundancy_evidence
        except Exception as error:  # pragma: no cover
            redundancy_evidence = {"measured": False, "why": str(error)[:120]}
    if _op == "crossover":
        try:
            from src.certification.redundancy import check_parents

            pack = [
                {
                    "name": str(getattr(p, "id", "") or ""),
                    "statement": str((getattr(p, "metadata", None) or {}).get("formal_statement") or ""),
                }
                for p in (parents or [])
            ]
            if len([p for p in pack if p["statement"]]) >= 2:
                redundancy_evidence = await check_parents(
                    verify_lean_proof,
                    generated.lean_header or THEOREM_CANONICAL_HEADER,
                    generated.formal_statement or "",
                    pack,
                    mechanism=str(
                        ((item or {}).get("fusion_contract") or {}).get("fusion_mechanism") or ""
                    ),
                    timeout=float(os.getenv("REDUNDANCY_TIMEOUT", "120")),
                )
                (item or {})["redundancy_evidence"] = redundancy_evidence
        except Exception as error:  # pragma: no cover
            redundancy_evidence = {"measured": False, "why": str(error)[:120]}

    # A silent mutation is gated on equivalence instead of on novelty. Every
    # other variant is asked to be different enough from its parent, which is a
    # question no check can settle and is why it ended up with a model judge.
    # This one inverts it into a question Lean can settle: is the child exactly
    # equivalent, in both directions? One direction alone would admit a
    # weakening, which is a different and easier theorem, and difficulty
    # invariance is the only reason this operator is worth having.
    #
    # An unproven equivalence is not a weak row to be flagged. It is an
    # unsupported claim, and the row is discarded.
    # The quality review runs last, on rows that have already proved themselves
    # sound. It is the only gate whose verdict rests on meaning rather than
    # structure, and it is also the only one that costs a model call, so it is
    # spent on the smallest set: rows that compile, are not vacuous, and would
    # otherwise be released.
    judge_evidence = await _review_problem_quality(
        generated, list(parents or [parent_input]), item
    )

    # Close the loop: if this row is itself a retry, the judgment that triggered
    # it now has an answer. Recording the answer is what makes the log evidence
    # about correctives rather than a list of opinions — a brief followed four
    # times by a rejected retry is worth knowing before issuing it a fifth.
    if int((item or {}).get("retry_count") or 0) > 0:
        try:
            from src.certification.judge_memory import record_retry_outcome

            source = str((item or {}).get("parent_problem_id") or "") or str(
                getattr(generated, "source_problem_id", "") or ""
            )
            if source:
                record_retry_outcome(
                    source,
                    "accepted" if judge_evidence.get("verdict") != "reject" else
                    f"rejected again ({judge_evidence.get('failure') or 'same'})",
                )
        except Exception:
            pass
    return build_result(
        status="certified",
        proof_summary=proof_summary,
        alignment=alignment,
        vacuity_evidence=vacuity_evidence,
        hypothesis_evidence=hypothesis_evidence,
        judge_evidence=judge_evidence,
        silent_evidence=silent_evidence,
        dedup_evidence=dedup_evidence,
        redundancy_evidence=redundancy_evidence,
    )


def _with_slot_metadata(
    problem: CertificationInput,
    item: Dict[str, Any],
    *,
    generation_count: int,
) -> CertificationInput:
    parent_cards = list(item.get("parent_context_cards") or [_parent_context_card(problem)])
    target_style = _target_style_for_item(item, [problem])
    operator_card = _operator_card({**item, "target_style": target_style, "parent_context_cards": parent_cards})
    return CertificationInput(
        id=problem.id,
        statement=problem.statement,
        answer=problem.answer,
        metadata={
            **problem.metadata,
            "slot": int(item["slot"]),
            "op_type": item["op_type"],
            "problem_style": _problem_style(problem),
            "target_style": target_style,
            "certification_route": _certification_route_for_style(target_style),
            "operator_variant": item.get("operator_variant") or _default_operator_variant(
                item["op_type"], target_family=item.get("target_family", ""), parent_family=_problem_family(problem)
            ),
            "parent_ids": list(item.get("parent_ids") or []),
            "parent_context_cards": parent_cards,
            "parents": [
                {
                    "id": problem.id,
                    "family": _problem_family(problem) or "unsupported",
                    "problem_style": _problem_style(problem),
                    "statement": problem.statement,
                    "answer": problem.answer,
                    "required_contribution": dict(item.get("parent_contributions") or {}).get(problem.id, ""),
                    "proof_context": _parent_proof_context(problem),
                }
            ],
            "variation_axis": item.get("variation_axis", ""),
            "reasoning_goal": item.get("reasoning_goal", ""),
            "target_family": item.get("target_family", ""),
            "required_params": dict(item.get("required_params") or {}),
            "composition_pattern": item.get("composition_pattern", ""),
            "parent_contributions": dict(item.get("parent_contributions") or {}),
            "avoid_patterns": list(item.get("avoid_patterns") or []),
            "quality_target": item.get("quality_target", ""),
            "operator_card": operator_card,
            "memory_delta_contract": dict(item.get("memory_delta_contract") or {}),
            "operator_goal": item.get("operator_goal")
            or item.get("reasoning_goal")
            or item.get("variation_axis", ""),
            "required_checkpoints": list(item.get("required_checkpoints") or []),
            "avoid_signatures": list(item.get("avoid_signatures") or []),
            "fusion_contract": dict(item.get("fusion_contract") or {}),
            "planner_source": item.get("planner_source", ""),
            "generation": generation_count,
            "retry_count": int(item.get("retry_count") or 0),
            "retry_feedback": item.get("retry_feedback", ""),
            "attempt_history": list(item.get("attempt_history") or []),
            "source_kind": item.get("source_kind")
            or ("survivor" if item.get("op_type") == "survivor" else "generated"),
        },
    )


def _crossover_parent_input(
    parents: List[CertificationInput],
    item: Dict[str, Any],
    *,
    generation_count: int,
) -> CertificationInput:
    joined_id = "__x__".join(parent.id for parent in parents)
    parent_cards = list(item.get("parent_context_cards") or _parent_context_cards(parents))
    target_style = _target_style_for_item(item, parents)
    operator_card = _operator_card({**item, "target_style": target_style, "parent_context_cards": parent_cards})
    structured_parents = [
        {
            "id": parent.id,
            "family": _problem_family(parent) or "unsupported",
            "problem_style": _problem_style(parent),
            "statement": parent.statement,
            "answer": parent.answer,
            "required_contribution": dict(item.get("parent_contributions") or {}).get(parent.id, ""),
            "proof_context": _parent_proof_context(parent),
        }
        for parent in parents
    ]
    return CertificationInput(
        id=joined_id,
        statement=(
            "Generate one Lean-certifiable harder child from these two parents, "
            "using a single supported arithmetic family.\n\n"
            f"Variation axis: {item.get('variation_axis', '')}\n\n"
            f"Reasoning goal: {item.get('reasoning_goal', '')}\n\n"
            f"Parents: {json.dumps(structured_parents, ensure_ascii=False)}"
            f"\n\nComposition pattern: {item.get('composition_pattern', '')}"
            f"\nParent contributions: "
            f"{json.dumps(item.get('parent_contributions') or {}, ensure_ascii=False)}"
            f"\nFusion contract: "
            f"{json.dumps(item.get('fusion_contract') or {}, ensure_ascii=False)}"
        ),
        answer=str(parents[0].answer),
        metadata={
            "slot": int(item["slot"]),
            "op_type": "crossover",
            "problem_style": "theorem_proof" if target_style == "theorem_proof" else "numeric_answer",
            "target_style": target_style,
            "certification_route": _certification_route_for_style(target_style),
            "operator_variant": item.get("operator_variant") or "crossover_hard",
            "parent_ids": [parent.id for parent in parents],
            "parent_context_cards": parent_cards,
            "parents": structured_parents,
            "variation_axis": item.get("variation_axis", ""),
            "reasoning_goal": item.get("reasoning_goal", ""),
            "target_family": item.get("target_family", ""),
            "required_params": dict(item.get("required_params") or {}),
            "composition_pattern": item.get("composition_pattern", ""),
            "parent_contributions": dict(item.get("parent_contributions") or {}),
            "avoid_patterns": list(item.get("avoid_patterns") or []),
            "quality_target": item.get("quality_target", ""),
            "operator_card": operator_card,
            "memory_delta_contract": dict(item.get("memory_delta_contract") or {}),
            "operator_goal": item.get("operator_goal")
            or item.get("reasoning_goal")
            or item.get("variation_axis", ""),
            "required_checkpoints": list(item.get("required_checkpoints") or []),
            "avoid_signatures": list(item.get("avoid_signatures") or []),
            "fusion_contract": dict(item.get("fusion_contract") or {}),
            "planner_source": item.get("planner_source", ""),
            "generation": generation_count,
            "retry_count": int(item.get("retry_count") or 0),
            "retry_feedback": item.get("retry_feedback", ""),
            "attempt_history": list(item.get("attempt_history") or []),
            "source_kind": item.get("source_kind") or "generated",
        },
    )


def _result_to_pool_problem(result: CertificationResult) -> Dict[str, Any]:
    return {
        "id": result.problem_id,
        "release_id": result.release_id,
        "statement": result.statement,
        "statement_sha256": result.statement_sha256,
        "answer": result.answer,
        "answer_sha256": result.answer_sha256,
        "solution": result.solution,
        "verification_code": result.verification_code,
        "formal_statement": result.formal_statement,
        "formal_statement_sha256": result.formal_statement_sha256,
        "lean_header": result.lean_header,
        "formal_status": result.formal_status,
        "benchmark": result.benchmark,
        "operation": result.operation,
        "family": result.family,
        "generation": result.generation,
        "slot": result.slot,
        "op_type": result.op_type,
        "operator_variant": result.operator_variant,
        "parent_ids": result.parent_ids,
        "ancestor_ids": result.ancestor_ids,
        "source_run": result.source_run,
        "source_file": result.source_file,
        "source_slot": result.source_slot,
        "target_family": result.target_family,
        "required_params": result.required_params,
        "generated_params": result.generated_params,
        "composition_pattern": result.composition_pattern,
        "parent_contributions": result.parent_contributions,
        "quality_target": result.quality_target,
        "reasoning_pattern": result.reasoning_pattern,
        "solution_skeleton": result.solution_skeleton,
        "projected_params": result.projected_params,
        "projection_check": result.projection_check,
        "semantic_parent_contribution": result.semantic_parent_contribution,
        "interestingness_features": result.interestingness_features,
        "quality_evidence": _compact_quality_evidence_for_pool(dict(result.quality_evidence or {})),
        "solution_verify_passed": result.solution_verify_passed,
        "solution_verify_flags": result.solution_verify_flags,
        "quality_verdict": result.quality_verdict,
        "quality_flags": result.quality_flags,
        "interestingness_score": result.interestingness_score,
        "feedback_for_next_generation": result.feedback_for_next_generation,
        "axis_applied": result.axis_applied,
        "axis_aligned": result.axis_aligned,
        "parent_eligible": result.parent_eligible,
        "selection_reason": result.selection_reason,
        "canonical_signature": result.canonical_signature,
        "retry_count": result.retry_count,
        "retry_reasons": result.retry_reasons,
        "quality_retry_count": result.quality_retry_count,
        "retry_exhausted": result.retry_exhausted,
        "replan_count": result.replan_count,
        "replan_reason": result.replan_reason,
        "replan_source": result.replan_source,
        "discarded_operator_card": _compact_operator_card_for_pool(dict(result.discarded_operator_card or {})),
        "survival_status": result.survival_status,
        "fallback_survivor_duplicate": result.fallback_survivor_duplicate,
        "failure_signature": result.failure_signature,
        "source_kind": result.source_kind,
        "problem_style": result.problem_style,
        "target_style": result.target_style,
        "certification_route": result.certification_route,
        "parent_context_cards": [],
        "operator_card": _compact_operator_card_for_pool(dict(result.operator_card or {})),
        "slot_outcome": result.slot_outcome,
        "proof_plan": result.proof_plan,
        "proof_obligations": result.proof_obligations,
        "proof_verify_summary": result.proof_verify_summary,
        "status": result.status,
        "lean_level": result.lean_level,
        "lean_code": result.lean_code,
        "difficulty": result.difficulty,
        "difficulty_label": result.difficulty_label,
        "input_metadata": {},
    }


def _theorem_survivor_result(
    parent: CertificationInput,
    item: Dict[str, Any],
    *,
    generation_count: int,
) -> CertificationResult:
    """Carry theorem parents without projecting them through numeric templates."""
    metadata = dict(parent.metadata or {})
    proof_context = _parent_proof_context(parent)
    proof_available = bool(proof_context.get("proof_body_available"))
    lean_code = str(proof_context.get("lean_code") or "")
    if lean_code == "not_available":
        lean_code = ""
    status = "survivor" if proof_available else "proof_failed"
    summary = (
        "theorem survivor carried forward unchanged"
        if proof_available
        else "theorem survivor lacks a complete proof body"
    )
    return CertificationResult(
        problem_id=parent.id,
        release_id=metadata.get("release_id"),
        source_problem_id=parent.id,
        generation=generation_count,
        slot=int(item["slot"]),
        operation="survivor",
        op_type="survivor",
        operator_variant=item.get("operator_variant") or "survivor",
        parent_ids=list(item.get("parent_ids") or [parent.id]),
        ancestor_ids=metadata.get("ancestor_ids"),
        family=metadata.get("family") or _problem_family(parent) or "theorem_proof",
        target_family="theorem_proof",
        status=status,
        lean_level=3 if proof_available else 0,
        lean_code=lean_code or metadata.get("lean_code"),
        anti_stub_passed=proof_available,
        aligned=proof_available,
        formal_status="survivor" if proof_available else "proof_failed",
        statement=parent.statement,
        answer=parent.answer,
        solution=metadata.get("solution"),
        verification_code=metadata.get("verification_code"),
        formal_statement=metadata.get("formal_statement"),
        formal_statement_sha256=metadata.get("formal_statement_sha256"),
        lean_header=metadata.get("lean_header"),
        benchmark=metadata.get("benchmark"),
        quality_verdict="acceptable" if proof_available else "weak",
        quality_flags=[] if proof_available else ["missing_parent_proof_body"],
        interestingness_score=float(metadata.get("interestingness_score") or 0.5),
        quality_evidence=dict(metadata.get("quality_evidence") or {}),
        problem_style="theorem_proof",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        parent_context_cards=list(item.get("parent_context_cards") or [_parent_context_card(parent)]),
        operator_card=_operator_card({**item, "target_style": "theorem_proof"}),
        slot_outcome=status,
        proof_plan=metadata.get("proof_plan") or metadata.get("solution"),
        proof_obligations=list(metadata.get("proof_obligations") or []),
        proof_verify_summary=metadata.get("proof_verify_summary") or summary,
        source_kind="survivor",
        parent_eligible=proof_available,
        selection_reason="survivor_carried" if proof_available else "missing_parent_proof_body",
        error=None if proof_available else summary,
        input_metadata=metadata,
    )


def _aggregate_llm_available() -> bool:
    return (
        os.getenv("GENERATION_PROVIDER", "").lower() == "codex_cli"
        or bool(os.getenv("OPENAI_API_KEY"))
        or bool(os.getenv("OPENROUTER_API_KEY"))
    )


def _parent_canonical_signature(parent: CertificationInput, family: str) -> str:
    metadata = parent.metadata or {}
    if _problem_style(parent) == "theorem_proof" or family == "theorem_proof":
        formal_surface = _canonical_theorem_surface(
            metadata.get("formal_statement") or metadata.get("lean_code") or parent.statement
        )
        if formal_surface:
            return json.dumps(
                {"family": "theorem_proof", "formal": formal_surface},
                sort_keys=True,
                ensure_ascii=False,
            )
    return json.dumps(
        {
            "family": family,
            "params": _pool_params(parent) or {"statement": parent.statement, "answer": parent.answer},
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _hash_preview(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "not_available"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _aggregate_selection_policy_surface() -> Dict[str, Any]:
    return {
        "final_authority": "aggregate_orchestrator_llm",
        "nonnegotiable_invariants": [
            "unknown_id",
            "not_certified_or_survivor",
            "exact_canonical_duplicate",
            "pool_size_exceeded",
            "backfill_ineligible",
        ],
        "orchestrator_judgment_factors": [
            "near_duplicate_memory",
            "structural_overlap",
            "repeated_reasoning_signature",
            "signature_group_cap",
            "family_cap",
            "lineage_cap",
            "weak_entropy_support",
        ],
        "family_cap": FAMILY_CAP,
        "theorem_lineage_cap": THEOREM_LINEAGE_CAP,
        "backfill_policy": "Use eligible previous seeds only when they are better next-generation parents than leaving the pool partial.",
    }


def _aggregate_candidate_mechanical_eligibility(
    result: CertificationResult,
    *,
    generation_count: int,
) -> Dict[str, Any]:
    family = result.family or result.target_family or detect_family(result.statement or "") or "unknown"
    evidence = dict(result.quality_evidence or {})
    novelty_memory = dict(evidence.get("novelty_memory") or {})
    blocking_invariants: List[str] = []
    selection_risks: List[str] = []
    if result.status not in {"certified", "survivor"}:
        blocking_invariants.append("not_certified_or_survivor")
    # Risks the judge cannot see, and only those. A risk derived from the
    # heuristic flag registry would reintroduce the filter that was just removed
    # from the verdict: the orchestrator is told not to select risky candidates,
    # so a `weak_quality` risk computed from those flags kept the same rows out
    # of the pool by a longer route.
    judge = dict(evidence.get("judge") or {})
    if "near_duplicate" in set(result.quality_flags or []):
        selection_risks.append("near_duplicate_memory")
    if novelty_memory.get("verdict") == "structural_overlap":
        selection_risks.append("structural_overlap")
    reasoning_signature = str(evidence.get("reasoning_signature") or "")
    if reasoning_signature and result.op_type != "survivor":
        selection_risks.append("repeated_reasoning_signature_sensitive")
    signature_group = str(evidence.get("signature_group") or "")
    if signature_group:
        selection_risks.append("signature_group_cap_sensitive")
    if family != "theorem_proof":
        selection_risks.append("family_cap_sensitive")
    if family == "theorem_proof" and result.op_type not in {"survivor", "fallback_survivor"}:
        selection_risks.append("lineage_cap_sensitive")
    # Weakness is now the judge's word. When it ran and kept the row, no
    # weakness risk is raised; when it ran and rejected, the row never reaches
    # here, because a rejection sets status `judge_rejected`. The heuristic
    # branch survives only for rows the judge did not see — the numeric route,
    # and any run with PROBLEM_JUDGE off.
    if not judge.get("ran"):
        if (
            result.quality_verdict == "weak"
            and _entropy_increase(result)
            and generation_count > 3
            and not _accepted_proxy_pass(result)
        ):
            selection_risks.append("weak_entropy_support_generation_cap")
        if result.quality_verdict == "weak" and (result.retry_exhausted or not _entropy_increase(result)):
            selection_risks.append(
                "weak_quality_after_retries"
                if result.retry_exhausted and result.quality_retry_count > 0
                else "weak_quality"
            )
    elif str(judge.get("quality") or "") == "weak":
        selection_risks.append("judge_quality_weak")
    return {
        "eligible": not blocking_invariants,
        "blocking_invariants": blocking_invariants,
        "selection_risks": sorted(set(selection_risks)),
        "cap_sensitive_keys": {
            "canonical_signature_hash": _hash_preview(result.canonical_signature or _canonical_signature(result)),
            "reasoning_signature": _prompt_text(evidence.get("reasoning_signature"), limit=220),
            "signature_group": _prompt_text(evidence.get("signature_group"), limit=120),
            "family": family,
            "lineage_roots": _result_root_lineages(result)[:4],
        },
        "final_gate_note": "Only blocking_invariants are code-enforced. Selection risks are final orchestrator judgment inputs.",
    }


def _aggregate_candidate_card(result: CertificationResult, *, generation_count: int) -> Dict[str, Any]:
    evidence = dict(result.quality_evidence or {})
    accepted_proxy = dict(evidence.get("accepted_proxy") or {})
    novelty_memory = dict(evidence.get("novelty_memory") or {})
    return {
        "candidate_id": result.problem_id,
        "source_problem_id": result.source_problem_id,
        "slot": result.slot,
        "op_type": result.op_type,
        "source_kind": result.source_kind or ("survivor" if result.op_type == "survivor" else "generated"),
        "status": result.status,
        "quality_verdict": result.quality_verdict,
        "quality_flags": list(result.quality_flags or [])[:8],
        "accepted_proxy": accepted_proxy,
        "novelty_verdict": novelty_memory.get("verdict"),
        "reasoning_signature": evidence.get("reasoning_signature"),
        "signature_group": evidence.get("signature_group"),
        "crossover_kind": evidence.get("crossover_kind"),
        "parent_ids": list(result.parent_ids or [])[:4],
        "family": result.family or result.target_family,
        "target_style": result.target_style or result.problem_style,
        "interestingness_score": result.interestingness_score,
        "entropy_direction": evidence.get("entropy_direction"),
        "selection_priority": list(_selection_priority(result)),
        "mechanical_eligibility": _aggregate_candidate_mechanical_eligibility(
            result,
            generation_count=generation_count,
        ),
        "statement_preview": _prompt_text(result.statement, limit=220),
        "formal_statement_preview": _prompt_text(result.formal_statement, limit=220),
        "feedback_for_next_generation": _prompt_text(result.feedback_for_next_generation, limit=260),
    }


def _aggregate_seed_card(parent: CertificationInput) -> Dict[str, Any]:
    metadata = dict(parent.metadata or {})
    evidence = dict(metadata.get("quality_evidence") or {})
    proof_context = _parent_proof_context(parent)
    family = _problem_family(parent) or metadata.get("family") or "unknown"
    return {
        "seed_id": parent.id,
        "family": family,
        "problem_style": _problem_style(parent),
        "status": metadata.get("status") or "certified",
        "quality_verdict": metadata.get("quality_verdict") or "acceptable",
        "quality_flags": list(metadata.get("quality_flags") or [])[:8],
        "proof_body_available": bool(proof_context.get("proof_body_available")),
        "reasoning_signature": evidence.get("reasoning_signature"),
        "signature_group": evidence.get("signature_group"),
        "crossover_kind": evidence.get("crossover_kind"),
        "interestingness_score": metadata.get("interestingness_score"),
        "prior_selection_reason": metadata.get("selection_reason"),
        "backfill_source_kind": metadata.get("backfill_source_kind") or "current_generation",
        "backfill_generation_distance": metadata.get("backfill_generation_distance", 1),
        "lineage_root": _root_lineage_id(parent.id),
        "mechanical_eligibility": {
            "eligible": True,
            "blocking_invariants": [],
            "selection_risks": [
                risk
                for risk in [
                    "repeated_reasoning_signature_sensitive"
                    if evidence.get("reasoning_signature")
                    else "",
                    "signature_group_cap_sensitive"
                    if evidence.get("signature_group")
                    else "",
                    "family_cap_sensitive" if family != "theorem_proof" else "lineage_cap_sensitive",
                ]
                if risk
            ],
            "cap_sensitive_keys": {
                "canonical_signature_hash": _hash_preview(_parent_canonical_signature(parent, family)),
                "reasoning_signature": _prompt_text(evidence.get("reasoning_signature"), limit=220),
                "signature_group": _prompt_text(evidence.get("signature_group"), limit=120),
                "family": family,
                "lineage_roots": [_root_lineage_id(parent.id)],
            },
            "final_gate_note": "This seed passed basic backfill eligibility. Selection risks are final orchestrator judgment inputs.",
        },
        "statement_preview": _prompt_text(parent.statement, limit=220),
        "formal_statement_preview": _prompt_text(
            metadata.get("formal_statement") or metadata.get("lean_code"), limit=220
        ),
    }


def _backfill_seed_from_row(row: Dict[str, Any], *, generation_distance: int) -> Optional[CertificationInput]:
    status = str(row.get("status") or "")
    if status not in {"certified", "survivor"}:
        return None
    if row.get("parent_eligible") is False:
        return None
    problem_id = str(row.get("id") or row.get("problem_id") or "").strip()
    statement = str(row.get("statement") or "").strip()
    if not problem_id or not statement:
        return None
    metadata = {
        k: v for k, v in dict(row).items() if k not in {"id", "statement", "answer"}
    }
    metadata.setdefault("status", status)
    metadata.setdefault("selection_reason", row.get("selection_reason") or "archived_parent")
    metadata["backfill_source_kind"] = "run_archive"
    metadata["backfill_generation_distance"] = generation_distance
    return CertificationInput(
        id=problem_id,
        statement=statement,
        answer=str(row.get("answer") or ""),
        metadata=metadata,
    )


def _build_backfill_seed_archive(
    current_generation: List[CertificationInput],
    *,
    output_path: Optional[Path],
    generation_count: int,
    max_history_generations: int = 2,
) -> List[CertificationInput]:
    archive: List[CertificationInput] = []
    seen_ids: set[str] = set()

    def add_seed(seed: CertificationInput, *, source_kind: str, generation_distance: int) -> None:
        if seed.id in seen_ids:
            return
        metadata = dict(seed.metadata or {})
        metadata.setdefault("backfill_source_kind", source_kind)
        metadata.setdefault("backfill_generation_distance", generation_distance)
        enriched = seed.model_copy(update={"metadata": metadata})
        if _parent_is_backfill_eligible(enriched):
            archive.append(enriched)
            seen_ids.add(enriched.id)

    for seed in current_generation:
        add_seed(seed, source_kind="current_generation", generation_distance=1)

    if output_path:
        min_generation = max(0, int(generation_count) - int(max_history_generations))
        rows = _read_jsonl_dict_rows(Path(output_path))
        for row in reversed(rows):
            try:
                row_generation = int(row.get("generation") or 0)
            except (TypeError, ValueError):
                row_generation = 0
            if row_generation >= int(generation_count) or row_generation < min_generation:
                continue
            generation_distance = max(1, int(generation_count) - row_generation)
            seed = _backfill_seed_from_row(row, generation_distance=generation_distance)
            if seed is not None:
                add_seed(seed, source_kind="run_archive", generation_distance=generation_distance)
    return archive


def _aggregate_selector_payload(
    *,
    results: List[CertificationResult],
    backfill_seed_archive: List[CertificationInput],
    pool_size: int,
    generation_count: int,
    target_accepted: int,
    generation_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generated = [result for result in results if _is_generated_result(result)]
    return {
        "pool_size": pool_size,
        "generation": generation_count,
        "target_accepted_per_generation": target_accepted,
        "selection_policy_surface": _aggregate_selection_policy_surface(),
        "current_candidates": [
            _aggregate_candidate_card(result, generation_count=generation_count) for result in results
        ],
        "previous_certified_seed_cards": [
            _aggregate_seed_card(parent)
            for parent in backfill_seed_archive
            if _parent_is_backfill_eligible(parent)
        ],
        "yield_funnel": _yield_funnel(results),
        "accepted_proxy_count": sum(1 for result in generated if _accepted_proxy_pass(result)),
        "accepted_grade_proxy_count": sum(1 for result in generated if _accepted_grade_proxy_pass(result)),
        "previous_generation_feedback": _generation_feedback_trace_manifest(
            dict(generation_feedback or {})
        )
        if generation_feedback
        else {},
        "instructions": [
            "Make final next-pool decisions. Code only invalidates nonnegotiable invariant violations.",
            "Do not select items with mechanical_eligibility.eligible=false.",
            "Treat selection_risks as judgment evidence, not automatic reject rules.",
            "Use previous seeds as backfill only when they are useful positive parents for the next generation.",
            "Prefer semantic novelty, accepted_proxy pass, proof availability, and useful family/lineage diversity.",
        ],
    }


def _aggregate_response_format() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "pool_decisions",
            "generation_survival_status",
            "pool_strategy_rationale",
            "warnings",
        ],
        "properties": {
            "pool_decisions": {
                "type": "array",
                "description": (
                    "Ordered final orchestrator decisions. Use select for current_candidates, backfill for previous_certified_seed_cards, "
                    "and reject for notable candidates or seeds that should not become next-generation parents. "
                    "Put select/backfill decisions first; include reject decisions only for notable evidence or risks."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "source_kind", "decision", "selection_reason", "expected_next_gen_role", "rationale"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "candidate_id from current_candidates or seed_id from previous_certified_seed_cards.",
                        },
                        "source_kind": {
                            "type": "string",
                            "enum": ["current_candidate", "previous_seed"],
                            "description": "Whether id refers to a current candidate or previous certified seed.",
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["select", "backfill", "reject"],
                            "description": (
                                "Final orchestrator decision for this id. Use select/backfill only when "
                                "mechanical_eligibility.eligible=true."
                            ),
                        },
                        "selection_reason": {
                            "type": "string",
                            "description": (
                                "Stable compact snake_case reason to write into selection_reason when selected/backfilled, "
                                "or to explain rejection when decision=reject. Do not emit invalidated_by_code:*; "
                                "that prefix is reserved for code invariant overrides."
                            ),
                        },
                        "expected_next_gen_role": {
                            "type": "string",
                            "description": "Expected role if selected, such as frontier_parent, scaffold_support, diversity_anchor, or none.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Short evidence-grounded rationale using quality, novelty, diversity, or invariant/risk evidence.",
                        },
                    },
                },
            },
            "generation_survival_status": {
                "type": "string",
                "enum": ["complete", "complete_with_backfill", "partial"],
                "description": "Final orchestrator survival status after its selected/backfill decisions.",
            },
            "pool_strategy_rationale": {
                "type": "string",
                "description": "One compact explanation of the pool composition strategy and diversity tradeoff.",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential insufficiency, ambiguity, fallback, or trace-worthy risks for downstream diagnostics.",
            },
        },
    }
    return _schema_response_format("aggregate_next_pool_selection", schema)


def _aggregate_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    # The project's working name appears in this string because the campaigns
    # ran with it. It is deliberately not renamed with the rest: this text was
    # sent to the model, so editing it would make the released code differ from
    # the code that produced the corpus, for a cosmetic gain.
    system = (
        "You are the EntropyMaLean next-pool aggregate orchestrator. "
        "Make final next-pool selection decisions from verified compact summaries. "
        "Return JSON only. Code only invalidates nonnegotiable invariant violations."
    )
    user = f"""
Task:
- HARD: Make ordered pool_decisions. Use decision=select for current_candidates and decision=backfill for previous_certified_seed_cards.
- HARD: Do not select/backfill items with mechanical_eligibility.eligible=false.
- IMPORTANT: nonnegotiable_invariants are enforced by code; orchestrator_judgment_factors are your final judgment inputs.
- IMPORTANT: If useful candidates/seeds are insufficient, return generation_survival_status=partial rather than duplicate padding.
- Prefer semantic novelty, accepted_proxy pass, proof availability, and useful family/lineage diversity.
- Use only the compact evidence below; do not include or request raw Lean/proof text.

Pool constraints:
- pool_size={payload.get("pool_size")}
- target_accepted_per_generation={payload.get("target_accepted_per_generation")}

AggregateSelectionPayload:
{json.dumps(payload, ensure_ascii=False, indent=2)[:12000]}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _aggregate_payload_sha256(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _aggregate_payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(payload.get("current_candidates") or [])
    seeds = list(payload.get("previous_certified_seed_cards") or [])
    blocked = Counter(
        reason
        for card in candidates
        for reason in list((card.get("mechanical_eligibility") or {}).get("blocking_invariants") or [])
    )
    risks = Counter(
        risk
        for card in candidates
        for risk in list((card.get("mechanical_eligibility") or {}).get("selection_risks") or [])
    )
    eligible_candidates = sum(
        1 for card in candidates if bool((card.get("mechanical_eligibility") or {}).get("eligible"))
    )
    return {
        "pool_size": payload.get("pool_size"),
        "generation": payload.get("generation"),
        "candidate_count": len(candidates),
        "eligible_candidate_count": eligible_candidates,
        "previous_seed_count": len(seeds),
        "blocking_invariant_counts": dict(blocked),
        "selection_risk_counts": dict(risks),
        "yield_funnel": dict(payload.get("yield_funnel") or {}),
    }


def llm_select_next_pool(
    payload: Dict[str, Any],
    *,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    config = default_generation_config(model=generation_model, temperature=generation_temperature)
    content = _chat_completion_text_sync(
        model=orchestrator_config(config).model,
        messages=_aggregate_messages(payload),
        temperature=0.1,
        response_format=_aggregate_response_format(),
        timeout_seconds=float(os.getenv("AGGREGATE_LLM_TIMEOUT", os.getenv("GENERATION_LLM_TIMEOUT", "180"))),
    )
    raw = _parse_json_object(content)
    return {"selector_source": "aggregate_orchestrator_llm", **raw}


def _deterministic_aggregate_selection(
    payload: Dict[str, Any],
    results: List[CertificationResult],
    backfill_seed_archive: List[CertificationInput],
) -> Dict[str, Any]:
    decisions: List[Dict[str, Any]] = []
    selected_signatures: set[str] = set()
    selected_reasoning_signatures: set[str] = set()
    signature_group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    root_lineage_counts: Counter[str] = Counter()
    used_ids: set[str] = set()
    pool_size = int(payload.get("pool_size") or POOL_SIZE)
    generation_count = int(payload.get("generation") or 0)
    selected_count = 0
    backfill_count = 0
    for result in sorted(results, key=_selection_priority):
        reason = (
            _result_rejection_reason(
                result,
                selected_signatures=selected_signatures,
                selected_reasoning_signatures=selected_reasoning_signatures,
                signature_group_counts=signature_group_counts,
                family_counts=family_counts,
                root_lineage_counts=root_lineage_counts,
                generation_count=generation_count,
                recommended=True,
            )
            if selected_count < pool_size
            else "not_recommended_by_orchestrator"
        )
        if reason == "selected":
            selection_reason = (
                "selected_entropy_increase_support"
                if result.quality_verdict == "weak" and _entropy_increase(result)
                else "selected_for_next_pool"
            )
            family = result.family or result.target_family or detect_family(result.statement or "") or "unknown"
            evidence = dict(result.quality_evidence or {})
            _record_selected_context(
                signature=result.canonical_signature or _canonical_signature(result),
                family=family,
                reasoning_signature=str(evidence.get("reasoning_signature") or ""),
                signature_group=str(evidence.get("signature_group") or ""),
                roots=_result_root_lineages(result),
                problem_id=result.problem_id,
                selected_signatures=selected_signatures,
                selected_reasoning_signatures=selected_reasoning_signatures,
                signature_group_counts=signature_group_counts,
                family_counts=family_counts,
                root_lineage_counts=root_lineage_counts,
                used_ids=used_ids,
            )
            selected_count += 1
            decision = "select"
        else:
            selection_reason = reason
            decision = "reject"
        decisions.append(
            {
                "id": result.problem_id,
                "source_kind": "current_candidate",
                "decision": decision,
                "selection_reason": selection_reason,
                "expected_next_gen_role": "frontier_parent" if decision == "select" else "none",
                "rationale": "Fallback deterministic ordering by certification, proxy quality, novelty, and diversity evidence.",
            }
        )
    for parent in backfill_seed_archive:
        if selected_count >= pool_size:
            break
        reason = _backfill_rejection_reason(
            parent,
            selected_signatures=selected_signatures,
            selected_reasoning_signatures=selected_reasoning_signatures,
            signature_group_counts=signature_group_counts,
            family_counts=family_counts,
            root_lineage_counts=root_lineage_counts,
            used_ids=used_ids,
        )
        if reason != "selected":
            continue
        family = _problem_family(parent) or "unknown"
        evidence = dict((parent.metadata or {}).get("quality_evidence") or {})
        _record_selected_context(
            signature=_parent_canonical_signature(parent, family),
            family=family,
            reasoning_signature=str(evidence.get("reasoning_signature") or ""),
            signature_group=str(evidence.get("signature_group") or ""),
            roots=[_root_lineage_id(parent.id)],
            problem_id=parent.id,
            selected_signatures=selected_signatures,
            selected_reasoning_signatures=selected_reasoning_signatures,
            signature_group_counts=signature_group_counts,
            family_counts=family_counts,
            root_lineage_counts=root_lineage_counts,
            used_ids=used_ids,
        )
        selected_count += 1
        backfill_count += 1
        decisions.append(
            {
                "id": parent.id,
                "source_kind": "previous_seed",
                "decision": "backfill",
                "selection_reason": "orchestrator_backfill",
                "expected_next_gen_role": "diversity_anchor",
                "rationale": "Fallback eligible previous seed for pool completion.",
            }
        )
    survival_status = (
        "complete_with_backfill"
        if selected_count == pool_size and backfill_count
        else ("complete" if selected_count == pool_size else "partial")
    )
    return {
        "selector_source": "deterministic_aggregate_fallback",
        "pool_decisions": decisions,
        "generation_survival_status": survival_status,
        "pool_strategy_rationale": "Fallback deterministic ordering by certification, proxy quality, novelty, and diversity caps.",
        "warnings": list(payload.get("selector_warnings") or []),
    }


def _decision_item(
    *,
    item_id: Any,
    source_kind: str,
    decision: str,
    selection_reason: str = "",
    expected_next_gen_role: str = "",
    rationale: str = "",
) -> Dict[str, str]:
    return {
        "id": str(item_id or "").strip(),
        "source_kind": source_kind if source_kind in {"current_candidate", "previous_seed"} else "current_candidate",
        "decision": decision if decision in {"select", "backfill", "reject"} else "reject",
        "selection_reason": _prompt_text(selection_reason or decision, limit=160),
        "expected_next_gen_role": _prompt_text(expected_next_gen_role or ("none" if decision == "reject" else "frontier_parent"), limit=120),
        "rationale": _prompt_text(rationale or selection_reason or decision, limit=500),
    }


def _normalize_aggregate_selection(raw: Dict[str, Any]) -> Dict[str, Any]:
    pool_decisions = [
        _decision_item(
            item_id=item.get("id") or item.get("candidate_id") or item.get("seed_id"),
            source_kind=str(item.get("source_kind") or "current_candidate"),
            decision=str(item.get("decision") or "reject"),
            selection_reason=str(item.get("selection_reason") or item.get("reason") or ""),
            expected_next_gen_role=str(item.get("expected_next_gen_role") or ""),
            rationale=str(item.get("rationale") or item.get("reason") or ""),
        )
        for item in list(raw.get("pool_decisions") or [])
        if isinstance(item, dict)
    ]
    if not pool_decisions:
        pool_decisions.extend(
            _decision_item(
                item_id=item,
                source_kind="current_candidate",
                decision="select",
                selection_reason="selected_for_next_pool",
                expected_next_gen_role="frontier_parent",
                rationale="Legacy selected_candidate_ids response.",
            )
            for item in list(raw.get("selected_candidate_ids") or [])
            if str(item).strip()
        )
        pool_decisions.extend(
            _decision_item(
                item_id=item,
                source_kind="previous_seed",
                decision="backfill",
                selection_reason="orchestrator_backfill",
                expected_next_gen_role="diversity_anchor",
                rationale="Legacy backfill_seed_ids response.",
            )
            for item in list(raw.get("backfill_seed_ids") or [])
            if str(item).strip()
        )
        pool_decisions.extend(
            _decision_item(
                item_id=item.get("candidate_id") or item.get("seed_id"),
                source_kind="current_candidate",
                decision="reject",
                selection_reason=str(item.get("reason") or "orchestrator_reject"),
                expected_next_gen_role="none",
                rationale=str(item.get("reason") or "Legacy rejected_candidate_rationales response."),
            )
            for item in list(raw.get("rejected_candidate_rationales") or [])
            if isinstance(item, dict)
        )
    survival_status = str(raw.get("generation_survival_status") or "").strip()
    if survival_status not in {"complete", "complete_with_backfill", "partial"}:
        survival_status = ""
    return {
        "selector_source": str(raw.get("selector_source") or "aggregate_selector"),
        "pool_decisions": [item for item in pool_decisions if item["id"]],
        "generation_survival_status": survival_status,
        "pool_strategy_rationale": _prompt_text(raw.get("pool_strategy_rationale"), limit=1000),
        "warnings": [str(item) for item in list(raw.get("warnings") or [])[:10]],
    }


def _result_candidate_ids(result: CertificationResult) -> List[str]:
    return [
        value
        for value in [
            result.problem_id,
            result.source_problem_id,
            f"slot:{result.slot}",
        ]
        if value not in (None, "")
    ]


def _append_unique_ids(base: List[str], extra: List[str]) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for value in base + extra:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            merged.append(text)
    return merged


def _candidate_invariant_violation(
    result: CertificationResult,
    *,
    selected_signatures: set[str],
) -> Optional[str]:
    if result.status not in {"certified", "survivor"}:
        return "not_certified_or_survivor"
    signature = result.canonical_signature or _canonical_signature(result)
    if signature in selected_signatures:
        return "exact_canonical_duplicate"
    return None


def _backfill_invariant_violation(
    parent: CertificationInput,
    *,
    selected_signatures: set[str],
    used_ids: set[str],
) -> Optional[str]:
    if not _parent_is_backfill_eligible(parent):
        return "backfill_ineligible"
    family = _problem_family(parent) or "unknown"
    signature = _parent_canonical_signature(parent, family)
    if parent.id in used_ids or signature in selected_signatures:
        return "exact_canonical_duplicate"
    return None


def _decision_selection_reason(decision: Dict[str, Any], default: str) -> str:
    reason = str(decision.get("selection_reason") or "").strip()
    if not reason or reason in {"select", "backfill", "reject"}:
        return default
    return _prompt_text(reason, limit=160)


def _result_rejection_reason(
    result: CertificationResult,
    *,
    selected_signatures: set[str],
    selected_reasoning_signatures: set[str],
    signature_group_counts: Counter[str],
    family_counts: Counter[str],
    root_lineage_counts: Counter[str],
    generation_count: int,
    recommended: bool,
) -> str:
    signature = result.canonical_signature or _canonical_signature(result)
    family = result.family or result.target_family or detect_family(result.statement or "") or "unknown"
    evidence = dict(result.quality_evidence or {})
    if result.status not in {"certified", "survivor"}:
        return "not_certified"
    if "near_duplicate" in set(result.quality_flags or []):
        return "near_duplicate_memory"
    if (
        result.quality_verdict == "weak"
        and _entropy_increase(result)
        and generation_count > 3
        and not _accepted_proxy_pass(result)
    ):
        return "entropy_support_generation_cap"
    if result.quality_verdict == "weak" and (result.retry_exhausted or not _entropy_increase(result)):
        return (
            "weak_quality_after_retries"
            if result.retry_exhausted and result.quality_retry_count > 0
            else "weak_quality"
        )
    if signature in selected_signatures:
        return "duplicate_signature"
    reasoning_signature = str(evidence.get("reasoning_signature") or "")
    if result.op_type != "survivor" and reasoning_signature in selected_reasoning_signatures:
        return "repeated_reasoning_signature"
    signature_group = str(evidence.get("signature_group") or "")
    if signature_group and signature_group_counts[signature_group] >= _signature_group_limit(signature_group):
        return "signature_group_cap"
    if family != "theorem_proof" and family_counts[family] >= FAMILY_CAP:
        return "family_cap"
    if (
        family == "theorem_proof"
        and result.op_type not in {"survivor", "fallback_survivor"}
        and any(root_lineage_counts[root] >= THEOREM_LINEAGE_CAP for root in _result_root_lineages(result))
    ):
        return "lineage_cap"
    return "selected" if recommended else "not_recommended_by_orchestrator"


def _backfill_rejection_reason(
    parent: CertificationInput,
    *,
    selected_signatures: set[str],
    selected_reasoning_signatures: set[str],
    signature_group_counts: Counter[str],
    family_counts: Counter[str],
    root_lineage_counts: Counter[str],
    used_ids: set[str],
) -> str:
    family = _problem_family(parent) or "unknown"
    signature = _parent_canonical_signature(parent, family)
    evidence = dict((parent.metadata or {}).get("quality_evidence") or {})
    if not _parent_is_backfill_eligible(parent):
        return "backfill_ineligible"
    if parent.id in used_ids or signature in selected_signatures:
        return "duplicate_signature"
    reasoning_signature = str(evidence.get("reasoning_signature") or "")
    if reasoning_signature and reasoning_signature in selected_reasoning_signatures:
        return "repeated_reasoning_signature"
    signature_group = str(evidence.get("signature_group") or "")
    if signature_group and signature_group_counts[signature_group] >= _signature_group_limit(signature_group):
        return "signature_group_cap"
    if family != "theorem_proof" and family_counts[family] >= FAMILY_CAP:
        return "family_cap"
    if family == "theorem_proof" and root_lineage_counts[_root_lineage_id(parent.id)] >= THEOREM_LINEAGE_CAP:
        return "lineage_cap"
    return "selected"


def _record_selected_context(
    *,
    signature: str,
    family: str,
    reasoning_signature: str,
    signature_group: str,
    roots: List[str],
    problem_id: str,
    selected_signatures: set[str],
    selected_reasoning_signatures: set[str],
    signature_group_counts: Counter[str],
    family_counts: Counter[str],
    root_lineage_counts: Counter[str],
    used_ids: set[str],
) -> None:
    selected_signatures.add(signature)
    if reasoning_signature:
        selected_reasoning_signatures.add(reasoning_signature)
    if signature_group:
        signature_group_counts[signature_group] += 1
    family_counts[family] += 1
    for root in roots:
        root_lineage_counts[root] += 1
    used_ids.add(problem_id)


def _result_failure_record(updated: CertificationResult) -> Dict[str, Any]:
    return {
        "slot": updated.slot,
        "op_type": updated.op_type,
        "parent_ids": updated.parent_ids,
        "target_family": updated.target_family,
        "required_params": updated.required_params,
        "generated_params": updated.generated_params,
        "composition_pattern": updated.composition_pattern,
        "parent_contributions": updated.parent_contributions,
        "quality_target": updated.quality_target,
        "quality_verdict": updated.quality_verdict,
        "quality_flags": updated.quality_flags,
        "feedback_for_next_generation": updated.feedback_for_next_generation,
        "status": updated.status,
        "error": updated.error,
        "retry_count": updated.retry_count,
        "retry_reasons": updated.retry_reasons,
        "quality_retry_count": updated.quality_retry_count,
        "retry_exhausted": updated.retry_exhausted,
        "replan_count": updated.replan_count,
        "replan_reason": updated.replan_reason,
        "replan_source": updated.replan_source,
        "failure_signature": updated.failure_signature,
    }


def _judge_failures_by_parents(results: List[CertificationResult]) -> List[Dict[str, Any]]:
    """Judge failure names grouped by the parent set that produced them."""
    grouped: Dict[tuple, Counter] = {}
    for result in results:
        judge = ((result.quality_evidence or {}).get("judge") or {})
        if judge.get("verdict") != "reject":
            continue
        failure = str(judge.get("failure") or "").strip()
        if not failure:
            continue
        key = tuple(sorted(str(p) for p in (result.parent_ids or [])))
        grouped.setdefault(key, Counter())[failure] += 1
    out = []
    for parents, failures in grouped.items():
        out.append(
            {
                "parent_ids": list(parents),
                "failures": dict(failures),
                "total": sum(failures.values()),
            }
        )
    out.sort(key=lambda row: row["total"], reverse=True)
    return out[:12]


def _result_rejection_record(updated: CertificationResult) -> Dict[str, Any]:
    return {
        "slot": updated.slot,
        "op_type": updated.op_type,
        "problem_id": updated.problem_id,
        "source_problem_id": updated.source_problem_id,
        # Present on the sibling `_result_failure_record` and omitted here, so
        # every rejection reached the next planner with `parent_ids: []`. The
        # planner could see that a slot had been rejected and not which parents
        # produced it, which is why one prime-pair lineage was planned three
        # generations running and only stopped when the judge -- who does see
        # siblings -- called the third a repeated device.
        "parent_ids": list(updated.parent_ids or []),
        "family": updated.family,
        "target_style": updated.target_style,
        "certification_route": updated.certification_route,
        "status": updated.status,
        "error": updated.error,
        "quality_verdict": updated.quality_verdict,
        "quality_flags": updated.quality_flags,
        "quality_evidence": updated.quality_evidence,
        "feedback_for_next_generation": updated.feedback_for_next_generation,
        "proof_verify_summary": updated.proof_verify_summary,
        "parent_contributions": updated.parent_contributions,
        "semantic_parent_contribution": updated.semantic_parent_contribution,
        "retry_count": updated.retry_count,
        "retry_reasons": updated.retry_reasons,
        "quality_retry_count": updated.quality_retry_count,
        "retry_exhausted": updated.retry_exhausted,
        "replan_count": updated.replan_count,
        "replan_reason": updated.replan_reason,
        "replan_source": updated.replan_source,
        "selection_reason": updated.selection_reason,
        "canonical_signature": updated.canonical_signature,
    }


def _append_orchestrator_backfill(
    parent: CertificationInput,
    *,
    slot: int,
    generation: int,
    approved: List[Dict[str, Any]],
    backfill_events: List[Dict[str, Any]],
    backfill_results: List[CertificationResult],
    selected_signatures: set[str],
    selected_reasoning_signatures: set[str],
    signature_group_counts: Counter[str],
    family_counts: Counter[str],
    root_lineage_counts: Counter[str],
    used_ids: set[str],
    selection_reason: str = "orchestrator_backfill",
) -> None:
    family = _problem_family(parent) or "unknown"
    signature = _parent_canonical_signature(parent, family)
    verdict = str(parent.metadata.get("quality_verdict") or "acceptable")
    backfill = {
        "id": parent.id,
        "statement": parent.statement,
        "answer": parent.answer,
        **dict(parent.metadata or {}),
    }
    backfill.update(
        {
            "family": family,
            "generation": generation,
            "slot": slot,
            "op_type": "fallback_survivor",
            "parent_ids": [parent.id],
            "status": "survivor",
            "lean_level": int(parent.metadata.get("lean_level") or 2),
            "quality_verdict": verdict,
            "quality_flags": list(parent.metadata.get("quality_flags") or []),
            "interestingness_score": float(parent.metadata.get("interestingness_score") or 0.5),
            "parent_eligible": True,
            "selection_reason": selection_reason,
            "canonical_signature": signature,
            "source_kind": "elite_backfill",
            "survival_status": selection_reason,
            "fallback_survivor_duplicate": False,
        }
    )
    approved.append(backfill)
    evidence = dict(parent.metadata.get("quality_evidence") or {})
    _record_selected_context(
        signature=signature,
        family=family,
        reasoning_signature=str(evidence.get("reasoning_signature") or ""),
        signature_group=str(evidence.get("signature_group") or ""),
        roots=[_root_lineage_id(parent.id)],
        problem_id=parent.id,
        selected_signatures=selected_signatures,
        selected_reasoning_signatures=selected_reasoning_signatures,
        signature_group_counts=signature_group_counts,
        family_counts=family_counts,
        root_lineage_counts=root_lineage_counts,
        used_ids=used_ids,
    )
    event = {
        "slot": slot,
        "source_id": parent.id,
        "family": family,
        "selection_reason": selection_reason,
        "fallback_survivor_duplicate": False,
    }
    backfill_events.append(event)
    backfill_results.append(
        CertificationResult(
            problem_id=f"{parent.id}__fallback_g{generation}_s{slot}",
            source_problem_id=parent.id,
            generation=generation,
            slot=slot,
            op_type="fallback_survivor",
            operator_variant="survivor",
            parent_ids=[parent.id],
            family=family,
            target_family=family,
            status="survivor",
            lean_level=int(parent.metadata.get("lean_level") or 2),
            lean_code=parent.metadata.get("lean_code"),
            anti_stub_passed=True,
            aligned=True,
            statement=parent.statement,
            answer=parent.answer,
            solution=parent.metadata.get("solution"),
            verification_code=parent.metadata.get("verification_code"),
            formal_statement=parent.metadata.get("formal_statement"),
            lean_header=parent.metadata.get("lean_header"),
            quality_verdict=verdict,
            quality_flags=list(parent.metadata.get("quality_flags") or []),
            interestingness_score=float(parent.metadata.get("interestingness_score") or 0.5),
            quality_evidence=dict(parent.metadata.get("quality_evidence") or {}),
            parent_eligible=True,
            selection_reason=selection_reason,
            canonical_signature=signature,
            source_kind="elite_backfill",
            problem_style=_problem_style(parent),
            target_style=_problem_style(parent),
            certification_route=_certification_route_for_style(_problem_style(parent)),
            parent_context_cards=list(parent.metadata.get("parent_context_cards") or [_parent_context_card(parent)]),
            operator_card={"op_type": "fallback_survivor", "parent_ids": [parent.id]},
            slot_outcome="survivor",
            survival_status=selection_reason,
            fallback_survivor_duplicate=False,
            input_metadata=dict(parent.metadata or {}),
        )
    )


def _build_generation_feedback_and_cards(
    *,
    results: List[CertificationResult],
    updated_results: List[CertificationResult],
    failed_slots: List[Dict[str, Any]],
    rejected_slots: List[Dict[str, Any]],
    backfill_events: List[Dict[str, Any]],
    target_accepted: int,
    planner: Dict[str, Any],
    signature_group_counts: Counter[str],
    generation_count: int,
) -> Dict[str, Any]:
    final_accepted_proxy_count = sum(
        1 for result in results if _is_generated_result(result) and _accepted_proxy_pass(result)
    )
    final_accepted_grade_proxy_count = sum(
        1 for result in results if _is_generated_result(result) and _accepted_grade_proxy_pass(result)
    )
    quality_flags = Counter(flag for result in results for flag in list(result.quality_flags or []))
    op_type_outcomes: Dict[str, Dict[str, int]] = {}
    planned_op_type_outcomes: Dict[str, Dict[str, int]] = {}
    for result in results:
        key = result.op_type or "unknown"
        op_type_outcomes.setdefault(key, {})
        outcome = result.quality_verdict or result.status
        op_type_outcomes[key][outcome] = op_type_outcomes[key].get(outcome, 0) + 1
        planned_key = result.planned_op_type or result.op_type or "unknown"
        planned_op_type_outcomes.setdefault(planned_key, {})
        planned_op_type_outcomes[planned_key][outcome] = planned_op_type_outcomes[planned_key].get(outcome, 0) + 1
    weak_slots = [
        {
            "slot": result.slot,
            "op_type": result.op_type,
            "operator_variant": result.operator_variant,
            "parent_ids": result.parent_ids,
            "quality_flags": result.quality_flags,
            "misformalization": dict((result.quality_evidence or {}).get("misformalization") or {}),
            "selection_reason": result.selection_reason,
            "feedback": result.feedback_for_next_generation,
        }
        for result in results
        if result.quality_verdict == "weak" or result.quality_flags
    ]
    generation_feedback = {
        "op_type_outcomes": op_type_outcomes,
        "planned_op_type_outcomes": planned_op_type_outcomes,
        "weak_slots": weak_slots,
        "failed_slots": failed_slots,
        "rejected_slots": rejected_slots,
        "backfill_events": backfill_events,
        "accepted_proxy_count": final_accepted_proxy_count,
        "accepted_grade_proxy_count": final_accepted_grade_proxy_count,
        "target_accepted_per_generation": target_accepted,
        "reserve_slots_run": sum(1 for result in results if result.source_kind == "reserve_generated"),
        "reserve_slots_selected": sum(
            1 for result in results if result.source_kind == "reserve_generated" and result.parent_eligible
        ),
        "yield_funnel": _yield_funnel(results),
        "planner_source": planner.get("planner_source"),
        "planner_warnings": list(planner.get("warnings") or []),
        "repeated_weak_patterns": [flag for flag, count in quality_flags.items() if count > 1],
        # What the judge actually said, per parent set. `quality_flags` carries
        # certification status -- `certification_not_successful` was the only
        # "repeated weak pattern" the planner ever saw -- and says nothing about
        # why a row was discarded. A planner that is told `repeated_device` came
        # from these two parents can stop planning a fourth variation of the same
        # device; without it, it planned three in a row.
        "judge_failures_by_parents": _judge_failures_by_parents(results),
        "quality_flags": dict(quality_flags),
        "dominant_signature_groups": {
            group: count for group, count in signature_group_counts.items() if count >= 2
        },
        "quality_retry_count": sum(int(result.quality_retry_count or 0) for result in results),
        "retry_exhausted_count": sum(1 for result in results if result.retry_exhausted),
        "replan_count": sum(int(result.replan_count or 0) for result in results),
        "replanned_slots": [
            {"slot": result.slot, "op_type": result.op_type, "replan_reason": result.replan_reason}
            for result in results
            if int(result.replan_count or 0) > 0
        ],
        "giveup_slots": [
            {"slot": result.slot, "op_type": result.op_type, "status": result.status}
            for result in updated_results
            if result.status not in {"certified", "survivor"}
        ],
        "backfilled_slots": backfill_events,
    }
    plan_outcome_cards = [
        {
            "generation": generation_count,
            "slot": result.slot,
            "op_type": result.op_type,
            "planned_op_type": result.planned_op_type,
            "planned_operator_variant": result.planned_operator_variant,
            "attempted_op_types": result.attempted_op_types,
            "parent_ids": result.parent_ids,
            "planned_variation_axis": result.variation_axis,
            "target_family": result.target_family or result.family,
            "status": result.status,
            "quality_verdict": result.quality_verdict,
            "quality_flags": result.quality_flags,
            "reasoning_pattern": result.reasoning_pattern,
            "projection_check": result.projection_check,
            "semantic_parent_contribution": result.semantic_parent_contribution,
            "interestingness_features": result.interestingness_features,
            "quality_evidence": result.quality_evidence,
            "operator_card": result.operator_card,
            "statement": result.statement,
            "answer": result.answer,
            "solution": result.solution,
            "formal_statement": result.formal_statement,
            "lean_code": result.lean_code,
            "proof_plan": result.proof_plan,
            "proof_obligations": result.proof_obligations,
            "error": result.error,
            "proof_verify_summary": result.proof_verify_summary,
            "solution_verify_passed": result.solution_verify_passed,
            "solution_verify_flags": result.solution_verify_flags,
            "retry_count": result.retry_count,
            "retry_reasons": result.retry_reasons,
            "attempt_history": result.attempt_history,
            "quality_retry_count": result.quality_retry_count,
            "retry_exhausted": result.retry_exhausted,
            "replan_count": result.replan_count,
            "replan_reason": result.replan_reason,
            "survival_status": result.survival_status,
            "failure_class": _failure_class(result),
            "failure_signature": result.failure_signature or _failure_signature(result),
            "selected_for_next_pool": bool(result.parent_eligible),
            "selection_reason": result.selection_reason,
        }
        for result in results
    ]
    for event in backfill_events:
        plan_outcome_cards.append(
            {
                "generation": generation_count,
                "slot": event["slot"],
                "op_type": "fallback_survivor",
                "parent_ids": [event["source_id"]],
                "status": "survivor",
                "quality_verdict": "acceptable",
                "failure_signature": "",
                "selected_for_next_pool": True,
                "selection_reason": event.get("selection_reason") or "orchestrator_backfill",
            }
        )
    generation_feedback["plan_outcome_summary"] = _plan_outcome_summary(plan_outcome_cards)
    return {"generation_feedback": generation_feedback, "plan_outcome_cards": plan_outcome_cards}


def select_next_pool_with_orchestrator(
    *,
    results: List[CertificationResult],
    current_generation: List[CertificationInput],
    pool_size: int,
    generation_count: int,
    target_accepted: int,
    planner: Dict[str, Any],
    generation_feedback: Optional[Dict[str, Any]] = None,
    backfill_seed_archive: Optional[List[CertificationInput]] = None,
    aggregate_selector: Optional[AggregateSelectorFn] = None,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    seed_archive = list(backfill_seed_archive or current_generation)
    payload = _aggregate_selector_payload(
        results=results,
        backfill_seed_archive=seed_archive,
        pool_size=pool_size,
        generation_count=generation_count,
        target_accepted=target_accepted,
        generation_feedback=generation_feedback,
    )
    selector_warnings: List[str] = []
    selector_source = "aggregate_orchestrator_llm"
    try:
        if aggregate_selector is not None:
            raw_selection = aggregate_selector(payload)
        elif _aggregate_llm_available():
            raw_selection = llm_select_next_pool(
                payload,
                generation_model=generation_model,
                generation_temperature=generation_temperature,
            )
        else:
            raise EnvironmentError("aggregate LLM unavailable")
        if inspect.isawaitable(raw_selection):
            raise TypeError("aggregate_selector callable must be synchronous")
        selection = _normalize_aggregate_selection(dict(raw_selection or {}))
        selector_source = selection["selector_source"]
    except Exception as exc:
        selector_warnings.append(f"aggregate selector fallback: {type(exc).__name__}: {exc}")
        payload["selector_warnings"] = selector_warnings
        selection = _normalize_aggregate_selection(
            _deterministic_aggregate_selection(payload, results, seed_archive)
        )
        selector_source = selection["selector_source"]

    result_by_id: Dict[str, CertificationResult] = {}
    for result in results:
        for candidate_id in _result_candidate_ids(result):
            result_by_id.setdefault(str(candidate_id), result)
    seed_by_id = {parent.id: parent for parent in seed_archive}
    decisions = list(selection["pool_decisions"])

    approved: List[Dict[str, Any]] = []
    failed_slots: List[Dict[str, Any]] = []
    rejected_slots: List[Dict[str, Any]] = []
    backfill_events: List[Dict[str, Any]] = []
    backfill_results: List[CertificationResult] = []
    updated_by_problem_id: Dict[str, CertificationResult] = {}
    selected_signatures: set[str] = set()
    selected_reasoning_signatures: set[str] = set()
    signature_group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    root_lineage_counts: Counter[str] = Counter()
    used_ids: set[str] = set()
    invariant_events: List[Dict[str, Any]] = []
    orchestrator_decision_events: List[Dict[str, Any]] = []

    for decision in decisions:
        item_id = str(decision.get("id") or "").strip()
        action = str(decision.get("decision") or "reject")
        source_kind = str(decision.get("source_kind") or "current_candidate")
        if not item_id:
            continue
        if action in {"select", "backfill"} and len(approved) >= pool_size:
            invariant_events.append(
                {
                    "id": item_id,
                    "source_kind": source_kind,
                    "decision": action,
                    "selection_reason": "invalidated_by_code:pool_size_exceeded",
                }
            )
            break
        if action == "backfill" or source_kind == "previous_seed":
            parent = seed_by_id.get(item_id)
            if parent is None:
                invariant_events.append(
                    {
                        "id": item_id,
                        "source_kind": "previous_seed",
                        "decision": action,
                        "selection_reason": "invalidated_by_code:unknown_id",
                    }
                )
                continue
            if action == "reject":
                orchestrator_decision_events.append(
                    {
                        "id": item_id,
                        "source_kind": "previous_seed",
                        "decision": "reject",
                        "selection_reason": _decision_selection_reason(decision, "orchestrator_reject"),
                    }
                )
                continue
            reason = _backfill_invariant_violation(
                parent,
                selected_signatures=selected_signatures,
                used_ids=used_ids,
            )
            if reason:
                invariant_events.append(
                    {
                        "id": item_id,
                        "source_kind": "previous_seed",
                        "decision": action,
                        "selection_reason": f"invalidated_by_code:{reason}",
                    }
                )
                continue
            selected_reason = _decision_selection_reason(decision, "orchestrator_backfill")
            _append_orchestrator_backfill(
                parent,
                slot=len(approved),
                generation=generation_count,
                approved=approved,
                backfill_events=backfill_events,
                backfill_results=backfill_results,
                selected_signatures=selected_signatures,
                selected_reasoning_signatures=selected_reasoning_signatures,
                signature_group_counts=signature_group_counts,
                family_counts=family_counts,
                root_lineage_counts=root_lineage_counts,
                used_ids=used_ids,
                selection_reason=selected_reason,
            )
            orchestrator_decision_events.append(
                {
                    "id": item_id,
                    "source_kind": "previous_seed",
                    "decision": "backfill",
                    "selection_reason": selected_reason,
                    "expected_next_gen_role": decision.get("expected_next_gen_role"),
                }
            )
            continue

        result = result_by_id.get(item_id)
        if result is None:
            invariant_events.append(
                {
                    "id": item_id,
                    "source_kind": "current_candidate",
                    "decision": action,
                    "selection_reason": "invalidated_by_code:unknown_id",
                }
            )
            continue
        if result.problem_id in updated_by_problem_id:
            invariant_events.append(
                {
                    "id": item_id,
                    "source_kind": "current_candidate",
                    "decision": action,
                    "selection_reason": "invalidated_by_code:exact_canonical_duplicate",
                }
            )
            continue
        signature = result.canonical_signature or _canonical_signature(result)
        if action == "select":
            reason = _candidate_invariant_violation(result, selected_signatures=selected_signatures)
            if reason:
                updated = result.model_copy(
                    update={
                        "parent_eligible": False,
                        "selection_reason": f"invalidated_by_code:{reason}",
                        "canonical_signature": signature,
                        "failure_signature": result.failure_signature or _failure_signature(result),
                        "source_kind": result.source_kind or ("survivor" if result.op_type == "survivor" else "generated"),
                    }
                )
                updated_by_problem_id[result.problem_id] = updated
                invariant_events.append(
                    {
                        "id": result.problem_id,
                        "source_kind": "current_candidate",
                        "decision": "select",
                        "selection_reason": f"invalidated_by_code:{reason}",
                    }
                )
                continue
            selected_reason = _decision_selection_reason(
                decision,
                "selected_entropy_increase_support"
                if result.quality_verdict == "weak" and _entropy_increase(result)
                else "selected_for_next_pool",
            )
            family = result.family or result.target_family or detect_family(result.statement or "") or "unknown"
            evidence = dict(result.quality_evidence or {})
            updated = result.model_copy(
                update={
                    "parent_eligible": True,
                    "selection_reason": selected_reason,
                    "canonical_signature": signature,
                    "source_kind": result.source_kind or ("survivor" if result.op_type == "survivor" else "generated"),
                }
            )
            approved.append(_result_to_pool_problem(updated))
            _record_selected_context(
                signature=signature,
                family=family,
                reasoning_signature=str(evidence.get("reasoning_signature") or ""),
                signature_group=str(evidence.get("signature_group") or ""),
                roots=_result_root_lineages(updated),
                problem_id=updated.problem_id,
                selected_signatures=selected_signatures,
                selected_reasoning_signatures=selected_reasoning_signatures,
                signature_group_counts=signature_group_counts,
                family_counts=family_counts,
                root_lineage_counts=root_lineage_counts,
                used_ids=used_ids,
            )
            updated_by_problem_id[result.problem_id] = updated
            orchestrator_decision_events.append(
                {
                    "id": result.problem_id,
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": selected_reason,
                    "expected_next_gen_role": decision.get("expected_next_gen_role"),
                }
            )
        else:
            reject_reason = _decision_selection_reason(decision, "orchestrator_reject")
            updated = result.model_copy(
                update={
                    "parent_eligible": False,
                    "selection_reason": reject_reason,
                    "canonical_signature": signature,
                    "failure_signature": result.failure_signature or _failure_signature(result),
                    "source_kind": result.source_kind or ("survivor" if result.op_type == "survivor" else "generated"),
                }
            )
            updated_by_problem_id[result.problem_id] = updated
            orchestrator_decision_events.append(
                {
                    "id": result.problem_id,
                    "source_kind": "current_candidate",
                    "decision": "reject",
                    "selection_reason": reject_reason,
                }
            )

    if len(approved) < pool_size:
        decided_seed_ids = {
            str(item.get("id") or "").strip()
            for item in decisions
            if str(item.get("source_kind") or "") == "previous_seed"
        }
        for parent in seed_archive:
            if len(approved) >= pool_size:
                break
            if parent.id in decided_seed_ids:
                continue
            reason = _backfill_invariant_violation(
                parent,
                selected_signatures=selected_signatures,
                used_ids=used_ids,
            )
            if reason:
                continue
            _append_orchestrator_backfill(
                parent,
                slot=len(approved),
                generation=generation_count,
                approved=approved,
                backfill_events=backfill_events,
                backfill_results=backfill_results,
                selected_signatures=selected_signatures,
                selected_reasoning_signatures=selected_reasoning_signatures,
                signature_group_counts=signature_group_counts,
                family_counts=family_counts,
                root_lineage_counts=root_lineage_counts,
                used_ids=used_ids,
                selection_reason="continuity_backfill",
            )
            orchestrator_decision_events.append(
                {
                    "id": parent.id,
                    "source_kind": "previous_seed",
                    "decision": "backfill",
                    "selection_reason": "continuity_backfill",
                    "expected_next_gen_role": "continuity_parent",
                }
            )

    for result in results:
        if result.problem_id in updated_by_problem_id:
            continue
        signature = result.canonical_signature or _canonical_signature(result)
        updated_by_problem_id[result.problem_id] = result.model_copy(
            update={
                "parent_eligible": False,
                "selection_reason": "not_selected_by_orchestrator",
                "canonical_signature": signature,
                "failure_signature": result.failure_signature or _failure_signature(result),
                "source_kind": result.source_kind or ("survivor" if result.op_type == "survivor" else "generated"),
            }
        )
        orchestrator_decision_events.append(
            {
                "id": result.problem_id,
                "source_kind": "current_candidate",
                "decision": "reject",
                "selection_reason": "not_selected_by_orchestrator",
            }
        )

    updated_results = [updated_by_problem_id[result.problem_id] for result in results]
    for updated in updated_results:
        if updated.parent_eligible:
            continue
        if updated.status not in {"certified", "survivor"}:
            failed_slots.append(_result_failure_record(updated))
        else:
            rejected_slots.append(_result_rejection_record(updated))
    final_results = sorted(
        updated_results + backfill_results,
        key=lambda result: ((result.slot or 0), 1 if result.op_type == "fallback_survivor" else 0),
    )
    feedback_and_cards = _build_generation_feedback_and_cards(
        results=final_results,
        updated_results=updated_results,
        failed_slots=failed_slots,
        rejected_slots=rejected_slots,
        backfill_events=backfill_events,
        target_accepted=target_accepted,
        planner=planner,
        signature_group_counts=signature_group_counts,
        generation_count=generation_count,
    )
    generation_feedback = feedback_and_cards["generation_feedback"]
    generation_feedback["aggregate_selector"] = {
        "selector_source": selector_source,
        "pool_strategy_rationale": selection.get("pool_strategy_rationale"),
        "warnings": selector_warnings + list(selection.get("warnings") or []),
        "generation_survival_status": selection.get("generation_survival_status"),
    }
    if len(approved) < pool_size:
        generation_survival_status = "partial"
    elif backfill_events:
        generation_survival_status = "complete_with_backfill"
    else:
        generation_survival_status = "complete"
    generation_feedback["generation_survival_status"] = generation_survival_status
    selected_candidate_ids = [
        item["id"]
        for item in selection["pool_decisions"]
        if item["source_kind"] == "current_candidate" and item["decision"] == "select"
    ]
    backfill_seed_ids = [
        item["id"]
        for item in selection["pool_decisions"]
        if item["source_kind"] == "previous_seed" and item["decision"] == "backfill"
    ]
    return {
        "results": final_results,
        "approved_candidates": approved,
        "failed_slots": failed_slots,
        "rejected_slots": rejected_slots,
        "backfill_events": backfill_events,
        "generation_feedback": generation_feedback,
        "plan_outcome_cards": feedback_and_cards["plan_outcome_cards"],
        "generation_survival_status": generation_survival_status,
        "selector_manifest": {
            "selector_source": selector_source,
            "payload_sha256": _aggregate_payload_sha256(payload),
            "payload_summary": _aggregate_payload_summary(payload),
            "pool_decisions": selection["pool_decisions"][:20],
            "selected_candidate_ids": selected_candidate_ids[:10],
            "backfill_seed_ids": backfill_seed_ids[:10],
            "backfill_archive_count": len(seed_archive),
            "orchestrator_decision_events": orchestrator_decision_events[:40],
            "pool_strategy_rationale": selection.get("pool_strategy_rationale"),
            "warnings": selector_warnings + list(selection.get("warnings") or []),
        },
        "invariant_events": invariant_events[:40],
        "hard_gate_events": invariant_events[:40],
    }


def build_pool_generation_graph(
    checker: Optional[LeanChecker] = None,
    *,
    planner: Optional[PlannerFn] = None,
    generator: Optional[SlotGeneratorFn] = None,
    theorem_generator: Optional[TheoremGeneratorFn] = None,
    theorem_verifier: Optional[TheoremVerifierFn] = None,
    theorem_alignment_verifier: Optional[TheoremAlignmentVerifierFn] = None,
    theorem_proof_repairer: Optional[Callable[..., Any]] = None,
    replanner: Optional[ReplannerFn] = None,
    aggregate_selector: Optional[AggregateSelectorFn] = None,
):
    """Build one-generation central-orchestrator graph."""
    lean_checker = checker or LeanChecker()
    graph = StateGraph(PoolState)

    def load_seed_pool_node(state: PoolState) -> PoolState:
        if state.get("current_generation"):
            return {
                "current_generation": state["current_generation"],
                "current_generation_size": len(state["current_generation"]),
            }
        pool_size = int(state.get("pool_size") or POOL_SIZE)
        loaded = load_seed_inputs(Path(state["input_path"]), pool_size=pool_size)
        return {"current_generation": loaded, "current_generation_size": len(loaded)}

    def plan_generation_node(state: PoolState) -> PoolState:
        pool = state["current_generation"]
        # Spend the offspring budget across the seed set rather than letting it
        # concentrate on whichever seed happens to breed easily.
        spent = _exhausted_seeds(_run_local_novelty_rows(state))
        if spent:
            before = len(pool)
            pool = _drop_exhausted_parents(pool, spent)
            if len(pool) != before:
                print(
                    f"[seed-cap] withheld {before - len(pool)} parent(s); "
                    f"{len(spent)} seed(s) at cap {SEED_OFFSPRING_CAP}",
                    flush=True,
                )
        parent_context_cards = _parent_context_cards(pool)
        with ls.trace(
            name="context_compile.parent_context_cards",
            run_type="tool",
            inputs={"pool_size": len(pool)},
            tags=["pool-generation", "context-compile"],
        ) as context_run:
            context_run.end(
                outputs={
                    "parent_context_cards": [
                        {
                            "id": card.get("id"),
                            "problem_style": card.get("problem_style"),
                            "certification_route": card.get("certification_route"),
                            "family": card.get("family"),
                            "proof_body_available": dict(card.get("proof_context") or {}).get(
                                "proof_body_available"
                            ),
                            "reusable_atoms": list(card.get("reusable_atoms") or [])[:8],
                            "template_fit": card.get("template_fit"),
                        }
                        for card in parent_context_cards
                    ]
                }
            )
        pool_size = int(state.get("pool_size") or POOL_SIZE)
        survivor_count = int(state.get("survivor_count", 1))
        crossover_count = int(state.get("crossover_count", 0))
        if crossover_count and len(pool) < 2:
            raise ValueError("crossover_count > 0 requires at least two parents")
        op_type_allocation_hint = _op_type_allocation_hint(
            list(state.get("plan_outcome_cards", []) or []),
            pool_size=pool_size,
            survivor_count=survivor_count,
            crossover_count=crossover_count,
        )
        planner_memory = select_planner_memory_cards(
            pool,
            memory_dir=Path(state.get("planner_memory_dir") or "data/certified"),
            limit=int(state.get("planner_memory_limit") or DEFAULT_PLANNER_MEMORY_LIMIT),
            enabled=not bool(state.get("disable_planner_memory")),
            exclude_paths=[Path(state["output_path"])],
            slot_op_types=(
                ["mutation", "crossover"]
                if int(state.get("crossover_count") or 0) > 0
                else ["mutation"]
            ),
        )
        with ls.trace(
            name="planner.memory_select",
            run_type="tool",
            inputs={
                "enabled": planner_memory.get("enabled"),
                "memory_dir": state.get("planner_memory_dir"),
                "limit": state.get("planner_memory_limit"),
            },
            tags=["pool-generation", "planner-memory"],
        ) as memory_run:
            memory_run.end(outputs=_planner_memory_trace_manifest(planner_memory))
        novelty_memory = _build_state_novelty_memory(state, pool)
        with ls.trace(
            name="planner.novelty_memory_select",
            run_type="tool",
            inputs={
                "accepted_ledger_path": novelty_memory.get("accepted_ledger_path"),
                "run_local_row_count": len(_run_local_novelty_rows(state)),
            },
            tags=["pool-generation", "novelty-memory"],
        ) as novelty_run:
            novelty_run.end(outputs=_novelty_memory_trace_manifest(novelty_memory))

        warnings: List[str] = []
        try:
            if planner is not None:
                maybe_plan = planner(
                    pool,
                    {
                        **dict(state),
                        "planner_memory": planner_memory,
                        "planner_case_pack": planner_memory,
                        "novelty_memory": novelty_memory,
                    },
                )
                raw_plan = maybe_plan
            else:
                raw_plan = llm_plan_generation(
                    pool,
                    pool_size=pool_size,
                    survivor_count=survivor_count,
                    crossover_count=crossover_count,
                    generation_model=state.get("generation_model"),
                    generation_temperature=state.get("generation_temperature"),
                    generation_feedback=state.get("generation_feedback"),
                    op_type_allocation_hint=op_type_allocation_hint,
                    planner_memory=planner_memory,
                    novelty_memory=novelty_memory,
                )
            if inspect.isawaitable(raw_plan):
                raise TypeError("planner callable must be synchronous")
            work_items = validate_pool_plan(
                raw_plan,
                pool,
                pool_size=pool_size,
                survivor_count=survivor_count,
                crossover_count=crossover_count,
            )
            work_items = _attach_novelty_contracts(work_items, novelty_memory)
            planner_payload = {
                "planner_source": raw_plan.get("planner_source", "orchestrator_llm"),
                "plan_rationale": raw_plan.get("plan_rationale", ""),
                "warnings": warnings + list(raw_plan.get("warnings") or []),
                "op_type_allocation_hint": op_type_allocation_hint,
                "parent_context_cards": parent_context_cards,
                "planner_memory": _planner_memory_trace_manifest(planner_memory),
                "planner_case_pack": _planner_memory_trace_manifest(planner_memory),
                "novelty_memory": _novelty_memory_trace_manifest(novelty_memory),
            }
        except Exception as exc:
            warnings.append(f"planner fallback: {type(exc).__name__}: {exc}")
            fallback = deterministic_fallback_plan(
                pool,
                pool_size=pool_size,
                survivor_count=survivor_count,
                crossover_count=crossover_count,
            )
            work_items = validate_pool_plan(
                fallback,
                pool,
                pool_size=pool_size,
                survivor_count=survivor_count,
                crossover_count=crossover_count,
            )
            work_items = _attach_novelty_contracts(work_items, novelty_memory)
            planner_payload = {
                "planner_source": "deterministic_fallback",
                "plan_rationale": fallback["plan_rationale"],
                "warnings": warnings + list(fallback.get("warnings") or []),
                "op_type_allocation_hint": op_type_allocation_hint,
                "parent_context_cards": parent_context_cards,
                "planner_memory": _planner_memory_trace_manifest(planner_memory),
                "planner_case_pack": _planner_memory_trace_manifest(planner_memory),
                "novelty_memory": _novelty_memory_trace_manifest(novelty_memory),
            }
        with ls.trace(
            name="planner.operator_cards",
            run_type="tool",
            # What the planner was working from, not only how many slots it
            # returned. Judging a plan means asking whether these parents were
            # the right ones to pair given what the pool held and what memory
            # said had already failed -- and a work_item_count answers none of
            # that.
            inputs={
                "work_item_count": len(work_items),
                "planner_source": planner_payload.get("planner_source"),
                "pool": [
                    {
                        "id": getattr(parent, "id", ""),
                        "statement": str(
                            (getattr(parent, "metadata", None) or {}).get("formal_statement") or ""
                        )[:400],
                    }
                    for parent in pool
                ],
                "op_type_allocation_hint": op_type_allocation_hint,
                "planner_memory": planner_payload.get("planner_memory"),
                "novelty_memory": planner_payload.get("novelty_memory"),
                "plan_rationale": planner_payload.get("plan_rationale"),
                "warnings": planner_payload.get("warnings"),
            },
            tags=["pool-generation", "operator-cards"],
        ) as cards_run:
            cards_run.end(
                outputs={
                    "operator_cards": [
                        _compact_operator_card_for_pool(item.get("operator_card", {}))
                        for item in work_items
                    ]
                }
            )
        return {
            "work_items": work_items,
            "planner": planner_payload,
            "planner_memory": _planner_memory_trace_manifest(planner_memory),
            "planner_case_pack": _planner_memory_trace_manifest(planner_memory),
            "novelty_memory": novelty_memory,
        }

    async def slot_dispatch_node(state: PoolState) -> PoolState:
        available = await lean_checker.health_check()
        if state.get("dispatch_mode") == "reserve":
            return {
                "slot_outputs": [],
                "lean_available": available,
                "dispatch_mode": "reserve_dispatched",
                "reserve_round_pending": False,
                "reserve_round_done": True,
            }
        return {"slot_outputs": [], "lean_available": available, "dispatch_mode": "primary_dispatched"}

    def slot_dispatch_route(state: PoolState):
        dispatch_items = (
            state.get("reserve_work_items", [])
            if state.get("dispatch_mode") == "reserve_dispatched"
            else state.get("work_items", [])
        )
        sends = [
            Send(
                "04_slot_unit",
                {
                    **dict(state),
                    "work_item": item,
                    "slot_outputs": [],
                },
            )
            for item in dispatch_items
        ]
        return sends if sends else "05_slot_aggregate"

    def slot_aggregate_route(state: PoolState):
        return "03_slot_dispatch" if state.get("reserve_round_pending") else "06_save_generation"

    async def slot_unit_node(state: PoolState) -> PoolState:
        from src.certification.graph import build_certification_graph

        item = dict(state["work_item"])
        if item.get("op_type") != "survivor" and not item.get("memory_delta_contract"):
            item = _attach_novelty_contracts([item], dict(state.get("novelty_memory") or {}))[0]
        slot = int(item["slot"])
        op_type = item["op_type"]
        parent_map = _problem_by_id(state["current_generation"])
        generation_count = int(state.get("generation_count", 0) or 0) + 1
        config = default_generation_config(
            model=state.get("generation_model"),
            temperature=state.get("generation_temperature"),
        )

        with ls.trace(
            name=f"slot_{slot}.{op_type}",
            run_type="chain",
            inputs={
                "slot": slot,
                "op_type": op_type,
                "operator_variant": item.get("operator_variant", ""),
                "parent_ids": item.get("parent_ids", []),
                "variation_axis": item.get("variation_axis", ""),
                "reasoning_goal": item.get("reasoning_goal", ""),
                "target_family": item.get("target_family", ""),
                "required_params": dict(item.get("required_params") or {}),
                "composition_pattern": item.get("composition_pattern", ""),
                "parent_contributions": dict(item.get("parent_contributions") or {}),
                "quality_target": item.get("quality_target", ""),
                "operator_card": _operator_card(item),
                "fusion_contract": dict(item.get("fusion_contract") or {}),
            },
            tags=["pool-generation", "slot", op_type],
            metadata={
                "slot": slot,
                "op_type": op_type,
                "operator_variant": item.get("operator_variant", ""),
                "parent_ids": item.get("parent_ids", []),
                "variation_axis": item.get("variation_axis", ""),
                "reasoning_goal": item.get("reasoning_goal", ""),
                "target_family": item.get("target_family", ""),
                "required_params": dict(item.get("required_params") or {}),
                "composition_pattern": item.get("composition_pattern", ""),
                "parent_contributions": dict(item.get("parent_contributions") or {}),
                "quality_target": item.get("quality_target", ""),
                "operator_card": _operator_card(item),
                "fusion_contract": dict(item.get("fusion_contract") or {}),
                "generation_count": generation_count,
            },
        ) as slot_run:
            try:
                planned_op_type = op_type
                planned_operator_variant = str(item.get("operator_variant") or "")
                last_result: Optional[CertificationResult] = None
                retry_reasons: List[str] = []
                failure_history: List[str] = []
                attempt_history: List[Dict[str, Any]] = []
                quality_retry_count = 0
                replan_count = 0
                replan_reason = ""
                replan_source = ""
                discarded_operator_card: Dict[str, Any] = {}
                attempt = 0
                max_retries = int(state.get("max_retries") or os.getenv("POOL_GENERATION_MAX_RETRIES", "3"))
                max_attempts = 1 if op_type == "survivor" else max_retries + 1
                for attempt in range(max_attempts):
                    attempt_item = dict(item)
                    current_op_type = str(attempt_item.get("op_type") or op_type)
                    current_parents = [
                        parent_map[parent_id] for parent_id in attempt_item.get("parent_ids", [])
                    ]
                    attempt_item["max_retries"] = max_retries
                    attempt_item["retry_count"] = attempt
                    attempt_item["retry_reasons"] = list(retry_reasons)
                    attempt_item["quality_retry_count"] = quality_retry_count
                    attempt_item["replan_count"] = replan_count
                    attempt_item["replan_reason"] = replan_reason
                    attempt_item["replan_source"] = replan_source
                    attempt_item["discarded_operator_card"] = dict(discarded_operator_card)
                    attempt_item["attempt_history"] = _attempt_history_summary(attempt_history)
                    attempt_item["leansearch_enabled"] = bool(
                        state.get("leansearch_enabled", DEFAULT_LEANSEARCH_ENABLED)
                    )
                    attempt_item["leansearch_limit"] = int(
                        state.get("leansearch_limit") or DEFAULT_LEANSEARCH_LIMIT
                    )
                    retry_feedback = ""
                    if attempt and last_result is not None:
                        retry_feedback = _retry_feedback_for_result(
                            last_result,
                            current_op_type,
                            attempt - 1,
                            attempt_history=attempt_history,
                        )
                        attempt_item["retry_feedback"] = retry_feedback
                    if current_op_type == "survivor":
                        cert_input = _with_slot_metadata(
                            current_parents[0],
                            attempt_item,
                            generation_count=generation_count,
                        )
                        cert_graph = None
                    else:
                        parent_input = (
                            _with_slot_metadata(current_parents[0], attempt_item, generation_count=generation_count)
                            if current_op_type == "mutation"
                            else _crossover_parent_input(
                                current_parents, attempt_item, generation_count=generation_count
                            )
                        )
                        cert_input = parent_input

                    if (
                        current_op_type == "survivor"
                        and current_parents
                        and _problem_style(current_parents[0]) == "theorem_proof"
                    ):
                        result = _theorem_survivor_result(
                            current_parents[0],
                            attempt_item,
                            generation_count=generation_count,
                        )
                    elif current_op_type != "survivor" and attempt_item.get("target_style") == "theorem_proof":
                        with ls.trace(
                            name=f"slot_{slot}.{current_op_type}.theorem_certify.attempt_{attempt}",
                            run_type="chain",
                            inputs={
                                "slot": slot,
                                "op_type": current_op_type,
                                "parent_ids": attempt_item.get("parent_ids", []),
                                "operator_card": _operator_card(attempt_item),
                            },
                            tags=["pool-generation", "theorem-certification", f"slot:{slot}", current_op_type],
                        ) as theorem_run:
                            result = await _certify_theorem_child(
                                parent_input=cert_input,
                                item=attempt_item,
                                parents=list(current_parents or []),
                                generation_count=generation_count,
                                config=config,
                                theorem_generator=theorem_generator,
                                theorem_verifier=theorem_verifier,
                                theorem_alignment_verifier=theorem_alignment_verifier,
                                theorem_proof_repairer=theorem_proof_repairer,
                            )
                            theorem_run.end(
                                outputs={
                                    "status": result.status,
                                    "problem_id": result.problem_id,
                                    "proof_verify_summary": result.proof_verify_summary,
                                }
                            )
                    else:
                        cert_graph = build_certification_graph(
                            checker=lean_checker,
                            generate_harder=(current_op_type != "survivor"),
                            generation_model=config.model,
                            generation_temperature=config.temperature,
                            generator=generator,
                        )
                        final_state = await cert_graph.ainvoke(
                            {"input": cert_input, "errors": []},
                            config={
                                "run_name": f"slot_{slot}.{current_op_type}.certify.attempt_{attempt}",
                                "tags": ["pool-generation", "certification", f"slot:{slot}", current_op_type],
                                "metadata": {
                                "slot": slot,
                                "op_type": current_op_type,
                                "operator_variant": attempt_item.get("operator_variant", ""),
                                "parent_ids": attempt_item.get("parent_ids", []),
                                    "variation_axis": attempt_item.get("variation_axis", ""),
                                    "reasoning_goal": attempt_item.get("reasoning_goal", ""),
                                    "target_family": attempt_item.get("target_family", ""),
                                    "target_style": attempt_item.get("target_style", "numeric_answer"),
                                    "required_params": dict(attempt_item.get("required_params") or {}),
                                    "composition_pattern": attempt_item.get("composition_pattern", ""),
                                    "parent_contributions": dict(attempt_item.get("parent_contributions") or {}),
                                    "quality_target": attempt_item.get("quality_target", ""),
                                    "operator_card": _operator_card(attempt_item),
                                    "fusion_contract": dict(attempt_item.get("fusion_contract") or {}),
                                    "generation_count": generation_count,
                                    "retry_count": attempt,
                                },
                            },
                        )
                        result = final_state["result"]
                    if current_op_type == "survivor" and result.status != "certified":
                        result = result.model_copy(update={"status": "survivor"})
                    result = result.model_copy(
                        update={
                            "slot": slot,
                            "op_type": current_op_type,
                            "operator_variant": attempt_item.get("operator_variant", ""),
                            "planned_op_type": planned_op_type,
                            "planned_operator_variant": planned_operator_variant,
                            "attempted_op_types": sorted(
                                {
                                    planned_op_type,
                                    current_op_type,
                                    *[
                                        str(card.get("op_type") or "")
                                        for card in attempt_history
                                        if card.get("op_type")
                                    ],
                                }
                            ),
                            "parent_ids": list(attempt_item.get("parent_ids") or []),
                            "variation_axis": attempt_item.get("variation_axis", ""),
                            "target_family": attempt_item.get("target_family", ""),
                            "required_params": dict(attempt_item.get("required_params") or {}),
                            "composition_pattern": attempt_item.get("composition_pattern", ""),
                            "parent_contributions": dict(attempt_item.get("parent_contributions") or {}),
                            "avoid_patterns": list(attempt_item.get("avoid_patterns") or []),
                            "quality_target": attempt_item.get("quality_target", ""),
                            "planner_source": attempt_item.get("planner_source", ""),
                            "generation": generation_count,
                            "retry_count": attempt,
                            "retry_reasons": list(retry_reasons),
                            "attempt_history": _attempt_history_summary(attempt_history),
                            "quality_retry_count": quality_retry_count,
                            "replan_count": replan_count,
                            "replan_reason": replan_reason or None,
                            "replan_source": replan_source or None,
                            "discarded_operator_card": dict(discarded_operator_card),
                            "source_kind": attempt_item.get("source_kind")
                            or ("survivor" if current_op_type == "survivor" else "generated"),
                            "problem_style": result.problem_style
                            or ("theorem_proof" if attempt_item.get("target_style") == "theorem_proof" else "numeric_answer"),
                            "target_style": attempt_item.get("target_style", "numeric_answer"),
                            "certification_route": result.certification_route
                            or _certification_route_for_style(attempt_item.get("target_style", "numeric_answer")),
                            "parent_context_cards": list(attempt_item.get("parent_context_cards") or []),
                            "operator_card": _operator_card(attempt_item),
                            "slot_outcome": result.status,
                        }
                    )
                    with ls.trace(
                        name=f"slot_{slot}.{current_op_type}.quality.attempt_{attempt}",
                        run_type="tool",
                        inputs={
                            "slot": slot,
                            "op_type": current_op_type,
                            "attempt": attempt,
                            "problem_id": result.problem_id,
                            "status": result.status,
                            "parent_ids": result.parent_ids,
                        },
                        tags=["pool-generation", "quality", f"slot:{slot}", current_op_type],
                        metadata={
                            "slot": slot,
                            "op_type": current_op_type,
                            "attempt": attempt,
                            "generation_count": generation_count,
                        },
                    ) as quality_run:
                        quality = verify_slot_quality(result, attempt_item, current_parents)
                        quality = _merge_novelty_memory_quality(
                            result,
                            quality,
                            dict(state.get("novelty_memory") or {}),
                        )
                        quality_run.end(
                            outputs={
                                "quality_verdict": quality.quality_verdict,
                                "quality_flags": quality.quality_flags,
                                "interestingness_score": quality.interestingness_score,
                                "semantic_parent_contribution": dict(
                                    quality.semantic_parent_contribution or {}
                                ),
                                "interestingness_features": dict(
                                    quality.interestingness_features or {}
                                ),
                                "quality_evidence": _compact_quality_evidence_for_pool(
                                    dict(quality.quality_evidence or {})
                                ),
                            }
                        )
                    result = result.model_copy(
                        update={
                            "quality_verdict": quality.quality_verdict,
                            "quality_flags": quality.quality_flags,
                            "interestingness_score": quality.interestingness_score,
                            "feedback_for_next_generation": quality.feedback_for_next_generation,
                            "semantic_parent_contribution": quality.semantic_parent_contribution,
                            "interestingness_features": quality.interestingness_features,
                            "quality_evidence": quality.quality_evidence,
                            "solution_verify_passed": (
                                quality.quality_evidence.get("solution_verification", {}).get("passed")
                            ),
                            "solution_verify_flags": list(
                                quality.quality_evidence.get("solution_verification", {}).get("flags", [])
                            ),
                            "canonical_signature": _canonical_signature(result),
                            "failure_signature": ",".join(sorted(set(quality.quality_flags)))
                            if quality.quality_flags
                            else _failure_signature(result),
                            "quality_retry_count": quality_retry_count,
                            "retry_reasons": list(retry_reasons),
                            "attempt_history": _attempt_history_summary(attempt_history),
                            "attempted_op_types": sorted(
                                {
                                    planned_op_type,
                                    current_op_type,
                                    *[
                                        str(card.get("op_type") or "")
                                        for card in attempt_history
                                        if card.get("op_type")
                                    ],
                                }
                            ),
                            "replan_count": replan_count,
                            "replan_reason": replan_reason or None,
                            "replan_source": replan_source or None,
                            "discarded_operator_card": dict(discarded_operator_card),
                            # The certify routes build results without the slot
                            # loop's attempt index; without this, every summary
                            # reported retry_count=0 even for retried slots.
                            "retry_count": attempt,
                            "retry_exhausted": False,
                        }
                    )
                    last_result = result
                    failure_signature = _failure_signature(last_result)
                    failure_history.append(failure_signature)
                    history_card = _attempt_history_card(
                        attempt=attempt,
                        item=attempt_item,
                        result=last_result,
                        retry_feedback=retry_feedback,
                    )
                    attempt_history.append(history_card)
                    last_result = last_result.model_copy(
                        update={"attempt_history": _attempt_history_summary(attempt_history)}
                    )
                    should_retry = (
                        _retryable_generation_failure(last_result)
                        or _retryable_lean_repair_failure(last_result, current_op_type)
                        or _retryable_theorem_failure(last_result, current_op_type)
                    )
                    if _retryable_generation_failure(last_result):
                        reason = f"contract:{last_result.status}"
                        if reason not in retry_reasons:
                            retry_reasons.append(reason)
                    elif _retryable_lean_repair_failure(last_result, current_op_type):
                        reason = "lean:type_check_failed"
                        if reason not in retry_reasons:
                            retry_reasons.append(reason)
                    elif _retryable_theorem_failure(last_result, current_op_type):
                        if last_result.status == "alignment_failed":
                            reason = "alignment:statement_lean_mismatch"
                        elif last_result.status == "statement_failed":
                            reason = "statement:typecheck_failed"
                        else:
                            reason = "proof:proof_failed"
                        if reason not in retry_reasons:
                            retry_reasons.append(reason)
                    quality_reasons = _quality_retry_reasons(last_result)
                    if not should_retry and quality_reasons:
                        should_retry = True
                        retry_reasons.extend(
                            reason for reason in quality_reasons if reason not in retry_reasons
                        )
                        if attempt < max_attempts - 1:
                            quality_retry_count += 1
                    if not should_retry:
                        break
                    replan_needed = (
                        replan_count == 0
                        and attempt < max_attempts - 1
                        and _is_plan_level_failure(
                            last_result,
                            op_type=current_op_type,
                            failure_history=failure_history,
                            attempt_history=attempt_history,
                        )
                    )
                    with ls.trace(
                        name=f"slot_{slot}.retry_decision",
                        run_type="tool",
                        inputs={
                            "slot": slot,
                            "attempt": attempt,
                            "op_type": current_op_type,
                            "status": last_result.status,
                            "failure_signature": failure_signature,
                            "failure_history": failure_history[-4:],
                            "attempt_history": _attempt_history_trace_summary(attempt_history),
                            "replan_count": replan_count,
                        },
                        tags=["pool-generation", "retry-decision", f"slot:{slot}"],
                    ) as decision_run:
                        decision_run.end(
                            outputs={
                                "should_retry": should_retry,
                                "replan_needed": replan_needed,
                                "failure_class": _failure_class(last_result),
                                "retry_reasons": retry_reasons,
                            }
                        )
                    if replan_needed:
                        with ls.trace(
                            name=f"slot_{slot}.replan_operator_card",
                            run_type="tool",
                            inputs={
                                "slot": slot,
                                "attempt": attempt,
                                "failure_signature": failure_signature,
                                "failure_class": _failure_class(last_result),
                                "attempt_history": _attempt_history_trace_summary(attempt_history),
                                "operator_card": _operator_card(attempt_item),
                            },
                            tags=["pool-generation", "replan", f"slot:{slot}"],
                        ) as replan_run:
                            replanned_item = await _replan_operator_card(
                                attempt_item,
                                last_result,
                                attempt_history,
                                config,
                                replanner=replanner,
                            )
                            replan_count += 1
                            replan_reason = str(
                                replanned_item.get("replan_reason") or failure_signature
                            )
                            replan_source = str(
                                replanned_item.get("replan_source") or "deterministic_fallback"
                            )
                            discarded_operator_card = dict(
                                replanned_item.get("discarded_operator_card")
                                or _operator_card(attempt_item)
                            )
                            item = _attach_novelty_contracts(
                                [replanned_item],
                                dict(state.get("novelty_memory") or {}),
                            )[0]
                            attempt_history[-1]["replan_decision"] = {
                                "replan_source": replan_source,
                                "replan_reason": replan_reason,
                                "operator_card": _operator_card(item),
                            }
                            last_result = last_result.model_copy(
                                update={"attempt_history": _attempt_history_summary(attempt_history)}
                            )
                            replan_run.end(
                                outputs={
                                    "replan_source": replan_source,
                                    "replan_reason": replan_reason,
                                    "operator_card": _operator_card(item),
                                }
                            )
                    if attempt >= max_attempts - 1:
                        exhausted_update = {"retry_exhausted": True}
                        exhausted_update["retry_count"] = attempt
                        exhausted_update["quality_retry_count"] = quality_retry_count
                        exhausted_update["retry_reasons"] = list(retry_reasons)
                        exhausted_update["attempt_history"] = _attempt_history_summary(attempt_history)
                        exhausted_update["attempted_op_types"] = sorted(
                            {
                                planned_op_type,
                                *[
                                    str(card.get("op_type") or "")
                                    for card in attempt_history
                                    if card.get("op_type")
                                ],
                            }
                        )
                        exhausted_update["replan_count"] = replan_count
                        exhausted_update["replan_reason"] = replan_reason or None
                        exhausted_update["replan_source"] = replan_source or None
                        exhausted_update["discarded_operator_card"] = dict(discarded_operator_card)
                        last_result = last_result.model_copy(update=exhausted_update)
                        break
                result = last_result
                if result is None:
                    raise RuntimeError("slot produced no certification result")
                slot_run.end(
                    outputs={
                        "status": result.status,
                        "problem_id": result.problem_id,
                        "lean_level": result.lean_level,
                        "quality_verdict": result.quality_verdict,
                        "quality_flags": result.quality_flags,
                        "retry_count": result.retry_count,
                        "quality_retry_count": result.quality_retry_count,
                        "attempt_history": _attempt_history_trace_summary(
                            list(result.attempt_history or [])
                        ),
                        "replan_count": result.replan_count,
                        "replan_reason": result.replan_reason,
                        "retry_exhausted": result.retry_exhausted,
                    }
                )
                return {
                    "slot_outputs": [_slot_output_ref(result, failed=result.status not in {"certified", "survivor"})]
                }
            except Exception as exc:
                failed_result = CertificationResult(
                    problem_id=f"slot_{slot}_{op_type}_failed",
                    generation=generation_count,
                    slot=slot,
                    op_type=op_type,
                    planned_op_type=op_type,
                    planned_operator_variant=str(item.get("operator_variant") or ""),
                    attempted_op_types=[op_type],
                    parent_ids=list(item.get("parent_ids") or []),
                    variation_axis=item.get("variation_axis", ""),
                    target_family=item.get("target_family", ""),
                    required_params=dict(item.get("required_params") or {}),
                    composition_pattern=item.get("composition_pattern", ""),
                    parent_contributions=dict(item.get("parent_contributions") or {}),
                    avoid_patterns=list(item.get("avoid_patterns") or []),
                    quality_target=item.get("quality_target", ""),
                    quality_verdict="weak",
                    quality_flags=["slot_exception"],
                    interestingness_score=0.0,
                    feedback_for_next_generation="Inspect slot exception before reusing this pattern.",
                    quality_evidence={
                        "checkpoint_coverage": 0.0,
                        "missing_checkpoints": ["slot_execution"],
                        "reasoning_signature": f"slot_exception:{op_type}",
                        "signature_group": "slot_exception",
                        "parent_contribution": {},
                        "feature_delta": {},
                        "novelty_flags": ["slot_exception"],
                    },
                    planner_source=item.get("planner_source", ""),
                    # `item` never carries the loop's attempt state; read the
                    # slot loop variables so slot_exception rows keep telemetry.
                    retry_count=int(attempt),
                    retry_reasons=list(retry_reasons),
                    attempt_history=_attempt_history_summary(attempt_history),
                    quality_retry_count=int(quality_retry_count),
                    retry_exhausted=True,
                    failure_signature="slot_exception",
                    source_kind=item.get("source_kind") or ("generated" if op_type != "survivor" else "survivor"),
                    status="slot_failed",
                    error=str(exc)[:500],
                )
                slot_run.end(
                    outputs={"status": "slot_failed"}, error=f"{type(exc).__name__}: {exc}"
                )
                return {
                    "slot_outputs": [_slot_output_ref(failed_result, failed=True)]
                }

    def slot_aggregate_node(state: PoolState) -> PoolState:
        outputs = sorted(state.get("slot_outputs", []), key=lambda item: int(item.get("slot", 0)))
        results = [_slot_output_result(output) for output in outputs]
        generation_count = int(state.get("generation_count", 0) or 0) + 1
        target_accepted = int(
            state.get("target_accepted_per_generation")
            or DEFAULT_TARGET_ACCEPTED_PER_GENERATION
        )
        reserve_budget = int(state.get("reserve_budget") or DEFAULT_RESERVE_BUDGET)
        generated_results = [result for result in results if _is_generated_result(result)]
        accepted_proxy_count = sum(1 for result in generated_results if _accepted_proxy_pass(result))
        accepted_grade_proxy_count = sum(
            1 for result in generated_results if _accepted_grade_proxy_pass(result)
        )
        if (
            not bool(state.get("disable_reserve_slots"))
            and not bool(state.get("reserve_round_done"))
            and bool(generated_results)
            and reserve_budget > 0
            and accepted_grade_proxy_count < target_accepted
        ):
            reserve_work_items = _build_reserve_work_items(
                state["current_generation"],
                results,
                target_accepted=target_accepted,
                reserve_budget=reserve_budget,
                crossover_count=int(state.get("crossover_count") or 0),
            )
            if reserve_work_items:
                reserve_novelty_memory = _extend_novelty_memory_with_rows(
                    dict(state.get("novelty_memory") or {}),
                    [result.model_dump() for result in results if result.status == "certified"],
                    source_kind="current_generation",
                )
                reserve_work_items = _attach_novelty_contracts(
                    reserve_work_items,
                    reserve_novelty_memory,
                )
                with ls.trace(
                    name="slot_aggregate.reserve_decision",
                    run_type="tool",
                    inputs={
                        "accepted_proxy_count": accepted_proxy_count,
                        "accepted_grade_proxy_count": accepted_grade_proxy_count,
                        "target_accepted_per_generation": target_accepted,
                        "reserve_budget": reserve_budget,
                    },
                    tags=["pool-generation", "reserve"],
                ) as reserve_run:
                    reserve_run.end(
                        outputs={
                            "reserve_slots_run": len(reserve_work_items),
                            "reserve_work_items": [
                                _compact_operator_card_for_pool(_operator_card(item))
                                for item in reserve_work_items
                            ],
                        }
                    )
                return {
                    "reserve_work_items": reserve_work_items,
                    "reserve_round_pending": True,
                    "dispatch_mode": "reserve",
                    "novelty_memory": reserve_novelty_memory,
                    "generation_feedback": {
                        "accepted_proxy_count": accepted_proxy_count,
                        "accepted_grade_proxy_count": accepted_grade_proxy_count,
                        "target_accepted_per_generation": target_accepted,
                        "reserve_slots_run": len(reserve_work_items),
                        "reserve_trigger": "accepted_grade_proxy_below_target",
                    },
                }
        pool_size = int(state.get("pool_size", POOL_SIZE))
        output_path_value = state.get("output_path")
        backfill_seed_archive = _build_backfill_seed_archive(
            state["current_generation"],
            output_path=Path(str(output_path_value)) if output_path_value else None,
            generation_count=generation_count,
        )
        with ls.trace(
            name="aggregate.orchestrator_select",
            run_type="tool",
            inputs={
                "slot_count": len(results),
                "pool_size": pool_size,
                "current_generation_seed_count": len(state["current_generation"]),
                "previous_seed_count": len(backfill_seed_archive),
            },
            tags=["pool-generation", "aggregate", "orchestrator-select"],
        ) as aggregate_run:
            aggregate = select_next_pool_with_orchestrator(
                results=results,
                current_generation=state["current_generation"],
                pool_size=pool_size,
                generation_count=generation_count,
                target_accepted=target_accepted,
                planner=dict(state.get("planner") or {}),
                generation_feedback=dict(state.get("generation_feedback") or {}),
                backfill_seed_archive=backfill_seed_archive,
                aggregate_selector=aggregate_selector,
                generation_model=state.get("generation_model"),
                generation_temperature=state.get("generation_temperature"),
            )
            aggregate_run.end(outputs=aggregate["selector_manifest"])
        results = list(aggregate["results"])
        approved = list(aggregate["approved_candidates"])
        failed_slots = list(aggregate["failed_slots"])
        generation_feedback = dict(aggregate["generation_feedback"])
        plan_outcome_cards = list(aggregate["plan_outcome_cards"])
        generation_survival_status = str(aggregate["generation_survival_status"])
        with ls.trace(
            name="aggregate.invariant_validation",
            run_type="tool",
            inputs={"event_count": len(aggregate["invariant_events"])},
            tags=["pool-generation", "aggregate", "invariant-validation"],
        ) as invariant_run:
            invariant_run.end(outputs={"events": aggregate["invariant_events"][:40]})
        with ls.trace(
            name="aggregate.backfill",
            run_type="tool",
            inputs={"approved_count": len(approved), "required_pool_size": pool_size},
            tags=["pool-generation", "aggregate", "backfill"],
        ) as backfill_run:
            backfill_run.end(
                outputs={
                    "approved_count": len(approved),
                    "backfill_count": len(aggregate["backfill_events"]),
                    "backfill_events": aggregate["backfill_events"],
                }
            )
        with ls.trace(
            name="slot_aggregate.outcome_cards",
            run_type="tool",
            inputs={"card_count": len(plan_outcome_cards)},
            tags=["pool-generation", "outcome-cards"],
        ) as outcome_run:
            outcome_run.end(outputs={"plan_outcome_summary": generation_feedback["plan_outcome_summary"]})
        with ls.trace(
            name="slot_aggregate.survival_decision",
            run_type="tool",
            inputs={
                "approved_count": len(approved),
                "pool_size": pool_size,
                "backfill_count": len(aggregate["backfill_events"]),
                "failed_count": len(failed_slots),
            },
            tags=["pool-generation", "survival-decision"],
        ) as survival_run:
            survival_run.end(
                outputs={
                    "generation_survival_status": generation_survival_status,
                    "continue_generation": generation_survival_status in {"complete", "complete_with_backfill"},
                }
            )
        return {
            "results": [_certification_result_manifest(result) for result in results],
            "results_ref": _results_state_ref(results),
            "approved_candidates": approved,
            "failed_slots": failed_slots,
            "generation_feedback": generation_feedback,
            "plan_outcome_cards": plan_outcome_cards,
            "current_generation": [
                CertificationInput(
                    id=str(item["id"]),
                    statement=str(item.get("statement") or ""),
                    answer=str(item.get("answer") or ""),
                    metadata={
                        k: v for k, v in item.items() if k not in {"id", "statement", "answer"}
                    },
                )
                for item in approved
            ],
            "generation_count": int(state.get("generation_count", 0) or 0) + 1,
            "generation_survival_status": generation_survival_status,
            "novelty_memory": dict(state.get("novelty_memory") or {}),
        }

    def save_generation_node(state: PoolState) -> PoolState:
        output_path = Path(state["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed_at = datetime.now().astimezone()
        completed_at_iso = completed_at.isoformat(timespec="seconds")
        completed_at_compact = completed_at.strftime("%Y%m%d_%H%M%S")
        run_manifest = _build_run_manifest(
            input_path=Path(state["input_path"]),
            output_path=output_path,
            run_name=str(state.get("run_name") or ""),
            generation_model=state.get("generation_model"),
            max_generations=int(state.get("max_generations") or state.get("generation_count") or 1),
            pool_size=int(state.get("pool_size", POOL_SIZE)),
            completed_at=completed_at_iso,
            completed_at_compact=completed_at_compact,
        )
        run_manifest_digest = _run_manifest_digest(run_manifest)
        results = sorted(_state_results(state), key=lambda result: result.slot or 0)
        for output in state.get("slot_outputs", []) or []:
            _runtime_store_pop(output.get("result_ref"), None)
        if state.get("results_ref"):
            _runtime_store_pop(state.get("results_ref"), None)
        # Public count fields still use "passed" for compatibility; these rows
        # are the full written generation result rows, not only next-pool rows.
        generation_result_rows = [
            {
                **result.model_dump(),
                "completed_at": completed_at_iso,
                "completed_at_compact": completed_at_compact,
            }
            for result in results
        ]
        pending_previous_results = list(state.get("all_passed_results", []) or [])
        previous_count = int(
            state.get("all_passed_results_count")
            or len(pending_previous_results)
            or 0
        )
        rows_to_write = pending_previous_results + generation_result_rows
        for row in rows_to_write:
            row["completed_at"] = completed_at_iso
            row["completed_at_compact"] = completed_at_compact
            if not row.get("release_id"):
                row["release_id"] = run_manifest["bench_version"]
            if not row.get("source_run"):
                row["source_run"] = run_manifest["run_name"]
        # Two slots can prove the same theorem, and both then certify. Keeping
        # both inflates the corpus without adding a problem, so the surface
        # decides identity here as it does in the novelty ledger: the first
        # writing of a statement stays and later repeats are dropped. Rows
        # already on disk from an earlier generation of this run are read back
        # so the check spans the whole file, not just this batch.
        append_only = output_path.exists() and not pending_previous_results
        already: set = set()
        if append_only:
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    already.add(_novelty_row_key(json.loads(line)))
        deduped_rows = []
        for row in rows_to_write:
            if str(row.get("status")) != "certified":
                deduped_rows.append(row)
                continue
            key = _novelty_row_key(row)
            if key and key in already:
                continue
            if key:
                already.add(key)
            deduped_rows.append(row)
        dropped = len(rows_to_write) - len(deduped_rows)
        if dropped:
            print(f"[dedup] dropped {dropped} certified row(s) already present", flush=True)
        rows_to_write = deduped_rows
        with output_path.open("a" if append_only else "w", encoding="utf-8") as fh:
            for row in rows_to_write:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Count what was written, not what was produced: rows_to_write has
        # already been deduplicated by key against the file, and a summary that
        # counted the pre-dedup list reported more rows than the file held.
        # `previous_count` already covers the pending previous results that
        # rows_to_write re-carries, so add only what this write puts on disk
        # beyond them.
        cumulative_count = previous_count - len(pending_previous_results) + len(rows_to_write)

        counts = dict(Counter(result.status for result in results))
        failed_slots = list(state.get("failed_slots", []) or [])
        generation_save_status = str(
            state.get("generation_survival_status")
            or (
                "complete"
                if len(state.get("approved_candidates", [])) == int(state.get("pool_size", POOL_SIZE))
                else "partial"
            )
        )
        generation_snapshot = {
            "generation": int(state.get("generation_count", 0) or 0),
            "planner": dict(state.get("planner", {}) or {}),
            "counts": counts,
            "accepted_proxy_count": int(
                (state.get("generation_feedback", {}) or {}).get("accepted_proxy_count", 0)
                or 0
            ),
            "accepted_grade_proxy_count": int(
                (state.get("generation_feedback", {}) or {}).get("accepted_grade_proxy_count", 0)
                or 0
            ),
            "reserve_slots_run": int(
                (state.get("generation_feedback", {}) or {}).get("reserve_slots_run", 0)
                or 0
            ),
            "reserve_slots_selected": int(
                (state.get("generation_feedback", {}) or {}).get("reserve_slots_selected", 0)
                or 0
            ),
            "yield_funnel": dict(
                (state.get("generation_feedback", {}) or {}).get("yield_funnel", {})
                or {}
            ),
            "saved_pool": _results_trace_manifest(list(state.get("approved_candidates", []) or []), limit=20),
            "passed_results_count": len(generation_result_rows),
            "cumulative_passed_results_count": cumulative_count,
            "deduplicated_on_write": dropped,
            "failed_slots": _results_trace_manifest(failed_slots, limit=20),
            "generation_feedback": _generation_feedback_trace_manifest(
                dict(state.get("generation_feedback", {}) or {})
            ),
            "planner_memory": _planner_memory_trace_manifest(
                dict(state.get("planner_memory", {}) or {})
            ),
            "planner_case_pack": _planner_memory_trace_manifest(
                dict(state.get("planner_case_pack", state.get("planner_memory", {})) or {})
            ),
            "novelty_memory": _novelty_memory_trace_manifest(
                dict(state.get("novelty_memory", {}) or {})
            ),
            "generation_save_status": generation_save_status,
            "run_manifest_digest": run_manifest_digest,
            "completed_at": completed_at_iso,
            "completed_at_compact": completed_at_compact,
        }
        generations = list(state.get("generations", []) or []) + [generation_snapshot]
        generation_zero = dict(state.get("generation_zero", {}) or {})
        if generation_zero:
            generation_zero["run_manifest_digest"] = run_manifest_digest
        summary = PoolRunResult(
            run_name=state.get("run_name", ""),
            generation_count=int(state.get("generation_count", 0) or 0),
            pool_size=int(state.get("pool_size", POOL_SIZE)),
            input_path=state.get("input_path", ""),
            output_path=str(output_path),
            summary_output_path=state.get("summary_output_path"),
            planner=dict(state.get("planner", {}) or {}),
            run_manifest=run_manifest,
            planner_memory=_planner_memory_trace_manifest(
                dict(state.get("planner_memory", {}) or {})
            ),
            planner_case_pack=_planner_memory_trace_manifest(
                dict(state.get("planner_case_pack", state.get("planner_memory", {})) or {})
            ),
            novelty_memory=_novelty_memory_trace_manifest(
                dict(state.get("novelty_memory", {}) or {})
            ),
            work_items=list(state.get("work_items", []) or []),
            counts=counts,
            saved_pool=_results_trace_manifest(list(state.get("approved_candidates", []) or []), limit=20),
            failed_slots=_results_trace_manifest(failed_slots, limit=20),
            passed_results_count=len(generation_result_rows),
            cumulative_passed_results_count=cumulative_count,
            deduplicated_on_write=dropped,
            generation_zero=generation_zero,
            gen0_enriched_input_path=state.get("gen0_enriched_input_path"),
            generation_feedback=_generation_feedback_trace_manifest(
                dict(state.get("generation_feedback", {}) or {})
            ),
            generations=generations,
            generation_save_status=generation_save_status,
            completed_at=completed_at_iso,
            completed_at_compact=completed_at_compact,
            langsmith_trace_hint={
                "root_run_name": f"pool_generation/{state.get('run_name', '')}",
                "generation_graph_run_name": f"generation.{state.get('generation_count', 0)}.graph",
            },
        ).model_dump()
        summary_output_path = state.get("summary_output_path")
        if summary_output_path:
            path = Path(summary_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "generations": generations,
            "run_manifest": run_manifest,
            "results_ref": "",
            "all_passed_results": [],
            "all_passed_results_count": cumulative_count,
            "novelty_memory": dict(state.get("novelty_memory") or {}),
        }

    graph.add_node("01_load_seed_pool", load_seed_pool_node)
    graph.add_node("02_orchestrator_plan_generation", plan_generation_node)
    graph.add_node("03_slot_dispatch", slot_dispatch_node)
    graph.add_node("04_slot_unit", slot_unit_node)
    graph.add_node("05_slot_aggregate", slot_aggregate_node)
    graph.add_node("06_save_generation", save_generation_node)

    graph.set_entry_point("01_load_seed_pool")
    graph.add_edge("01_load_seed_pool", "02_orchestrator_plan_generation")
    graph.add_edge("02_orchestrator_plan_generation", "03_slot_dispatch")
    graph.add_conditional_edges(
        "03_slot_dispatch",
        slot_dispatch_route,
        ["04_slot_unit", "05_slot_aggregate"],
    )
    graph.add_edge("04_slot_unit", "05_slot_aggregate")
    graph.add_conditional_edges(
        "05_slot_aggregate",
        slot_aggregate_route,
        ["03_slot_dispatch", "06_save_generation"],
    )
    graph.add_edge("06_save_generation", END)
    compiled = graph.compile()
    compiled.slot_dispatch_route = slot_dispatch_route  # test hook
    compiled.slot_aggregate_route = slot_aggregate_route  # test hook
    return compiled


def _needs_generation_zero_proof_completion(problem: CertificationInput) -> bool:
    return (
        _problem_style(problem) == "theorem_proof"
        and not bool(_parent_proof_context(problem).get("proof_body_available"))
        and _prompt_text((problem.metadata or {}).get("formal_statement") or (problem.metadata or {}).get("lean_code"))
        != "not_available"
    )


def _lean_statement_without_proof(source: Any) -> str:
    text = str(source or "").strip()
    if not text:
        return ""
    theorem_match = re.search(r"\b(?:theorem|lemma|def)\s+", text)
    if theorem_match:
        prefix = text[: theorem_match.start()]
        declaration = text[theorem_match.start() :]
        proof_match = re.search(r":=\s*by\b", declaration)
        if proof_match:
            declaration = declaration[: proof_match.start()].rstrip() + " :="
        elif ":=" in declaration:
            declaration = declaration.split(":=", 1)[0].rstrip() + " :="
        return f"{prefix}{declaration}".strip()
    return text


def _lean_decl_name(source: Any) -> str:
    match = re.search(r"\b(?:theorem|lemma|def)\s+([A-Za-z0-9_'.]+)", str(source or ""))
    return match.group(1) if match else ""


def _normalize_lean_sum_notation(text: str) -> str:
    """Normalize a small common Finset-sum notation difference.

    Gen0 should accept a proof that changes only from `∑ k in s, f k` to
    `s.sum fun k => f k`; local Lean verification already proved the code.
    This normalization is deliberately narrow so unrelated statements do not
    pass the alignment gate.
    """
    pattern = re.compile(
        r"∑\s+([A-Za-z_][A-Za-z0-9_']*)\s+(?:in|∈)\s+([^,=:]+?),\s*([^)=:]+)"
    )

    def replace(match: re.Match[str]) -> str:
        var = match.group(1).strip()
        domain = match.group(2).strip()
        body = match.group(3).strip()
        return f"({domain}).sum fun {var} => {body}"

    return pattern.sub(replace, text)


def _lean_alignment_surface(source: Any) -> str:
    text = _lean_statement_without_proof(source)
    text = _normalize_lean_sum_notation(text)
    text = re.sub(r"^\s*(?:import|open|set_option)\b.*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"\b(?:noncomputable[ \t]+)?abbrev[ \t]+([A-Za-z0-9_'.]*_solution)[ \t]*:[ \t]*([^:=\n]+?)[ \t]*:=[ \t]*[^\n]*",
        r"abbrev \1 : \2 := _",
        text,
    )
    text = re.sub(r"\b(theorem|lemma|def)\s+[A-Za-z0-9_'.]+", r"\1 _", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("(", "").replace(")", "")
    return text.lower()


def _gen0_proof_matches_formal_statement(formal_statement: Any, lean_code: Any) -> bool:
    formal = str(formal_statement or "").strip()
    code = str(lean_code or "").strip()
    if not formal or not code:
        return False
    lowered = code.lower()
    if any(marker in lowered for marker in ("placeholder", "not provided", "not included", "not visible")):
        return False
    decl_name = _lean_decl_name(formal)
    if decl_name and _lean_decl_name(code) != decl_name:
        return False
    formal_surface = _lean_alignment_surface(formal)
    code_surface = _lean_alignment_surface(code)
    return bool(decl_name) and bool(formal_surface) and formal_surface == code_surface


def _gen0_latest_turn(completion: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in reversed(list(completion.get("attempts") or [])):
        turns = list(attempt.get("turn_diagnostics") or [])
        if turns:
            return dict(turns[-1])
    return {}


def _gen0_failure_packet(completion: Dict[str, Any]) -> Dict[str, Any]:
    latest = _gen0_latest_turn(completion)
    summaries: List[str] = []
    best_partial = ""
    for attempt in completion.get("attempts") or []:
        for turn in attempt.get("turn_diagnostics") or []:
            summary = str(turn.get("summary") or "")
            if summary:
                summaries.append(summary[:240])
    for attempt in completion.get("attempts") or []:
        final_proof = str(attempt.get("final_proof") or "")
        if final_proof:
            best_partial = final_proof
            break
    return {
        "failure_class": "proof_failed" if latest else "generation_failed",
        "latest_summary": str(latest.get("summary") or "no verifier diagnostics captured")[:800],
        "recent_signatures": summaries[-4:],
        "attempt_count": len(completion.get("attempts") or []),
        "best_partial_proof": _prompt_text(best_partial, limit=4000),
    }


def _gen0_status(problem: CertificationInput) -> str:
    metadata = problem.metadata or {}
    completion = _metadata_dict(metadata.get("gen0_proof_completion"))
    if _metadata_bool(metadata.get("gen0_proof_completed")):
        return "certified"
    if completion.get("status") == "timeout":
        return "timeout"
    if completion.get("status") == "skipped":
        return "skipped"
    return "proof_failed"


def _gen0_target_count(pool: List[CertificationInput]) -> int:
    return sum(1 for problem in pool if _needs_generation_zero_proof_completion(problem))


def _effective_gen0_parallelism(requested: int, target_seed_count: int) -> int:
    if target_seed_count <= 0:
        return 0
    return min(max(1, int(requested)), target_seed_count, GEN0_MAX_PARALLEL_CAP)


def _generation_zero_work_items(
    pool: List[CertificationInput],
    *,
    config: GenerationConfig,
    proof_k: int,
    proof_turns: int,
    requested_parallelism: int,
    effective_parallelism: int,
) -> List[Gen0SeedWorkItem]:
    items: List[Gen0SeedWorkItem] = []
    for problem in pool:
        if not _needs_generation_zero_proof_completion(problem):
            continue
        metadata = dict(problem.metadata or {})
        items.append(
            Gen0SeedWorkItem(
                seed_id=problem.id,
                problem_style=_problem_style(problem),
                formal_statement=str(metadata.get("formal_statement") or ""),
                lean_header=str(metadata.get("lean_header") or THEOREM_CANONICAL_HEADER),
                proof_body_available_before=bool(_parent_proof_context(problem).get("proof_body_available")),
                proof_k=proof_k,
                proof_turns=proof_turns,
                model=config.model,
                lane=(len(items) % max(1, effective_parallelism)) if effective_parallelism else 0,
                requested_parallelism=max(1, int(requested_parallelism)),
                effective_parallelism=effective_parallelism,
            )
        )
    return items


def _generation_zero_summary(
    original_pool: List[CertificationInput],
    completed_pool: List[CertificationInput],
) -> Dict[str, Any]:
    by_id = {problem.id: problem for problem in completed_pool}
    seed_summaries: List[Dict[str, Any]] = []
    missing_count = 0
    completed_count = 0
    failed_count = 0
    target_count = 0
    work_items: List[Dict[str, Any]] = []
    worker_count_requested = 0
    worker_count_effective = 0
    for original in original_pool:
        completed = by_id.get(original.id, original)
        completed_metadata = completed.metadata or {}
        was_target = _metadata_bool(completed_metadata.get("gen0_target"))
        if _needs_generation_zero_proof_completion(original):
            missing_count += 1
        status = _gen0_status(completed) if was_target else "not_needed"
        if was_target:
            target_count += 1
        if was_target and status == "certified":
            completed_count += 1
        elif was_target and status != "skipped":
            failed_count += 1
        completion = _metadata_dict(completed_metadata.get("gen0_proof_completion"))
        failure_packet = _metadata_dict(completed_metadata.get("gen0_failure_packet"))
        work_item = _metadata_dict(completed_metadata.get("gen0_work_item"))
        if work_item:
            work_items.append(work_item)
            worker_count_requested = int(completed_metadata.get("gen0_worker_count_requested") or worker_count_requested)
            worker_count_effective = int(completed_metadata.get("gen0_worker_count_effective") or worker_count_effective)
        seed_summaries.append(
            {
                "seed_id": completed.id,
                "problem_style": _problem_style(completed),
                "was_target": was_target,
                "proof_body_available_before": bool(_parent_proof_context(original).get("proof_body_available")),
                "proof_body_available_after": bool(_parent_proof_context(completed).get("proof_body_available")),
                "status": status,
                "attempts": len(completion.get("attempts") or []),
                "min_turns_to_success": completion.get("min_turns_to_success"),
                "elapsed_seconds": completion.get("total_elapsed_seconds", 0.0),
                "failure_class": "" if status == "certified" else failure_packet.get("failure_class", ""),
                "lane": work_item.get("lane", completed_metadata.get("gen0_lane", "")) if work_item else "",
            }
        )
    return {
        "total_seed_count": len(original_pool),
        "missing_proof_body_count": missing_count,
        "target_seed_count": target_count,
        "worker_count_requested": worker_count_requested,
        "worker_count_effective": worker_count_effective,
        "worker_count_source": "cli_or_default",
        "work_items": work_items,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "seeds": seed_summaries,
    }


def _generation_zero_rows(pool: List[CertificationInput]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slot, problem in enumerate(pool):
        metadata = dict(problem.metadata or {})
        if not _metadata_bool(metadata.get("gen0_target")):
            continue
        proof_context = _parent_proof_context(problem)
        status = _gen0_status(problem)
        completion = _metadata_dict(metadata.get("gen0_proof_completion"))
        failure_packet = _metadata_dict(metadata.get("gen0_failure_packet"))
        failure_class = str(failure_packet.get("failure_class") or "")
        quality_flags = [] if status == "certified" else [failure_class or "seed_proof_completion_failed"]
        quality_evidence = {
            "checkpoint_coverage": 1.0 if status == "certified" else 0.0,
            "missing_checkpoints": [] if status == "certified" else ["seed_proof_completion"],
            "reasoning_signature": f"gen0:{problem.id}",
            "signature_group": "gen0_seed_proof_completion",
            "parent_contribution": {},
            "feature_delta": {},
            "novelty_flags": quality_flags,
            "failure_class": failure_class,
        }
        quality_evidence["misformalization"] = derive_misformalization_taxonomy(
            CertificationResult(
                problem_id=problem.id,
                status=status,
                error=failure_packet.get("latest_summary"),
                proof_verify_summary=failure_packet.get("latest_summary"),
            ),
            quality_flags,
            quality_evidence,
        )
        rows.append(
            CertificationResult(
                problem_id=problem.id,
                generation=0,
                slot=slot,
                operation="seed_proof_completion",
                op_type="seed_proof_completion",
                parent_ids=[],
                parent_eligible=bool(proof_context.get("proof_body_available")),
                selection_reason="seed_proof_completed" if status == "certified" else "seed_proof_completion_failed",
                quality_verdict="acceptable" if status == "certified" else "weak",
                quality_flags=quality_flags,
                quality_evidence=quality_evidence,
                source_kind="seed",
                problem_style=_problem_style(problem),
                target_style="theorem_proof",
                certification_route="theorem_prover",
                slot_outcome=status,
                statement=problem.statement,
                answer=problem.answer,
                solution=metadata.get("solution"),
                verification_code=metadata.get("verification_code"),
                formal_statement=metadata.get("formal_statement"),
                lean_header=metadata.get("lean_header"),
                formal_status="certified" if status == "certified" else status,
                benchmark=metadata.get("benchmark"),
                family="theorem_proof",
                status=status,
                lean_level=3 if status == "certified" else 0,
                lean_code=metadata.get("lean_code"),
                anti_stub_passed=status == "certified",
                aligned=status == "certified",
                llm_used=True,
                llm_model=completion.get("provider_slug"),
                error=None if status == "certified" else failure_packet.get("latest_summary"),
                elapsed_seconds=float(completion.get("total_elapsed_seconds") or 0.0),
                proof_verify_summary=failure_packet.get("latest_summary"),
                gen0_failure_packet=failure_packet,
                input_metadata=metadata,
            ).model_dump()
        )
    return rows


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _write_generation_zero_enriched_seed_csv(
    input_path: Path,
    output_path: Path,
    completed_pool: List[CertificationInput],
) -> Path:
    """Persist Gen0 proof completions in a reusable seed CSV without mutating the raw file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    completed_by_id = {problem.id: problem for problem in completed_pool}
    with input_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        original_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    extra_fieldnames = [
        "problem_style",
        "certification_route",
        "formal_statement",
        "lean_header",
        "lean_code",
        "formal_status",
        "proof_body_available",
        "seed_formal_statement_original",
        "gen0_target",
        "gen0_proof_completed",
        "gen0_failure_packet",
        "gen0_proof_completion",
    ]
    fieldnames = list(original_fieldnames)
    for name in extra_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)

    enriched_rows: List[Dict[str, str]] = []
    for row in rows:
        problem_id = str(row.get("id") or row.get("release_id") or "").strip()
        completed = completed_by_id.get(problem_id)
        enriched = dict(row)
        if completed is not None:
            metadata = dict(completed.metadata or {})
            proof_context = _parent_proof_context(completed)
            updates = {
                "problem_style": _problem_style(completed),
                "certification_route": _certification_route_for_style(_problem_style(completed)),
                "formal_statement": metadata.get("formal_statement"),
                "lean_header": metadata.get("lean_header"),
                "lean_code": metadata.get("lean_code"),
                "formal_status": metadata.get("formal_status") or _gen0_status(completed),
                "proof_body_available": bool(proof_context.get("proof_body_available")),
                "seed_formal_statement_original": metadata.get("seed_formal_statement_original"),
                "gen0_target": _metadata_bool(metadata.get("gen0_target")),
                "gen0_proof_completed": _metadata_bool(metadata.get("gen0_proof_completed")),
                "gen0_failure_packet": _metadata_dict(metadata.get("gen0_failure_packet")),
                "gen0_proof_completion": _metadata_dict(metadata.get("gen0_proof_completion")),
            }
            enriched.update({key: _csv_cell(value) for key, value in updates.items()})
        enriched_rows.append({field: _csv_cell(enriched.get(field, "")) for field in fieldnames})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    return output_path


def _parent_is_backfill_eligible(parent: CertificationInput) -> bool:
    metadata = dict(parent.metadata or {})
    status = str(metadata.get("status") or "certified")
    if str(metadata.get("quality_verdict") or "acceptable") == "weak":
        return False
    if status not in {"certified", "survivor"}:
        return False
    if _problem_style(parent) == "theorem_proof":
        proof_context = _parent_proof_context(parent)
        if not bool(proof_context.get("proof_body_available")):
            return False
        if _metadata_bool(metadata.get("gen0_target")) and not _metadata_bool(metadata.get("gen0_proof_completed")):
            return False
    return True


async def _call_seed_proof_completer(
    completer: SeedProofCompleterFn,
    problem: CertificationInput,
    config: GenerationConfig,
    **kwargs: Any,
) -> CertificationInput:
    signature = inspect.signature(completer)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported_kwargs = {
        key: value for key, value in kwargs.items() if accepts_kwargs or key in signature.parameters
    }
    result = completer(problem, config, **supported_kwargs) if supported_kwargs else completer(problem, config)
    enriched = await _maybe_await(result)
    if not isinstance(enriched, CertificationInput):
        enriched = CertificationInput.model_validate(enriched)
    return enriched


async def _default_seed_proof_completer(
    problem: CertificationInput,
    config: GenerationConfig,
    *,
    proof_k: int = 4,
    proof_turns: int = 6,
    llm_timeout: float = 240.0,
    lean_timeout: float = 300.0,
    leansearch_enabled: bool = DEFAULT_LEANSEARCH_ENABLED,
    leansearch_limit: int = DEFAULT_LEANSEARCH_LIMIT,
) -> CertificationInput:
    """Fill missing theorem seed proof bodies with the existing Lean proof loop."""
    use_codex_cli = os.getenv("GENERATION_PROVIDER", "").lower() == "codex_cli"
    if not use_codex_cli and not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return problem.model_copy(
            update={
                "metadata": {
                    **dict(problem.metadata or {}),
                    "gen0_target": True,
                    "gen0_proof_completion": {
                        "status": "skipped",
                        "reason": "no LLM API key available",
                    },
                }
            }
        )

    from src.evaluation.model_runner import ModelConfig
    from src.evaluation.multi_turn_prover import ModelTurnResponse, prove_with_refinement
    from src.evaluation.proof_orchestrator import _select_verifier

    client = None if use_codex_cli else _openai_client(config.model)

    async def model_call(model_config, system, user, temperature, max_tokens):
        if use_codex_cli:
            with ls.trace(
                name="codex_cli_call",
                run_type="llm",
                inputs={
                    "model": model_config.provider_slug,
                    "system_chars": len(system),
                    "user_chars": len(user),
                    "schema_enabled": False,
                    "phase": "gen0_seed_proof_completion",
                },
                tags=["codex-cli", "llm", "gen0"],
            ) as codex_run:
                response = await call_codex_cli(
                    model=model_config.provider_slug,
                    system=system,
                    user=user,
                    timeout_seconds=llm_timeout,
                    cwd=Path.cwd(),
                )
                codex_run.end(
                    outputs={
                        "finish_reason": response.finish_reason,
                        "elapsed_seconds": response.elapsed_seconds,
                        "stdout_chars": len(response.raw_text or ""),
                        "stderr_tail": (response.stderr or "")[-1000:],
                        "stdout_tail": (response.stdout or "")[-1000:],
                        "error": response.error,
                    }
                )
            return ModelTurnResponse(
                raw_text=response.raw_text,
                finish_reason=response.finish_reason,
                elapsed_seconds=response.elapsed_seconds,
                error=response.error,
            )
        started = time.monotonic()
        request: Dict[str, Any] = {
            "model": model_config.provider_slug,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        extra_body = _openrouter_extra_body(model_config.provider_slug)
        if extra_body:
            request["extra_body"] = extra_body
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(client.chat.completions.create, **request),
                timeout=llm_timeout,
            )
            choice = response.choices[0]
            return ModelTurnResponse(
                raw_text=choice.message.content or "",
                finish_reason=getattr(choice, "finish_reason", None),
                elapsed_seconds=time.monotonic() - started,
            )
        except asyncio.TimeoutError:
            return ModelTurnResponse(
                raw_text="",
                finish_reason="timeout",
                elapsed_seconds=time.monotonic() - started,
                error=f"timeout after {llm_timeout}s",
            )
        except Exception as exc:
            return ModelTurnResponse(
                raw_text="",
                finish_reason="error",
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    metadata = dict(problem.metadata or {})
    formal_prefix = str(metadata.get("formal_statement") or metadata.get("lean_code") or "").strip()
    formal_statement = str(metadata.get("formal_statement") or _lean_statement_without_proof(formal_prefix)).strip()
    header = str(metadata.get("lean_header") or THEOREM_CANONICAL_HEADER).strip()
    proof_context = json.dumps(
        {
            "solution": _prompt_text(metadata.get("solution")),
            "verification_code": {
                "kind": _verification_code_kind(metadata.get("verification_code")),
                "content": _prompt_text(metadata.get("verification_code"), limit=3000),
            },
            "existing_lean_code": _prompt_text(metadata.get("lean_code"), limit=3000),
            "rule": "Use artifacts as hints only. Preserve the theorem statement; complete the proof body.",
        },
        ensure_ascii=False,
        indent=2,
    )
    premise_query_counter = {"count": 0}

    async def premise_context_provider(**kwargs: Any) -> Dict[str, Any]:
        phase = str(kwargs.get("phase") or "initial")
        return await _theorem_premise_pack_payload(
            statement=str(kwargs.get("statement") or problem.statement),
            formal_statement=str(kwargs.get("formal_prefix") or formal_statement or formal_prefix),
            diagnostics=str(kwargs.get("diagnostics") or ""),
            leansearch_enabled=leansearch_enabled,
            leansearch_limit=leansearch_limit,
            phase=phase,
            query_counter=premise_query_counter,
        )

    proof_attempt = await prove_with_refinement(
        benchmark=str(metadata.get("benchmark") or "seed_gen0"),
        arm="generation_zero_seed_completion",
        problem_id=problem.id,
        statement=problem.statement,
        formal_prefix=formal_prefix,
        header=header,
        model_config=ModelConfig(label="Gen0ProofCompleter", provider_slug=config.model, temperature=0.0),
        model_call=model_call,
        K=proof_k,
        T_max=proof_turns,
        first_temperature=0.0,
        refine_temperature=0.0,
        max_tokens=None,
        lean_timeout=lean_timeout,
        proof_context=proof_context,
        # Gen-0 runs K attempts x T turns per seed, so a cold `lake env lean`
        # per turn (~25 s) dominates the wall clock. Honour LEAN_VERIFIER so a
        # warm REPL can be used instead.
        verifier=_select_verifier(),
        trace_run_prefix=f"gen0.seed.{problem.id}",
        premise_context_provider=premise_context_provider if leansearch_enabled else None,
    )
    completion = proof_attempt.to_summary()
    if proof_attempt.pass_at_k and proof_attempt.attempts and proof_attempt.attempts[-1].final_proof:
        completed_code = proof_attempt.attempts[-1].final_proof or ""
        if not _gen0_proof_matches_formal_statement(formal_statement, completed_code):
            completion["pass_at_k"] = False
            completion["alignment_failed"] = True
            completion["alignment_reason"] = "completed Lean code does not prove the requested formal_statement"
            return problem.model_copy(
                update={
                    "metadata": {
                        **metadata,
                        "gen0_target": True,
                        "seed_formal_statement_original": metadata.get("formal_statement"),
                        "gen0_proof_completed": False,
                        "gen0_proof_completion": completion,
                        "gen0_failure_packet": {
                            **_gen0_failure_packet(completion),
                            "failure_class": "alignment_failed",
                            "latest_summary": "Gen0 Lean code was complete but did not match the requested formal_statement.",
                            "requested_formal_statement": _prompt_text(formal_statement, limit=4000),
                            "completed_lean_code": _prompt_text(completed_code, limit=8000),
                            "requested_surface": _prompt_text(_lean_alignment_surface(formal_statement), limit=1200),
                            "completed_surface": _prompt_text(_lean_alignment_surface(completed_code), limit=1200),
                        },
                    }
                }
            )
        return problem.model_copy(
            update={
                "metadata": {
                    **metadata,
                    "gen0_target": True,
                    "seed_formal_statement_original": metadata.get("formal_statement"),
                    "lean_code": completed_code,
                    "formal_statement": formal_statement,
                    "formal_status": "certified",
                    "gen0_failure_packet": {},
                    "gen0_proof_completed": True,
                    "gen0_proof_completion": completion,
                    "status": "certified",
                    "quality_verdict": metadata.get("quality_verdict") or "acceptable",
                }
            }
        )
    return problem.model_copy(
        update={
            "metadata": {
                **metadata,
                "gen0_target": True,
                "gen0_proof_completed": False,
                "gen0_proof_completion": completion,
                "gen0_failure_packet": _gen0_failure_packet(completion),
            }
        }
    )


async def _complete_generation_zero_proofs(
    pool: List[CertificationInput],
    *,
    config: GenerationConfig,
    seed_proof_completer: Optional[SeedProofCompleterFn] = None,
    proof_k: int = 4,
    proof_turns: int = 6,
    max_seed_seconds: float = 3600.0,
    max_parallel: int = DEFAULT_GEN0_MAX_PARALLEL,
    llm_timeout: float = 240.0,
    lean_timeout: float = 300.0,
    leansearch_enabled: bool = DEFAULT_LEANSEARCH_ENABLED,
    leansearch_limit: int = DEFAULT_LEANSEARCH_LIMIT,
) -> List[CertificationInput]:
    completer = seed_proof_completer or _default_seed_proof_completer
    completed: List[Optional[CertificationInput]] = [None for _ in pool]
    target_seed_count = _gen0_target_count(pool)
    requested_parallelism = max(1, int(max_parallel))
    effective_parallelism = _effective_gen0_parallelism(requested_parallelism, target_seed_count)
    work_items = _generation_zero_work_items(
        pool,
        config=config,
        proof_k=proof_k,
        proof_turns=proof_turns,
        requested_parallelism=requested_parallelism,
        effective_parallelism=effective_parallelism,
    )
    work_items_by_seed = {item.seed_id: item for item in work_items}
    semaphore = asyncio.Semaphore(max(1, effective_parallelism))

    async def complete_one(index: int, problem: CertificationInput) -> None:
        if not _needs_generation_zero_proof_completion(problem):
            completed[index] = problem
            return
        work_item = work_items_by_seed.get(problem.id)
        seed_started = time.monotonic()
        async with semaphore:
            with ls.trace(
                name=f"gen0.seed.{problem.id}",
                run_type="chain",
                inputs={
                    "problem_id": problem.id,
                    "lane": work_item.lane if work_item else 0,
                    "work_item": work_item.model_dump() if work_item else {},
                    "effective_parallelism": effective_parallelism,
                    "proof_body_available": False,
                    "formal_statement_chars": len(str((problem.metadata or {}).get("formal_statement") or "")),
                    "proof_k": proof_k,
                    "proof_turns": proof_turns,
                    "max_seed_seconds": max_seed_seconds,
                    "llm_timeout": llm_timeout,
                    "lean_timeout": lean_timeout,
                    "leansearch_enabled": leansearch_enabled,
                    "leansearch_limit": leansearch_limit,
                },
                tags=["pool-generation", "generation-0", "seed-proof-completion"],
            ) as seed_run:
                try:
                    enriched = await asyncio.wait_for(
                        _call_seed_proof_completer(
                            completer,
                            problem,
                            config,
                            proof_k=proof_k,
                            proof_turns=proof_turns,
                            llm_timeout=llm_timeout,
                            lean_timeout=lean_timeout,
                            leansearch_enabled=leansearch_enabled,
                            leansearch_limit=leansearch_limit,
                        ),
                        timeout=max_seed_seconds,
                    )
                except asyncio.TimeoutError:
                    metadata = dict(problem.metadata or {})
                    enriched = problem.model_copy(
                        update={
                            "metadata": {
                                **metadata,
                                "gen0_target": True,
                                "gen0_proof_completed": False,
                                "gen0_proof_completion": {
                                    "status": "timeout",
                                    "problem_id": problem.id,
                                    "provider_slug": config.model,
                                    "total_elapsed_seconds": max_seed_seconds,
                                    "attempts": [],
                                },
                                "gen0_failure_packet": {
                                    "failure_class": "timeout",
                                    "latest_summary": f"Gen0 seed proof completion timed out after {max_seed_seconds}s",
                                    "recent_signatures": ["gen0_seed_timeout"],
                                    "attempt_count": 0,
                                    "best_partial_proof": "not_available",
                                },
                            }
                        }
                    )
                except Exception as exc:
                    elapsed = time.monotonic() - seed_started
                    metadata = dict(problem.metadata or {})
                    enriched = problem.model_copy(
                        update={
                            "metadata": {
                                **metadata,
                                "gen0_target": True,
                                "gen0_proof_completed": False,
                                "gen0_proof_completion": {
                                    "status": "exception",
                                    "problem_id": problem.id,
                                    "provider_slug": config.model,
                                    "total_elapsed_seconds": elapsed,
                                    "attempts": [],
                                },
                                "gen0_failure_packet": {
                                    "failure_class": "gen0_exception",
                                    "latest_summary": f"{type(exc).__name__}: {exc}",
                                    "recent_signatures": ["gen0_seed_exception"],
                                    "attempt_count": 0,
                                    "best_partial_proof": "not_available",
                                },
                            }
                        }
                    )
                if not isinstance(enriched, CertificationInput):
                    enriched = CertificationInput.model_validate(enriched)
                metadata = dict(enriched.metadata or {})
                original_metadata = dict(problem.metadata or {})
                metadata["gen0_target"] = True
                metadata.setdefault(
                    "seed_formal_statement_original",
                    original_metadata.get("formal_statement"),
                )
                original_formal = str(original_metadata.get("formal_statement") or "").strip()
                current_formal = str(metadata.get("formal_statement") or "").strip()
                if original_formal and ":= by" in current_formal:
                    metadata["formal_statement"] = original_formal
                if work_item is not None:
                    metadata["gen0_work_item"] = work_item.model_dump()
                    metadata["gen0_lane"] = work_item.lane
                    metadata["gen0_worker_count_requested"] = requested_parallelism
                    metadata["gen0_worker_count_effective"] = effective_parallelism
                    metadata["gen0_worker_count_source"] = "cli_or_default"
                enriched = enriched.model_copy(update={"metadata": metadata})
                completed[index] = enriched
                seed_run.end(
                    outputs={
                        "problem_id": enriched.id,
                        "lane": work_item.lane if work_item else 0,
                        "status": _gen0_status(enriched),
                        "proof_body_available": bool(_parent_proof_context(enriched).get("proof_body_available")),
                        "gen0_proof_completed": _metadata_bool((enriched.metadata or {}).get("gen0_proof_completed")),
                        "elapsed_seconds": time.monotonic() - seed_started,
                        "failure_class": _metadata_dict((enriched.metadata or {}).get("gen0_failure_packet")).get("failure_class", ""),
                    }
                )

    with ls.trace(
        name="generation.0.seed_proof_completion",
        run_type="chain",
        inputs={
            "pool_size": len(pool),
            "missing_proof_body_count": target_seed_count,
            "target_seed_count": target_seed_count,
            "worker_count_requested": requested_parallelism,
            "worker_count_effective": effective_parallelism,
            "worker_count_source": "cli_or_default",
            "leansearch_enabled": leansearch_enabled,
            "leansearch_limit": leansearch_limit,
            "work_items": [item.model_dump() for item in work_items],
        },
        tags=["pool-generation", "generation-0", "seed-proof-completion"],
    ) as gen0_run:
        await asyncio.gather(*(complete_one(index, problem) for index, problem in enumerate(pool)))
        completed_pool = [problem if problem is not None else pool[index] for index, problem in enumerate(completed)]
        gen0_run.end(
            outputs={
                "completed_count": sum(
                    1 for problem in completed_pool if _metadata_bool((problem.metadata or {}).get("gen0_proof_completed"))
                ),
                "proof_body_available_count": sum(
                    1 for problem in completed_pool if bool(_parent_proof_context(problem).get("proof_body_available"))
                ),
                "failed_count": sum(
                    1
                    for problem in completed_pool
                    if _metadata_bool((problem.metadata or {}).get("gen0_target"))
                    and not _metadata_bool((problem.metadata or {}).get("gen0_proof_completed"))
                ),
                "target_seed_count": target_seed_count,
                "worker_count_requested": requested_parallelism,
                "worker_count_effective": effective_parallelism,
                "worker_count_source": "cli_or_default",
                "leansearch_enabled": leansearch_enabled,
                "leansearch_limit": leansearch_limit,
            }
        )
    return completed_pool


#: Consecutive generations that measured nothing before the run stops. Two is
#: enough to separate a provider outage from a hard pool: a generation where
#: every worker refuses on its own contract still *measured* something, so it
#: never counts here.
BARREN_GENERATION_LIMIT = 2

#: Failures that mean the provider never answered. A slot that failed for any
#: other reason -- malformed JSON, a contract the worker declined, a proof that
#: would not close -- did get an answer, and is evidence about the pool.
_NO_ANSWER = (
    "exited with code 1",
    "session has ended",
    "401",
    "refresh token",
    "unauthorized",
    "connection refused",
    "timed out",
)


def _generation_produced_nothing(state: Dict[str, Any]) -> str:
    """Whether a generation measured nothing at all, and why.

    Defined by the absence of measurement, not by the presence of failure. A
    generation in which every worker returned a real answer that did not
    certify is a fact about the pool and must not stop the run; one in which
    the provider never answered is not a measurement at all.

    Written after a revoked codex token let a group run four more generations,
    each recording every slot as `generation_failed` with
    `"Your session has ended"` in the error. The existing guard only fires on a
    group that finishes with zero certified rows, so a group that had already
    banked a few certifications sailed past it and burned the rest of its
    budget on a provider that was answering 401.

    Returns a short reason when nothing was measured, and "" otherwise.
    """
    counts = dict((state.get("summary") or {}).get("counts") or {})
    if counts.get("certified"):
        return ""
    results = _state_results(state) or []
    if not results:
        return "no slot results"
    answered, silent = 0, []
    for result in results:
        # A survivor is last generation's row carried forward. It is not
        # something this generation measured, and counting it as an answer made
        # the guard silent on the real case: the generation that recorded five
        # survivors and seven `generation_failed`, every one of them a revoked
        # token, read as "the provider answered".
        if getattr(result, "status", "") == "survivor":
            continue
        if getattr(result, "status", "") == "certified":
            answered += 1
            continue
        text = " ".join(
            str(value or "").lower()
            for value in (getattr(result, "error", ""),
                          getattr(result, "failure_signature", ""),
                          getattr(result, "proof_verify_summary", ""))
        )
        if any(marker in text for marker in _NO_ANSWER):
            silent.append(getattr(result, "slot", "?"))
        else:
            # A refusal, a parse error, a failed proof -- the provider spoke.
            answered += 1
    if answered or not silent:
        return ""
    return f"all {len(silent)} slots got no answer from the provider"


async def run_pool_generation_async(
    input_path: Path,
    output_path: Path,
    summary_output_path: Optional[Path] = None,
    *,
    max_generations: int = 1,
    pool_size: int = POOL_SIZE,
    survivor_count: int = 1,
    crossover_count: int = DEFAULT_CROSSOVER_COUNT,
    max_parallel: int = POOL_SIZE,
    max_retries: int = 3,
    target_accepted_per_generation: int = DEFAULT_TARGET_ACCEPTED_PER_GENERATION,
    reserve_budget: int = DEFAULT_RESERVE_BUDGET,
    disable_reserve_slots: bool = False,
    gen0_proof_k: int = 4,
    gen0_proof_turns: int = 6,
    gen0_max_seed_seconds: float = 3600.0,
    gen0_max_parallel: int = DEFAULT_GEN0_MAX_PARALLEL,
    skip_gen0: bool = False,
    planner_memory_dir: Optional[Path] = Path("data/certified"),
    planner_memory_limit: int = DEFAULT_PLANNER_MEMORY_LIMIT,
    disable_planner_memory: bool = False,
    disable_leansearch: bool = False,
    leansearch_limit: int = DEFAULT_LEANSEARCH_LIMIT,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    run_name: Optional[str] = None,
    project_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    checker: Optional[LeanChecker] = None,
    planner: Optional[PlannerFn] = None,
    generator: Optional[SlotGeneratorFn] = None,
    theorem_generator: Optional[TheoremGeneratorFn] = None,
    theorem_verifier: Optional[TheoremVerifierFn] = None,
    theorem_alignment_verifier: Optional[TheoremAlignmentVerifierFn] = None,
    theorem_proof_repairer: Optional[Callable[..., Any]] = None,
    replanner: Optional[ReplannerFn] = None,
    seed_proof_completer: Optional[SeedProofCompleterFn] = None,
    aggregate_selector: Optional[AggregateSelectorFn] = None,
) -> PoolRunResult:
    if max_generations < 1:
        raise ValueError("max_generations must be >= 1")
    if pool_size != POOL_SIZE:
        raise ValueError("MVP pool generation keeps pool_size fixed at 5")
    if crossover_count and pool_size < 2:
        raise ValueError("crossover requires at least two parent slots")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if target_accepted_per_generation < 0:
        raise ValueError("target_accepted_per_generation must be >= 0")
    if reserve_budget < 0:
        raise ValueError("reserve_budget must be >= 0")
    if gen0_proof_k < 1:
        raise ValueError("gen0_proof_k must be >= 1")
    if gen0_proof_turns < 1:
        raise ValueError("gen0_proof_turns must be >= 1")
    if gen0_max_seed_seconds <= 0:
        raise ValueError("gen0_max_seed_seconds must be > 0")
    if gen0_max_parallel < 1:
        raise ValueError("gen0_max_parallel must be >= 1")
    if planner_memory_limit < 0:
        raise ValueError("planner_memory_limit must be >= 0")
    if leansearch_limit < 1:
        raise ValueError("leansearch_limit must be >= 1")
    run_name = run_name or f"pool_gen{max_generations}_{Path(input_path).stem}"
    graph = build_pool_generation_graph(
        checker=checker,
        planner=planner,
        generator=generator,
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        theorem_proof_repairer=theorem_proof_repairer,
        replanner=replanner,
        aggregate_selector=aggregate_selector,
    )
    project = project_name or os.getenv("LANGSMITH_PROJECT")
    root_name = f"pool_generation/{run_name}"
    root_metadata = {
        "bench_version": DEFAULT_BENCH_VERSION,
        "pool_size": pool_size,
        "survivor_count": survivor_count,
        "crossover_count": crossover_count,
        "max_parallel": max_parallel,
        "max_retries": max_retries,
        "target_accepted_per_generation": target_accepted_per_generation,
        "reserve_budget": reserve_budget,
        "reserve_slots_enabled": not disable_reserve_slots,
        "gen0_proof_k": gen0_proof_k,
        "gen0_proof_turns": gen0_proof_turns,
        "gen0_max_seed_seconds": gen0_max_seed_seconds,
        "gen0_max_parallel": gen0_max_parallel,
        "planner_memory_dir": str(planner_memory_dir) if planner_memory_dir else "",
        "planner_memory_limit": planner_memory_limit,
        "planner_memory_enabled": not disable_planner_memory,
        "leansearch_enabled": not disable_leansearch,
        "leansearch_limit": leansearch_limit,
        "generation_model": generation_model,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    with ls.trace(
        name=root_name,
        run_type="chain",
        inputs={"input_path": str(input_path), "max_generations": max_generations},
        project_name=project,
        tags=["pool-generation", "langgraph"] + list(tags or []),
        metadata=root_metadata,
    ) as root_run:
        state: PoolState = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "summary_output_path": str(summary_output_path) if summary_output_path else None,
            "run_name": run_name,
            "pool_size": pool_size,
            "max_generations": max_generations,
            "survivor_count": survivor_count,
            "crossover_count": crossover_count,
            "max_parallel": max_parallel,
            "max_retries": max_retries,
            "target_accepted_per_generation": target_accepted_per_generation,
            "reserve_budget": reserve_budget,
            "disable_reserve_slots": disable_reserve_slots,
            "gen0_proof_k": gen0_proof_k,
            "gen0_proof_turns": gen0_proof_turns,
            "gen0_max_seed_seconds": gen0_max_seed_seconds,
            "gen0_max_parallel": gen0_max_parallel,
            "planner_memory_dir": str(planner_memory_dir) if planner_memory_dir else "",
            "planner_memory_limit": planner_memory_limit,
            "disable_planner_memory": disable_planner_memory,
            "leansearch_enabled": not disable_leansearch,
            "leansearch_limit": leansearch_limit,
            "generation_model": generation_model,
            "generation_temperature": generation_temperature,
            "generation_count": 0,
            "slot_outputs": [],
            "reserve_work_items": [],
            "reserve_round_pending": False,
            "reserve_round_done": False,
            "dispatch_mode": "",
            "plan_outcome_cards": [],
            "novelty_memory": {},
            "generations": [],
            "all_passed_results": [],
            "all_passed_results_count": 0,
        }
        seed_pool = load_seed_inputs(input_path, pool_size=pool_size)
        config = default_generation_config(model=generation_model, temperature=generation_temperature)
        if skip_gen0:
            state["current_generation"] = seed_pool
            state["generation_zero"] = {
                "skipped": True,
                "reason": "skip_gen0_requested",
                "total_seed_count": len(seed_pool),
                "missing_proof_body_count": _gen0_target_count(seed_pool),
            }
        else:
            completed_seed_pool = await _complete_generation_zero_proofs(
                seed_pool,
                config=config,
                seed_proof_completer=seed_proof_completer,
                proof_k=gen0_proof_k,
                proof_turns=gen0_proof_turns,
                max_seed_seconds=gen0_max_seed_seconds,
                max_parallel=gen0_max_parallel,
                llm_timeout=float(os.getenv("GENERATION_LLM_TIMEOUT", "240")),
                lean_timeout=float(os.getenv("GEN0_LEAN_TIMEOUT", "300")),
                leansearch_enabled=not disable_leansearch,
                leansearch_limit=leansearch_limit,
            )
            gen0_enriched_input_path = Path(output_path).with_suffix(".gen0_seeds.csv")
            _write_generation_zero_enriched_seed_csv(
                input_path,
                gen0_enriched_input_path,
                completed_seed_pool,
            )
            state["current_generation"] = completed_seed_pool
            generation_zero = _generation_zero_summary(seed_pool, completed_seed_pool)
            generation_zero["enriched_seed_csv_path"] = str(gen0_enriched_input_path)
            state["generation_zero"] = generation_zero
            state["gen0_enriched_input_path"] = str(gen0_enriched_input_path)
            state["all_passed_results"] = _generation_zero_rows(completed_seed_pool)
            state["all_passed_results_count"] = len(state["all_passed_results"])
        final_state: PoolState = state
        barren_streak = 0
        for generation_index in range(1, max_generations + 1):
            state = {
                **final_state,
                "slot_outputs": [],
                "work_items": [],
                "results": [],
                "approved_candidates": [],
                "failed_slots": [],
                "reserve_work_items": [],
                "reserve_round_pending": False,
                "reserve_round_done": False,
                "dispatch_mode": "",
            }
            graph_output = await graph.ainvoke(
                state,
                config={
                    "run_name": f"generation.{generation_index}.graph",
                    "tags": ["pool-generation", f"generation-{generation_index}"],
                    "metadata": {
                        **root_metadata,
                        "generation_index": generation_index,
                        "generation_feedback_present": bool(state.get("generation_feedback")),
                    },
                    "max_concurrency": max_parallel,
                    "configurable": {"thread_id": f"pool:{run_name}:generation:{generation_index}"},
                },
                output_keys=[
                    "summary",
                    "current_generation",
                    "generation_count",
                    "generation_feedback",
                    "generation_survival_status",
                    "generations",
                    "all_passed_results_count",
                    "run_manifest",
                    "results_ref",
                    "novelty_memory",
                ],
            )
            final_state = {**state, **dict(graph_output)}
            if final_state.get("summary", {}).get("generation_save_status") not in {
                "complete",
                "complete_with_backfill",
            }:
                break
            unproductive = _generation_produced_nothing(final_state)
            if unproductive:
                barren_streak += 1
                if barren_streak >= BARREN_GENERATION_LIMIT:
                    print(
                        f"[halt] {barren_streak} consecutive generations produced nothing: "
                        f"{unproductive}. Stopping rather than writing more of them.",
                        flush=True,
                    )
                    break
            else:
                barren_streak = 0
        summary = dict(final_state["summary"])
        counts = dict(summary.get("counts", {}) or {})
        generation_feedback = dict(summary.get("generation_feedback", {}) or {})
        quality_flags = generation_feedback.get("quality_flags", {}) or {}
        weak_quality_count = len(generation_feedback.get("weak_slots", []) or [])
        strong_crossover_count = 0
        for item in summary.get("saved_pool", []) or []:
            if item.get("op_type") == "crossover" and item.get("quality_verdict") == "strong":
                strong_crossover_count += 1
        root_run.end(
            outputs={
                "counts": counts,
                "certified_count": counts.get("certified", 0),
                "failed_count": len(summary.get("failed_slots", []) or []),
                "weak_quality_count": weak_quality_count,
                "strong_crossover_count": strong_crossover_count,
                "planner_fallback_count": 1
                if summary.get("planner", {}).get("planner_source") == "deterministic_fallback"
                else 0,
                "quality_flags": quality_flags,
                "generation_feedback_present": bool(generation_feedback),
                "generation_save_status": summary.get("generation_save_status"),
                "generation_zero": summary.get("generation_zero", {}),
                "gen0_enriched_input_path": summary.get("gen0_enriched_input_path"),
                "run_manifest": summary.get("run_manifest", {}),
                "planner_memory": summary.get("planner_memory", {}),
                "planner_case_pack": summary.get("planner_case_pack", {}),
                "novelty_memory": summary.get("novelty_memory", {}),
                "output_path": str(output_path),
            }
        )
    langsmith_upload = _verify_langsmith_trace_upload(
        root_run_name=root_name,
        project_name=project,
    )
    summary["langsmith_upload"] = langsmith_upload
    if summary_output_path:
        Path(summary_output_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return PoolRunResult.model_validate(summary)


def run_pool_generation(
    input_path: Path,
    output_path: Path,
    summary_output_path: Optional[Path] = None,
    *,
    max_generations: int = 1,
    pool_size: int = POOL_SIZE,
    survivor_count: int = 1,
    crossover_count: int = DEFAULT_CROSSOVER_COUNT,
    max_parallel: int = POOL_SIZE,
    max_retries: int = 3,
    target_accepted_per_generation: int = DEFAULT_TARGET_ACCEPTED_PER_GENERATION,
    reserve_budget: int = DEFAULT_RESERVE_BUDGET,
    disable_reserve_slots: bool = False,
    gen0_proof_k: int = 4,
    gen0_proof_turns: int = 6,
    gen0_max_seed_seconds: float = 3600.0,
    gen0_max_parallel: int = DEFAULT_GEN0_MAX_PARALLEL,
    skip_gen0: bool = False,
    planner_memory_dir: Optional[Path] = Path("data/certified"),
    planner_memory_limit: int = DEFAULT_PLANNER_MEMORY_LIMIT,
    disable_planner_memory: bool = False,
    disable_leansearch: bool = False,
    leansearch_limit: int = DEFAULT_LEANSEARCH_LIMIT,
    generation_model: Optional[str] = None,
    generation_temperature: Optional[float] = None,
    run_name: Optional[str] = None,
    project_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    checker: Optional[LeanChecker] = None,
    planner: Optional[PlannerFn] = None,
    generator: Optional[SlotGeneratorFn] = None,
    theorem_generator: Optional[TheoremGeneratorFn] = None,
    theorem_verifier: Optional[TheoremVerifierFn] = None,
    theorem_alignment_verifier: Optional[TheoremAlignmentVerifierFn] = None,
    theorem_proof_repairer: Optional[Callable[..., Any]] = None,
    replanner: Optional[ReplannerFn] = None,
    seed_proof_completer: Optional[SeedProofCompleterFn] = None,
    aggregate_selector: Optional[AggregateSelectorFn] = None,
) -> PoolRunResult:
    return asyncio.run(
        run_pool_generation_async(
            input_path,
            output_path,
            summary_output_path,
            max_generations=max_generations,
            pool_size=pool_size,
            survivor_count=survivor_count,
            crossover_count=crossover_count,
            max_parallel=max_parallel,
            max_retries=max_retries,
            target_accepted_per_generation=target_accepted_per_generation,
            reserve_budget=reserve_budget,
            disable_reserve_slots=disable_reserve_slots,
            gen0_proof_k=gen0_proof_k,
            gen0_proof_turns=gen0_proof_turns,
            gen0_max_seed_seconds=gen0_max_seed_seconds,
            gen0_max_parallel=gen0_max_parallel,
            skip_gen0=skip_gen0,
            planner_memory_dir=planner_memory_dir,
            planner_memory_limit=planner_memory_limit,
            disable_planner_memory=disable_planner_memory,
            disable_leansearch=disable_leansearch,
            leansearch_limit=leansearch_limit,
            generation_model=generation_model,
            generation_temperature=generation_temperature,
            run_name=run_name,
            project_name=project_name,
            tags=tags,
            checker=checker,
            planner=planner,
            generator=generator,
            theorem_generator=theorem_generator,
            theorem_verifier=theorem_verifier,
            theorem_alignment_verifier=theorem_alignment_verifier,
            theorem_proof_repairer=theorem_proof_repairer,
            replanner=replanner,
            seed_proof_completer=seed_proof_completer,
            aggregate_selector=aggregate_selector,
        )
    )
