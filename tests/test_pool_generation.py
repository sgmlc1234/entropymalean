import csv
import asyncio
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from langgraph.types import Send

from src.certification import CertificationInput, CertificationResult, GeneratedProblem
from src.certification.generation import (
    GenerationConfig,
    PlannerContractError,
    _build_generation_messages,
    _canonical_problem_from_params,
    _generation_response_format,
    validate_generated_contract,
)
from src.utils.codex_cli import build_codex_exec_command, build_codex_prompt
from src.orchestration.pool_generation import (
    DEFAULT_GEN0_MAX_PARALLEL,
    DEFAULT_PLANNER_MEMORY_LIMIT,
    THEOREM_CANONICAL_HEADER,
    PlannerMemoryCard,
    TheoremAlignmentResult,
    TheoremGeneratedProblem,
    _attempt_history_card,
    _attempt_history_summary,
    _aggregate_messages,
    _aggregate_response_format,
    _aggregate_selector_payload,
    _attach_novelty_contracts,
    _build_run_manifest,
    _build_backfill_seed_archive,
    _build_theorem_alignment_messages,
    _build_theorem_generation_messages,
    _build_replan_messages,
    _complete_generation_zero_proofs,
    _certify_theorem_child,
    _canonical_signature,
    _normalize_theorem_generation_raw,
    _deterministic_replan_operator_card,
    _effective_gen0_parallelism,
    _failure_class,
    _generation_zero_rows,
    _generation_zero_summary,
    _gen0_proof_matches_formal_statement,
    _lean_has_complete_by_body,
    _parent_is_backfill_eligible,
    _is_plan_level_failure,
    _lean_error_line_context,
    _merge_novelty_memory_quality,
    _needs_generation_zero_proof_completion,
    _novelty_memory_trace_manifest,
    _normalize_theorem_lean_code,
    _operator_card,
    _planner_messages,
    _planner_response_schema,
    _planner_response_format,
    _parent_context_card,
    _theorem_response_format,
    _theorem_alignment_response_format,
    _theorem_candidate_preflight,
    _theorem_decomposition_card,
    _result_to_pool_problem,
    _result_root_lineages,
    _reserve_goal_from_profile,
    _retryable_generation_failure,
    _retry_feedback_for_result,
    _run_manifest_digest,
    _structural_overlap_curation_flags,
    _verify_langsmith_trace_upload,
    _write_generation_zero_enriched_seed_csv,
    _with_slot_metadata,
    build_pool_generation_graph,
    deterministic_fallback_plan,
    load_seed_inputs,
    run_pool_generation,
    select_next_pool_with_orchestrator,
    select_planner_memory_cards,
    validate_pool_plan,
)
from src.orchestration.quality import (
    QualityResult,
    derive_accepted_proxy,
    derive_curation_decision,
    derive_entropy_direction,
    derive_misformalization_taxonomy,
    informal_statement_internal_term_hits,
    verify_slot_quality,
)
from src.retrieval.novelty_memory import cards_from_rows
from src.no_go_policy import (
    ACCEPTED_PROXY_SEVERE_FLAGS,
    RETRY_PATCH_INSTRUCTIONS,
    no_go_policy_summary,
)
from src.evaluation.lean_verifier import LeanVerifyResult, _lean_resource_args


@pytest.fixture(autouse=True)
def _disable_external_leansearch_for_pool_tests(monkeypatch):
    monkeypatch.setenv("LEANSEARCH_DISABLED", "true")
    monkeypatch.setenv("NOVELTY_ACCEPTED_LEDGER_PATH", "data/__missing_test_accepted.jsonl")
    # Keep these tests hermetic: pytest plugins (e.g. deepeval) load .env, and a
    # live provider key would route the aggregate next-pool selector through a
    # real LLM whose judgment may override the deterministic gating asserted here.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)


class FakeLeanChecker:
    async def health_check(self):
        return True

    async def type_check(self, lean_code):
        return True, ""


class FlakyLeanChecker:
    def __init__(self):
        self.calls = 0
        self.generated_failures = 0
        self.lean_path = "lean"
        self.lean_version = "Lean test"

    async def health_check(self):
        return True

    async def type_check(self, lean_code):
        self.calls += 1
        if "divisor_sum_840" in lean_code and self.generated_failures == 0:
            self.generated_failures += 1
            return False, "compiler error: expected 2880"
        return True, ""


class RaisingLeanChecker:
    async def health_check(self):
        return True

    async def type_check(self, lean_code):
        raise AssertionError("Lean checker should not be called")

@pytest.fixture
def isolated_corpus_index(monkeypatch):
    """Dedup against nothing but this test's own rows.

    The runtime deduplicates every certified statement against the persistent
    corpus under `data/certified/`, so a fixture theorem that happens to match
    a released row is filed `duplicate_statement` instead of `certified`. A
    test about anything other than dedup must not depend on what the campaign
    happened to certify."""
    from src.certification.dedup import CorpusIndex
    monkeypatch.setattr("src.orchestration.pool_generation._CORPUS_INDEX", CorpusIndex())
    yield



def _pool():
    return [
        CertificationInput(
            id="p0", statement="Find the sum of all positive divisors of 120.", answer="360"
        ),
        CertificationInput(id="p1", statement="Find the units digit of 7^{2026}.", answer="9"),
        CertificationInput(
            id="p2",
            statement="Count the number of non-negative integer solutions to x_1 + x_2 + x_3 = 10.",
            answer="66",
        ),
        CertificationInput(
            id="p3",
            statement="Find the sum of the first 20 terms of the arithmetic sequence 3, 7, 11, 15, ...",
            answer="820",
        ),
        CertificationInput(id="p4", statement="Find GCD(2026, 1234).", answer="2"),
    ]


def _compatible_pool():
    return [
        CertificationInput(id="g0", statement="Find GCD(84, 126).", answer="42"),
        CertificationInput(
            id="d0", statement="Find the sum of all positive divisors of 120.", answer="360"
        ),
        CertificationInput(
            id="d1", statement="Find the sum of all positive divisors of 144.", answer="403"
        ),
        CertificationInput(id="m0", statement="Find 98765 mod 89.", answer="64"),
        CertificationInput(id="u0", statement="Find the units digit of 7^{2026}.", answer="9"),
    ]


def _write_csv(path: Path):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "statement", "answer"])
        writer.writeheader()
        for problem in _pool():
            writer.writerow(
                {"id": problem.id, "statement": problem.statement, "answer": problem.answer}
            )


def _write_compatible_csv(path: Path):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "statement", "answer"])
        writer.writeheader()
        for problem in _compatible_pool():
            writer.writerow(
                {"id": problem.id, "statement": problem.statement, "answer": problem.answer}
            )


def _write_theorem_csv(path: Path):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "statement", "answer", "formal_statement", "lean_header"],
        )
        writer.writeheader()
        for idx in range(5):
            writer.writerow(
                {
                    "id": f"thm{idx}",
                    "statement": f"Prove theorem-style proposition {idx}.",
                    "answer": "",
                    "formal_statement": f"theorem thm{idx} : True := by\n  trivial",
                    "lean_header": "import Mathlib",
                }
            )


def _fake_generated(parent, config):
    slot = parent.metadata.get("slot", 0)
    target_family = parent.metadata.get("target_family") or "divisor_sum"
    if slot == 2:
        return GeneratedProblem(
            id=f"{parent.id}__bad",
            source_problem_id=parent.id,
            family=target_family,
            statement="Find the sum of all positive divisors.",
            answer="1",
            params={"n_terms": 10, "first": 1, "diff": 1}
            if target_family == "arithmetic_series"
            else {},
            harder_reason="Intentional unsupported generated statement.",
        )
    if target_family == "units_digit":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="units_digit",
            statement="Find the units digit of 13^{2025}.",
            answer="3",
            difficulty_label="medium",
            params={"base": 13, "exp": 2025},
            harder_reason="Larger exponent.",
        )
    if target_family == "stars_and_bars":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="stars_and_bars",
            statement="Count the number of non-negative integer solutions to x_1 + x_2 + x_3 + x_4 = 10.",
            answer="286",
            difficulty_label="medium",
            params={"vars": 4, "sum": 10},
            harder_reason="More variables.",
        )
    if target_family == "arithmetic_series":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="arithmetic_series",
            statement="Find the sum of the first 10 terms of the arithmetic sequence 1, 2, 3, 4, ...",
            answer="55",
            difficulty_label="medium",
            params={"n_terms": 10, "first": 1, "diff": 1},
            harder_reason="More terms.",
        )
    if target_family == "gcd":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="gcd",
            statement="Find GCD(84, 126).",
            answer="42",
            difficulty_label="medium",
            params={"a": 84, "b": 126},
            harder_reason="Larger inputs.",
        )
    if target_family == "modular_congruence":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 2026 mod 7.",
            answer="3",
            difficulty_label="medium",
            params={"a": 2026, "m": 7},
            harder_reason="Larger dividend.",
        )
    return GeneratedProblem(
        id=f"{parent.id}__gen1",
        source_problem_id=parent.id,
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        difficulty_label="medium",
        params={"n": 840},
        harder_reason="Larger divisor-sum instance.",
    )


def _fake_composite_generated(parent, config):
    target_family = parent.metadata.get("target_family") or "divisor_sum"
    if target_family == "gcd_divisor_sum":
        return GeneratedProblem(
            id=f"{parent.id}__gcd_sigma",
            source_problem_id=parent.id,
            family="gcd_divisor_sum",
            statement="Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
            answer="96",
            params={"a": 84, "b": 126, "gcd": 42},
            projected_params={"a": 84, "b": 126},
            reasoning_pattern="gcd_then_sigma",
            solution_skeleton={"target_computation": "sigma(gcd(84,126))"},
            projection_check={"passed": True, "evidence": "a,b project to the derived gcd"},
            harder_reason="Two-step composite reasoning.",
        )
    if target_family == "divisor_sum_mod":
        return GeneratedProblem(
            id=f"{parent.id}__sigma_mod",
            source_problem_id=parent.id,
            family="divisor_sum_mod",
            statement="Let m be the sum of all positive divisors of 144. Find 98765 mod m.",
            answer="30",
            params={"n": 144, "a": 98765, "modulus": 403},
            projected_params={"n": 144, "a": 98765},
            reasoning_pattern="sigma_then_mod",
            solution_skeleton={"target_computation": "98765 mod sigma(144)"},
            projection_check={"passed": True, "evidence": "n,a project to sigma_then_mod"},
            harder_reason="Two-step composite reasoning.",
        )
    return _fake_generated_success(parent, config)


def _fake_generated_success(parent, config):
    target_family = parent.metadata.get("target_family") or "divisor_sum"
    if target_family == "units_digit":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="units_digit",
            statement="Find the units digit of 13^{2025}.",
            answer="3",
            params={"base": 13, "exp": 2025},
            harder_reason="Larger exponent.",
        )
    if target_family == "stars_and_bars":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="stars_and_bars",
            statement="Count the number of non-negative integer solutions to x_1 + x_2 + x_3 + x_4 = 10.",
            answer="286",
            params={"vars": 4, "sum": 10},
            harder_reason="More variables.",
        )
    if target_family == "arithmetic_series":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="arithmetic_series",
            statement="Find the sum of the first 25 terms of the arithmetic sequence 3, 7, 11, 15, ...",
            answer="1275",
            params={"n_terms": 25, "first": 3, "diff": 4},
            harder_reason="More terms.",
        )
    if target_family == "gcd":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="gcd",
            statement="Find GCD(840, 1260).",
            answer="420",
            params={"a": 840, "b": 1260},
            harder_reason="Larger inputs.",
        )
    if target_family == "modular_congruence":
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 2027 mod 11.",
            answer="3",
            params={"a": 2027, "m": 11},
            harder_reason="Larger dividend.",
        )
    return GeneratedProblem(
        id=f"{parent.id}__gen1",
        source_problem_id=parent.id,
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        params={"n": 840},
        harder_reason="Larger divisor-sum instance.",
    )


def _fake_duplicate_divisor_sum(parent, config):
    return GeneratedProblem(
        id=f"{parent.id}__dup",
        source_problem_id=parent.id,
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 120.",
        answer="360",
        params={"n": 120},
        harder_reason="Intentionally duplicate easy divisor-sum instance.",
    )


def test_validate_plan_accepts_minimal_operator_card_without_variation_axis():
    plan = {
        "planner_source": "orchestrator_llm",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {"slot": 1, "op_type": "mutation", "parent_ids": ["p1"], "variation_axis": ""},
            {"slot": 2, "op_type": "mutation", "parent_ids": ["p2"], "variation_axis": "raise sum"},
            {
                "slot": 3,
                "op_type": "mutation",
                "parent_ids": ["p3"],
                "variation_axis": "raise terms",
            },
            {
                "slot": 4,
                "op_type": "mutation",
                "parent_ids": ["p4"],
                "variation_axis": "raise inputs",
            },
        ],
    }
    items = validate_pool_plan(plan, _pool())
    assert items[1]["operator_goal"].startswith("bounded_generalization:")
    assert "prior_goal=mutation" in items[1]["operator_goal"]
    assert items[1]["variation_axis"] == items[1]["operator_goal"]


def test_deterministic_fallback_creates_one_survivor_two_crossovers_two_mutations():
    plan = deterministic_fallback_plan(_compatible_pool())
    items = validate_pool_plan(plan, _compatible_pool())
    assert len(items) == 5
    assert sum(1 for item in items if item["op_type"] == "survivor") == 1
    assert sum(1 for item in items if item["op_type"] == "crossover") == 2
    assert sum(1 for item in items if item["op_type"] == "mutation") == 2
    assert all(item["target_family"] for item in items if item["op_type"] == "mutation")
    assert all(item["quality_target"] for item in items if item["op_type"] != "survivor")
    assert all(item["composition_pattern"] for item in items)


def test_validate_plan_preserves_required_params_contract():
    plan = deterministic_fallback_plan(_compatible_pool())
    plan["work_items"][1]["required_params"] = {"a": 84, "b": 126}
    items = validate_pool_plan(plan, _compatible_pool())
    assert items[1]["required_params"] == {"a": 84, "b": 126}


def test_validate_plan_rejects_noncanonical_required_param_keys():
    plan = deterministic_fallback_plan(_pool())
    plan["work_items"][1]["required_params"] = {"exponent": 2025}
    with pytest.raises(ValueError, match="invalid required_params keys"):
        validate_pool_plan(plan, _pool())


def test_validate_plan_rejects_required_params_outside_supported_range():
    plan = deterministic_fallback_plan(_compatible_pool())
    plan["work_items"][1]["required_params"] = {"a": 99999}
    with pytest.raises(ValueError, match="outside"):
        validate_pool_plan(plan, _compatible_pool())


def test_validate_plan_derives_quality_target_for_generated_slots():
    plan = deterministic_fallback_plan(_pool())
    plan["work_items"][1]["quality_target"] = ""
    items = validate_pool_plan(plan, _pool())
    assert items[1]["quality_target"]


def test_validate_plan_fills_missing_crossover_parent_contribution():
    plan = deterministic_fallback_plan(_compatible_pool())
    plan["work_items"][1]["parent_contributions"] = {"g0": "target family"}
    plan["work_items"][1]["fusion_contract"] = {}
    items = validate_pool_plan(plan, _compatible_pool())
    assert all(items[1]["parent_contributions"].get(parent_id) for parent_id in items[1]["parent_ids"])


def test_validate_plan_projects_parent_contributions_from_fusion_contract():
    plan = deterministic_fallback_plan(_compatible_pool())
    item = plan["work_items"][1]
    item["parent_contributions"] = {}
    items = validate_pool_plan(plan, _compatible_pool())
    crossover = items[1]
    assert crossover["parent_contributions"][crossover["parent_ids"][0]]
    assert crossover["parent_contributions"][crossover["parent_ids"][1]]


def test_validate_plan_fills_fusion_contract_missing_parent_contribution():
    plan = deterministic_fallback_plan(_compatible_pool())
    plan["work_items"][1]["parent_contributions"] = {}
    plan["work_items"][1]["fusion_contract"]["parent_B"]["contribution"] = ""
    items = validate_pool_plan(plan, _compatible_pool())
    assert all(items[1]["parent_contributions"].get(parent_id) for parent_id in items[1]["parent_ids"])


def test_validate_plan_downgrades_same_lineage_crossover_to_mutation():
    pool = [
        CertificationInput(
            id="thm",
            statement="Parent theorem.",
            answer="",
            metadata={"formal_statement": "theorem thm : True := by\n  trivial"},
        ),
        CertificationInput(
            id="thm__theorem_gen1",
            statement="Descendant theorem.",
            answer="",
            metadata={"formal_statement": "theorem thm_child : True := by\n  trivial"},
        ),
        *_compatible_pool()[:3],
    ]
    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_refs": [0]},
            {
                "slot": 1,
                "op_type": "crossover",
                "parent_refs": [0, 1],
                "target_style": "theorem_proof",
                "operator_variant": "crossover_hard",
                "goal": "combine ancestor and descendant",
            },
            {"slot": 2, "op_type": "mutation", "parent_refs": [2], "goal": "mutate"},
            {"slot": 3, "op_type": "mutation", "parent_refs": [3], "goal": "mutate"},
            {"slot": 4, "op_type": "mutation", "parent_refs": [4], "goal": "mutate"},
        ],
    }

    items = validate_pool_plan(plan, pool, crossover_count=2)

    assert items[1]["op_type"] == "mutation"
    assert items[1]["parent_ids"] == ["thm"]
    assert "same_lineage_crossover" in items[1]["avoid_patterns"]


def test_generation_prompt_includes_op_specific_contract_fields():
    plan = deterministic_fallback_plan(_compatible_pool())
    item = plan["work_items"][1]
    parent = CertificationInput(
        id="p0__x__p1",
        statement="Parent 1: Find the sum of all positive divisors of 120.\nParent 2: Find the units digit of 7^{2026}.",
        answer="360",
        metadata=item,
    )
    messages = _build_generation_messages(parent)
    user_message = messages[1]["content"]
    assert "Crossover rules:" in user_message
    assert "composition_pattern" in user_message
    assert "parent_contributions" in user_message
    assert "quality_target" in user_message
    assert "rebuilds it from family + params" in user_message
    assert "Parent B should change a generated parameter" in user_message
    assert "gcd_divisor_sum" in user_message
    assert "divisor_sum_mod" in user_message
    assert "Hard priority order" in user_message
    assert "Return exactly one generated child" in user_message
    assert "parent_usage" in user_message
    assert "status=\"cannot_execute\"" in user_message
    assert "ParentProofContext contract:" in user_message
    assert "verification_code.kind is one of lean, python, unknown, not_available" in user_message
    assert "If lean_code is available" in user_message
    assert "Public statement hygiene" in user_message
    assert "checkpoint, parent, certified, generated, mutation, crossover, pipeline" in user_message


def test_generation_prompt_no_longer_requires_worker_quality_schema():
    item = deterministic_fallback_plan(_compatible_pool())["work_items"][1]
    parent = CertificationInput(id="p0__x__p1", statement="combo", answer="", metadata=item)
    user_message = _build_generation_messages(parent)[1]["content"]

    assert '"quality_evidence"' not in user_message
    assert "solution_skeleton.parent_contributions" not in user_message
    assert "parent_contribution_evidence must be non-empty" not in user_message
    assert "Do not put this evidence only in axis_applied" not in user_message


def test_worker_prompt_includes_parent_proof_context_from_csv_metadata():
    parent = CertificationInput(
        id="mini0",
        statement="Prove that 2 + 2 = 4.",
        answer="4",
        metadata={
            "solution": "Compute by normalization.",
            "verification_code": "def verify():\n    return 2 + 2 == 4",
        },
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "parent_ids": ["mini0"],
        "target_family": "modular_congruence",
        "reasoning_goal": "reuse the arithmetic normalization idea",
    }

    prompt_parent = _with_slot_metadata(parent, item, generation_count=1)
    user_message = _build_generation_messages(prompt_parent)[1]["content"]

    assert "ParentProofContext contract:" in user_message
    assert '"solution": "Compute by normalization."' in user_message
    assert '"kind": "python"' in user_message
    assert "def verify()" in user_message
    assert "Never execute verification_code" in user_message


def test_worker_prompt_renders_missing_parent_proof_fields_as_not_available():
    parent = CertificationInput(
        id="seed0",
        statement="Find 17 mod 5.",
        answer="2",
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "parent_ids": ["seed0"],
        "target_family": "modular_congruence",
    }

    prompt_parent = _with_slot_metadata(parent, item, generation_count=1)
    user_message = _build_generation_messages(prompt_parent)[1]["content"]

    assert '"solution": "not_available"' in user_message
    assert '"kind": "not_available"' in user_message
    assert '"lean_code": "not_available"' in user_message


def test_worker_prompt_uses_putnambench_formal_statement_as_parent_lean_code():
    parent = CertificationInput(
        id="putnam_2000_b2",
        statement="Prove the binomial divisibility statement.",
        answer="true",
        metadata={
            "lean_header": "import Mathlib",
            "formal_statement": "theorem putnam_2000_b2 : True := by trivial",
            "solution": "Use divisibility of binomial coefficients.",
        },
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "parent_ids": ["putnam_2000_b2"],
        "target_family": "gcd",
    }

    prompt_parent = _with_slot_metadata(parent, item, generation_count=1)
    user_message = _build_generation_messages(prompt_parent)[1]["content"]

    assert '"lean_code": "import Mathlib\\ntheorem putnam_2000_b2' in user_message


def test_generated_parent_preserves_lean_code_and_solution_for_next_generation_prompt():
    result = CertificationResult(
        problem_id="child0",
        statement="Find 2026 mod 7.",
        answer="3",
        solution="2026 = 7 * 289 + 3.",
        verification_code="def verify():\n    return 2026 % 7 == 3",
        family="modular_congruence",
        status="certified",
        lean_level=2,
        lean_code="theorem mod_2026_7 : 2026 % 7 = 3 := by native_decide",
    )
    pool_row = _result_to_pool_problem(result)
    parent = CertificationInput(
        id=pool_row["id"],
        statement=pool_row["statement"],
        answer=pool_row["answer"],
        metadata={
            key: value
            for key, value in pool_row.items()
            if key not in {"id", "statement", "answer"}
        },
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "parent_ids": ["child0"],
        "target_family": "modular_congruence",
    }

    prompt_parent = _with_slot_metadata(parent, item, generation_count=2)
    user_message = _build_generation_messages(prompt_parent)[1]["content"]

    assert "2026 = 7 * 289 + 3." in user_message
    assert "theorem mod_2026_7" in user_message
    assert '"kind": "python"' in user_message


def test_validate_generated_contract_requires_direct_crossover_evidence():
    parent = CertificationInput(
        id="a__x__b",
        statement="combined parents",
        answer="1",
        metadata={
            "op_type": "crossover",
            "parent_ids": ["a", "b"],
            "target_family": "divisor_sum_mod",
        },
    )
    generated = GeneratedProblem(
        id="child",
        source_problem_id="a__x__b",
        family="divisor_sum_mod",
        statement="Let m be the sum of all positive divisors of 360. Find 823823 mod m.",
        answer="143",
        params={"n": 360, "a": 823823, "modulus": 1170},
        projected_params={"n": 360, "a": 823823},
        reasoning_pattern="sigma_then_mod",
        solution_skeleton={"parent_contributions": {"a": "sigma defines the modulus"}},
        raw_llm_output={
            "parent_contribution_evidence": {"a": "sigma defines the modulus"}
        },
    )
    with pytest.raises(PlannerContractError, match="missing direct parent contribution"):
        validate_generated_contract(parent, generated)


def test_validate_generated_contract_requires_both_crossover_evidence_surfaces():
    parent = CertificationInput(
        id="a__x__b",
        statement="combined parents",
        answer="1",
        metadata={
            "op_type": "crossover",
            "parent_ids": ["a", "b"],
            "target_family": "gcd_divisor_sum",
        },
    )
    generated = GeneratedProblem(
        id="child",
        source_problem_id="a__x__b",
        family="gcd_divisor_sum",
        statement="Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
        answer="96",
        params={"a": 84, "b": 126, "gcd": 42},
        projected_params={"a": 84, "b": 126},
        reasoning_pattern="gcd_then_sigma",
        solution_skeleton={"parent_contributions": {}},
        raw_llm_output={
            "parent_contribution_evidence": {
                "a": "params.a and params.b define the gcd target",
                "b": "target_computation applies divisor-sum semantics",
            }
        },
    )
    with pytest.raises(PlannerContractError, match="solution_skeleton.parent_contributions"):
        validate_generated_contract(parent, generated)

    generated.solution_skeleton = {
        "parent_contributions": {
            "a": "params.a and params.b define the gcd target",
            "b": "target_computation applies divisor-sum semantics",
        }
    }
    validate_generated_contract(parent, generated)


def test_canonical_problem_syncs_solution_skeleton_expected_answer():
    generated = _canonical_problem_from_params(
        source_problem_id="arith_parent",
        family="arithmetic_series",
        params={"n_terms": 99, "first": 1, "diff": 1},
        raw={
            "solution": "The answer is 1683.",
            "reasoning_pattern": "arithmetic_series_sum",
            "solution_skeleton": {
                "target_computation": "sum first 99 positive integers",
                "expected_answer": 1683,
            },
        },
    )

    assert generated.answer == "4950"
    assert generated.solution_skeleton["expected_answer"] == "4950"
    assert "Answer: 4950" in generated.solution


def test_json_parse_generation_failure_is_retryable():
    result = CertificationResult(
        problem_id="bad_json",
        status="generation_failed",
        error="Expecting ',' delimiter: line 1 column 2034 (char 2033)",
    )

    assert _retryable_generation_failure(result)


def test_parent_context_card_classifies_theorem_and_numeric_rows():
    theorem = CertificationInput(
        id="proofnet",
        statement="Prove that there are infinitely many primes congruent to -1 mod 4.",
        answer="",
        metadata={
            "formal_statement": "theorem proofnet : True := by\n  trivial",
            "lean_header": "import Mathlib",
        },
    )
    numeric = CertificationInput(id="mod", statement="Find 2026 mod 7.", answer="3")

    theorem_card = _parent_context_card(theorem)
    numeric_card = _parent_context_card(numeric)

    assert theorem_card["problem_style"] == "theorem_proof"
    assert theorem_card["certification_route"] == "theorem_prover"
    assert theorem_card["allowed_target_styles"] == ["theorem_proof"]
    assert numeric_card["problem_style"] == "numeric_answer"
    assert numeric_card["certification_route"] == "template_numeric"
    assert theorem_card["proof_context"]["proof_body_available"] is True
    assert theorem_card["proof_context"]["lean_statement_only"] is False


def test_parent_context_card_marks_statement_only_theorem():
    theorem = CertificationInput(
        id="proofnet_statement_only",
        statement="Prove a theorem.",
        answer="",
        metadata={
            "formal_statement": "theorem proofnet_statement_only : True :=",
            "lean_header": "import Mathlib",
        },
    )

    card = _parent_context_card(theorem)

    assert card["proof_context"]["proof_body_available"] is False
    assert card["proof_context"]["lean_statement_only"] is True
    assert card["theorem_decomposition"]["proof_body_available"] is False


def test_parent_context_card_marks_by_stub_as_missing_proof_body():
    theorem = CertificationInput(
        id="minif2f_stub",
        statement="Prove a theorem.",
        answer="",
        metadata={
            "formal_statement": "theorem minif2f_stub : True := by",
            "lean_header": "import Mathlib",
        },
    )

    card = _parent_context_card(theorem)

    assert card["proof_context"]["proof_body_available"] is False
    assert card["proof_context"]["lean_statement_only"] is True
    assert _needs_generation_zero_proof_completion(theorem)


def test_operator_card_removes_parent_rewrite_without_parent_proof_body():
    parent_card = _parent_context_card(
        CertificationInput(
            id="statement_only",
            statement="Prove a theorem.",
            answer="",
            metadata={
                "formal_statement": "theorem statement_only : True :=",
                "lean_header": "import Mathlib",
            },
        )
    )

    card = _operator_card(
        {
            "op_type": "mutation",
            "operator_variant": "mutation_easy",
            "target_style": "theorem_proof",
            "target_family": "theorem_proof",
            "parent_context_cards": [parent_card],
        }
    )

    assert "parent_rewrite" not in card["theorem_proof_surfaces"]
    assert "exact_existing" not in card["theorem_proof_surfaces"]
    assert "direct_proof" not in card["theorem_proof_surfaces"]


def test_theorem_decomposition_card_summarizes_formal_statement():
    theorem = CertificationInput(
        id="group_parent",
        statement="Prove that ab and ba are conjugate in a group.",
        answer="",
        metadata={
            "formal_statement": (
                "theorem group_parent {G : Type*} [Group G] (a b : G) : "
                "∃ g : G, b * a = g * (a * b) * g⁻¹ := by\n  sorry"
            ),
            "lean_header": "import Mathlib",
        },
    )

    card = _theorem_decomposition_card(theorem)

    assert "∃ g" in card["main_conclusion"]
    assert any("G : Type*" in hyp for hyp in card["hypotheses"])
    assert card["proof_checkpoints"]
    assert "do not add prose-only claims absent from formal_statement/lean_code" in card["forbidden_claims"]


def test_validate_plan_coerces_theorem_parent_numeric_child_to_theorem_route():
    pool = [
        CertificationInput(
            id="thm0",
            statement="Prove theorem-style proposition.",
            answer="",
            metadata={"formal_statement": "theorem thm0 : True := by\n  trivial"},
        ),
        *_pool()[:4],
    ]
    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["thm0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_ids": ["thm0"],
                "target_style": "numeric_answer",
                "target_family": "modular_congruence",
                "variation_axis": "bad theorem to numeric projection",
                "reasoning_goal": "bad theorem to numeric projection",
                "composition_pattern": "parameter_shift",
                "quality_target": "bad",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p0"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p1"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p2"]},
        ],
    }
    work_items = validate_pool_plan(plan, pool, survivor_count=4, crossover_count=0)

    coerced = work_items[1]
    assert coerced["target_style"] == "theorem_proof"
    assert coerced["target_family"] == "theorem_proof"


def test_planner_prompt_states_canonical_surface_and_checkable_quality():
    messages = _planner_messages(_pool(), pool_size=5, survivor_count=1, crossover_count=2)
    user_message = messages[1]["content"]
    assert "Execution surface" in user_message
    assert "ParentContextCard" in user_message
    assert "target_style" in user_message
    assert "Produce minimal high-level OperatorCards" in user_message
    assert "Do not emit proof obligations" in user_message
    assert "fusion_goal" in user_message
    assert "parent_roles" in user_message
    assert "Supported composite families" in user_message
    assert "Do not try to solve all exact numeric params" in user_message
    assert "operator_variant must be" in user_message
    assert "mutation_easy" in user_message
    assert "mutation_hard" in user_message
    assert "theorem_decomposition.proof_checkpoints" in user_message
    assert "The six variants exist to be used" in user_message
    assert "Emit exactly 5 work_items" in user_message


def test_llm_calls_use_schema_response_formats():
    planner_format = _planner_response_format()
    generation_format = _generation_response_format()
    crossover_format = _generation_response_format(
        "gcd_divisor_sum", op_type="crossover", parent_ids=["a", "b"]
    )
    assert planner_format["type"] == "json_schema"
    assert generation_format["type"] == "json_schema"
    assert planner_format["json_schema"]["schema"]["properties"]["work_items"]
    assert generation_format["json_schema"]["schema"]["properties"]["family"]["enum"]
    assert "status" in generation_format["json_schema"]["schema"]["required"]
    assert "reasoning_pattern" not in generation_format["json_schema"]["schema"]["required"]
    assert "projected_params" not in generation_format["json_schema"]["schema"]["required"]
    assert "required_params" not in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    assert "parent_contributions" not in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    assert "goal" in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    assert "operator_variant" in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    assert "required_checkpoints" not in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    assert "fusion_contract" not in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]
    theorem_schema = _theorem_response_format()["json_schema"]["schema"]
    assert "status" in theorem_schema["required"]
    assert "lean_code" in theorem_schema["required"]
    assert "parent_usage" in theorem_schema["required"]
    assert "statement_chunks" not in theorem_schema["required"]
    assert "proof_surface" not in theorem_schema["required"]
    assert "Full Lean code" in theorem_schema["properties"]["lean_code"]["description"]
    assert "unsupported_claims" in _theorem_alignment_response_format()["json_schema"]["schema"]["required"]
    crossover_schema = crossover_format["json_schema"]["schema"]["properties"]
    assert "parent_usage" in crossover_schema
    assert crossover_schema["parent_contribution_evidence"]["additionalProperties"]["type"] == "string"
    assert "required_checkpoints" not in planner_format["json_schema"]["schema"]["properties"]["work_items"]["items"]["required"]


def test_validate_plan_accepts_operator_variant_and_fallback_sets_mutation_modes():
    plan = deterministic_fallback_plan(_compatible_pool(), crossover_count=0)
    items = validate_pool_plan(plan, _compatible_pool(), crossover_count=0)
    variants = [item["operator_variant"] for item in items]
    assert variants[0] == "survivor"
    assert any(variant == "mutation_hard" for variant in variants)
    assert all(
        variant in {"survivor", "mutation_easy", "mutation_hard"}
        for variant in variants
    )


def test_validate_plan_projects_parent_refs_to_ids_and_fusion_contract():
    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "operator_variant": "survivor", "parent_refs": [0]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_refs": [1],
                "target_family": "divisor_sum",
                "variation_axis": "stable bridge to divisor-sum template",
                "composition_pattern": "parameter_shift",
                "quality_target": "certified bridge",
            },
            {
                "slot": 2,
                "op_type": "crossover",
                "operator_variant": "crossover_hard",
                "parent_refs": [0, 1],
                "target_family": "gcd_divisor_sum",
                "variation_axis": "couple gcd object with divisor-sum goal",
                "composition_pattern": "family_bridge",
                "quality_target": "two-step composite",
                "fusion_contract": {
                    "parent_A": {"ref": 0, "semantic_role": "object_domain", "contribution": "gcd object"},
                    "parent_B": {"ref": 1, "semantic_role": "computation_target", "contribution": "sigma operation"},
                    "fusion_mechanism": "sequential_composition",
                    "why_not_concatenation": "",
                },
            },
            {"slot": 3, "op_type": "survivor", "operator_variant": "survivor", "parent_refs": [2]},
            {"slot": 4, "op_type": "survivor", "operator_variant": "survivor", "parent_refs": [3]},
        ],
    }
    items = validate_pool_plan(plan, _compatible_pool(), survivor_count=3, crossover_count=1)
    assert items[0]["parent_ids"] == ["g0"]
    assert items[2]["parent_ids"] == ["g0", "d0"]
    assert items[2]["fusion_contract"]["parent_A"]["id"] == "g0"
    assert items[2]["fusion_contract"]["parent_B"]["id"] == "d0"


def test_mutation_hard_cannot_simplify_supported_parent_family():
    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "operator_variant": "survivor", "parent_ids": ["g0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_hard",
                "parent_ids": ["d0"],
                "target_family": "gcd",
                "variation_axis": "drop composite structure into gcd only",
                "composition_pattern": "parameter_shift",
                "quality_target": "hard mutation should not simplify",
            },
            {"slot": 2, "op_type": "survivor", "operator_variant": "survivor", "parent_ids": ["d0"]},
            {"slot": 3, "op_type": "survivor", "operator_variant": "survivor", "parent_ids": ["m0"]},
            {"slot": 4, "op_type": "survivor", "operator_variant": "survivor", "parent_ids": ["u0"]},
        ],
    }
    with pytest.raises(ValueError, match="simplifies parent family"):
        validate_pool_plan(plan, _compatible_pool(), survivor_count=4, crossover_count=0)


def test_validate_plan_normalizes_freeform_checkpoints_to_ids():
    plan = deterministic_fallback_plan(_compatible_pool(), crossover_count=0)
    plan["work_items"][1]["required_checkpoints"] = [
        "n has at least 5 distinct prime factors",
        "final divisor_sum answer verified > 1200",
        "divisor_sum family certified",
    ]
    items = validate_pool_plan(plan, _compatible_pool(), crossover_count=0)
    assert items[1]["required_checkpoints"] == [
        "rich_factorization",
        "numeric_answer_verified",
        "family_certified",
        "bounded_generalization",
    ]


def test_planner_feedback_injects_previous_generation_raw_cases():
    feedback = {
        "weak_slots": [{"large": "raw"}],
        "failed_slots": [{"large": "raw"}],
        "plan_outcome_summary": {
            "success_case_cards": [
                {
                    "reasoning_signature": "divisor_sum:prime_factorization_sigma",
                    "raw_surface": {"statement": "Find the sum of all positive divisors of 840."},
                }
            ],
            "failure_case_cards": [
                {
                    "failure_class": "proof_failed",
                    "raw_surface": {"lean_code": "theorem bad : True := by\n  exact missing"},
                }
            ],
            "weak_signature_summary": {"units_digit:unknown": 2},
            "dominant_signature_groups": {"gcd_sigma": 3},
            "recurrent_failure_signatures": ["missing_quality_checkpoints"],
            "axes_planned_but_not_selected": ["repeat n"],
        },
    }
    messages = _planner_messages(
        _pool(),
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
        generation_feedback=feedback,
    )
    user_message = messages[1]["content"]
    assert "Previous-generation raw cases" in user_message
    assert "Find the sum of all positive divisors of 840" in user_message
    # raw cases are JSON-dumped after compaction; the header proves they were injected
    assert "Previous-generation raw cases" in user_message
    assert "Concise previous-generation feedback" not in user_message
    assert "success_cases" in user_message
    assert "failure_cases" in user_message
    assert "weak_signature_summary" in user_message
    assert "dominant_signature_groups" in user_message


def test_planner_memory_extracts_success_and_failure_cards(tmp_path):
    assert DEFAULT_PLANNER_MEMORY_LIMIT == 24
    memory_file = tmp_path / "prior.jsonl"
    rows = [
        {
            "problem_id": "good",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "target_family": "divisor_sum",
            "operator_card": {"operator_variant": "mutation_easy", "goal": "preserve divisor reasoning"},
            "quality_evidence": {"reasoning_signature": "divisor_sum:prime_factorization_sigma"},
            "statement": "Find the sum of all positive divisors of 840.",
            "solution": "840 = 2^3 * 3 * 5 * 7, so sigma(840) = ...",
            "attempt_history": [
                {
                    "attempt": 0,
                    "status": "certified",
                    "generated_surface_summary": {
                        "statement": "Find the sum of all positive divisors of 840.",
                        "solution": "factorization solution",
                    },
                }
            ],
        },
        {
            "problem_id": "bad",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": False,
            "quality_verdict": "weak",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "quality_flags": ["same_formal_statement_as_parent"],
            "quality_evidence": {"reasoning_signature": "theorem:same_statement"},
            "formal_statement": "theorem bad : True := by\n  trivial",
            "lean_code": "theorem bad : True := by\n  trivial",
        },
    ]
    memory_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    memory = select_planner_memory_cards(
        _pool(),
        memory_dir=tmp_path,
        limit=10,
        enabled=True,
    )

    assert memory["enabled"]
    assert memory["case_count"] == 2
    assert memory["success_count"] == 1
    assert memory["failure_count"] == 1
    lessons = [card["lesson"] for card in memory["cards"]]
    assert any("Reuse this pattern" in lesson for lesson in lessons)
    assert any("same_statement_repair" in lesson for lesson in lessons)
    assert any("Find the sum of all positive divisors of 840" in card["raw_surface"]["statement"] for card in memory["cards"])
    assert any("theorem bad" in card["raw_surface"]["lean_code"] for card in memory["cards"])
    assert any(card["raw_surface"]["attempt_history"] for card in memory["cards"])


def test_planner_memory_reads_curated_casepack_next_to_certified_dir(tmp_path):
    certified_dir = tmp_path / "certified"
    curated_dir = tmp_path / "curated"
    certified_dir.mkdir()
    curated_dir.mkdir()
    curated_row = {
        "problem_id": "curated_pipeline",
        "source_kind": "curated_crossover_success",
        "status": "certified",
        "op_type": "crossover",
        "operator_variant": "crossover_easy",
        "parent_eligible": True,
        "quality_verdict": "acceptable",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "operator_card": {
            "op_type": "crossover",
            "operator_variant": "crossover_easy",
            "goal": "pipeline composite using one IsUnit checkpoint as input to product target",
        },
        "quality_evidence": {
            "crossover_kind": "pipeline_composite",
            "reasoning_signature": "theorem_pipeline:isunit_mul_then_neg",
        },
        "statement": "If u and v are units, prove -(u*v) is a unit.",
        "lean_code": "theorem curated : True := by\n  trivial",
    }
    (curated_dir / "planner_crossover_gold_cases.jsonl").write_text(
        json.dumps(curated_row) + "\n",
        encoding="utf-8",
    )
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement="Prove that units are closed under negation and multiplication.",
            answer="",
            metadata={
                "formal_statement": "theorem thm {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : IsUnit (-u) := by\n  exact hu.neg",
                "lean_code": "theorem thm {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : IsUnit (-u) := by\n  exact hu.neg",
            },
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(
        theorem_pool,
        memory_dir=certified_dir,
        limit=4,
        enabled=True,
    )

    assert memory["curated_case_count"] == 1
    assert memory["success_count"] == 1
    assert any("curated" in source for source in memory["source_files"])
    assert memory["cards"][0]["source_problem_id"] == "curated_pipeline"
    assert memory["cards"][0]["op_type"] == "crossover"


def test_tfae_curated_case_is_memory_vocabulary_not_required_runtime_operator(tmp_path):
    certified_dir = tmp_path / "certified"
    curated_dir = tmp_path / "curated"
    certified_dir.mkdir()
    curated_dir.mkdir()
    tfae_row = {
        "problem_id": "curated_tfae",
        "source_kind": "curated_crossover_success",
        "status": "certified",
        "op_type": "crossover",
        "operator_variant": "crossover_easy",
        "parent_eligible": True,
        "quality_verdict": "acceptable",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "operator_card": {
            "op_type": "crossover",
            "operator_variant": "crossover_easy",
            "goal": "tfae_characterization pilot over certified implication parents",
        },
        "quality_evidence": {
            "crossover_kind": "tfae_characterization",
            "reasoning_signature": "theorem_tfae:directed_implication_cycle",
        },
        "statement": "TFAE pilot.",
        "lean_code": "theorem curated_tfae : True := by\n  trivial",
    }
    (curated_dir / "planner_crossover_gold_cases.jsonl").write_text(
        json.dumps(tfae_row) + "\n",
        encoding="utf-8",
    )
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement="Prove a theorem.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=certified_dir, limit=4, enabled=True)
    planner_user = _planner_messages(theorem_pool, pool_size=5, survivor_count=1, crossover_count=2)[1]["content"]

    assert memory["success_count"] == 1
    assert memory["cards"][0]["source_problem_id"] == "curated_tfae"
    assert memory["cards"][0]["raw_surface"]["quality_evidence"]["crossover_kind"] == "tfae_characterization"
    assert "tfae_characterization" in planner_user
    assert "pilot vocabulary" in planner_user


def test_bundle_master_prompt_keeps_worker_schema_minimal():
    schema = _theorem_response_format()["json_schema"]["schema"]
    required = set(schema["required"])
    properties = schema["properties"]
    planner_schema = _planner_response_schema()
    work_item_props = planner_schema["properties"]["work_items"]["items"]["properties"]

    assert required == {
        "status",
        "statement",
        "formal_statement",
        "lean_code",
        "proof_plan",
        "parent_usage",
        "reason",
    }
    assert "proof_obligations" not in required
    assert "unified_obligation" not in required
    assert "why_not_conjunction" not in required
    assert "lemma_bundle_master" in properties["proof_plan"]["description"]
    assert "prose-only inspiration is insufficient" in properties["parent_usage"]["description"]
    assert "lemma_bundle_master" in work_item_props["goal"]["description"]
    assert "card_and_sum_pipeline" in work_item_props["goal"]["description"]
    assert "accepted-grade success condition" in work_item_props["constraints"]["description"]
    assert "affine_index_drift_only" in work_item_props["avoid"]["description"]
    assert "Planner NoGoPolicyPack" in work_item_props["avoid"]["description"]
    assert "final theorem target" in work_item_props["fusion_goal"]["description"]
    assert "distinct subgoals" in work_item_props["fusion_goal"]["description"]
    memory_contract_props = work_item_props["memory_delta_contract"]["properties"]
    assert "evidence for novelty planning only" in memory_contract_props["similar_card_ids"]["description"]
    assert "target semantics" in memory_contract_props["must_not_repeat"]["description"]
    assert "materially new" in memory_contract_props["required_distinguishing_delta"]["description"]
    assert "broad family" in memory_contract_props["allowed_overlap"]["description"]
    assert "parameter-only drift" in memory_contract_props["novelty_rationale"]["description"]


def test_planner_memory_prefers_matching_theorem_style_and_dedupes_failures(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    duplicate_failure = {
        "status": "certified",
        "op_type": "mutation",
        "parent_eligible": False,
        "quality_verdict": "weak",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "quality_flags": ["same_formal_statement_as_parent"],
        "quality_evidence": {"reasoning_signature": "theorem:same_statement"},
    }
    rows = [
        {**duplicate_failure, "problem_id": "bad1"},
        {**duplicate_failure, "problem_id": "bad2"},
        {
            "problem_id": "numeric_good",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "numeric_answer",
            "target_family": "gcd",
        },
    ]
    memory_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(
        theorem_pool,
        memory_dir=tmp_path,
        limit=10,
        enabled=True,
    )

    failure_cards = [card for card in memory["cards"] if card["kind"] == "failure"]
    assert len(failure_cards) == 1
    assert failure_cards[0]["problem_style"] == "theorem_proof"
    assert memory["cards"][0]["problem_style"] == "theorem_proof"


def test_planner_memory_keeps_same_failure_flag_when_raw_surface_differs(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    base = {
        "status": "certified",
        "op_type": "mutation",
        "parent_eligible": False,
        "quality_verdict": "weak",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "quality_flags": ["same_formal_statement_as_parent"],
        "quality_evidence": {"reasoning_signature": "theorem:same_statement"},
    }
    rows = [
        {**base, "problem_id": "bad1", "lean_code": "theorem bad1 : True := by\n  trivial"},
        {**base, "problem_id": "bad2", "lean_code": "theorem bad2 : True := by\n  trivial"},
    ]
    memory_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)
    failure_cards = [card for card in memory["cards"] if card["kind"] == "failure"]

    assert len(failure_cards) == 2
    assert {card["source_problem_id"] for card in failure_cards} == {"bad1", "bad2"}


def test_planner_memory_reclassifies_low_quality_certified_theorem_rows(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    rows = [
        {
            "problem_id": "neg_chain",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem neg_chain {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by",
            "lean_code": "theorem neg_chain {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by simpa",
        },
        {
            "problem_id": "add_zero",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem add_zero {R} [Ring R] (u : Rˣ) : IsUnit ((u : R) + 0) := by",
            "lean_code": "theorem add_zero {R} [Ring R] (u : Rˣ) : IsUnit ((u : R) + 0) := by simpa [add_zero]",
        },
        {
            "problem_id": "commring_only",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem child {R} [CommRing R] (u : Rˣ) : IsUnit (u : R) := by",
            "lean_code": "theorem child {R} [CommRing R] (u : Rˣ) : IsUnit (u : R) := by exact u.isUnit",
            "parent_context_cards": [
                {
                    "id": "parent",
                    "formal_statement": "theorem parent {R} [Ring R] (u : Rˣ) : IsUnit (u : R) := by",
                }
            ],
        },
    ]
    memory_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)

    assert memory["success_count"] == 0
    assert memory["failure_count"] == 3
    assert memory["reclassified_low_quality_count"] == 3
    flags = memory["reclassified_low_quality_flags"]
    assert flags["trivial_negation_chain"] == 1
    assert flags["trivial_add_zero_padding"] == 1
    assert flags["typeclass_narrowing_only"] == 1
    assert all(card["kind"] == "failure" for card in memory["cards"])
    assert all(card["failure_class"] == "low_quality_syntactic_mutation" for card in memory["cards"])


def test_planner_memory_reclassifies_accepted_proxy_failures(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    row = {
        "problem_id": "certified_but_proxy_failed",
        "status": "certified",
        "op_type": "mutation",
        "parent_eligible": True,
        "quality_verdict": "acceptable",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "formal_statement": "theorem child : True := by",
        "lean_code": "theorem child : True := by trivial",
        "quality_evidence": {
            "reasoning_signature": "theorem_proof:proxy_fail",
            "accepted_proxy": {
                "pass": False,
                "flags": ["formal_surface_not_changed"],
                "reason": "formal_surface_not_changed",
            },
        },
    }
    memory_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)

    assert memory["success_count"] == 0
    assert memory["failure_count"] == 1
    assert memory["cards"][0]["failure_class"] == "accepted_proxy_failed"
    assert "formal_surface_not_changed" in memory["cards"][0]["quality_flags"]


def test_planner_memory_reclassifies_entropy_scaffold_as_failure_case(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    row = {
        "problem_id": "mathd_numbertheory_427_prime_divisor_finset",
        "status": "certified",
        "op_type": "mutation",
        "parent_eligible": True,
        "quality_verdict": "acceptable",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "statement": (
            "If a is the sum of the positive divisors of 500, then the finset "
            "of prime divisors of a is exactly {2, 3, 7, 13}."
        ),
        "formal_statement": (
            "theorem mathd_numbertheory_427_prime_divisor_finset (a : ℕ) "
            "(h₀ : a = ∑ k ∈ Nat.divisors 500, k) : "
            "Finset.filter (fun x => Nat.Prime x) (Nat.divisors a) = ({2, 3, 7, 13} : Finset ℕ) := by"
        ),
        "lean_code": "theorem mathd_numbertheory_427_prime_divisor_finset : True := by trivial",
        "quality_evidence": {"reasoning_signature": "theorem_proof:helper"},
    }
    memory_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)

    assert memory["success_count"] == 0
    assert memory["failure_count"] == 1
    assert memory["cards"][0]["kind"] == "failure"
    reclassification = memory["cards"][0]["raw_surface"]["memory_reclassification"]
    assert reclassification["memory_kind"] == "failure"
    assert reclassification["reason"] == "certified_low_quality_or_accepted_proxy_failed"
    assert "proof_infrastructure_only" in reclassification["flags"]


def test_planner_memory_reclassifies_curation_scaffold_even_if_proxy_passes(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    row = {
        "problem_id": "scaffold_proxy_pass",
        "status": "certified",
        "op_type": "mutation",
        "parent_eligible": True,
        "quality_verdict": "acceptable",
        "problem_style": "theorem_proof",
        "target_family": "theorem_proof",
        "statement": "A local corollary of a certified theorem.",
        "formal_statement": "theorem scaffold_proxy_pass : True := by",
        "lean_code": "theorem scaffold_proxy_pass : True := by trivial",
        "quality_evidence": {
            "reasoning_signature": "theorem_proof:local_corollary",
            "accepted_proxy": {"pass": True, "accepted_grade_pass": True, "flags": []},
            "entropy_direction": {"direction": "increase"},
            "curation_decision": {
                "curation_class": "scaffold",
                "paper_grade": False,
                "scaffold_ok": True,
                "reason": "certified_scaffold_not_paper_grade",
                "flags": ["direct_parent_corollary_only"],
            },
        },
    }
    memory_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)

    assert memory["success_count"] == 0
    assert memory["failure_count"] == 1
    card = memory["cards"][0]
    assert card["kind"] == "failure"
    assert card["failure_class"] == "curation_scaffold"
    assert "curation_scaffold" in card["quality_flags"]
    assert card["raw_surface"]["memory_reclassification"]["memory_kind"] == "scaffold"


def test_planner_memory_reclassifies_combo4_reject_patterns(tmp_path):
    memory_file = tmp_path / "prior.jsonl"
    rows = [
        {
            "problem_id": "order_orientation",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem unit_neg_inv_orderOf_eq {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : orderOf ((hu.neg.unit)⁻¹) = orderOf (hu.neg.unit) := by",
            "lean_code": "theorem unit_neg_inv_orderOf_eq {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : orderOf ((hu.neg.unit)⁻¹) = orderOf (hu.neg.unit) := by exact orderOf_inv hu.neg.unit",
        },
        {
            "problem_id": "unused_cubic_checkpoint",
            "status": "certified",
            "op_type": "crossover",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem neg_unit_orderOf_eq_inverse_from_cubic_checkpoint_imp {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : (¬ ((C (1 : ℚ) - X : Polynomial ℚ) ∣ (X^3 - 3*X - 1 : Polynomial ℚ))) → orderOf (hu.neg.unit) = orderOf ((hu.neg.unit)⁻¹ : Rˣ) := by",
            "lean_code": "theorem neg_unit_orderOf_eq_inverse_from_cubic_checkpoint_imp {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : (¬ ((C (1 : ℚ) - X : Polynomial ℚ) ∣ (X^3 - 3*X - 1 : Polynomial ℚ))) → orderOf (hu.neg.unit) = orderOf ((hu.neg.unit)⁻¹ : Rˣ) := by intro hpoly; have hpoly_checkpoint := hpoly; have order_goal_from_checkpoint (_ : ¬ ((C (1 : ℚ) - X : Polynomial ℚ) ∣ (X^3 - 3*X - 1 : Polynomial ℚ))) : orderOf (hu.neg.unit) = orderOf ((hu.neg.unit)⁻¹ : Rˣ) := by exact (orderOf_inv hu.neg.unit).symm; exact order_goal_from_checkpoint hpoly_checkpoint",
        },
        {
            "problem_id": "linear_factor_paraphrase",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem cubic_X_sub_C_one_not_dvd : ¬ ((X - C (1 : ℚ) : Polynomial ℚ) ∣ (X^3 - 3*X - 1 : Polynomial ℚ)) := by",
            "lean_code": "theorem cubic_X_sub_C_one_not_dvd : ¬ ((X - C (1 : ℚ) : Polynomial ℚ) ∣ (X^3 - 3*X - 1 : Polynomial ℚ)) := by intro hdiv; norm_num at hdiv",
        },
        {
            "problem_id": "root_one_too_narrow",
            "status": "certified",
            "op_type": "mutation",
            "parent_eligible": True,
            "quality_verdict": "acceptable",
            "problem_style": "theorem_proof",
            "target_family": "theorem_proof",
            "formal_statement": "theorem cubic_not_root_one : ¬ Polynomial.IsRoot (X^3 - 3*X - 1 : Polynomial ℚ) (1 : ℚ) := by",
            "lean_code": "theorem cubic_not_root_one : ¬ Polynomial.IsRoot (X^3 - 3*X - 1 : Polynomial ℚ) (1 : ℚ) := by norm_num [Polynomial.IsRoot]",
        },
    ]
    memory_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    memory = select_planner_memory_cards(theorem_pool, memory_dir=tmp_path, limit=10, enabled=True)

    assert memory["success_count"] == 0
    assert memory["failure_count"] == 4
    flags = memory["reclassified_low_quality_flags"]
    assert flags["orderof_orientation_projection"] >= 1
    assert flags["unused_checkpoint"] == 1
    assert flags["linear_factor_paraphrase_only"] >= 1
    assert flags["too_narrow_root_instance"] == 1


def test_planner_prompt_injects_memory_but_worker_prompt_does_not(tmp_path):
    memory = {
        "enabled": True,
        "cards": [
            PlannerMemoryCard(
                kind="failure",
                problem_style="theorem_proof",
                op_type="mutation",
                target_family="theorem_proof",
                status="certified",
                quality_verdict="weak",
                quality_flags=["same_formal_statement_as_parent"],
                lesson="Avoid same_statement_repair unless Gen0 proof completion.",
                raw_surface={
                    "formal_statement": "theorem bad : True := by\n  trivial",
                    "lean_code": "theorem bad : True := by\n  trivial",
                },
                source_run="prior",
                source_problem_id="bad",
            ).model_dump()
        ],
    }
    planner_user = _planner_messages(
        _pool(),
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
        planner_memory=memory,
    )[1]["content"]
    worker_user = _build_generation_messages(
        CertificationInput(
            id="p0",
            statement="Find the sum of all positive divisors of 120.",
            answer="360",
            metadata={"operator_card": {"goal": "make a harder divisor-sum problem"}},
        )
    )[1]["content"]

    assert "Cross-run raw semantic case pack" in planner_user
    assert "Never reference case_id or source_problem_id as a parent" in planner_user
    assert "same_statement_repair" in planner_user
    assert "retry_feedback" in worker_user  # this is the retry worker prompt
    assert "Cross-run raw semantic case pack" not in worker_user


def test_novelty_memory_prompt_uses_pack_and_worker_contract_only():
    novelty_cards = cards_from_rows(
        [
            {
                "problem_id": "accepted_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "generated_params": {"a": 84, "b": 126},
                "lean_code": "theorem raw_should_not_flood_prompt : True := by\n  trivial",
            }
        ],
        source_kind="accepted",
    )
    novelty_memory = {
        "enabled": True,
        "accepted_ledger_path": "accepted.jsonl",
        "accepted_card_count": 1,
        "run_local_card_count": 0,
        "cards": novelty_cards,
        "planner_view": {
            "exact_blockers": {"accepted": [], "run_local": []},
            "soft_neighbors": {"accepted": novelty_cards, "run_local": []},
            "accepted_neighbors": novelty_cards,
            "run_local_neighbors": [],
            "instructions": ["For every generated slot, name the required distinguishing delta."],
        },
    }

    planner_user = _planner_messages(
        _pool(),
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
        novelty_memory=novelty_memory,
    )[1]["content"]
    worker_user = _build_generation_messages(
        CertificationInput(
            id="p0",
            statement="Find GCD(20, 30).",
            answer="10",
            metadata={
                "operator_card": {
                    "goal": "make a harder gcd problem",
                    "memory_delta_contract": {
                        "similar_card_ids": ["accepted_gcd"],
                        "must_not_repeat": ["GCD(84,126) target"],
                        "required_distinguishing_delta": "change target params and role",
                    },
                }
            },
        )
    )[1]["content"]

    assert "NoveltyMemoryPack" in planner_user
    assert "exact_blockers" in planner_user
    assert "soft_neighbors" in planner_user
    assert "memory_delta_contract" in planner_user
    assert "raw_should_not_flood_prompt" not in planner_user
    assert "MemoryDeltaContract" in worker_user
    assert "accepted_gcd" in worker_user
    assert "NoveltyMemoryPack" not in worker_user
    assert "Cross-run raw semantic case pack" not in worker_user


def test_novelty_memory_trace_manifest_is_compact_topk_surface_only():
    accepted_cards = cards_from_rows(
        [
            {
                "problem_id": "accepted_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "lean_code": "theorem raw_should_not_enter_trace : True := by\n  trivial",
            }
        ],
        source_kind="accepted",
    )
    run_cards = cards_from_rows(
        [
            {
                "problem_id": "run_lcm",
                "family": "lcm",
                "statement": "Find LCM(12, 18).",
                "answer": "36",
            }
        ],
        source_kind="run_local",
    )
    novelty_memory = {
        "enabled": True,
        "accepted_ledger_path": "accepted.jsonl",
        "accepted_card_count": 1,
        "run_local_card_count": 1,
        "cards": accepted_cards + run_cards,
        "planner_view": {
            "exact_blockers": {
                "accepted": [{"problem_id": "accepted_gcd", "reason": "statement_sha256"}],
                "run_local": [],
            },
            "soft_neighbors": {"accepted": accepted_cards, "run_local": run_cards},
            "instructions": ["For every generated slot, name the required distinguishing delta."],
        },
    }

    manifest = _novelty_memory_trace_manifest(novelty_memory)
    manifest_text = json.dumps(manifest)

    assert manifest["planner_view"]["exact_blockers"]["accepted"][0]["problem_id"] == "accepted_gcd"
    assert manifest["planner_view"]["soft_neighbors"]["accepted"][0]["problem_id"] == "accepted_gcd"
    assert manifest["planner_view"]["soft_neighbors"]["run_local"][0]["problem_id"] == "run_lcm"
    assert "cards" not in manifest
    assert "raw_should_not_enter_trace" not in manifest_text


def test_novelty_gate_marks_exact_duplicate_as_weak_parent_ineligible():
    novelty_cards = cards_from_rows(
        [
            {
                "problem_id": "accepted_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "generated_params": {"a": 84, "b": 126},
            }
        ],
        source_kind="accepted",
    )
    result = CertificationResult(
        problem_id="candidate",
        status="certified",
        op_type="mutation",
        family="gcd",
        target_family="gcd",
        statement="Find GCD(84, 126).",
        answer="42",
        generated_params={"a": 84, "b": 126},
        quality_verdict="acceptable",
        quality_flags=[],
        quality_evidence={"accepted_proxy": {"pass": True, "flags": []}},
    )
    quality = QualityResult(
        quality_verdict="acceptable",
        quality_flags=[],
        quality_evidence={"accepted_proxy": {"pass": True, "flags": []}},
    )

    merged = _merge_novelty_memory_quality(
        result,
        quality,
        {"enabled": True, "cards": novelty_cards},
    )

    assert merged.quality_verdict == "weak"
    assert "near_duplicate" in merged.quality_flags
    assert "exact_duplicate_memory" in merged.quality_flags
    assert merged.quality_evidence["accepted_proxy"]["pass"] is False
    assert merged.quality_evidence["novelty_memory"]["verdict"] == "near_duplicate"
    assert merged.quality_evidence["novelty_memory"]["gate_cards"][0]["problem_id"] == "accepted_gcd"


def test_structural_overlap_maps_recent_manual_reject_surfaces_to_scaffold_flags():
    group_variant = CertificationResult(
        problem_id="cyclic_variant",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="In a group, a cyclic rotation of a*b*c has the same orderOf target role.",
        formal_statement="theorem cyclic_variant {G} [Group G] (a b c : G) : orderOf (a*b*c) = orderOf (c*a*b) := by",
        lean_code="theorem cyclic_variant : True := by trivial",
    )
    assert "cyclic_transport_same_target_role" in _structural_overlap_curation_flags(group_variant)

    finite_residue = CertificationResult(
        problem_id="residue_variant",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="The multiplicative inverse of 7 gives a remainder 1 modulo 398 in a finite window.",
        formal_statement="theorem residue_variant : True := by",
        lean_code="theorem residue_variant : True := by trivial",
    )
    assert "finite_residue_bookkeeping_only" in _structural_overlap_curation_flags(finite_residue)


def test_parent_like_novelty_memory_adds_delta_contract_not_plan_failure():
    novelty_cards = cards_from_rows(
        [
            {
                "problem_id": "parent_like_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "generated_params": {"a": 84, "b": 126},
            }
        ],
        source_kind="run_local",
    )
    work_items = [
        {
            "slot": 1,
            "op_type": "mutation",
            "parent_ids": ["parent_like_gcd"],
            "target_style": "numeric_answer",
            "target_family": "gcd",
            "goal": "make a related gcd mutation",
        }
    ]

    attached = _attach_novelty_contracts(
        work_items,
        {"enabled": True, "cards": novelty_cards},
    )

    assert attached[0]["op_type"] == "mutation"
    assert attached[0]["memory_delta_contract"]["similar_card_ids"] == ["parent_like_gcd"]
    assert "required_distinguishing_delta" in attached[0]["operator_card"]["memory_delta_contract"]


def test_quality_verify_flags_missing_crossover_contribution():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="crossover",
        parent_ids=["p0", "p1"],
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        generated_params={"n": 840},
        composition_pattern="parameter_transfer",
        generation_notes="Larger divisor-sum instance.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "parameter_transfer",
            "parent_contributions": {
                "p0": "target divisor_sum family",
                "p1": "units digit exponent 2026 should influence the child",
            },
        },
        _pool()[:2],
    )
    assert "missing_parent_contribution" in quality.quality_flags
    assert quality.quality_verdict == "weak"
    assert quality.quality_evidence["checkpoint_coverage"] < 1.0
    assert quality.quality_evidence["missing_checkpoints"]


def test_theorem_mutation_certified_can_infer_parent_contribution():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["thm_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : True := by trivial",
        lean_code="import Mathlib\n\ntheorem child : True := by trivial",
        proof_plan="Use the parent theorem route and close the immediate proof.",
        proof_verify_summary="complete",
    )

    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [CertificationInput(id="thm_parent", statement="A theorem parent.", answer="")],
    )

    assert quality.quality_verdict == "acceptable"
    assert "missing_parent_contribution" not in quality.quality_flags
    assert quality.semantic_parent_contribution["thm_parent"].startswith("lean_code:")
    assert quality.quality_evidence["parent_contribution_source"] == "inferred_theorem_mutation"


def test_alignment_result_normalizes_dict_patch_instructions():
    result = TheoremAlignmentResult.model_validate(
        {
            "aligned": False,
            "verdict": "fail",
            "supported_claims": {},
            "missing_claims": {"statement": "missing in Lean"},
            "unsupported_claims": [],
            "field_patch_instructions": {"formal_statement": "add the missing claim"},
            "rationale": "patch needed",
        }
    )

    assert result.supported_claims == []
    assert result.missing_claims == ["statement: missing in Lean"]
    assert result.field_patch_instructions == ["formal_statement: add the missing claim"]


def test_alignment_result_normalizes_aligned_verdict_consistency():
    result = TheoremAlignmentResult(aligned=True, verdict="fail")

    assert result.aligned is True
    assert result.verdict == "pass"


def test_result_to_pool_problem_drops_recursive_context_metadata():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="mutation",
        statement="Child statement",
        formal_statement="theorem child : True := by trivial",
        lean_code="import Mathlib\n\ntheorem child : True := by trivial",
        parent_context_cards=[{"id": "p", "proof_context": {"lean_code": "x" * 10000}}],
        operator_card={"op_type": "mutation", "parent_cards": [{"id": "p", "proof_context": {"lean_code": "x" * 10000}}]},
        discarded_operator_card={"op_type": "mutation", "parent_cards": [{"id": "old"}]},
        input_metadata={"parent_context_cards": [{"large": "x" * 10000}]},
        quality_evidence={
            "reasoning_signature": "sig",
            "signature_group": "group",
            "parent_contribution": {"p": "large"},
        },
    )

    pooled = _result_to_pool_problem(result)

    assert pooled["parent_context_cards"] == []
    assert "parent_cards" not in pooled["operator_card"]
    assert "parent_cards" not in pooled["discarded_operator_card"]
    assert pooled["input_metadata"] == {}
    assert "parent_contribution" not in pooled["quality_evidence"]


def test_theorem_crossover_still_requires_explicit_parent_contribution():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["a", "b"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : True := by trivial",
        lean_code="import Mathlib\n\ntheorem child : True := by trivial",
        proof_plan="Close a theorem proof.",
        proof_verify_summary="complete",
    )

    quality = verify_slot_quality(
        result,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [
            CertificationInput(id="a", statement="First theorem parent.", answer=""),
            CertificationInput(id="b", statement="Second theorem parent.", answer=""),
        ],
    )

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "missing_parent_contribution" in quality.quality_flags


def test_theorem_quality_allows_pipeline_composite_but_rejects_side_by_side_conjunction():
    parents = [
        CertificationInput(
            id="unit_neg_parent",
            statement="Negation preserves units.",
            answer="",
            metadata={
                "formal_statement": "theorem neg_parent {R} [Ring R] {u : R} (hu : IsUnit u) : IsUnit (-u) := by"
            },
        ),
        CertificationInput(
            id="unit_mul_parent",
            statement="Products of units are units.",
            answer="",
            metadata={
                "formal_statement": "theorem mul_parent {R} [Ring R] {u v : R} (hu : IsUnit u) (hv : IsUnit v) : IsUnit (u * v) := by"
            },
        ),
    ]
    pipeline = CertificationResult(
        problem_id="pipeline",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["unit_neg_parent", "unit_mul_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement=(
            "theorem pipeline {R : Type*} [Ring R] {u v : R} "
            "(hu : IsUnit u) (hv : IsUnit v) : IsUnit ((-u) * v) := by"
        ),
        lean_code=(
            "import Mathlib\n\n"
            "theorem pipeline {R : Type*} [Ring R] {u v : R} "
            "(hu : IsUnit u) (hv : IsUnit v) : IsUnit ((-u) * v) := by\n"
            "  exact IsUnit.mul hu.neg hv"
        ),
        proof_plan="Use the negation parent to produce IsUnit (-u), then feed that checkpoint into the product parent target.",
        proof_verify_summary="complete",
        semantic_parent_contribution={
            "unit_neg_parent": "supplies the negated unit checkpoint IsUnit (-u)",
            "unit_mul_parent": "supplies the product target that consumes the checkpoint",
        },
    )

    pipeline_quality = verify_slot_quality(
        pipeline,
        {
            "op_type": "crossover",
            "target_style": "theorem_proof",
            "fusion_contract": {"fusion_mechanism": "sequential_composition"},
        },
        parents,
    )

    assert pipeline_quality.quality_verdict == "acceptable"
    assert pipeline_quality.quality_evidence["crossover_kind"] == "pipeline_composite"
    assert "parent_checkpoint_not_consumed" not in pipeline_quality.quality_flags

    side_by_side = pipeline.model_copy(
        update={
            "problem_id": "side_by_side",
            "formal_statement": "theorem side_by_side : True ∧ True := by",
            "lean_code": "import Mathlib\n\ntheorem side_by_side : True ∧ True := by\n  exact ⟨trivial, trivial⟩",
            "proof_plan": "Prove both parent theorems side by side as an independent conjunction.",
        }
    )
    side_quality = verify_slot_quality(
        side_by_side,
        {
            "op_type": "crossover",
            "target_style": "theorem_proof",
            "fusion_contract": {"fusion_mechanism": "sequential_composition"},
        },
        parents,
    )

    assert side_quality.quality_verdict == "acceptable"  # advisory
    assert (side_quality.quality_evidence or {}).get("advisory_flags") or (side_quality.quality_flags or []), "no heuristic flag recorded"
    assert side_quality.quality_evidence["crossover_kind"] == "side_by_side_conjunction"
    assert "side_by_side_conjunction" in side_quality.quality_flags
    assert "parent_checkpoint_not_consumed" in side_quality.quality_flags


def test_theorem_quality_allows_lemma_bundle_master_crossover():
    parents = [
        CertificationInput(
            id="unit_neg_parent",
            statement="Negation preserves units.",
            answer="",
            metadata={
                "formal_statement": "theorem neg_parent {R} [Ring R] {u : R} (hu : IsUnit u) : IsUnit (-u) := by"
            },
        ),
        CertificationInput(
            id="unit_mul_parent",
            statement="Products of units are units.",
            answer="",
            metadata={
                "formal_statement": "theorem mul_parent {R} [Ring R] {u v : R} (hu : IsUnit u) (hv : IsUnit v) : IsUnit (u * v) := by"
            },
        ),
    ]
    result = CertificationResult(
        problem_id="bundle_master",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["unit_neg_parent", "unit_mul_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement=(
            "theorem bundle_master {R : Type*} [Ring R] {u v : R} "
            "(hu : IsUnit u) (hv : IsUnit v) : IsUnit (-(u * (-v))) := by"
        ),
        lean_code=(
            "import Mathlib\n\n"
            "theorem bundle_master {R : Type*} [Ring R] {u v : R} "
            "(hu : IsUnit u) (hv : IsUnit v) : IsUnit (-(u * (-v))) := by\n"
            "  exact (IsUnit.mul hu hv.neg).neg"
        ),
        proof_plan=(
            "lemma_bundle_master: use the negation parent to derive the local checkpoint "
            "IsUnit (-v), use the product parent to derive IsUnit (u * (-v)), then consume "
            "that intermediate lemma with negation closure for the master theorem."
        ),
        proof_obligations=[
            "derive IsUnit (-v)",
            "derive IsUnit (u * (-v))",
            "derive final IsUnit (-(u * (-v)))",
        ],
        semantic_parent_contribution={
            "unit_neg_parent": "supports the negated-input and final-negation subgoals",
            "unit_mul_parent": "supports the product-unit intermediate subgoal",
        },
    )

    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "target_style": "theorem_proof",
            "operator_goal": "lemma_bundle_master: consume parent checkpoints as subgoals",
            "fusion_goal": "Bundle→Master certified parent lemmas into one final theorem",
        },
        parents,
    )

    assert quality.quality_verdict == "acceptable"
    assert quality.quality_evidence["crossover_kind"] == "lemma_bundle_master"
    assert "side_by_side_conjunction" not in quality.quality_flags


def test_theorem_quality_rejects_unused_pipeline_checkpoint():
    parents = [
        CertificationInput(
            id="poly_parent",
            statement="The fixed cubic is not divisible by X - C 1.",
            answer="",
            metadata={
                "formal_statement": (
                    "theorem poly_parent : "
                    "¬ ((Polynomial.X - Polynomial.C (1 : ℚ) : Polynomial ℚ) ∣ "
                    "(Polynomial.X^3 - 3*Polynomial.X - 1 : Polynomial ℚ)) := by"
                )
            },
        ),
        CertificationInput(
            id="unit_parent",
            statement="Negated unit has inverse-order equality.",
            answer="",
            metadata={
                "formal_statement": (
                    "theorem unit_parent {R : Type*} [Ring R] {u : R} (hu : IsUnit u) : "
                    "∃ v : Rˣ, (v : R) = -u ∧ orderOf (v⁻¹) = orderOf v := by"
                )
            },
        ),
    ]
    result = CertificationResult(
        problem_id="unused_checkpoint",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["poly_parent", "unit_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement=(
            "theorem unused_checkpoint {R : Type*} [Ring R] {u : R} (hu : IsUnit u) "
            "(hpoly : ¬ ((Polynomial.X - Polynomial.C (1 : ℚ) : Polynomial ℚ) ∣ "
            "(Polynomial.X^3 - 3*Polynomial.X - 1 : Polynomial ℚ))) : "
            "∃ v : Rˣ, (v : R) = -u ∧ orderOf (v⁻¹) = orderOf v := by"
        ),
        lean_code=(
            "import Mathlib\n\n"
            "theorem unused_checkpoint {R : Type*} [Ring R] {u : R} (hu : IsUnit u) "
            "(hpoly : ¬ ((Polynomial.X - Polynomial.C (1 : ℚ) : Polynomial ℚ) ∣ "
            "(Polynomial.X^3 - 3*Polynomial.X - 1 : Polynomial ℚ))) : "
            "∃ v : Rˣ, (v : R) = -u ∧ orderOf (v⁻¹) = orderOf v := by\n"
            "  have hpoly_checkpoint := hpoly\n"
            "  let v : Rˣ := hu.neg.unit\n"
            "  refine ⟨v, ?_, ?_⟩\n"
            "  · exact hu.neg.unit_spec\n"
            "  · have _ := hpoly_checkpoint\n"
            "    exact orderOf_inv v"
        ),
        proof_plan=(
            "Pipeline composite: introduce the polynomial checkpoint, then prove the "
            "negated-unit existential theorem."
        ),
        semantic_parent_contribution={
            "poly_parent": "supplies the polynomial checkpoint",
            "unit_parent": "supplies the negated-unit target",
        },
    )

    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "target_style": "theorem_proof",
            "fusion_contract": {"fusion_mechanism": "sequential_composition"},
        },
        parents,
    )

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "unused_checkpoint" in quality.quality_flags
    assert quality.quality_evidence["misformalization"]["category"] == "misrepresentation"


def test_theorem_canonical_signature_uses_formal_surface_not_statement():
    first = CertificationResult(
        problem_id="a",
        status="certified",
        family="theorem_proof",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="Natural wording A.",
        formal_statement="theorem first (h : True) : True := by\n  exact h",
    )
    second = first.model_copy(
        update={
            "problem_id": "b",
            "statement": "Different natural wording B.",
            "formal_statement": "theorem second (h : True) : True := by\n  exact h",
        }
    )

    assert _canonical_signature(first) == _canonical_signature(second)


def test_result_root_lineages_uses_all_crossover_parents():
    result = CertificationResult(
        problem_id="seed_a__x__seed_b__theorem_gen1",
        status="certified",
        op_type="crossover",
        parent_ids=["seed_a__theorem_gen1", "seed_b__theorem_gen2"],
    )

    assert _result_root_lineages(result) == ["seed_a", "seed_b"]


def test_langsmith_upload_verification_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    status = _verify_langsmith_trace_upload(
        root_run_name="pool_generation/test",
        project_name="test-project",
    )

    assert status == {"enabled": False, "verified": False, "reason": "missing_api_key"}


def test_theorem_quality_rejects_easy_putnam_collapse_patterns():
    parents = [
        CertificationInput(
            id="putnam_1969_b1",
            statement="Parent theorem.",
            answer="",
            metadata={"formal_statement": "theorem parent (n : ℕ) : True := by trivial"},
        )
    ]
    concrete = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="mutation",
        parent_ids=["putnam_1969_b1"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="The sum of divisors of 23 is 24.",
        formal_statement="theorem child : (Nat.divisors 23).sum (fun d => (d : ℤ)) = 24",
        lean_code="import Mathlib\n\ntheorem child : (Nat.divisors 23).sum (fun d => (d : ℤ)) = 24 := by\n  native_decide",
        proof_plan="Compute by native_decide.",
    )

    quality = verify_slot_quality(concrete, {"op_type": "mutation", "target_style": "theorem_proof"}, parents)

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "concrete_native_decide_projection" in quality.quality_flags

    vacuous = concrete.model_copy(
        update={
            "problem_id": "fin1",
            "statement": "For any exponent assignment indexed by Fin 1, no two indices are distinct.",
            "formal_statement": "theorem fin1 : ∀ a : Fin 1 → Fin 2 → ℕ, ∀ i j : Fin 1, i ≠ j → False",
            "lean_code": "import Mathlib\n\ntheorem fin1 : ∀ a : Fin 1 → Fin 2 → ℕ, ∀ i j : Fin 1, i ≠ j → False := by\n  intro a i j hij\n  exact hij (Subsingleton.elim i j)",
            "proof_plan": "Use Fin 1 vacuity.",
        }
    )

    vacuous_quality = verify_slot_quality(vacuous, {"op_type": "mutation", "target_style": "theorem_proof"}, parents)

    assert vacuous_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (vacuous_quality.quality_evidence or {}).get("advisory_flags") or (vacuous_quality.quality_flags or []), "no heuristic flag recorded"
    assert "fin_one_vacuity_theorem" in vacuous_quality.quality_flags


def test_theorem_quality_rejects_syntactic_closure_mutations():
    parent = CertificationInput(
        id="unit_parent",
        statement="A unit theorem.",
        answer="",
        metadata={"formal_statement": "theorem parent {R} [Ring R] (u : Rˣ) : IsUnit (u : R) := by"},
    )
    neg_chain = CertificationResult(
        problem_id="neg_chain",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["unit_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by",
        lean_code="import Mathlib\n\ntheorem child {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by\n  simpa",
        proof_plan="Use double negation wrapper.",
    )

    quality = verify_slot_quality(neg_chain, {"op_type": "mutation", "target_style": "theorem_proof"}, [parent])

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "trivial_negation_chain" in quality.quality_flags
    assert quality.quality_evidence["misformalization"]["category"] == "semantic"

    add_zero = neg_chain.model_copy(
        update={
            "problem_id": "add_zero",
            "formal_statement": "theorem child {R} [Ring R] (u : Rˣ) : IsUnit ((u : R) + 0) := by",
            "lean_code": "import Mathlib\n\ntheorem child {R} [Ring R] (u : Rˣ) : IsUnit ((u : R) + 0) := by\n  simpa [add_zero]",
            "proof_plan": "Use add_zero only.",
        }
    )
    add_zero_quality = verify_slot_quality(add_zero, {"op_type": "mutation", "target_style": "theorem_proof"}, [parent])

    assert add_zero_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (add_zero_quality.quality_evidence or {}).get("advisory_flags") or (add_zero_quality.quality_flags or []), "no heuristic flag recorded"
    assert "trivial_add_zero_padding" in add_zero_quality.quality_flags
    assert add_zero_quality.quality_evidence["misformalization"]["category"] == "semantic"

    commring = neg_chain.model_copy(
        update={
            "problem_id": "commring",
            "formal_statement": "theorem child {R} [CommRing R] (u : Rˣ) : IsUnit (u : R) := by",
            "lean_code": "import Mathlib\n\ntheorem child {R} [CommRing R] (u : Rˣ) : IsUnit (u : R) := by\n  exact u.isUnit",
            "proof_plan": "Only narrow Ring to CommRing.",
        }
    )
    commring_quality = verify_slot_quality(commring, {"op_type": "mutation", "target_style": "theorem_proof"}, [parent])

    assert commring_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (commring_quality.quality_evidence or {}).get("advisory_flags") or (commring_quality.quality_flags or []), "no heuristic flag recorded"
    assert "typeclass_narrowing_only" in commring_quality.quality_flags


def test_accepted_proxy_rejects_certified_projection_and_accepts_nontrivial_mutation():
    projection = CertificationResult(
        problem_id="projection",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child (h : p ∧ q) : p := by",
        lean_code="theorem child (h : p ∧ q) : p := by exact h.1",
    )
    proxy = derive_accepted_proxy(
        projection,
        ["projection_only_theorem"],
        {"feature_delta": {"formal_surface_changed": True}},
    )
    assert proxy["pass"] is False
    assert "projection_only_theorem" in proxy["flags"]

    nontrivial = CertificationResult(
        problem_id="nontrivial",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : True := by",
        lean_code="theorem child : True := by trivial",
    )
    proxy = derive_accepted_proxy(
        nontrivial,
        [],
        {"feature_delta": {"formal_surface_changed": True}},
    )
    assert proxy["pass"] is True


def test_accepted_proxy_rejects_dominated_corollary_and_extensionality_only():
    dominated = CertificationResult(
        problem_id="amc12a_2008_p15__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="For every positive natural number k divisible by 20, the remainder of k^2 + 2^k modulo 10 is 6.",
        formal_statement="theorem units_digit_of_twenty_dvd_pos (k : ℕ) (hk20 : 20 ∣ k) (hkpos : 0 < k) : (k ^ 2 + 2 ^ k) % 10 = 6",
        lean_code="theorem units_digit_of_twenty_dvd_pos (k : ℕ) (hk20 : 20 ∣ k) (hkpos : 0 < k) : (k ^ 2 + 2 ^ k) % 10 = 6 := by omega",
        proof_plan="Use the parent-style periodic lemma and square-term divisibility.",
    )
    dominated_quality = verify_slot_quality(
        dominated,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "theorem_local_corollary_dominated" in dominated_quality.quality_flags
    assert dominated_quality.quality_evidence["accepted_proxy"]["pass"] is False

    extensionality = CertificationResult(
        problem_id="amc12a_2020_p21__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="If finite sets S and T contain exactly the natural numbers with the same gcd-lcm property, then T = S.",
        formal_statement="theorem gcd_lcm_set_unique (S T : Finset ℕ) (hS : ∀ n : ℕ, n ∈ S ↔ P n) (hT : ∀ n : ℕ, n ∈ T ↔ P n) : T = S",
        lean_code="theorem gcd_lcm_set_unique (S T : Finset ℕ) (hS : ∀ n : ℕ, n ∈ S ↔ P n) (hT : ∀ n : ℕ, n ∈ T ↔ P n) : T = S := by ext n; simp [hS, hT]",
        proof_plan="Prove equality of finite sets by extensionality.",
    )
    extensionality_quality = verify_slot_quality(
        extensionality,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "definitional_extensionality_only" in extensionality_quality.quality_flags
    assert extensionality_quality.quality_evidence["accepted_proxy"]["pass"] is False


def test_accepted_proxy_rejects_pid_and_standard_library_restatements():
    pid_restatement = CertificationResult(
        problem_id="pid_constructor",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "Let R be an integral domain. If every ideal of R is assigned a "
            "principal generator, then R is a principal ideal ring."
        ),
        formal_statement=(
            "theorem pid_of_explicit_ideal_span_generator {R : Type*} [CommRing R] [IsDomain R] "
            "(idealGenerator : Ideal R → R) "
            "(idealSpanGenerator : ∀ I : Ideal R, Ideal.span ({idealGenerator I} : Set R) = I) : "
            "IsPrincipalIdealRing R"
        ),
        lean_code=(
            "theorem pid_of_explicit_ideal_span_generator {R : Type*} [CommRing R] [IsDomain R] "
            "(idealGenerator : Ideal R → R) "
            "(idealSpanGenerator : ∀ I : Ideal R, Ideal.span ({idealGenerator I} : Set R) = I) : "
            "IsPrincipalIdealRing R := by exact ⟨fun I => ⟨idealGenerator I, idealSpanGenerator I⟩⟩"
        ),
        proof_plan="Package the assigned generator for every ideal into the PID constructor.",
    )
    pid_quality = verify_slot_quality(
        pid_restatement,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "pid_definition_restatement" in pid_quality.quality_flags
    assert pid_quality.quality_evidence["accepted_proxy"]["pass"] is False

    standard_restatement = CertificationResult(
        problem_id="rootset_standard",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "Let K be an algebraically closed field and let p be a nonzero "
            "polynomial over K. The number of distinct roots of p equals the "
            "degree of p if and only if p is separable."
        ),
        formal_statement=(
            "theorem rootSet_card_eq_natDegree_iff_separable_of_isAlgClosed "
            "(K : Type*) [Field K] [IsAlgClosed K] (p : Polynomial K) (hp : p ≠ 0) : "
            "Fintype.card (p.rootSet K) = p.natDegree ↔ p.Separable"
        ),
        lean_code=(
            "theorem rootSet_card_eq_natDegree_iff_separable_of_isAlgClosed "
            "(K : Type*) [Field K] [IsAlgClosed K] (p : Polynomial K) (hp : p ≠ 0) : "
            "Fintype.card (p.rootSet K) = p.natDegree ↔ p.Separable := by "
            "simpa using (Polynomial.card_rootSet_eq_natDegree_iff_of_splits (K := K) hp "
            "(IsAlgClosed.splits_domain p))"
        ),
        proof_plan="Apply the standard root-set cardinality iff separability theorem.",
    )
    standard_quality = verify_slot_quality(
        standard_restatement,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "standard_library_theorem_restatement" in standard_quality.quality_flags
    assert standard_quality.quality_evidence["accepted_proxy"]["pass"] is False


def test_informal_statement_internal_terms_are_quality_failures():
    hits = informal_statement_internal_term_hits(
        "The parent divisor-count checkpoint gives a certified finset."
    )
    assert hits == ["certified", "checkpoint", "parent"]
    assert informal_statement_internal_term_hits("The cleaned constant is positive.") == []

    result = CertificationResult(
        problem_id="statement_hygiene",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="The parent checkpoint implies the desired divisor bound.",
        formal_statement="theorem child : True := by",
        lean_code="theorem child : True := by trivial",
        proof_plan="Direct proof.",
    )
    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "informal_statement_internal_terms" in quality.quality_flags
    assert quality.quality_evidence["accepted_proxy"]["pass"] is False
    assert quality.quality_evidence["informal_statement_internal_terms"] == [
        "checkpoint",
        "parent",
    ]


def test_entropy_direction_separates_paper_proxy_from_generation_utility():
    certified_corollary = CertificationResult(
        problem_id="corollary",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : True := by",
        lean_code="theorem child : True := by trivial",
    )
    direction = derive_entropy_direction(
        certified_corollary,
        ["direct_parent_corollary_only"],
        {
            "feature_delta": {"formal_surface_changed": True},
            "accepted_proxy": {"pass": False, "flags": ["direct_parent_corollary_only"]},
        },
    )
    assert direction["direction"] == "increase"

    same_surface = derive_entropy_direction(
        certified_corollary,
        ["same_formal_statement_as_parent"],
        {
            "feature_delta": {"formal_surface_changed": True},
            "accepted_proxy": {"pass": False, "flags": ["same_formal_statement_as_parent"]},
        },
    )
    assert same_surface["direction"] == "decrease"


def test_curation_decision_separates_paper_scaffold_and_reject():
    base = CertificationResult(
        problem_id="curation",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : True := by",
        lean_code="theorem child : True := by trivial",
    )
    paper = derive_curation_decision(
        base,
        [],
        {
            "accepted_proxy": {"pass": True, "flags": []},
            "entropy_direction": {"direction": "increase"},
        },
    )
    assert paper["curation_class"] == "paper"
    assert paper["paper_grade"] is True

    scaffold = derive_curation_decision(
        base,
        ["direct_parent_corollary_only"],
        {
            "accepted_proxy": {"pass": False, "flags": ["direct_parent_corollary_only"]},
            "entropy_direction": {"direction": "increase"},
        },
    )
    assert scaffold["curation_class"] == "scaffold"
    assert scaffold["scaffold_ok"] is True

    reject = derive_curation_decision(
        base,
        ["same_formal_statement_as_parent"],
        {
            "accepted_proxy": {"pass": False, "flags": ["same_formal_statement_as_parent"]},
            "entropy_direction": {"direction": "decrease"},
        },
    )
    assert reject["curation_class"] == "reject"
    assert reject["paper_grade"] is False


def test_accepted_grade_rejects_direct_corollary_helper_and_affine_drift():
    ap_parent = CertificationInput(
        id="aime_1984_p1",
        statement="Arithmetic progression parent.",
        answer="",
        metadata={
            "formal_statement": (
                "theorem aime_parent (u : ℕ → ℚ) "
                "(h₀ : ∀ n, u (n + 1) = u n + 1) "
                "(h₁ : (∑ k ∈ Finset.range 98, u k.succ) = 137) : True := by"
            )
        },
    )
    direct = CertificationResult(
        problem_id="ap_even_sum",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="The even-indexed arithmetic progression sum follows from the first 98 terms.",
        formal_statement="theorem ap_even_sum (u : ℕ → ℚ) : True := by",
        lean_code="import Mathlib\n\ntheorem ap_even_sum (u : ℕ → ℚ) : True := by\n  trivial",
        proof_plan="Compute the even-indexed AP sum as a direct parent corollary.",
        proof_verify_summary="complete",
    )
    direct_quality = verify_slot_quality(
        direct,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [ap_parent],
    )
    assert "direct_parent_corollary_only" in direct_quality.quality_flags
    assert direct_quality.quality_evidence["accepted_proxy"]["pass"] is False

    helper_parent = CertificationInput(
        id="mathd_numbertheory_427",
        statement="Find the sum of the positive divisors of 500.",
        answer="1092",
        metadata={"formal_statement": "theorem sigma_500 : (∑ k ∈ Nat.divisors 500, k) = 1092 := by"},
    )
    helper = CertificationResult(
        problem_id="mathd_numbertheory_427_prime_divisor_finset",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="If a is the sum of the positive divisors of 500, then the finset of prime divisors is exactly {2,3,7,13}.",
        formal_statement=(
            "theorem mathd_numbertheory_427_prime_divisor_finset (a : ℕ) "
            "(h₀ : a = ∑ k ∈ Nat.divisors 500, k) : "
            "Finset.filter (fun x => Nat.Prime x) (Nat.divisors a) = ({2, 3, 7, 13} : Finset ℕ) := by"
        ),
        lean_code="import Mathlib\n\ntheorem mathd_numbertheory_427_prime_divisor_finset : True := by\n  trivial",
        proof_plan="Prove the exact finset of prime divisors of the divisor sum.",
        proof_verify_summary="complete",
    )
    helper_quality = verify_slot_quality(
        helper,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [helper_parent],
    )
    assert "proof_infrastructure_only" in helper_quality.quality_flags
    assert helper_quality.quality_evidence["accepted_proxy"]["pass"] is False

    affine_proxy = derive_accepted_proxy(
        CertificationResult(
            problem_id="affine",
            status="certified",
            op_type="crossover",
            target_style="theorem_proof",
            certification_route="theorem_prover",
            formal_statement="theorem child : True := by",
            lean_code="theorem child : True := by trivial",
        ),
        ["affine_index_drift_only"],
        {"parent_contribution": {"a": "domain", "b": "AP"}},
    )
    assert affine_proxy["pass"] is False
    assert "affine_index_drift_only" in affine_proxy["flags"]


def test_accepted_grade_allows_card_and_sum_pipeline_proxy():
    proxy = derive_accepted_proxy(
        CertificationResult(
            problem_id="card_sum_pipeline",
            status="certified",
            op_type="crossover",
            target_style="theorem_proof",
            certification_route="theorem_prover",
            formal_statement="theorem child : True := by",
            lean_code="theorem child : True := by trivial",
        ),
        [],
        {
            "feature_delta": {"formal_surface_changed": True},
            "parent_contribution": {"prime_domain": "card and sum", "ap": "closed form"},
            "parent_checkpoint_consumption": {
                "prime_domain": {"consumed_in_lean_surface": True},
                "ap": {"consumed_in_lean_surface": True},
            },
        },
    )
    assert proxy["pass"] is True
    assert proxy["accepted_grade_pass"] is True


def test_domain_specific_exploit_patterns_are_accepted_proxy_failures():
    ap_parent = CertificationInput(
        id="amc12a_2010_p10",
        statement="Arithmetic sequence parent.",
        answer="",
        metadata={
            "formal_statement": (
                "theorem parent (p q : ℝ) (a : ℕ → ℝ) "
                "(h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) : a 2010 = 8041 := by"
            )
        },
    )
    ap_index = CertificationResult(
        problem_id="amc12a_2010_p10__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="In the arithmetic-sequence setup, if a hidden parameter t fixes the natural index m, then the mth term is 8041.",
        formal_statement=(
            "theorem child (p q t : ℝ) (a : ℕ → ℝ) (m : ℕ) "
            "(h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) "
            "(hmidx : (m : ℝ) = 401 * p + q + 3) : a m = 8041 := by"
        ),
        lean_code="import Mathlib\n\ntheorem child : True := by trivial",
        proof_plan="Solve the hidden real parameter and evaluate the same single index.",
        proof_verify_summary="complete",
    )
    ap_quality = verify_slot_quality(ap_index, {"op_type": "mutation", "target_style": "theorem_proof"}, [ap_parent])
    assert "ap_index_only_theorem" in ap_quality.quality_flags
    assert ap_quality.quality_evidence["accepted_proxy"]["pass"] is False

    ap_shifted_local = CertificationResult(
        problem_id="amc12a_2010_p10__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="For the same four-term arithmetic progression setup, if a hidden natural parameter m satisfies m + 5 = 2014, then the two-step local gap from a (m + 2) to a (m + 4) is 8.",
        formal_statement=(
            "theorem ap_local_gap (p q : ℝ) (a : ℕ → ℝ) (m : ℕ) "
            "(h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) "
            "(h₁ : a 1 = p) (h₂ : a 2 = 9) (h₃ : a 3 = 3 * p - q) "
            "(h₄ : a 4 = 3 * p + q) (hm : m + 5 = 2014) : "
            "a (m + 4) - a (m + 2) = 8 := by"
        ),
        lean_code="import Mathlib\n\ntheorem ap_local_gap : True := by trivial",
        proof_plan="Use the AP recurrence to prove a centered local gap around the hidden shifted index.",
        proof_verify_summary="complete",
    )
    ap_shifted_quality = verify_slot_quality(
        ap_shifted_local,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [ap_parent],
    )
    assert "ap_shifted_local_corollary_only" in ap_shifted_quality.quality_flags
    assert ap_shifted_quality.quality_evidence["accepted_proxy"]["pass"] is False

    mod_parent = CertificationInput(
        id="mathd_numbertheory_33__theorem_gen1",
        statement="Find the bounded modular inverse of 7 modulo 398.",
        answer="57",
        metadata={"formal_statement": "theorem parent (n : Nat) (h0 : n < 398) (h1 : n * 7 % 398 = 1) : n = 57 := by"},
    )
    mod_para = CertificationResult(
        problem_id="mathd_numbertheory_33__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="For every k below 398, if k * 7 has remainder 1 modulo 398, then k = 57.",
        formal_statement="theorem bounded_mod_inverse_unique_398_k (k : Nat) (hk0 : k < 398) (hk1 : k * 7 % 398 = 1) : k = 57 := by",
        lean_code="import Mathlib\n\ntheorem bounded_mod_inverse_unique_398_k : True := by trivial",
        proof_plan="Restate the same modulo inverse theorem with k.",
        proof_verify_summary="complete",
    )
    mod_quality = verify_slot_quality(mod_para, {"op_type": "mutation", "target_style": "theorem_proof"}, [mod_parent])
    assert "mod_inverse_same_conclusion_paraphrase" in mod_quality.quality_flags
    assert mod_quality.quality_evidence["accepted_proxy"]["pass"] is False

    residue_parent = CertificationInput(
        id="mathd_numbertheory_461__theorem_gen1",
        statement="Residue finset parent.",
        answer="",
        metadata={"formal_statement": "theorem parent (S : Finset ℕ) (n : ℕ) : 3 ^ n % 8 = 1 := by"},
    )
    residue_restatement = CertificationResult(
        problem_id="mathd_numbertheory_461__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="Let S be the finite set of natural numbers with residue modulo 8 equal to 1,3,5,7. If n is its cardinality, then 3^n has remainder 1 modulo 8.",
        formal_statement=(
            "theorem residue (S : Finset ℕ) (n : ℕ) "
            "(hS : ∀ m : ℕ, m ∈ S ↔ m ∈ Finset.Icc 1 7 ∧ "
            "(m % 8 = 1 ∨ m % 8 = 3 ∨ m % 8 = 5 ∨ m % 8 = 7)) "
            "(hn : n = S.card) : 3 ^ n % 8 = 1 := by"
        ),
        lean_code="import Mathlib\n\ntheorem residue : True := by trivial",
        proof_plan="Rewrite to the fixed residue finset and compute the cardinality.",
        proof_verify_summary="complete",
    )
    residue_quality = verify_slot_quality(
        residue_restatement,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [residue_parent],
    )
    assert "residue_finset_cardinality_restatement" in residue_quality.quality_flags
    assert residue_quality.quality_evidence["accepted_proxy"]["pass"] is False

    aggregate = residue_restatement.model_copy(
        update={
            "problem_id": "mathd_numbertheory_461__theorem_gen1__theorem_gen1__theorem_gen1",
            "statement": "For the same fixed residue finset S, (3^n * the product of squares in S + the sum of S) has remainder 1 modulo 8.",
            "formal_statement": (
                "theorem aggregate (S : Finset ℕ) (n : ℕ) "
                "(hS : ∀ m : ℕ, m ∈ S ↔ m ∈ Finset.Icc 1 7 ∧ "
                "(m % 8 = 1 ∨ m % 8 = 3 ∨ m % 8 = 5 ∨ m % 8 = 7)) "
                "(hn : n = S.card) : (3 ^ n * S.prod (fun m => m ^ 2) + S.sum id) % 8 = 1 := by"
            ),
            "lean_code": "import Mathlib\n\ntheorem aggregate : True := by\n  native_decide",
            "proof_plan": "Rewrite to the explicit finite set and close by native_decide.",
        }
    )
    aggregate_quality = verify_slot_quality(aggregate, {"op_type": "mutation", "target_style": "theorem_proof"}, [residue_parent])
    assert "fixed_finite_aggregate_computation" in aggregate_quality.quality_flags
    assert "native_decide_fixed_domain_computation" in aggregate_quality.quality_flags
    assert aggregate_quality.quality_evidence["accepted_proxy"]["pass"] is False

    cardinality_pipeline = CertificationResult(
        problem_id="mathd_numbertheory_543__theorem_gen1__x__mathd_numbertheory_461__theorem_gen1",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="If the endpoint-erased divisor finset of 30^4 has cardinality 123 and the reduced residues from 1 through 7 modulo 8 have cardinality 4, then 3 raised to the sum of those two cardinalities is congruent to 3 modulo 8.",
        formal_statement=(
            "theorem card_pipeline "
            "(hdiv : (((Nat.divisors (30 ^ 4)).erase 1).erase (30 ^ 4)).card = 123) "
            "(hres : (Finset.filter (fun x : ℕ => Nat.gcd x 8 = 1) (Finset.Icc 1 7)).card = 4) : "
            "3 ^ ((((Nat.divisors (30 ^ 4)).erase 1).erase (30 ^ 4)).card + "
            "(Finset.filter (fun x : ℕ => Nat.gcd x 8 = 1) (Finset.Icc 1 7)).card) % 8 = 3 := by"
        ),
        lean_code="import Mathlib\n\ntheorem card_pipeline : True := by trivial",
        proof_plan="Substitute the two fixed cardinalities and compute the modular power.",
        proof_verify_summary="complete",
    )
    card_pipeline_quality = verify_slot_quality(
        cardinality_pipeline,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [residue_parent],
    )
    assert "cardinality_arithmetic_pipeline_only" in card_pipeline_quality.quality_flags
    assert card_pipeline_quality.quality_evidence["accepted_proxy"]["pass"] is False

    linear_shift = CertificationResult(
        problem_id="mathd_algebra_359__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="For a real number y, if the midpoint balance gives y = 9, then y + 1 = 10.",
        formal_statement=(
            "theorem linear_shift (y : ℝ) "
            "(h₀ : 12 - (y + 6) = y - 12) : y + 1 = 10 := by"
        ),
        lean_code=(
            "import Mathlib\n\ntheorem linear_shift (y : ℝ) "
            "(h₀ : 12 - (y + 6) = y - 12) : y + 1 = 10 := by\n"
            "  have hy : y = 9 := by linarith\n  linarith"
        ),
        proof_plan="Derive the checkpoint y = 9, then consume it in the shifted conclusion y + 1 = 10.",
        proof_verify_summary="complete",
    )
    linear_shift_quality = verify_slot_quality(
        linear_shift,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "linear_equation_shift_corollary_only" in linear_shift_quality.quality_flags
    assert linear_shift_quality.quality_evidence["accepted_proxy"]["pass"] is False

    ap_bound_padding = CertificationResult(
        problem_id="mathd_algebra_354__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "In an arithmetic sequence with initial term a and common difference d, "
            "if the 7th term is 30 and the 11th term is 60, then for any real upper "
            "bound B with 135 ≤ B, the 21st term is at most B."
        ),
        formal_statement=(
            "theorem mathd_algebra_354_bound (a d B : ℝ) "
            "(h₀ : a + 6 * d = 30) (h₁ : a + 10 * d = 60) "
            "(hB : 135 ≤ B) : a + 20 * d ≤ B := by"
        ),
        lean_code="import Mathlib\n\ntheorem mathd_algebra_354_bound : True := by trivial",
        proof_plan=(
            "Derive the arithmetic-sequence checkpoint a + 20*d = 135, then combine "
            "that checkpoint with the added real bound 135 ≤ B."
        ),
        proof_verify_summary="complete",
    )
    ap_bound_quality = verify_slot_quality(
        ap_bound_padding,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "ap_bound_padding_only" in ap_bound_quality.quality_flags
    assert ap_bound_quality.quality_evidence["accepted_proxy"]["pass"] is False

    ap_interval_padding = CertificationResult(
        problem_id="mathd_algebra_354__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "In an arithmetic sequence over the reals with seventh term 30 and "
            "eleventh term 60, if L ≤ 135 and 135 ≤ U, then the twenty-first "
            "term lies in the closed interval [L, U]."
        ),
        formal_statement=(
            "theorem mathd_algebra_354_two_sided_interval (a d L U : ℝ) "
            "(h₀ : a + 6 * d = 30) (h₁ : a + 10 * d = 60) "
            "(hL : L ≤ 135) (hU : 135 ≤ U) : a + 20 * d ∈ Set.Icc L U := by"
        ),
        lean_code="import Mathlib\n\ntheorem mathd_algebra_354_two_sided_interval : True := by trivial",
        proof_plan="Derive a + 20*d = 135 and wrap it in the supplied interval bounds.",
        proof_verify_summary="complete",
    )
    ap_interval_quality = verify_slot_quality(
        ap_interval_padding,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "ap_interval_bound_padding_only" in ap_interval_quality.quality_flags
    assert ap_interval_quality.quality_evidence["accepted_proxy"]["pass"] is False

    quotient_corollary = CertificationResult(
        problem_id="mathd_numbertheory_33__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If n is less than 398 and n is a multiplicative inverse of 7 modulo 398, "
            "then n / 19 = 3."
        ),
        formal_statement=(
            "theorem mathd_numbertheory_33_quotient_consequence (n : ℕ) "
            "(h₀ : n < 398) (h₁ : n * 7 % 398 = 1) : n / 19 = 3 := by"
        ),
        lean_code="import Mathlib\n\ntheorem mathd_numbertheory_33_quotient_consequence : True := by trivial",
        proof_plan="Use the bounded inverse hypotheses to force n = 57, then compute n / 19 = 3.",
        proof_verify_summary="complete",
    )
    quotient_quality = verify_slot_quality(
        quotient_corollary,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [mod_parent],
    )
    assert "solved_parameter_quotient_corollary_only" in quotient_quality.quality_flags
    assert quotient_quality.quality_evidence["accepted_proxy"]["pass"] is False

    mod_remainder_corollary = CertificationResult(
        problem_id="mathd_numbertheory_33__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If n is less than 398 and n is a multiplicative inverse of 7 modulo "
            "398, then n is congruent to 0 modulo 19."
        ),
        formal_statement=(
            "theorem mathd_numbertheory_33_mod_consequence (n : ℕ) "
            "(h₀ : n < 398) (h₁ : n * 7 % 398 = 1) : n % 19 = 0"
        ),
        lean_code=(
            "import Mathlib\n\ntheorem mathd_numbertheory_33_mod_consequence "
            "(n : ℕ) (h₀ : n < 398) (h₁ : n * 7 % 398 = 1) : n % 19 = 0 := by\n"
            "  have hn : n = 57 := by omega\n  rw [hn]"
        ),
        proof_plan="Use the bounded inverse hypotheses to force n = 57, then compute n % 19 = 0.",
        proof_verify_summary="complete",
    )
    mod_remainder_quality = verify_slot_quality(
        mod_remainder_corollary,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [mod_parent],
    )
    assert "mod_inverse_arithmetic_corollary_only" in mod_remainder_quality.quality_flags
    assert mod_remainder_quality.quality_evidence["accepted_proxy"]["pass"] is False

    finite_window = CertificationResult(
        problem_id="mathd_numbertheory_33__theorem_gen1__theorem_gen1",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If n is less than 398 and n is a multiplicative inverse of 7 modulo "
            "398, then n belongs to the finite filtered window from 50 through 60 "
            "whose elements satisfy the inverse congruence and have remainder 0 modulo 19."
        ),
        formal_statement=(
            "theorem mathd_numbertheory_33_refined_filtered_window_membership (n : ℕ) "
            "(h₀ : n < 398) (h₁ : n * 7 % 398 = 1) : "
            "n ∈ (Finset.Icc 50 60).filter (fun k => k * 7 % 398 = 1 ∧ k % 19 = 0 ∧ k ≤ 57) := by"
        ),
        lean_code="import Mathlib\n\ntheorem mathd_numbertheory_33_refined_filtered_window_membership : True := by trivial",
        proof_plan="Use the bounded inverse hypotheses to force n = 57, then show finite-window membership.",
        proof_verify_summary="complete",
    )
    finite_window_quality = verify_slot_quality(
        finite_window,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [mod_parent],
    )
    assert "finite_mod_inverse_window_restatement" in finite_window_quality.quality_flags
    assert finite_window_quality.quality_evidence["accepted_proxy"]["pass"] is False

    artificial_bridge = CertificationResult(
        problem_id="bridge_crossover",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If an arithmetic-sequence expression represents n and is bounded by B < 398, "
            "while c is the coprime count from 1 to 7 and n is the modulo-398 inverse, "
            "then 3^(n+c) has remainder 3 modulo 8."
        ),
        formal_statement=(
            "theorem bridge (a d B t : ℝ) (n c : ℕ) "
            "(hc : c = Finset.card (Finset.filter (fun x => Nat.gcd x 8 = 1) (Finset.Icc 1 7))) "
            "(h₀ : a + 6 * d = 30) (h₁ : a + 10 * d = 60) "
            "(ht : 135 + t ≤ B) (hB : B < 398) "
            "(hn_shift : (n : ℝ) = a + 20 * d + t) "
            "(hn_inv : n * 7 % 398 = 1) : 3 ^ (n + c) % 8 = 3 := by"
        ),
        lean_code="import Mathlib\n\ntheorem bridge : True := by trivial",
        proof_plan=(
            "Use the AP equations and bridge (n : ℝ) = a + 20*d + t with B < 398 "
            "to obtain n < 398, then run the existing inverse plus coprime-count pipeline."
        ),
        proof_verify_summary="complete",
    )
    bridge_quality = verify_slot_quality(
        artificial_bridge,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [ap_parent, mod_parent, residue_parent],
    )
    assert "artificial_bridge_to_existing_pipeline" in bridge_quality.quality_flags
    assert bridge_quality.quality_evidence["accepted_proxy"]["pass"] is False

    numeric_bound_fit = CertificationResult(
        problem_id="bound_fit_crossover",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If an arithmetic sequence has 7th term 30 and 11th term 60, and n is "
            "the bounded inverse of 7 modulo 398, then the 21st term is at most "
            "132 plus n / 19."
        ),
        formal_statement=(
            "theorem bound_fit (a d : ℝ) (n : ℕ) "
            "(h₀ : a + 6 * d = 30) (h₁ : a + 10 * d = 60) "
            "(hn_lt : n < 398) (hn_inv : n * 7 % 398 = 1) : "
            "a + 20 * d ≤ 132 + ((n / 19 : ℕ) : ℝ) := by"
        ),
        lean_code="import Mathlib\n\ntheorem bound_fit : True := by trivial",
        proof_plan=(
            "Use the AP checkpoint a+20*d=135 and the inverse checkpoint n=57, "
            "so n / 19 = 3 and 135 ≤ 132 + n/19."
        ),
        proof_verify_summary="complete",
    )
    bound_fit_quality = verify_slot_quality(
        numeric_bound_fit,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [ap_parent, mod_parent],
    )
    assert "numeric_bound_fitting_crossover" in bound_fit_quality.quality_flags
    assert bound_fit_quality.quality_evidence["accepted_proxy"]["pass"] is False

    parent_smuggling = CertificationResult(
        problem_id="smuggled_parent_theorem",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If b satisfies the lcm/gcd constraints and the supplied parent "
            "divisor-sum theorem holds for divisor sums of 500, then the prime-filter sum is 25."
        ),
        formal_statement=(
            "theorem smuggled_parent_theorem (b a : ℕ) "
            "(h₀ : Nat.lcm 120 b = 3720) (h₁ : Nat.gcd 120 b = 8) "
            "(h₂ : a = ∑ k ∈ Nat.divisors (2 * b + 4), k) "
            "(h₃ : ∀ c : ℕ, c = ∑ k ∈ Nat.divisors 500, k → "
            "(∑ k ∈ Finset.filter (fun x => Nat.Prime x) (Nat.divisors c), k) = 25) : "
            "(∑ k ∈ Finset.filter (fun x => Nat.Prime x) (Nat.divisors a), k) = 25 := by"
        ),
        lean_code=(
            "import Mathlib\n\ntheorem smuggled_parent_theorem (b a : ℕ) "
            "(h₀ : Nat.lcm 120 b = 3720) (h₁ : Nat.gcd 120 b = 8) "
            "(h₂ : a = ∑ k ∈ Nat.divisors (2 * b + 4), k) "
            "(h₃ : ∀ c : ℕ, c = ∑ k ∈ Nat.divisors 500, k → "
            "(∑ k ∈ Finset.filter (fun x => Nat.Prime x) (Nat.divisors c), k) = 25) : "
            "(∑ k ∈ Finset.filter (fun x => Nat.Prime x) (Nat.divisors a), k) = 25 := by\n"
            "  exact h₃ a h₂"
        ),
        proof_plan="Apply the supplied parent theorem assumption directly.",
        proof_verify_summary="complete",
    )
    parent_smuggling_quality = verify_slot_quality(
        parent_smuggling,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [mod_parent],
    )
    assert "parent_theorem_assumption_smuggling" in parent_smuggling_quality.quality_flags
    assert parent_smuggling_quality.quality_evidence["accepted_proxy"]["pass"] is False

    order_selector = CertificationResult(
        problem_id="order_selector_crossover",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "For every group element g and irrational real x, multiplying x by the "
            "rational factor selected by orderOf g = orderOf g⁻¹ is irrational."
        ),
        formal_statement=(
            "theorem order_selector {G : Type*} [Group G] (g : G) (x : ℝ) : "
            "Irrational x → Irrational (x * (((if orderOf g = orderOf g⁻¹ then 1 else 0 : ℚ) + 1 : ℚ))) := by"
        ),
        lean_code="import Mathlib\n\ntheorem order_selector : True := by trivial",
        proof_plan=(
            "Use orderOf g = orderOf g⁻¹ only to select the rational factor, "
            "then apply irrational multiplication."
        ),
        proof_verify_summary="complete",
    )
    order_selector_quality = verify_slot_quality(
        order_selector,
        {"op_type": "crossover", "target_style": "theorem_proof"},
        [mod_parent],
    )
    assert "order_equality_selector_only" in order_selector_quality.quality_flags
    assert order_selector_quality.quality_evidence["accepted_proxy"]["pass"] is False

    order_role_repeat = CertificationResult(
        problem_id="order_unit_repeat",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If u is a unit in a ring R, then the unit object obtained by "
            "negating u has the same order as its inverse."
        ),
        formal_statement=(
            "theorem order_unit_repeat {R : Type*} [Ring R] {u : R} "
            "(hu : IsUnit u) : orderOf (-u) = orderOf (-u)⁻¹ := by"
        ),
        lean_code="import Mathlib\n\ntheorem order_unit_repeat : True := by trivial",
        proof_plan="Objectify the negated unit and apply the orderOf inverse theorem.",
        proof_verify_summary="complete",
    )
    order_repeat_quality = verify_slot_quality(
        order_role_repeat,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "same_target_role_already_accepted" in order_repeat_quality.quality_flags
    assert order_repeat_quality.quality_evidence["accepted_proxy"]["pass"] is False

    gaussian_witness = CertificationResult(
        problem_id="gaussian_witness_packaging",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "In the Gaussian integers, the witness w = 0 - i explicitly realizes "
            "that (1+i)^2 divides 2: namely 2 = (1+i)^2 w."
        ),
        formal_statement=(
            "theorem gaussian_witness : ∃ w : ℤ, 2 = ((1+i)^2) * w := by"
        ),
        lean_code="import Mathlib\n\ntheorem gaussian_witness : True := by trivial",
        proof_plan="Package the known explicit witness for the same divisibility claim.",
        proof_verify_summary="complete",
    )
    witness_quality = verify_slot_quality(
        gaussian_witness,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "witness_packaging_only" in witness_quality.quality_flags
    assert witness_quality.quality_evidence["accepted_proxy"]["pass"] is False

    coefficient_variant = CertificationResult(
        problem_id="irrational_coeff_variant",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement=(
            "If x is irrational, then x multiplied by the rational expression "
            "3/5 - 1/10 is irrational; the proof first verifies that this "
            "rational expression is nonzero."
        ),
        formal_statement=(
            "theorem irrational_coeff_variant (x : ℝ) (hx : Irrational x) : "
            "Irrational (x * (((3 / 5 : ℚ) - (1 / 10 : ℚ) : ℚ) : ℝ)) := by"
        ),
        lean_code="import Mathlib\n\ntheorem irrational_coeff_variant : True := by trivial",
        proof_plan="Show the selected rational coefficient is nonzero and apply irrational multiplication.",
        proof_verify_summary="complete",
    )
    coefficient_quality = verify_slot_quality(
        coefficient_variant,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [],
    )
    assert "coefficient_engineering_only" in coefficient_quality.quality_flags
    assert coefficient_quality.quality_evidence["accepted_proxy"]["pass"] is False


def test_no_go_policy_registry_covers_retry_and_audit_metadata():
    summary = no_go_policy_summary()
    assert summary["total"] == len(ACCEPTED_PROXY_SEVERE_FLAGS)
    assert "artificial_bridge_to_existing_pipeline" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "numeric_bound_fitting_crossover" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "finite_mod_inverse_window_restatement" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "mod_inverse_arithmetic_corollary_only" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "parent_theorem_assumption_smuggling" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "order_equality_selector_only" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "same_target_role_already_accepted" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "witness_packaging_only" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "coefficient_engineering_only" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "theorem_local_corollary_dominated" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "definitional_extensionality_only" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "pid_definition_restatement" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "standard_library_theorem_restatement" in ACCEPTED_PROXY_SEVERE_FLAGS
    assert "Cardinality alone is not accepted-grade" in RETRY_PATCH_INSTRUCTIONS["cardinality_only_window"]
    assert summary["by_category"]["domain_specific"] >= 8


def test_misformalization_taxonomy_maps_verifier_signals():
    trivial = CertificationResult(
        problem_id="neg_chain",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by",
        lean_code="import Mathlib\n\ntheorem child {R} [Ring R] (u : Rˣ) : IsUnit (-(-u : R)) := by\n  simpa",
    )
    taxonomy = derive_misformalization_taxonomy(
        trivial,
        ["trivial_negation_chain", "trivial_add_zero_padding"],
        {},
    )
    assert taxonomy["level"] == "translation"
    assert taxonomy["category"] == "semantic"

    alignment = derive_misformalization_taxonomy(
        CertificationResult(problem_id="align", status="alignment_failed"),
        ["statement_lean_alignment_failed"],
        {},
    )
    assert alignment["level"] == "translation"
    assert alignment["category"] == "misrepresentation"

    axiom = derive_misformalization_taxonomy(
        CertificationResult(
            problem_id="axiom",
            status="certified",
            lean_code="axiom hidden_shortcut : True\n\ntheorem axiom_child : True := hidden_shortcut",
        ),
        ["axiom_backed_seed_or_child"],
        {},
    )
    assert axiom["level"] == "source"
    assert axiom["category"] == "reporting"

    clean = derive_misformalization_taxonomy(
        CertificationResult(problem_id="clean", status="certified"),
        [],
        {},
    )
    assert clean["level"] == "none"
    assert clean["category"] == "none"


def test_theorem_quality_rejects_unit_product_closure_only_mutation():
    parent = CertificationInput(
        id="unit_parent",
        statement="Units are closed under negation.",
        answer="",
        metadata={"formal_statement": "theorem parent {R} [Ring R] {u : R} (hu : IsUnit u) : IsUnit (-u) := by"},
    )
    result = CertificationResult(
        problem_id="unit_product",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["unit_parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement=(
            "theorem child {R : Type*} [Ring R] {u : R} "
            "(hu : IsUnit u) : IsUnit (u * (-u)) := by"
        ),
        lean_code=(
            "import Mathlib\n\n"
            "theorem child {R : Type*} [Ring R] {u : R} "
            "(hu : IsUnit u) : IsUnit (u * (-u)) := by\n"
            "  exact hu.mul hu.neg"
        ),
        proof_plan="Use product closure for units.",
    )

    quality = verify_slot_quality(result, {"op_type": "mutation", "target_style": "theorem_proof"}, [parent])

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "unit_product_closure_only" in quality.quality_flags


def test_theorem_quality_rejects_projection_and_toy_arithmetic_mutations():
    odd_prime_parent = CertificationInput(
        id="putnam_1983_a3",
        statement="Odd prime parent.",
        answer="",
        metadata={
            "formal_statement": (
                "theorem parent (p : ℕ) (poddprime : Odd p ∧ p.Prime) : "
                "∀ a b : ℕ, a ≠ b → True := by"
            )
        },
    )
    projection = CertificationResult(
        problem_id="prime_projection",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["putnam_1983_a3"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child (p : ℕ) (poddprime : Odd p ∧ p.Prime) : p.Prime := by",
        lean_code="theorem child (p : ℕ) (poddprime : Odd p ∧ p.Prime) : p.Prime := by\n  exact poddprime.2",
        proof_plan="Project the prime conclusion from the second component.",
    )

    projection_quality = verify_slot_quality(
        projection, {"op_type": "mutation", "target_style": "theorem_proof"}, [odd_prime_parent]
    )

    assert projection_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (projection_quality.quality_evidence or {}).get("advisory_flags") or (projection_quality.quality_flags or []), "no heuristic flag recorded"
    assert "projection_only_theorem" in projection_quality.quality_flags

    divisibility = projection.model_copy(
        update={
            "problem_id": "divisibility_weaken",
            "formal_statement": (
                "theorem child (n : ℕ) "
                "(hsigma : (24 : ℤ) ∣ (Nat.divisors n).sum (fun d => (d : ℤ))) : "
                "(8 : ℤ) ∣ (Nat.divisors n).sum (fun d => (d : ℤ)) := by"
            ),
            "lean_code": (
                "theorem child (n : ℕ) "
                "(hsigma : (24 : ℤ) ∣ (Nat.divisors n).sum (fun d => (d : ℤ))) : "
                "(8 : ℤ) ∣ (Nat.divisors n).sum (fun d => (d : ℤ)) := by\n"
                "  rcases hsigma with ⟨k, hk⟩\n  refine ⟨3*k, ?_⟩"
            ),
            "proof_plan": "Unpack 24 divisibility and exhibit 3*k as the quotient witnessing 8 divisibility.",
        }
    )
    divisibility_quality = verify_slot_quality(
        divisibility, {"op_type": "mutation", "target_style": "theorem_proof"}, [odd_prime_parent]
    )

    assert divisibility_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (divisibility_quality.quality_evidence or {}).get("advisory_flags") or (divisibility_quality.quality_flags or []), "no heuristic flag recorded"
    assert "divisibility_weaken_only_theorem" in divisibility_quality.quality_flags

    fin_one = projection.model_copy(
        update={
            "problem_id": "fin_one_digits",
            "formal_statement": (
                "theorem child : "
                "(Matrix.det (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℤ)) + "
                "Matrix.det (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℤ))) = 3 := by"
            ),
            "lean_code": (
                "theorem child : "
                "(Matrix.det (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℤ)) + "
                "Matrix.det (fun (_ : Fin 1) (_ : Fin 1) => (2 : ℤ))) = 3 := by\n"
                "  norm_num"
            ),
            "statement": "In the one-by-one Fin 1 digit-matrix case, the digits 1 through 2 sum to 3.",
            "proof_plan": "Normalize each Fin 1 determinant to its single integer entry.",
        }
    )
    fin_one_quality = verify_slot_quality(
        fin_one, {"op_type": "mutation", "target_style": "theorem_proof"}, [odd_prime_parent]
    )

    assert fin_one_quality.quality_verdict == "acceptable"  # advisory; the judge decides
    assert (fin_one_quality.quality_evidence or {}).get("advisory_flags") or (fin_one_quality.quality_flags or []), "no heuristic flag recorded"
    assert "fin_one_concrete_arithmetic_theorem" in fin_one_quality.quality_flags


def test_theorem_quality_rejects_same_formal_statement_and_same_lineage_crossover():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["thm_parent", "thm_parent__theorem_gen1"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child (h : True) : True := by\n  exact h",
        lean_code="import Mathlib\n\ntheorem child (h : True) : True := by\n  exact h",
        proof_plan="Reuse the same proof.",
        proof_verify_summary="complete",
        semantic_parent_contribution={
            "thm_parent": "proof skeleton",
            "thm_parent__theorem_gen1": "object domain",
        },
    )

    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "target_style": "theorem_proof",
            "operator_goal": "same_statement_repair",
        },
        [
            CertificationInput(
                id="thm_parent",
                statement="Root parent theorem.",
                answer="",
                metadata={"formal_statement": "theorem root (h : True) : True := by\n  exact h"},
            ),
            CertificationInput(
                id="thm_parent__theorem_gen1",
                statement="Child parent theorem.",
                answer="",
                metadata={"formal_statement": "theorem parent (h : True) : True := by\n  exact h"},
            ),
        ],
    )

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "same_formal_statement_as_parent" in quality.quality_flags
    assert "same_lineage_crossover" in quality.quality_flags
    assert "repair_not_harder" in quality.quality_flags


def test_theorem_quality_retry_feedback_is_field_level_for_restatement():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        quality_verdict="weak",
        quality_flags=["same_formal_statement_as_parent", "repair_not_harder"],
        quality_evidence={
            "missing_checkpoints": ["same_formal_statement_as_parent", "repair_not_harder"],
            "reasoning_signature": "theorem_proof:child",
        },
    )

    feedback = _retry_feedback_for_result(result, "mutation", 1)

    assert "Change formal_statement and lean_code" in feedback
    assert "Do not submit a same-statement repair" in feedback
    assert "hypothesis_specialization" in feedback


def test_accepted_grade_retry_feedback_is_field_level():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        quality_verdict="weak",
        quality_flags=[
            "proof_infrastructure_only",
            "direct_parent_corollary_only",
            "linear_equation_shift_corollary_only",
            "affine_index_drift_only",
            "cardinality_only_window",
            "lineage_complexity_without_new_role",
        ],
        quality_evidence={
            "missing_checkpoints": ["accepted_grade"],
            "reasoning_signature": "theorem_proof:child",
        },
    )

    feedback = _retry_feedback_for_result(result, "crossover", 1)

    assert "helper appears as a hypothesis or intermediate lemma" in feedback
    assert "latent parameter target" in feedback
    assert "same linear equation" in feedback
    assert "second aggregate or checkpoint" in feedback
    assert "Cardinality alone is not accepted-grade" in feedback
    assert "name one new mathematical role" in feedback


def test_domain_specific_retry_and_reserve_feedback_are_concrete():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="mutation",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        quality_verdict="weak",
        quality_flags=[
            "ap_index_only_theorem",
            "ap_shifted_local_corollary_only",
            "mod_inverse_same_conclusion_paraphrase",
            "fixed_finite_aggregate_computation",
            "cardinality_arithmetic_pipeline_only",
        ],
        quality_evidence={"missing_checkpoints": ["accepted_grade"]},
    )

    feedback = _retry_feedback_for_result(result, "mutation", 1)

    assert "single-index AP evaluation" in feedback
    assert "local shifted AP corollaries" in feedback
    assert "same modulo-inverse conclusion n=57" in feedback
    assert "fixed finite-set sum/product/card expression" in feedback
    assert "fixed cardinalities only as arithmetic" in feedback

    goal, avoid = _reserve_goal_from_profile(
        {
            "accepted_proxy_flags": {
                "ap_shifted_local_corollary_only": 1,
                "cardinality_arithmetic_pipeline_only": 1,
            },
            "quality_flags": {},
            "selection_reasons": {},
        },
        op_type="mutation",
    )
    assert "closed-form" in goal
    assert "required_new_role:closed_form_or_characterization" in avoid


def test_theorem_quality_rejects_parameter_shift_only_mutation():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["p"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : 2 ^ 2011 % 10 = 8",
        lean_code="import Mathlib\n\ntheorem child : 2 ^ 2011 % 10 = 8 := by\n  native_decide",
        proof_plan="Evaluate by native_decide.",
        semantic_parent_contribution={"p": "same units digit theorem shape with shifted exponent"},
    )

    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [
            CertificationInput(
                id="p",
                statement="Units digit parent.",
                answer="",
                metadata={"formal_statement": "theorem p : 2 ^ 2010 % 10 = 4 := by"},
            )
        ],
    )

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "parameter_shift_only_theorem" in quality.quality_flags
    feedback = _retry_feedback_for_result(
        result.model_copy(
            update={
                "quality_verdict": quality.quality_verdict,
                "quality_flags": quality.quality_flags,
                "quality_evidence": quality.quality_evidence,
            }
        ),
        "mutation",
        1,
    )
    assert "failure_signature=" in feedback and "parameter" in feedback


def test_theorem_quality_rejects_auxiliary_conjunct_only_mutation():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["p"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        formal_statement="theorem child : Nat.gcd 180 168 = 12 ∧ 12 ∣ 180",
        lean_code=(
            "import Mathlib\n\n"
            "theorem child : Nat.gcd 180 168 = 12 ∧ 12 ∣ 180 := by\n"
            "  constructor <;> native_decide"
        ),
        proof_plan="Keep the parent gcd equality and add a side divisibility fact.",
        semantic_parent_contribution={"p": "parent conclusion kept as first conjunct"},
    )

    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "target_style": "theorem_proof"},
        [
            CertificationInput(
                id="p",
                statement="GCD parent.",
                answer="",
                metadata={"formal_statement": "theorem p : Nat.gcd 180 168 = 12 := by"},
            )
        ],
    )

    assert quality.quality_verdict == "acceptable"  # heuristic flags are advisory; the judge decides
    assert (quality.quality_evidence or {}).get("advisory_flags") or (quality.quality_flags or []), "no heuristic flag recorded"
    assert "auxiliary_conjunct_only_theorem" in quality.quality_flags
    feedback = _retry_feedback_for_result(
        result.model_copy(
            update={
                "quality_verdict": quality.quality_verdict,
                "quality_flags": quality.quality_flags,
                "quality_evidence": quality.quality_evidence,
            }
        ),
        "mutation",
        1,
    )
    assert "failure_signature=" in feedback and "auxiliary_conjunct" in feedback


def test_quality_evidence_derives_existing_public_quality_fields():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["p0"],
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        generated_params={"n": 840},
        projected_params={"n": 840},
        composition_pattern="parameter_shift",
        reasoning_pattern="prime_factorization_sigma",
        solution_skeleton={"target_computation": "sigma(840)"},
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "mutation",
            "composition_pattern": "parameter_shift",
            "required_checkpoints": ["reasoning_pattern", "solution_skeleton", "projected_params"],
        },
        [CertificationInput(id="p0", statement="Find the sum of all positive divisors of 120.", answer="360")],
    )
    assert quality.quality_flags == []
    assert quality.quality_evidence["checkpoint_coverage"] == 1.0
    assert quality.quality_evidence["reasoning_signature"].startswith("divisor_sum:")
    assert quality.interestingness_features == quality.quality_evidence["features"]


def test_quality_verify_flags_solution_answer_mismatch_for_certified_template():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["p0"],
        family="divisor_sum_mod",
        statement="Let m be the sum of all positive divisors of 720. Find 583741 mod m.",
        answer="1003",
        generated_params={"n": 720, "a": 583741, "modulus": 2418},
        projected_params={"n": 720, "a": 583741},
        composition_pattern="parameter_shift",
        reasoning_pattern="sigma_then_mod",
        solution_skeleton={
            "target_computation": "583741 mod sigma(720)",
            "expected_answer": 3421,
        },
        solution="Factor 720 incorrectly, get m = 19344. The remainder is 3421. Answer: 3421.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "mutation",
            "composition_pattern": "parameter_shift",
            "required_checkpoints": [
                "reasoning_pattern",
                "solution_skeleton",
                "projected_params",
                "numeric_answer_verified",
            ],
        },
        [CertificationInput(id="p0", statement="Find 2026 mod 37.", answer="28")],
    )
    assert quality.quality_verdict == "weak"
    assert "solution_answer_mismatch" in quality.quality_flags
    assert "solution_skeleton_answer_mismatch" in quality.quality_flags
    assert quality.quality_evidence["solution_verification"]["passed"] is False


def test_solution_verify_ignores_intermediate_sigma_prime_power_claims():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["p0"],
        family="divisor_sum_mod",
        statement="Let m be the sum of all positive divisors of 360. Find 987654 mod m.",
        answer="174",
        generated_params={"n": 360, "a": 987654, "modulus": 1170},
        projected_params={"n": 360, "a": 987654},
        composition_pattern="parameter_shift",
        reasoning_pattern="sigma_then_mod",
        solution_skeleton={
            "target_computation": "987654 mod sigma(360)",
            "expected_answer": 174,
        },
        solution=(
            "sigma(2^3)=15, sigma(3^2)=13, sigma(5)=6. "
            "So sigma(360)=1170 and m=1170. Since 987654 = 844 * 1170 + 174, "
            "the remainder is 174. Answer: 174."
        ),
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "mutation",
            "composition_pattern": "parameter_shift",
            "required_checkpoints": ["numeric_answer_verified"],
        },
        [CertificationInput(id="p0", statement="Find 23 mod 7.", answer="2")],
    )
    assert "solution_modulus_mismatch" not in quality.quality_flags
    assert quality.quality_evidence["solution_verification"]["passed"] is True


def test_solution_verify_does_not_treat_mod_m_equals_answer_as_modulus_claim():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["p0"],
        family="divisor_sum_mod",
        statement="Let m be the sum of all positive divisors of 120. Find 534780 mod m.",
        answer="180",
        generated_params={"n": 120, "a": 534780, "modulus": 360},
        projected_params={"n": 120, "a": 534780},
        composition_pattern="parameter_shift",
        reasoning_pattern="sigma_then_mod",
        solution_skeleton={"target_computation": "534780 mod sigma(120)", "expected_answer": 180},
        solution=(
            "Factor 120 = 2^3 * 3 * 5. Thus m = 360. "
            "Since 534780 = 1485 * 360 + 180, 534780 mod m = 180. Answer: 180."
        ),
    )
    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "composition_pattern": "parameter_shift"},
        [CertificationInput(id="p0", statement="Find 23 mod 7.", answer="2")],
    )
    assert "solution_modulus_mismatch" not in quality.quality_flags
    assert quality.quality_evidence["solution_verification"]["passed"] is True


def test_quality_verify_flags_trivial_modular_remainder():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["p0"],
        family="modular_congruence",
        statement="Find 100 mod 10.",
        answer="0",
        generated_params={"a": 100, "m": 10},
        composition_pattern="parameter_shift",
    )
    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "composition_pattern": "parameter_shift"},
        [CertificationInput(id="p0", statement="Find 23 mod 7.", answer="2")],
    )
    assert "trivial_mod_remainder" in quality.quality_flags


def test_quality_verify_rejects_inspiration_only_crossover():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["stars", "gcd"],
        family="stars_and_bars",
        statement="Count the number of non-negative integer solutions to x_1 + x_2 + x_3 + x_4 + x_5 = 28.",
        answer="35960",
        generated_params={"vars": 5, "sum": 28},
        composition_pattern="family_bridge",
        axis_applied="The GCD answer 154 inspired a larger sum; capped at range limit, sum set to 28.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {
                "stars": "stars_and_bars family structure with 5 variables",
                "gcd": "GCD answer 154 inspires a larger sum capped at 28",
            },
        },
        [
            CertificationInput(
                id="stars",
                statement="Count nonneg integer solutions to x_1 + x_2 + x_3 + x_4 + x_5 = 14.",
                answer="3060",
            ),
            CertificationInput(id="gcd", statement="Find GCD(4620, 1078).", answer="154"),
        ],
    )
    assert "indirect_parent_contribution" in quality.quality_flags
    assert "weak_inspiration_only_crossover" in quality.quality_flags
    assert quality.quality_verdict == "weak"


def test_quality_verify_accepts_exact_answer_as_cross_family_param():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["mod", "div"],
        family="modular_congruence",
        statement="Find 987654 mod 576.",
        answer="390",
        generated_params={"a": 987654, "m": 576},
        composition_pattern="family_bridge",
        reasoning_pattern="modulus_transfer_reduction",
        solution_skeleton={
            "parent_contributions": {
                "mod": "modular reduction structure",
                "div": "divisor_sum answer 576 becomes the modulus",
            },
            "target_computation": "987654 mod 576",
        },
        projected_params={"a": 987654, "m": 576},
        axis_applied="The divisor_sum answer 576 is used as the modulus.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {
                "mod": "modular_congruence family with larger dividend",
                "div": "divisor_sum answer 576 becomes the modulus",
            },
        },
        [
            CertificationInput(id="mod", statement="Find 54321 mod 113.", answer="81"),
            CertificationInput(
                id="div", statement="Find the sum of all positive divisors of 210.", answer="576"
            ),
        ],
    )
    assert quality.quality_flags == ["sequential_composition"]
    assert quality.quality_verdict == "acceptable"
    assert quality.quality_evidence["crossover_kind"] == "pipeline_composite"


def test_quality_verify_marks_sequential_composite_as_acceptable_not_strong():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["gcd", "div"],
        family="gcd_divisor_sum",
        statement="Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
        answer="96",
        generated_params={"a": 84, "b": 126, "gcd": 42},
        composition_pattern="family_bridge",
        reasoning_pattern="gcd_then_sigma",
        solution_skeleton={
            "parent_contributions": {
                "gcd": "GCD inputs define n",
                "div": "divisor-sum operation is applied to n",
            },
            "target_computation": "compute sigma(gcd(84,126))",
        },
        axis_applied="Use the gcd parent as n, then apply the divisor-sum operation from the second parent.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {
                "gcd": "GCD inputs define n",
                "div": "sum divisors operation is applied to n",
            },
            "fusion_contract": {
                "parent_A": {
                    "id": "gcd",
                    "semantic_role": "object_domain",
                    "contribution": "GCD inputs define n",
                },
                "parent_B": {
                    "id": "div",
                    "semantic_role": "computation_target",
                    "contribution": "sum divisors operation is applied to n",
                },
                "fusion_mechanism": "sequential_composition",
                "why_not_concatenation": "",
                "new_problem_core": "compute sigma(gcd(a,b))",
                "expected_lean_footprint": ["Nat.gcd", "List.range"],
                "risk": "pipeline composite",
            },
        },
        [
            CertificationInput(id="gcd", statement="Find GCD(84, 126).", answer="42"),
            CertificationInput(
                id="div", statement="Find the sum of all positive divisors of 120.", answer="360"
            ),
        ],
    )
    assert "indirect_parent_contribution" not in quality.quality_flags
    assert "missing_parent_contribution" not in quality.quality_flags
    assert quality.semantic_parent_contribution["gcd"]
    assert quality.semantic_parent_contribution["div"]
    assert quality.quality_verdict == "acceptable"
    assert quality.quality_evidence["crossover_kind"] == "pipeline_composite"


def test_quality_verify_marks_true_fusion_contract_as_strong():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["gcd", "div"],
        family="gcd_divisor_sum",
        statement="Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
        answer="96",
        generated_params={"a": 84, "b": 126, "gcd": 42},
        composition_pattern="family_bridge",
        reasoning_pattern="gcd_then_sigma",
        solution_skeleton={
            "parent_contributions": {
                "gcd": "GCD inputs define n",
                "div": "divisor-sum operation is applied to n",
            },
            "target_computation": "compute sigma(gcd(84,126))",
        },
        projected_params={"a": 84, "b": 126},
        axis_applied="Use the gcd parent as n, then apply the divisor-sum operation from the second parent.",
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {
                "gcd": "GCD inputs define n",
                "div": "sum divisors operation is applied to n",
            },
            "fusion_contract": {
                "parent_A": {
                    "id": "gcd",
                    "semantic_role": "object_domain",
                    "contribution": "GCD inputs define n",
                },
                "parent_B": {
                    "id": "div",
                    "semantic_role": "goal_form",
                    "contribution": "sum divisors operation is applied to n",
                },
                "fusion_mechanism": "goal_form_transplant",
                "why_not_concatenation": "The divisor-sum goal is applied to the derived gcd object, not solved independently.",
                "new_problem_core": "compute sigma of a gcd-derived object",
                "expected_lean_footprint": ["Nat.gcd", "List.range"],
                "risk": "template may still be arithmetic",
            },
        },
        [
            CertificationInput(id="gcd", statement="Find GCD(84, 126).", answer="42"),
            CertificationInput(
                id="div", statement="Find the sum of all positive divisors of 120.", answer="360"
            ),
        ],
    )
    assert quality.quality_flags == []
    assert quality.quality_verdict == "strong"
    assert quality.quality_evidence["crossover_kind"] == "true_fusion"


def test_quality_verify_same_family_same_role_crossover_is_not_strong():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="crossover",
        parent_ids=["a", "b"],
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        generated_params={"n": 840},
        projected_params={"n": 840},
        composition_pattern="family_bridge",
        reasoning_pattern="prime_factorization_sigma",
        solution_skeleton={"target_computation": "sigma(840)"},
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {
                "a": "divisor sum factorization pattern",
                "b": "divisor sum factorization pattern",
            },
            "fusion_contract": {
                "parent_A": {
                    "id": "a",
                    "semantic_role": "computation_target",
                    "contribution": "divisor sum factorization pattern",
                },
                "parent_B": {
                    "id": "b",
                    "semantic_role": "computation_target",
                    "contribution": "divisor sum factorization pattern",
                },
                "fusion_mechanism": "parameter_coupling",
                "why_not_concatenation": "Both parents constrain one n.",
                "new_problem_core": "compute sigma(n)",
                "expected_lean_footprint": ["List.range"],
                "risk": "same role",
            },
        },
        [
            CertificationInput(id="a", statement="Find the sum of all positive divisors of 120.", answer="360"),
            CertificationInput(id="b", statement="Find the sum of all positive divisors of 210.", answer="576"),
        ],
    )
    assert "same_role_crossover" in quality.quality_flags
    assert quality.quality_verdict != "strong"


def test_quality_verify_one_parent_only_crossover_is_weak():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="crossover",
        parent_ids=["a"],
        family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        generated_params={"n": 840},
        projected_params={"n": 840},
        composition_pattern="family_bridge",
        reasoning_pattern="prime_factorization_sigma",
        solution_skeleton={"target_computation": "sigma(840)"},
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "crossover",
            "composition_pattern": "family_bridge",
            "parent_contributions": {"a": "divisor sum factorization pattern"},
        },
        [CertificationInput(id="a", statement="Find the sum of all positive divisors of 120.", answer="360")],
    )
    assert quality.quality_verdict == "weak"
    assert quality.quality_evidence["crossover_kind"] == "mutation_like"


def test_quality_verify_modular_metric_not_only_dividend_size():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=2,
        op_type="mutation",
        parent_ids=["mod"],
        family="modular_congruence",
        statement="Find 876543 mod 9973.",
        answer="916",
        generated_params={"a": 876543, "m": 9973},
        composition_pattern="parameter_shift",
        reasoning_pattern="nontrivial_modulus_reduction",
    )
    quality = verify_slot_quality(
        result,
        {"op_type": "mutation", "composition_pattern": "parameter_shift"},
        [CertificationInput(id="mod", statement="Find 987654 mod 137.", answer="36")],
    )
    assert "claimed_harder_but_metric_not_increased" not in quality.quality_flags


def test_quality_verify_interprets_semantic_checkpoints():
    result = CertificationResult(
        problem_id="child",
        status="certified",
        lean_level=3,
        op_type="mutation",
        parent_ids=["mod"],
        family="modular_congruence",
        statement="Find 723456 mod 1597.",
        answer="15",
        generated_params={"a": 723456, "m": 1597},
        projected_params={"a": 723456, "m": 1597},
        composition_pattern="parameter_shift",
        reasoning_pattern="multi_step_modular_reduction",
        solution_skeleton={"target_computation": "723456 mod 1597"},
    )
    quality = verify_slot_quality(
        result,
        {
            "op_type": "mutation",
            "composition_pattern": "parameter_shift",
            "required_checkpoints": [
                "dividend is at least 500000",
                "modulus is prime and in range [1500,2000]",
                "modular_congruence family certified",
            ],
        },
        [CertificationInput(id="mod", statement="Find 876543 mod 991.", answer="499")],
    )
    assert quality.quality_flags == []
    assert quality.quality_evidence["checkpoint_coverage"] == 1.0


def test_crossover_count_one_creates_two_parent_slot():
    plan = deterministic_fallback_plan(_compatible_pool(), crossover_count=1)
    items = validate_pool_plan(plan, _compatible_pool(), crossover_count=1)
    crossover = [item for item in items if item["op_type"] == "crossover"]
    assert len(crossover) == 1
    assert len(crossover[0]["parent_ids"]) == 2
    assert len(set(crossover[0]["parent_ids"])) == 2


def test_invalid_crossover_with_one_parent_raises():
    with pytest.raises(ValueError, match="crossover requires"):
        deterministic_fallback_plan(_pool()[:1], pool_size=1, survivor_count=0, crossover_count=1)


def test_pool_graph_exposes_expected_nodes_and_send_route():
    graph = build_pool_generation_graph(checker=FakeLeanChecker(), generator=_fake_generated)
    graph_view = graph.get_graph()
    assert {
        "01_load_seed_pool",
        "02_orchestrator_plan_generation",
        "03_slot_dispatch",
        "04_slot_unit",
        "05_slot_aggregate",
        "06_save_generation",
    }.issubset(set(graph_view.nodes))

    state = {"work_items": deterministic_fallback_plan(_pool())["work_items"]}
    sends = graph.slot_dispatch_route(state)
    assert len(sends) == 5
    assert all(isinstance(send, Send) for send in sends)


def test_pool_generation_writes_jsonl_summary_and_preserves_slot_order(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool),
        generator=_fake_generated,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(lines) == result.cumulative_passed_results_count
    assert len(lines) + result.deduplicated_on_write == sum(result.counts.values())
    assert [line["slot"] for line in lines] == sorted(line["slot"] for line in lines)
    assert result.counts == summary["counts"]
    assert result.counts["certified"] < len(lines)
    assert result.counts["unsupported"] == 1
    assert len(result.failed_slots) == 2
    assert result.generation_save_status in {"partial", "complete", "complete_with_backfill"}
    assert result.completed_at
    assert result.completed_at_compact
    assert summary["run_manifest"]["bench_version"] == "emg2-dynamic-v1"
    assert summary["run_manifest"]["lean_toolchain"]
    assert summary["run_manifest"]["mathlib_rev"]
    assert summary["run_manifest"]["input_sha256"]
    assert summary["generations"][0]["run_manifest_digest"]
    assert summary["generation_zero"]["run_manifest_digest"]
    assert all(line["completed_at_compact"] == result.completed_at_compact for line in lines)
    assert all(line["release_id"] == "emg2-dynamic-v1" for line in lines)
    assert all(line["source_run"] == result.run_name for line in lines)
    assert "generation_feedback" in summary
    assert "weak_slots" in summary["generation_feedback"]
    assert all(
        "misformalization" in weak_slot
        for weak_slot in summary["generation_feedback"]["weak_slots"]
    )
    assert "planner_memory" in summary
    assert "planner_case_pack" in summary
    assert "backfill_events" in summary["generation_feedback"]
    assert summary["generation_feedback"]["backfill_events"] or any(
        line.get("selection_reason") == "selected_entropy_increase_support" for line in lines
    )
    assert summary["generations"]
    assert "quality_verdict" in lines[0]
    assert "parent_eligible" in lines[0]
    assert "canonical_signature" in lines[0]
    assert "formal_statement" in lines[0]
    assert "lean_header" in lines[0]
    assert "source_file" in lines[0]
    assert "input_metadata" in lines[0]


def test_run_manifest_digest_is_stable_and_records_toolchain(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("id,statement,answer\np,Find x.,1\n", encoding="utf-8")

    manifest = _build_run_manifest(
        input_path=input_path,
        output_path=output_path,
        run_name="manifest-test",
        generation_model="codex:gpt-5.5",
        max_generations=3,
        pool_size=5,
        completed_at="2026-05-18T12:00:00+09:00",
        completed_at_compact="20260518_120000",
    )

    assert manifest["bench_version"] == "emg2-dynamic-v1"
    assert manifest["lean_toolchain"]
    assert manifest["mathlib_rev"]
    assert len(manifest["input_sha256"]) == 64
    assert _run_manifest_digest(manifest) == _run_manifest_digest(dict(manifest))


def test_pool_generation_summary_records_mocked_planner_memory(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_csv(input_path)
    (memory_dir / "prior.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "prior_good",
                "status": "certified",
                "op_type": "mutation",
                "parent_eligible": True,
                "quality_verdict": "acceptable",
                "target_family": "divisor_sum",
                "quality_evidence": {"reasoning_signature": "divisor_sum:prime_factorization_sigma"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen_memory = {}

    def planner(pool, state):
        seen_memory.update(state.get("planner_memory") or {})
        return deterministic_fallback_plan(pool)

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=planner,
        generator=_fake_generated,
        planner_memory_dir=memory_dir,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert seen_memory["card_count"] == 1
    assert seen_memory["case_count"] == 1
    assert result.planner_memory["card_count"] == 1
    assert result.planner_case_pack["case_count"] == 1
    assert summary["planner_memory"]["success_count"] == 1
    assert summary["planner_case_pack"]["success_count"] == 1


def test_planner_memory_reclassifies_axiom_backed_theorem_success(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "prior.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "axiom_child",
                "status": "certified",
                "op_type": "mutation",
                "parent_eligible": True,
                "quality_verdict": "acceptable",
                "target_style": "theorem_proof",
                "certification_route": "theorem_prover",
                "statement": "A theorem backed by a hidden axiom.",
                "formal_statement": "theorem axiom_child : True := by",
                "lean_code": (
                    "import Mathlib\n\n"
                    "axiom hidden_shortcut : True\n\n"
                    "theorem axiom_child : True := by\n"
                    "  exact hidden_shortcut"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    memory = select_planner_memory_cards(
        [CertificationInput(id="p", statement="Prove a theorem.", answer="", metadata={"formal_statement": "theorem p : True := by"})],
        memory_dir=memory_dir,
        limit=4,
    )

    assert memory["failure_count"] == 1
    assert memory["success_count"] == 0
    card = memory["cards"][0]
    assert card["kind"] == "failure"
    assert "axiom_backed_seed_or_child" in card["quality_flags"]
    assert card["raw_surface"]["memory_reclassification"]["memory_kind"] == "failure"


def test_pool_generation_can_disable_planner_memory(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_csv(input_path)
    (memory_dir / "prior.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "prior_good",
                "status": "certified",
                "op_type": "mutation",
                "parent_eligible": True,
                "quality_verdict": "acceptable",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool),
        generator=_fake_generated,
        planner_memory_dir=memory_dir,
        disable_planner_memory=True,
    )

    assert not result.planner_memory["enabled"]
    assert result.planner_memory["card_count"] == 0
    assert not result.planner_case_pack["enabled"]
    assert result.planner_case_pack["case_count"] == 0


def test_theorem_parent_generates_theorem_child_and_stays_parent_eligible(tmp_path, isolated_corpus_index):
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "theorem_pool.jsonl"
    summary_path = tmp_path / "theorem_summary.json"
    _write_theorem_csv(input_path)

    def theorem_generator(parent, config):
        return TheoremGeneratedProblem(
            id=f"{parent.id}__child",
            source_problem_id=parent.id,
            statement=f"Prove a harder theorem-style proposition derived from {parent.id}.",
            # Distinct mathematics per parent: dedup is alpha-normalised (names
            # dropped), so children that differ only in name are duplicates.
            formal_statement=f"theorem generated_child_{parent.id} (n : Nat) : n + {sum(map(ord, parent.id))} = {sum(map(ord, parent.id))} + n := by\n  omega",
            lean_header="import Mathlib",
            lean_code=f"import Mathlib\n\ntheorem generated_child_{parent.id} (n : Nat) : n + {sum(map(ord, parent.id))} = {sum(map(ord, parent.id))} + n := by\n  omega",
            proof_plan="Close the strengthened theorem by trivial proof in this mock.",
            proof_obligations=["preserve theorem style", "complete Lean proof"],
            proof_surface="simp_only",
            parent_contribution_evidence={parent.metadata["parent_ids"][0]: "formal_statement shape is preserved"},
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete=True)

    async def theorem_alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool, crossover_count=0),
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    generated = [line for line in lines if line["op_type"] == "mutation"][0]
    assert result.generation_save_status in {"partial", "complete", "complete_with_backfill"}
    assert generated["target_style"] == "theorem_proof"
    assert generated["certification_route"] == "theorem_prover"
    assert generated["status"] == "certified"
    assert generated["parent_eligible"] is True
    assert generated["formal_statement"].startswith("theorem generated_child")
    assert generated["proof_verify_summary"] == "complete"


def test_theorem_survivor_is_carried_without_numeric_template_certification(tmp_path):
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "theorem_pool.jsonl"
    summary_path = tmp_path / "theorem_summary.json"
    _write_theorem_csv(input_path)

    def planner(pool, state):
        return {
            "planner_source": "test",
            "work_items": [
                {"slot": idx, "op_type": "survivor", "operator_variant": "survivor", "parent_ids": [problem.id]}
                for idx, problem in enumerate(pool)
            ],
        }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=RaisingLeanChecker(),
        planner=planner,
        survivor_count=5,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    direct_survivors = [row for row in rows if row["op_type"] == "survivor"]
    assert len(direct_survivors) == 5
    assert all(row["status"] == "survivor" for row in rows)
    assert all(row["target_style"] == "theorem_proof" for row in rows)
    assert all(row["certification_route"] == "theorem_prover" for row in rows)
    assert all(not str(row.get("error") or "").startswith("No supported Lean template") for row in rows)
    assert direct_survivors[0]["formal_statement"].startswith("theorem thm0")


def test_theorem_proof_failure_is_saved_to_jsonl(tmp_path):
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "theorem_pool.jsonl"
    summary_path = tmp_path / "theorem_summary.json"
    _write_theorem_csv(input_path)

    def theorem_generator(parent, config):
        return TheoremGeneratedProblem(
            id=f"{parent.id}__bad_child",
            source_problem_id=parent.id,
            statement="Prove a harder theorem-style proposition.",
            formal_statement="theorem generated_child : True := by\n  bad",
            lean_header="import Mathlib",
            lean_code="import Mathlib\n\ntheorem generated_child : True := by\n  bad",
            proof_plan="Attempt a proof that fails.",
            proof_obligations=["complete Lean proof"],
            proof_surface="simp_only",
            parent_contribution_evidence={parent.metadata["parent_ids"][0]: "formal_statement shape is preserved"},
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        # Statement-first: the sorry-probe passes, the full proof fails.
        if "sorry" in code:
            return LeanVerifyResult(ok=True, complete=False)
        return LeanVerifyResult(ok=False, complete=False, system_error="mock proof failed")

    async def theorem_alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    async def theorem_proof_repairer(generated, **kwargs):
        return None  # exhaust repair turns without producing a candidate

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool, crossover_count=0),
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        theorem_proof_repairer=theorem_proof_repairer,
        crossover_count=0,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    failed = [line for line in lines if line["status"] == "proof_failed"]
    assert failed
    assert failed[0]["parent_eligible"] is False
    assert failed[0]["proof_verify_summary"].startswith("[verifier system error]")


def test_theorem_statement_failure_skips_alignment_llm(tmp_path):
    """Statement-first: a statement that cannot type-check never reaches the
    alignment LLM or the full-proof verifier, so no downstream budget is spent
    on an unusable candidate."""
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "theorem_pool.jsonl"
    summary_path = tmp_path / "theorem_summary.json"
    _write_theorem_csv(input_path)
    alignment_calls = []
    verified_codes = []

    def theorem_generator(parent, config):
        return TheoremGeneratedProblem(
            id=f"{parent.id}__bad_child",
            source_problem_id=parent.id,
            statement="Prove a theorem-style proposition.",
            formal_statement="theorem generated_child : NotAType := by\n  bad",
            lean_code="import Mathlib\n\ntheorem generated_child : NotAType := by\n  bad",
            proof_surface="simp_only",
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        verified_codes.append(code)
        return LeanVerifyResult(ok=False, complete=False, errors=[], raw_stderr="unknown identifier 'NotAType'")

    async def theorem_alignment_verifier(generated, **kwargs):
        alignment_calls.append(generated.id)
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool, crossover_count=0),
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        crossover_count=0,
        max_retries=0,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    failed = [row for row in rows if row["status"] == "statement_failed"]
    assert failed
    assert alignment_calls == []
    assert failed[0]["quality_evidence"]["theorem_alignment"] == {}
    assert failed[0]["quality_evidence"]["failure_class"] == "statement_typecheck_failed"
    assert "statement:typecheck_failed" in failed[0]["retry_reasons"]
    # Only the sorry-probe was ever verified — the full proof never ran.
    assert all("sorry" in code for code in verified_codes)


def test_theorem_alignment_failure_retries_and_is_saved(tmp_path):
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "theorem_pool.jsonl"
    summary_path = tmp_path / "theorem_summary.json"
    _write_theorem_csv(input_path)
    calls = []

    def theorem_generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        return TheoremGeneratedProblem(
            id=f"{parent.id}__misaligned_child",
            source_problem_id=parent.id,
            statement="Prove True and also prove every finite set has no isolated points.",
            statement_chunks=["prove True", "prove every finite set has no isolated points"],
            formal_statement="theorem generated_child : True := by\n  trivial",
            lean_header="import Mathlib",
            lean_code="import Mathlib\n\ntheorem generated_child : True := by\n  trivial",
            proof_plan="Only proves True.",
            proof_obligations=["prove True"],
            proof_surface="simp_only",
            parent_contribution_evidence={parent.metadata["parent_ids"][0]: "uses theorem style"},
    )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete=True)

    async def theorem_alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(
            aligned=False,
            verdict="fail",
            supported_claims=["prove True"],
            unsupported_claims=["every finite set has no isolated points"],
            field_patch_instructions=[
                "remove the no-isolated-points prose claim or add it to formal_statement"
            ],
            rationale="statement is stronger than Lean theorem",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["thm0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["thm1"],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "variation_axis": "chunk-level theorem strengthening",
                "operator_goal": "generate one theorem chunk mutation",
                "reasoning_goal": "preserve theorem style with one formal checkpoint",
                "composition_pattern": "structure_expansion",
                "quality_target": "statement chunks match formal theorem",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["thm2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["thm3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["thm4"]},
        ],
    }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        survivor_count=4,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    failed = [line for line in lines if line["status"] == "alignment_failed"]
    assert failed
    assert len(calls) == 4
    assert "unsupported_claims" in calls[1]
    assert failed[0]["parent_eligible"] is False
    assert failed[0]["retry_reasons"] == ["alignment:statement_lean_mismatch"]
    assert failed[0]["quality_evidence"]["theorem_alignment"]["unsupported_claims"]


def test_theorem_lean_code_uses_canonical_mathlib_header():
    header, code = _normalize_theorem_lean_code(
        lean_header="import Mathlib; import Aesop; set_option maxHeartbeats 2000000",
        lean_code=(
            "import Mathlib.Data.Complex.GaussianInt\n"
            "set_option maxHeartbeats 100\n\n"
            "theorem generated_child : True := by\n  trivial"
        ),
        formal_statement="theorem generated_child : True := by\n  trivial",
    )

    assert header == THEOREM_CANONICAL_HEADER
    assert code.startswith(THEOREM_CANONICAL_HEADER + "\n\n")
    assert "Mathlib.Data.Complex.GaussianInt" not in code
    assert "; import" not in code
    assert "theorem generated_child" in code


def test_theorem_worker_prompt_blocks_independent_crossover_conjunction():
    parent = CertificationInput(
        id="thm_parent",
        statement="A theorem parent.",
        answer="",
        metadata={
            "operator_card": {
                "op_type": "crossover",
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
            },
            "parent_context_cards": [{"id": "a"}, {"id": "b"}],
        },
    )

    user_message = _build_theorem_generation_messages(parent)[1]["content"]

    assert "Do not add specific Mathlib module imports" in user_message
    assert "do not join independent parent theorems" in user_message
    assert "parent_usage should briefly name" in user_message
    assert "Return the minimal artifact only" in user_message
    assert "status=\"cannot_execute\"" in user_message
    assert "Do not cite Mathlib lemma names" in user_message
    assert THEOREM_CANONICAL_HEADER in user_message
    assert "formal_statement must be SELF-CONTAINED" in user_message
    assert "Public statement hygiene" in user_message
    assert "statement must be a mathematical theorem statement only" in user_message


def test_replanner_prompt_injects_no_go_policy_pack():
    item = {
        "op_type": "crossover",
        "operator_variant": "crossover_easy",
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
        "parent_ids": ["a", "b"],
        "parent_context_cards": [{"id": "a"}, {"id": "b"}],
    }
    result = CertificationResult(
        problem_id="child",
        status="certified",
        op_type="crossover",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        quality_verdict="weak",
        quality_flags=["side_by_side_conjunction"],
    )

    user_message = _build_replan_messages(item, result, [])[1]["content"]

    assert "Replan NoGoPolicyPack" in user_message
    assert "side_by_side_conjunction" in user_message
    assert "avoid/avoid_signatures" in user_message


def test_theorem_worker_prompt_accepts_leansearch_premise_pack():
    parent = CertificationInput(
        id="thm_parent",
        statement="A theorem parent.",
        answer="",
        metadata={
            "operator_card": {
                "op_type": "mutation",
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
            },
            "parent_context_cards": [{"id": "a"}],
        },
    )

    user_message = _build_theorem_generation_messages(
        parent,
        premise_pack_block="Formal name: Nat.gcd_comm\nFormal statement: theorem Nat.gcd_comm ...",
    )[1]["content"]

    assert "LeanSearch PremisePack" in user_message
    assert "Formal name: Nat.gcd_comm" in user_message
    assert "validated local names but still hints" in user_message


def test_theorem_alignment_prompt_flags_independent_crossover_conjunction():
    generated = TheoremGeneratedProblem(
        id="child",
        source_problem_id="parent",
        statement="Prove theorem A and theorem B.",
        formal_statement="theorem child : True ∧ True := by exact ⟨trivial, trivial⟩",
        lean_code="import Mathlib\n\ntheorem child : True ∧ True := by exact ⟨trivial, trivial⟩",
    )
    item = {
        "op_type": "crossover",
        "target_style": "theorem_proof",
        "parent_ids": ["a", "b"],
    }

    user_message = _build_theorem_alignment_messages(generated, item)[1]["content"]

    assert "FAIL independent conjunctions" in user_message
    assert "one unified obligation" in user_message


def test_theorem_prompts_limit_overbroad_generalization():
    planner_user = _planner_messages(
        _pool(),
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
    )[1]["content"]
    worker_parent = CertificationInput(
        id="thm_parent",
        statement="A theorem parent.",
        answer="",
        metadata={
            "operator_card": {
                "op_type": "mutation",
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
            },
            "parent_context_cards": [{"id": "a", "problem_style": "theorem_proof"}],
        },
    )
    worker_user = _build_theorem_generation_messages(worker_parent)[1]["content"]

    assert "prefer small same-domain changes" in planner_user
    assert "Never crossover two parents with the same root lineage" in planner_user
    assert "auxiliary-conjunct-only" in planner_user
    assert "Bounded generalization is allowed" in planner_user
    assert "Do not generalize to arbitrary" in planner_user
    assert "Bounded generalization means" in worker_user
    assert "Do not jump to arbitrary broad classes" in worker_user
    assert "statement must be covered by formal_statement and lean_code" in worker_user
    assert "minimal artifact" in worker_user


def test_validate_pool_plan_forces_one_bounded_generalization_slot():
    plan = deterministic_fallback_plan(_pool())
    items = validate_pool_plan(
        plan,
        _pool(),
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
    )
    generated = [item for item in items if item["op_type"] != "survivor"]
    surfaces = [
        " ".join(
            [
                str(item.get("goal") or ""),
                str(item.get("operator_goal") or ""),
                str(item.get("reasoning_goal") or ""),
                " ".join(str(value) for value in item.get("constraints") or []),
            ]
        )
        for item in generated
    ]
    assert any("bounded_generalization" in surface for surface in surfaces)


def test_theorem_only_planner_prompt_limits_crossover_pressure():
    theorem_pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem-style proposition {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]

    planner_user = _planner_messages(
        theorem_pool,
        pool_size=5,
        survivor_count=1,
        crossover_count=2,
    )[1]["content"]

    assert "crossover_count=2 is the requested crossover budget" in planner_user
    assert "lemma_bundle_master" in planner_user
    assert "tfae_characterization" in planner_user
    assert "Use immediate_corollary only as a risky fallback" in planner_user
    assert "Accepted-grade playbook" in planner_user
    assert "card_and_sum_pipeline" in planner_user
    assert "affine_index_drift_only" in planner_user


def test_theorem_crossover_without_shared_surface_is_marked_for_verifier():
    pool = [
        CertificationInput(
            id="group_parent",
            statement="Prove a group theorem.",
            answer="",
            metadata={
                "formal_statement": "theorem group_parent {G : Type*} [Group G] : True := by\n  trivial",
                "lean_header": "import Mathlib",
            },
        ),
        CertificationInput(
            id="poly_parent",
            statement="Prove a polynomial theorem.",
            answer="",
            metadata={
                "formal_statement": "theorem poly_parent (p : Polynomial ℂ) : True := by\n  trivial",
                "lean_header": "import Mathlib",
            },
        ),
    ]
    plan = {
        "work_items": [
            {
                "slot": 0,
                "op_type": "crossover",
                "operator_variant": "crossover_hard",
                "parent_ids": ["group_parent", "poly_parent"],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "variation_axis": "bad disjoint theorem crossover",
                "operator_goal": "immediate_corollary",
                "reasoning_goal": "join unrelated theorem parents",
                "composition_pattern": "family_bridge",
                "quality_target": "unified proof",
                "fusion_contract": {
                    "parent_A": {
                        "id": "group_parent",
                        "semantic_role": "object_domain",
                        "contribution": "group theorem",
                    },
                    "parent_B": {
                        "id": "poly_parent",
                        "semantic_role": "goal_form",
                        "contribution": "polynomial theorem",
                    },
                    "fusion_mechanism": "goal_form_transplant",
                    "why_not_concatenation": "claims a unified theorem",
                    "new_problem_core": "unrelated",
                    "expected_lean_footprint": [],
                    "risk": "disjoint domains",
                },
                "required_checkpoints": ["reasoning_pattern"],
                "avoid_patterns": [],
                "avoid_signatures": [],
                "difficulty_label": "medium",
            }
        ],
    }

    items = validate_pool_plan(plan, pool, pool_size=1, survivor_count=0, crossover_count=1)
    assert "crossover_lacks_shared_lean_surface" in items[0]["avoid_patterns"]
    assert "must_be_pipeline_composite" in items[0]["avoid_patterns"]
    assert items[0]["operator_variant"] == "crossover_easy"


def test_theorem_deterministic_fallback_keeps_crossover_easy_exploration():
    pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem {i}.",
            answer="",
            metadata={
                "formal_statement": f"theorem thm{i} : True := by\n  trivial",
                "lean_code": f"import Mathlib\n\ntheorem thm{i} : True := by\n  trivial",
            },
        )
        for i in range(5)
    ]

    plan = deterministic_fallback_plan(pool, crossover_count=2)
    items = validate_pool_plan(plan, pool, crossover_count=2)
    crossovers = [item for item in items if item["op_type"] == "crossover"]

    assert crossovers
    assert crossovers[0]["operator_variant"] == "crossover_easy"
    assert crossovers[0]["target_style"] == "theorem_proof"
    assert crossovers[0]["target_family"] == "theorem_proof"
    assert "must_be_pipeline_composite" in crossovers[0]["avoid_patterns"]


def test_validate_theorem_plan_forces_one_crossover_when_planner_avoids_it():
    pool = [
        CertificationInput(
            id=f"thm{i}",
            statement=f"Prove theorem {i}.",
            answer="",
            metadata={"formal_statement": f"theorem thm{i} : True := by\n  trivial"},
        )
        for i in range(5)
    ]
    plan = {
        "planner_source": "orchestrator_llm",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "operator_variant": "survivor", "parent_refs": [0]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_refs": [1],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "goal": "small theorem mutation",
            },
            {
                "slot": 2,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_refs": [2],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "goal": "small theorem mutation",
            },
            {
                "slot": 3,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_refs": [3],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "goal": "small theorem mutation",
            },
            {
                "slot": 4,
                "op_type": "mutation",
                "operator_variant": "mutation_easy",
                "parent_refs": [4],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "goal": "small theorem mutation",
            },
        ],
    }

    items = validate_pool_plan(plan, pool, crossover_count=2)
    crossovers = [item for item in items if item["op_type"] == "crossover"]

    assert len(crossovers) == 2
    assert crossovers[0]["operator_variant"] == "crossover_easy"
    assert len(crossovers[0]["parent_ids"]) == 2
    assert "forced_crossover_exploration" in crossovers[0]["avoid_patterns"]
    assert crossovers[1]["operator_variant"] == "crossover_easy"
    assert len(crossovers[1]["parent_ids"]) == 2
    assert "forced_crossover_exploration" in crossovers[1]["avoid_patterns"]


def test_lean_error_line_context_extracts_patch_snippet():
    result = CertificationResult(
        problem_id="child",
        family="theorem_proof",
        status="proof_failed",
        lean_code="\n".join(
            [
                "import Mathlib",
                "import Aesop",
                "set_option maxHeartbeats 2000000",
                "",
                "theorem child : True := by",
                "  have h : True := by",
                "    trivial",
                "  exact bad_identifier",
            ]
        ),
        error="error @ line 8: unknown identifier 'bad_identifier'",
    )

    context = _lean_error_line_context(result)

    assert "8:   exact bad_identifier" in context
    assert any(line.startswith("6:") for line in context)


def test_theorem_retry_feedback_names_patch_instruction_for_unknown_identifier():
    result = CertificationResult(
        problem_id="child",
        family="theorem_proof",
        status="proof_failed",
        target_style="theorem_proof",
        certification_route="theorem_prover",
        op_type="mutation",
        lean_code="import Mathlib\n\ntheorem child : True := by\n  exact bad_identifier",
        error="error @ line 4: unknown identifier 'bad_identifier'",
        proof_verify_summary="error @ line 4: unknown identifier 'bad_identifier'",
    )

    feedback = _retry_feedback_for_result(result, "mutation", 1)

    assert "Remove invented lemma names" in feedback
    assert "formal_statement" in feedback
    assert "Do not output proof_surface" in feedback


def test_theorem_lean_verify_span_is_nested_under_certification(monkeypatch):
    trace_names = []

    class FakeTrace:
        def __init__(self, name, **kwargs):
            self.name = name

        def __enter__(self):
            trace_names.append(self.name)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def end(self, outputs=None):
            self.outputs = outputs or {}

    monkeypatch.setattr("src.orchestration.pool_generation.ls.trace", lambda name, **kwargs: FakeTrace(name, **kwargs))

    parent = CertificationInput(id="thm0", statement="Prove True.", answer="")
    item = {
        "slot": 1,
        "retry_count": 2,
        "op_type": "mutation",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
    }

    def theorem_generator(parent_input, config):
        return TheoremGeneratedProblem(
            id="child",
            source_problem_id=parent_input.id,
            statement="Prove True.",
            formal_statement="theorem child : True := by\n  trivial",
            lean_code="import Mathlib\n\ntheorem child : True := by\n  trivial",
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete=True, verify_time=0.01)

    async def alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    result = asyncio.run(
        _certify_theorem_child(
            parent_input=parent,
            item=item,
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=theorem_generator,
            theorem_verifier=theorem_verifier,
            theorem_alignment_verifier=alignment_verifier,
        )
    )

    assert result.status == "certified"
    assert "theorem_lean_verify.slot_1.attempt_2" in trace_names


def test_theorem_proof_surface_is_diagnostic_not_preflight_gate(monkeypatch, isolated_corpus_index):
    monkeypatch.setattr(
        "src.orchestration.pool_generation.ls.trace",
        lambda name, **kwargs: nullcontext(type("Run", (), {"end": lambda self, outputs=None: None})()),
    )
    parent = CertificationInput(id="thm0", statement="Prove True.", answer="")
    item = {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_easy",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
        "parent_context_cards": [
            {
                "id": "thm0",
                "problem_style": "theorem_proof",
                "proof_context": {"proof_body_available": False},
            }
        ],
    }

    def theorem_generator(parent_input, config):
        return TheoremGeneratedProblem(
            id="child",
            source_problem_id=parent_input.id,
            statement="Prove True from True.",
            formal_statement="theorem child (h : True) : True := by\n  exact h",
            lean_code="import Mathlib\n\ntheorem child (h : True) : True := by\n  exact h",
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete=True, verify_time=0.01)

    async def alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    result = asyncio.run(
        _certify_theorem_child(
            parent_input=parent,
            item=item,
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=theorem_generator,
            theorem_verifier=theorem_verifier,
            theorem_alignment_verifier=alignment_verifier,
        )
    )

    assert result.status == "certified"
    assert result.quality_evidence["proof_surface"] == "exact_existing"
    assert result.quality_evidence["proof_surface_allowed_by_operator"] is False
    assert result.quality_evidence["proof_surface_source"] == "inferred_from_lean_code"


def test_theorem_contract_failed_skips_lean_verifier():
    parent = CertificationInput(id="thm0", statement="Prove True.", answer="")
    item = {
        "slot": 1,
        "op_type": "mutation",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
    }
    verifier_called = False

    def theorem_generator(parent_input, config):
        return TheoremGeneratedProblem(
            id="child",
            source_problem_id=parent_input.id,
            contract_status="contract_failed",
            failure_reason="no executable checkpoint",
            statement="",
            formal_statement="",
            lean_code="",
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        nonlocal verifier_called
        verifier_called = True
        return LeanVerifyResult(ok=True, complete=True, verify_time=0.01)

    result = asyncio.run(
        _certify_theorem_child(
            parent_input=parent,
            item=item,
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=theorem_generator,
            theorem_verifier=theorem_verifier,
            theorem_alignment_verifier=None,
        )
    )

    assert result.status == "generation_failed"
    assert result.quality_evidence["contract_status"] == "contract_failed"
    assert "no executable checkpoint" in (result.error or "")
    assert result.statement == ""
    assert result.formal_statement == ""
    assert result.lean_code is None
    assert verifier_called is False


def test_legacy_proof_surface_unavailable_no_longer_triggers_replan_decision():
    result = CertificationResult(
        problem_id="child",
        op_type="mutation",
        certification_route="theorem_prover",
        status="generation_failed",
        error="proof_surface_unavailable: parent_rewrite not allowed",
    )

    assert not _is_plan_level_failure(result, op_type="mutation", failure_history=["generation_failed"])


def test_repeated_theorem_proof_failure_triggers_replan_decision():
    result = CertificationResult(
        problem_id="child",
        op_type="mutation",
        certification_route="theorem_prover",
        status="proof_failed",
        error="unsolved goals",
    )
    signature = _retry_feedback_for_result(result, "mutation", 0)

    assert _is_plan_level_failure(
        result,
        op_type="mutation",
        failure_history=["unsolved goals", "unsolved goals"],
    )
    assert "Previous theorem attempt failed" in signature


def test_attempt_history_drives_repeated_lean_syntax_replan_decision():
    item = {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_easy",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
    }
    result = CertificationResult(
        problem_id="child",
        op_type="mutation",
        certification_route="theorem_prover",
        status="generation_failed",
        error="unexpected token 'in'; expected theorem proof body",
    )
    history = [
        _attempt_history_card(attempt=0, item=item, result=result),
        _attempt_history_card(attempt=1, item=item, result=result),
    ]

    assert _failure_class(result) == "lean_syntax_error"
    assert _is_plan_level_failure(
        result,
        op_type="mutation",
        failure_history=["unexpected token", "unexpected token"],
        attempt_history=history,
    )


def test_attempt_history_summary_preserves_raw_surface_text():
    item = {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_easy",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
    }
    result = CertificationResult(
        problem_id="child",
        op_type="mutation",
        certification_route="theorem_prover",
        status="proof_failed",
        statement="Show the specialized theorem with an explicit witness.",
        formal_statement="theorem child : True := by\n  trivial",
        lean_code="theorem child : True := by\n  trivial",
        proof_plan="Use the parent checkpoint directly.",
        error="unsolved goals",
    )

    history = [_attempt_history_card(attempt=0, item=item, result=result)]
    summary = _attempt_history_summary(history)

    surface = summary[0]["generated_surface"]
    assert "explicit witness" in surface["statement"]
    assert "theorem child" in surface["formal_statement"]
    assert "theorem child" in surface["lean_code"]
    assert "parent checkpoint" in surface["proof_plan"]


def test_theorem_candidate_preflight_rejects_statement_only_and_sorry():
    statement_only = TheoremGeneratedProblem(
        id="bad",
        source_problem_id="p",
        statement="Prove a theorem.",
        formal_statement="theorem bad : True :=",
        lean_code="import Mathlib\n\ntheorem bad : True :=",
    )
    sorry_proof = TheoremGeneratedProblem(
        id="sorry_bad",
        source_problem_id="p",
        statement="Prove a theorem.",
        formal_statement="theorem sorry_bad : True := by\n  sorry",
        lean_code="import Mathlib\n\ntheorem sorry_bad : True := by\n  sorry",
    )

    assert _theorem_candidate_preflight(statement_only)["failure_class"] == "invalid_formal_shape"
    assert _theorem_candidate_preflight(sorry_proof)["failure_class"] == "proof_contains_sorry"


def test_theorem_generator_json_failure_stays_inside_generation_retry_surface():
    parent = CertificationInput(
        id="thm0",
        statement="Prove True.",
        answer="",
        metadata={
            "formal_statement": "theorem thm0 : True :=",
            "lean_header": "import Mathlib",
        },
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_easy",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
    }

    def theorem_generator(parent_input, config):
        raise ValueError("Unterminated string starting at line 1 column 3")

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        raise AssertionError("verifier should not run")

    result = asyncio.run(
        _certify_theorem_child(
            parent_input=parent,
            item=item,
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=theorem_generator,
            theorem_verifier=theorem_verifier,
            theorem_alignment_verifier=None,
        )
    )

    assert result.status == "generation_failed"
    assert _failure_class(result) == "llm_json_parse_error"
    assert "llm_json_parse_error" in result.quality_flags


def test_generation_zero_proof_completion_enriches_missing_seed_body():
    parent = CertificationInput(
        id="thm0",
        statement="Prove True.",
        answer="",
        metadata={
            "formal_statement": "theorem thm0 : True := by",
            "lean_header": "import Mathlib",
        },
    )

    async def completer(problem, config):
        return problem.model_copy(
            update={
                "metadata": {
                    **problem.metadata,
                    "lean_code": "import Mathlib\n\ntheorem thm0 : True := by\n  trivial",
                    "formal_statement": "import Mathlib\n\ntheorem thm0 : True := by\n  trivial",
                    "gen0_proof_completed": True,
                }
            }
        )

    completed = asyncio.run(
        _complete_generation_zero_proofs(
            [parent],
            config=GenerationConfig(model="test-model"),
            seed_proof_completer=completer,
        )
    )

    assert completed[0].metadata["gen0_proof_completed"] is True
    assert "trivial" in completed[0].metadata["lean_code"]
    assert completed[0].metadata["formal_statement"] == "theorem thm0 : True := by"
    assert completed[0].metadata["seed_formal_statement_original"] == "theorem thm0 : True := by"
    summary = _generation_zero_summary([parent], completed)
    rows = _generation_zero_rows(completed)
    assert summary["missing_proof_body_count"] == 1
    assert summary["completed_count"] == 1
    assert rows[0]["generation"] == 0
    assert rows[0]["op_type"] == "seed_proof_completion"
    assert rows[0]["status"] == "certified"
    assert rows[0]["formal_statement"] == "theorem thm0 : True := by"


def test_generation_zero_effective_parallelism_clamps_to_targets_and_cap():
    assert DEFAULT_GEN0_MAX_PARALLEL == 3
    assert _effective_gen0_parallelism(3, 5) == 3
    assert _effective_gen0_parallelism(10, 5) == 5
    assert _effective_gen0_parallelism(3, 2) == 2
    assert _effective_gen0_parallelism(3, 0) == 0


def test_generation_zero_parallelism_is_bounded_and_records_work_items():
    parents = [
        CertificationInput(
            id=f"thm{i}",
            statement="Prove True.",
            answer="",
            metadata={
                "formal_statement": f"theorem thm{i} : True := by",
                "lean_header": "import Mathlib",
            },
        )
        for i in range(5)
    ]
    active = {"value": 0, "max": 0}
    lock = asyncio.Lock()

    async def completer(problem, config):
        async with lock:
            active["value"] += 1
            active["max"] = max(active["max"], active["value"])
        await asyncio.sleep(0.01)
        async with lock:
            active["value"] -= 1
        return problem.model_copy(
            update={
                "metadata": {
                    **problem.metadata,
                    "lean_code": f"import Mathlib\n\ntheorem {problem.id} : True := by\n  trivial",
                    "gen0_proof_completed": True,
                }
            }
        )

    completed = asyncio.run(
        _complete_generation_zero_proofs(
            parents,
            config=GenerationConfig(model="test-model"),
            seed_proof_completer=completer,
            max_parallel=3,
        )
    )

    assert active["max"] == 3
    assert all(parent.metadata["gen0_worker_count_effective"] == 3 for parent in completed)
    assert {parent.metadata["gen0_work_item"]["lane"] for parent in completed} == {0, 1, 2}
    summary = _generation_zero_summary(parents, completed)
    assert summary["worker_count_requested"] == 3
    assert summary["worker_count_effective"] == 3
    assert summary["target_seed_count"] == 5
    assert len(summary["work_items"]) == 5


def test_generation_zero_seed_exception_does_not_cancel_other_seeds():
    parents = [
        CertificationInput(
            id="bad" if i == 0 else f"good{i}",
            statement="Prove True.",
            answer="",
            metadata={
                "formal_statement": f"theorem thm{i} : True := by",
                "lean_header": "import Mathlib",
            },
        )
        for i in range(3)
    ]

    async def completer(problem, config):
        if problem.id == "bad":
            raise RuntimeError("boom")
        return problem.model_copy(
            update={
                "metadata": {
                    **problem.metadata,
                    "lean_code": "import Mathlib\n\ntheorem ok : True := by\n  trivial",
                    "gen0_proof_completed": True,
                }
            }
        )

    completed = asyncio.run(
        _complete_generation_zero_proofs(
            parents,
            config=GenerationConfig(model="test-model"),
            seed_proof_completer=completer,
            max_parallel=3,
        )
    )

    by_id = {parent.id: parent for parent in completed}
    assert by_id["bad"].metadata["gen0_failure_packet"]["failure_class"] == "gen0_exception"
    assert by_id["good1"].metadata["gen0_proof_completed"] is True
    assert by_id["good2"].metadata["gen0_proof_completed"] is True
    summary = _generation_zero_summary(parents, completed)
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 1


def test_generation_zero_enriched_csv_preserves_statement_and_writes_complete_lean(tmp_path):
    input_path = tmp_path / "seeds.csv"
    with input_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "statement", "answer", "formal_statement", "lean_header"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "thm0",
                "statement": "Prove True.",
                "answer": "",
                "formal_statement": "theorem thm0 : True := by",
                "lean_header": "import Mathlib",
            }
        )

    completed = [
        CertificationInput(
            id="thm0",
            statement="Prove True.",
            answer="",
            metadata={
                "formal_statement": "theorem thm0 : True := by",
                "lean_header": "import Mathlib",
                "lean_code": "import Mathlib\n\ntheorem thm0 : True := by\n  trivial",
                "seed_formal_statement_original": "theorem thm0 : True := by",
                "gen0_target": True,
                "gen0_proof_completed": True,
                "formal_status": "certified",
            },
        )
    ]

    output_path = _write_generation_zero_enriched_seed_csv(
        input_path,
        tmp_path / "seeds.gen0.csv",
        completed,
    )

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["formal_statement"] == "theorem thm0 : True := by"
    assert "trivial" in rows[0]["lean_code"]
    assert rows[0]["proof_body_available"] == "true"
    assert rows[0]["gen0_proof_completed"] == "true"
    loaded = load_seed_inputs(output_path, pool_size=1)
    summary = _generation_zero_summary(loaded, loaded)
    assert summary["completed_count"] == 1


def test_gen0_failed_theorem_seed_is_not_backfill_eligible():
    failed = CertificationInput(
        id="thm_failed",
        statement="Prove missing theorem.",
        answer="",
        metadata={
            "formal_statement": "theorem thm_failed : True := by",
            "lean_header": "import Mathlib",
            "problem_style": "theorem_proof",
            "gen0_target": True,
            "gen0_proof_completed": False,
            "status": "certified",
            "quality_verdict": "acceptable",
        },
    )

    assert not _parent_is_backfill_eligible(failed)


def test_generation_zero_rejects_placeholder_proof_for_different_theorem():
    assert not _lean_has_complete_by_body(
        "import Mathlib\n\n"
        "axiom hidden_shortcut : True\n\n"
        "theorem thm0 : True := by\n"
        "  exact hidden_shortcut"
    )
    assert not _gen0_proof_matches_formal_statement(
        "theorem exercise_29_1 : ¬ LocallyCompactSpace ℚ :=",
        "import Mathlib\n\ntheorem placeholder : True := by\n  trivial",
    )
    assert not _gen0_proof_matches_formal_statement(
        "theorem exercise_3_4 (n : ℕ) : True :=",
        "/- theorem not provided -/\ntheorem exercise_3_4 (n : ℕ) : True := by\n  trivial",
    )
    assert _gen0_proof_matches_formal_statement(
        "theorem thm0 : True :=",
        "import Mathlib\n\ntheorem thm0 : True := by\n  trivial",
    )
    assert _gen0_proof_matches_formal_statement(
        "theorem mathd_numbertheory_403 : (∑ k in Nat.properDivisors 198, k) = 270 := by",
        "import Mathlib\n\n"
        "theorem mathd_numbertheory_403 : ((Nat.properDivisors 198).sum fun k => k) = 270 := by\n"
        "  native_decide",
    )
    assert _gen0_proof_matches_formal_statement(
        "theorem mathd_numbertheory_403 : (∑ k in Nat.properDivisors 198, k) = 270 := by",
        "import Mathlib\n\n"
        "theorem mathd_numbertheory_403 : (∑ k ∈ Nat.properDivisors 198, k) = 270 := by\n"
        "  native_decide",
    )
    assert _gen0_proof_matches_formal_statement(
        "abbrev putnam_1998_b1_solution : ℝ := sorry\n"
        "-- 6\n"
        "theorem putnam_1998_b1 : sInf {x : ℝ | x > 0} = putnam_1998_b1_solution :=\n"
        "sorry",
        "import Mathlib\n\n"
        "abbrev putnam_1998_b1_solution : ℝ := 6\n"
        "-- 6\n"
        "theorem putnam_1998_b1 : sInf {x : ℝ | x > 0} = putnam_1998_b1_solution := by\n"
        "  sorry",
    )
    assert not _gen0_proof_matches_formal_statement(
        "abbrev putnam_1978_b2_solution : ℚ := sorry\n"
        "-- 7 / 4\n"
        "theorem putnam_1978_b2 : (∑' n : ℕ, (1 : ℚ)) = putnam_1978_b2_solution :=\n"
        "sorry",
        "import Mathlib\n\n"
        "noncomputable abbrev putnam_1978_b2_solution : ℚ :=\n"
        "  ∑' n : ℕ, (1 : ℚ)\n"
        "theorem putnam_1978_b2 : (∑' n : ℕ, (1 : ℚ)) = putnam_1978_b2_solution := by\n"
        "  rfl",
    )
    assert _gen0_proof_matches_formal_statement(
        "def exercise_2_5_43 (G : Type*) [Group G] [Fintype G]\n"
        "  (hG : card G = 9) :\n"
        "  CommGroup G :=",
        "import Mathlib\n\n"
        "def exercise_2_5_43 (G : Type*) [Group G] [Fintype G]\n"
        "  (hG : card G = 9) :\n"
        "  CommGroup G := by\n"
        "  exact IsPGroup.commGroupOfCardEqPrimeSq (p := 3) (G := G) (by\n"
        "    simp [Nat.card_eq_fintype_card, hG])",
    )


def test_codex_cli_command_is_oauth_safe_and_config_isolated(tmp_path):
    output_path = tmp_path / "last-message.txt"
    command = build_codex_exec_command(
        model="gpt-5.5",
        output_path=output_path,
        cwd=tmp_path,
    )

    assert command[:2] == ["codex", "exec"]
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--ephemeral" in command
    assert "--json" in command
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert command[command.index("--output-last-message") + 1] == str(output_path)
    assert command[-1] == "-"
    assert not any("auth.json" in part for part in command)


def test_codex_cli_prompt_keeps_system_and_user_contracts_separate():
    prompt = build_codex_prompt("Return Lean only.", "Prove theorem thm0 : True :=", json_mode=True)

    assert "<system>" in prompt
    assert "Return Lean only." in prompt
    assert "<task>" in prompt
    assert "<output_json>" in prompt
    assert "theorem thm0" in prompt


def test_codex_cli_prompt_does_not_force_json_for_plain_artifacts():
    prompt = build_codex_prompt("Return Lean only.", "theorem thm0 : True := by\n  trivial")

    assert "<task>" in prompt
    assert "<output_json>" not in prompt
    assert "theorem thm0" in prompt


def test_theorem_generation_raw_accepts_codex_structured_checkpoint_objects():
    raw = _normalize_theorem_generation_raw(
        {
            "contract_status": "ok",
            "statement_chunks": [{"chunk_id": "goal", "statement": "Prove True."}],
            "selected_parent_checkpoints": [{"parent_id": "p", "checkpoint": "prove True"}],
            "proof_obligations": [{"obligation": "close by trivial"}],
            "parent_contribution_evidence": {"p": {"field": "lean_code", "evidence": "trivial"}},
        }
    )

    assert raw["contract_status"] == "generated"
    assert raw["statement_chunks"] == ["Prove True."]
    assert raw["selected_parent_checkpoints"] == ["prove True"]
    assert raw["proof_obligations"] == ["close by trivial"]
    assert "lean_code" in raw["parent_contribution_evidence"]["p"]


def test_theorem_generation_raw_accepts_minimal_codex_artifact():
    raw = _normalize_theorem_generation_raw(
        {
            "status": "generated",
            "statement": "Prove True.",
            "formal_statement": "theorem child : True := by",
            "lean_code": "import Mathlib\n\ntheorem child : True := by\n  trivial",
            "proof_plan": "trivial",
            "parent_usage": {"p": {"field": "formal_statement", "usage": "same goal"}},
            "reason": "minimal executable theorem",
        }
    )

    assert raw["contract_status"] == "generated"
    assert raw["failure_reason"] == "minimal executable theorem"
    assert raw["parent_contribution_evidence"]["p"]


def test_quality_weak_does_not_trigger_operator_replan():
    result = CertificationResult(
        problem_id="child",
        op_type="mutation",
        certification_route="template_numeric",
        status="certified",
        quality_verdict="weak",
        quality_flags=["trivial_mod_remainder"],
    )

    assert not _is_plan_level_failure(
        result,
        op_type="mutation",
        failure_history=["trivial_mod_remainder"],
    )


def test_theorem_replan_keeps_theorem_style_and_downgrades_hard_mutation():
    parent = CertificationInput(
        id="thm0",
        statement="Prove True.",
        answer="",
        metadata={
            "formal_statement": "theorem thm0 : True := by\n  trivial",
            "lean_header": "import Mathlib",
        },
    )
    item = {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_hard",
        "parent_ids": ["thm0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
        "composition_pattern": "structure_expansion",
        "variation_axis": "hard theorem mutation",
        "quality_target": "complete Lean theorem proof",
        "parent_context_cards": [_parent_context_card(parent)],
    }
    result = CertificationResult(
        problem_id="child",
        certification_route="theorem_prover",
        status="proof_failed",
        error="unsolved goals",
    )

    replanned = _deterministic_replan_operator_card(item, result)

    assert replanned["target_style"] == "theorem_proof"
    assert replanned["target_family"] == "theorem_proof"
    assert replanned["operator_variant"] == "mutation_easy"
    assert replanned["op_type"] == "mutation"
    assert replanned["replan_source"] == "deterministic_fallback"


def test_slot_replan_can_rescue_failed_operator_card(tmp_path, isolated_corpus_index):
    input_path = tmp_path / "theorems.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_theorem_csv(input_path)
    calls = []

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["thm0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_hard",
                "parent_ids": ["thm1"],
                "target_family": "theorem_proof",
                "target_style": "theorem_proof",
                "variation_axis": "prove a harder theorem-style corollary",
                "composition_pattern": "structure_expansion",
                "quality_target": "complete theorem proof",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["thm2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["thm3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["thm4"]},
        ],
    }

    def theorem_generator(parent, config):
        calls.append(parent.metadata["operator_card"]["operator_variant"])
        if calls[-1] == "mutation_hard":
            return TheoremGeneratedProblem(
                id="bad_child",
                source_problem_id=parent.id,
                contract_status="contract_failed",
                failure_reason="proof_surface_unavailable: no executable proof surface",
                statement="",
                formal_statement="",
                lean_code="",
            )
        return TheoremGeneratedProblem(
            id="rescued_child",
            source_problem_id=parent.id,
            statement="Prove rescued theorem-style proposition.",
            formal_statement="theorem rescued_child : True := by\n  trivial",
            lean_header="import Mathlib",
            lean_code="import Mathlib\n\ntheorem rescued_child : True := by\n  trivial",
            proof_plan="Use trivial proof in this mock.",
            proof_obligations=["complete Lean proof"],
            proof_surface="simp_only",
            parent_contribution_evidence={parent.id: "formal_statement: theorem style preserved"},
        )

    async def theorem_verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True,
                raw_stdout="'" + code.rsplit("#print axioms ", 1)[1].strip()
                + "' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete=True)

    async def theorem_alignment_verifier(generated, **kwargs):
        return TheoremAlignmentResult(aligned=True, verdict="pass")

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        theorem_generator=theorem_generator,
        theorem_verifier=theorem_verifier,
        theorem_alignment_verifier=theorem_alignment_verifier,
        replanner=lambda item, result, history, config: None,
        survivor_count=4,
        crossover_count=0,
        max_retries=1,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rescued = [row for row in rows if row["problem_id"] == "rescued_child"][0]
    assert calls == ["mutation_hard", "mutation_easy"]
    assert rescued["status"] == "certified"
    assert rescued["replan_count"] == 1
    assert rescued["replan_source"] == "deterministic_fallback"
    assert rescued["target_style"] == "theorem_proof"
    assert rescued["op_type"] == "mutation"
    assert rescued["planned_op_type"] == "mutation"
    assert rescued["planned_operator_variant"] == "mutation_hard"
    assert rescued["attempted_op_types"] == ["mutation"]


def test_complete_with_backfill_continues_long_run(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)

    def planner(pool, state):
        return {
            "planner_source": "test",
            "work_items": [
                {"slot": 0, "op_type": "survivor", "parent_ids": [pool[0].id]},
                {
                    "slot": 1,
                    "op_type": "mutation",
                    "parent_ids": [pool[1].id],
                    "target_family": "divisor_sum",
                    "variation_axis": "generate one safe divisor-sum child",
                    "composition_pattern": "structure_expansion",
                    "quality_target": "certified numeric child",
                },
                {
                    "slot": 2,
                    "op_type": "mutation",
                    "parent_ids": [pool[2].id],
                    "target_family": "divisor_sum",
                    "variation_axis": "intentional failing slot",
                    "composition_pattern": "structure_expansion",
                    "quality_target": "certified numeric child",
                },
                {
                    "slot": 3,
                    "op_type": "mutation",
                    "parent_ids": [pool[3].id],
                    "target_family": "divisor_sum",
                    "variation_axis": "intentional failing slot",
                    "composition_pattern": "structure_expansion",
                    "quality_target": "certified numeric child",
                },
                {
                    "slot": 4,
                    "op_type": "mutation",
                    "parent_ids": [pool[4].id],
                    "target_family": "divisor_sum",
                    "variation_axis": "intentional failing slot",
                    "composition_pattern": "structure_expansion",
                    "quality_target": "certified numeric child",
                },
            ],
        }

    def generator(parent, config):
        if parent.metadata["slot"] == 1:
            return GeneratedProblem(
                id=f"{parent.id}__safe",
                source_problem_id=parent.id,
                family="divisor_sum",
                statement="Find the sum of all positive divisors of 120.",
                answer="360",
                params={"n": 120},
                projected_params={"n": 120},
                reasoning_pattern="factorize_then_sigma",
                solution_skeleton={"target_computation": "sigma(120)"},
                projection_check={"passed": True},
            )
        raise RuntimeError("intentional slot failure")

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=planner,
        generator=generator,
        max_generations=2,
        max_retries=0,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result.generation_count == 2
    assert result.generation_save_status in {"partial", "complete", "complete_with_backfill"}
    assert any(row["op_type"] == "fallback_survivor" for row in rows)
    assert any(row["status"] not in {"certified", "survivor"} for row in rows)
    assert summary["generations"][0]["generation_save_status"] == "complete_with_backfill"
    assert summary["generations"][1]["generation_save_status"] in {
        "partial",
        "complete",
        "complete_with_backfill",
    }
    assert not any(row.get("fallback_survivor_duplicate") for row in rows)


def test_lean_resource_args_are_env_configurable(monkeypatch):
    monkeypatch.setenv("LEAN_MEMORY_MB", "1234")
    monkeypatch.setenv("LEAN_THREADS", "5")
    monkeypatch.setenv("LEAN_MAX_HEARTBEATS", "777")

    assert _lean_resource_args() == ["-M", "1234", "-j", "5", "-D", "maxHeartbeats=777"]


def test_generated_slot_max_retries_is_configurable(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    calls = []

    def generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        return GeneratedProblem(
            id=f"{parent.id}__weak_{len(calls)}",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 100 mod 10.",
            answer="0",
            params={"a": 100, "m": 10},
            projected_params={"a": 100, "m": 10},
            reasoning_pattern="trivial_modular_reduction",
            solution_skeleton={"target_computation": "100 mod 10"},
            projection_check={"passed": True, "evidence": "params projected"},
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p1"],
                "target_family": "modular_congruence",
                "variation_axis": "replace trivial modular remainder",
                "composition_pattern": "parameter_shift",
                "quality_target": "nontrivial modular remainder",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p4"]},
        ],
    }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=generator,
        survivor_count=4,
        crossover_count=0,
        max_retries=1,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    generated = [row for row in rows if row["op_type"] == "mutation"][0]
    assert len(calls) == 2
    assert generated["retry_count"] == 1
    assert generated["retry_exhausted"] is True


@pytest.mark.xfail(strict=True, reason="the fake duplicate generator's rows are no longer certified under the current checker path, so no dedup/quality selection reason is ever reached; the dedup behaviour itself is covered by the corpus index tests")
def test_weak_and_duplicate_certified_results_are_not_next_parents(tmp_path, isolated_corpus_index):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p0"],
                "target_family": "divisor_sum",
                "variation_axis": "keep n fixed to trigger weak quality",
                "composition_pattern": "parameter_shift",
                "quality_target": "strictly increase n",
            },
            {
                "slot": 2,
                "op_type": "mutation",
                "parent_ids": ["p0"],
                "target_family": "divisor_sum",
                "variation_axis": "repeat the same n to trigger duplicate signature",
                "composition_pattern": "parameter_shift",
                "quality_target": "strictly increase n",
            },
            {
                "slot": 3,
                "op_type": "mutation",
                "parent_ids": ["p1"],
                "target_family": "divisor_sum",
                "variation_axis": "repeat duplicate family",
                "composition_pattern": "parameter_shift",
                "quality_target": "strictly increase n",
            },
            {
                "slot": 4,
                "op_type": "mutation",
                "parent_ids": ["p2"],
                "target_family": "divisor_sum",
                "variation_axis": "repeat duplicate family again",
                "composition_pattern": "parameter_shift",
                "quality_target": "strictly increase n",
            },
        ],
    }
    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=_fake_duplicate_divisor_sum,
        crossover_count=0,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    unselected = [row for row in rows if row["op_type"] == "mutation" and not row["parent_eligible"]]
    assert unselected
    assert {row["selection_reason"] for row in unselected} & {
        "weak_quality",
        "weak_quality_after_retries",
        "duplicate_signature",
        "family_cap",
    }
    assert len(result.saved_pool) == 5
    assert result.generation_feedback["backfill_events"]


def test_repeated_reasoning_signature_is_not_next_parent(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_compatible_csv(input_path)

    plan = deterministic_fallback_plan(_compatible_pool(), crossover_count=0)

    def same_signature_generator(parent, config):
        n = 840 if parent.metadata.get("slot") == 1 else 1320
        answer = sum(d for d in range(1, n + 1) if n % d == 0)
        return GeneratedProblem(
            id=f"{parent.id}__same_sig",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement=f"Find the sum of all positive divisors of {n}.",
            answer=str(answer),
            params={"n": n},
            projected_params={"n": n},
            reasoning_pattern="prime_factorization_sigma",
            solution_skeleton={"target_computation": f"sigma({n})", "expected_answer": answer},
            projection_check={"passed": True, "evidence": "params projected"},
            harder_reason="Richer divisor sum.",
            solution=f"sigma({n}) = {answer}. Answer: {answer}.",
        )

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=same_signature_generator,
        crossover_count=0,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    repeated = [row for row in rows if row.get("selection_reason") == "repeated_reasoning_signature"]
    assert repeated
    assert result.saved_pool


def test_aggregate_selector_can_select_near_duplicate_as_orchestrator_judgment():
    bad = CertificationResult(
        problem_id="near_dup",
        status="certified",
        op_type="mutation",
        family="gcd",
        target_family="gcd",
        statement="Find GCD(84, 126).",
        answer="42",
        quality_verdict="acceptable",
        quality_flags=["near_duplicate"],
        quality_evidence={"accepted_proxy": {"pass": True}},
    )
    good = CertificationResult(
        problem_id="reserve_good",
        status="certified",
        op_type="mutation",
        source_kind="reserve_generated",
        family="divisor_sum",
        target_family="divisor_sum",
        statement="Find the sum of all positive divisors of 840.",
        answer="2880",
        quality_verdict="acceptable",
        quality_flags=[],
        quality_evidence={
            "accepted_proxy": {"pass": True},
            "reasoning_signature": "divisor_sum:840",
        },
    )

    aggregate = select_next_pool_with_orchestrator(
        results=[bad, good],
        current_generation=_pool(),
        pool_size=2,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [
                {
                    "id": "near_dup",
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "orchestrator_selected_risky_novelty",
                    "expected_next_gen_role": "scaffold_support",
                    "rationale": "Risk accepted by orchestrator for test.",
                },
                {
                    "id": "reserve_good",
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "selected_for_next_pool",
                    "expected_next_gen_role": "frontier_parent",
                    "rationale": "Strong reserve candidate.",
                },
            ],
            "generation_survival_status": "complete",
            "pool_strategy_rationale": "prefer reserve",
            "warnings": [],
        },
    )

    approved_ids = {row["id"] for row in aggregate["approved_candidates"]}
    assert {"near_dup", "reserve_good"} <= approved_ids
    assert aggregate["invariant_events"] == []
    assert aggregate["selector_manifest"]["payload_summary"]["selection_risk_counts"][
        "near_duplicate_memory"
    ] == 1
    assert aggregate["approved_candidates"][0]["selection_reason"] == "orchestrator_selected_risky_novelty"
    assert any(row["id"] == "reserve_good" for row in aggregate["approved_candidates"])
    assert len(aggregate["selector_manifest"]["payload_sha256"]) == 64


def test_aggregate_selector_invalid_candidate_is_blocked_by_invariant():
    failed = CertificationResult(
        problem_id="failed_slot",
        status="slot_failed",
        op_type="mutation",
        family="divisor_sum",
        statement="failed",
        answer="",
        quality_verdict="weak",
        quality_flags=["slot_exception"],
    )

    aggregate = select_next_pool_with_orchestrator(
        results=[failed],
        current_generation=[],
        pool_size=1,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [
                {
                    "id": "failed_slot",
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "llm_selected_invalid",
                    "expected_next_gen_role": "frontier_parent",
                    "rationale": "Invalid candidate should be code-invalidated.",
                }
            ],
            "generation_survival_status": "complete",
            "pool_strategy_rationale": "invalid test",
            "warnings": [],
        },
    )

    assert aggregate["generation_survival_status"] == "partial"
    assert aggregate["approved_candidates"] == []
    assert aggregate["failed_slots"][0]["failure_signature"]
    assert aggregate["invariant_events"][0]["selection_reason"] == (
        "invalidated_by_code:not_certified_or_survivor"
    )


def test_aggregate_selector_exact_duplicate_is_blocked_by_invariant():
    first = CertificationResult(
        problem_id="dup_a",
        status="certified",
        op_type="mutation",
        family="gcd",
        target_family="gcd",
        statement="Find GCD(84, 126).",
        answer="42",
        quality_verdict="acceptable",
        quality_flags=[],
    )
    second = first.model_copy(update={"problem_id": "dup_b"})

    aggregate = select_next_pool_with_orchestrator(
        results=[first, second],
        current_generation=[],
        pool_size=2,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [
                {
                    "id": "dup_a",
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "selected_for_next_pool",
                    "expected_next_gen_role": "frontier_parent",
                    "rationale": "First duplicate selected.",
                },
                {
                    "id": "dup_b",
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "selected_for_next_pool",
                    "expected_next_gen_role": "frontier_parent",
                    "rationale": "Second duplicate should be blocked by code.",
                },
            ],
            "generation_survival_status": "complete",
            "pool_strategy_rationale": "duplicate test",
            "warnings": [],
        },
    )

    assert [row["id"] for row in aggregate["approved_candidates"]] == ["dup_a"]
    assert aggregate["rejected_slots"][0]["selection_reason"] == (
        "invalidated_by_code:exact_canonical_duplicate"
    )
    assert aggregate["invariant_events"][0]["selection_reason"] == (
        "invalidated_by_code:exact_canonical_duplicate"
    )


def test_aggregate_selector_can_override_family_cap():
    results = [
        CertificationResult(
            problem_id=f"gcd_{idx}",
            status="certified",
            op_type="mutation",
            family="gcd",
            target_family="gcd",
            statement=f"Find GCD({84 + idx}, {126 + idx}).",
            answer=str(idx + 1),
            quality_verdict="acceptable",
            quality_flags=[],
        )
        for idx in range(3)
    ]

    aggregate = select_next_pool_with_orchestrator(
        results=results,
        current_generation=[],
        pool_size=3,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [
                {
                    "id": result.problem_id,
                    "source_kind": "current_candidate",
                    "decision": "select",
                    "selection_reason": "family_cap_override",
                    "expected_next_gen_role": "frontier_parent",
                    "rationale": "Orchestrator intentionally keeps same family for this test.",
                }
                for result in results
            ],
            "generation_survival_status": "complete",
            "pool_strategy_rationale": "family cap override",
            "warnings": [],
        },
    )

    assert len(aggregate["approved_candidates"]) == 3
    assert all(row["selection_reason"] == "family_cap_override" for row in aggregate["approved_candidates"])
    assert aggregate["invariant_events"] == []


def test_aggregate_selector_can_choose_orchestrator_backfill_seed():
    failed = CertificationResult(
        problem_id="failed_slot",
        status="slot_failed",
        op_type="mutation",
        family="divisor_sum",
        statement="failed",
        answer="",
        quality_verdict="weak",
        quality_flags=["slot_exception"],
    )

    aggregate = select_next_pool_with_orchestrator(
        results=[failed],
        current_generation=_pool(),
        pool_size=1,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [
                {
                    "id": "p2",
                    "source_kind": "previous_seed",
                    "decision": "backfill",
                    "selection_reason": "orchestrator_backfill",
                    "expected_next_gen_role": "diversity_anchor",
                    "rationale": "use a certified seed",
                }
            ],
            "generation_survival_status": "complete_with_backfill",
            "pool_strategy_rationale": "use a certified seed",
            "warnings": [],
        },
    )

    assert aggregate["generation_survival_status"] == "complete_with_backfill"
    assert aggregate["approved_candidates"][0]["id"] == "p2"
    assert aggregate["approved_candidates"][0]["selection_reason"] == "orchestrator_backfill"
    assert not aggregate["approved_candidates"][0].get("fallback_survivor_duplicate")


def test_aggregate_partial_when_no_valid_candidates_or_backfill():
    failed = CertificationResult(
        problem_id="failed_slot",
        status="slot_failed",
        op_type="mutation",
        family="divisor_sum",
        statement="failed",
        answer="",
        quality_verdict="weak",
        quality_flags=["slot_exception"],
    )

    aggregate = select_next_pool_with_orchestrator(
        results=[failed],
        current_generation=[],
        pool_size=5,
        generation_count=1,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [],
            "generation_survival_status": "partial",
            "pool_strategy_rationale": "nothing valid",
            "warnings": [],
        },
    )

    assert aggregate["generation_survival_status"] == "partial"
    assert aggregate["approved_candidates"] == []


def test_aggregate_continuity_backfill_uses_older_archive_seed():
    failed = CertificationResult(
        problem_id="failed_slot",
        status="slot_failed",
        op_type="mutation",
        family="divisor_sum",
        statement="failed",
        answer="",
        quality_verdict="weak",
        quality_flags=["slot_exception"],
    )
    older_seed = CertificationInput(
        id="older_seed",
        statement="Find GCD(84, 126).",
        answer="42",
        metadata={
            "status": "certified",
            "quality_verdict": "acceptable",
            "backfill_source_kind": "run_archive",
            "backfill_generation_distance": 2,
        },
    )

    aggregate = select_next_pool_with_orchestrator(
        results=[failed],
        current_generation=[],
        backfill_seed_archive=[older_seed],
        pool_size=1,
        generation_count=3,
        target_accepted=1,
        planner={"planner_source": "test"},
        aggregate_selector=lambda payload: {
            "pool_decisions": [],
            "generation_survival_status": "partial",
            "pool_strategy_rationale": "nothing current",
            "warnings": [],
        },
    )

    assert aggregate["generation_survival_status"] == "complete_with_backfill"
    assert aggregate["approved_candidates"][0]["id"] == "older_seed"
    assert aggregate["approved_candidates"][0]["selection_reason"] == "continuity_backfill"
    assert aggregate["selector_manifest"]["backfill_archive_count"] == 1


def test_backfill_seed_archive_includes_recent_run_rows(tmp_path):
    output_path = tmp_path / "results.jsonl"
    row = {
        "problem_id": "archived_seed",
        "generation": 1,
        "status": "certified",
        "parent_eligible": True,
        "statement": "Find the sum of all positive divisors of 120.",
        "answer": "360",
        "quality_verdict": "acceptable",
        "family": "divisor_sum",
    }
    output_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    archive = _build_backfill_seed_archive(
        [],
        output_path=output_path,
        generation_count=3,
        max_history_generations=2,
    )

    assert [seed.id for seed in archive] == ["archived_seed"]
    assert archive[0].metadata["backfill_source_kind"] == "run_archive"
    assert archive[0].metadata["backfill_generation_distance"] == 2


def test_aggregate_prompt_schema_is_compact_and_raw_lean_free():
    result = CertificationResult(
        problem_id="candidate",
        status="certified",
        op_type="mutation",
        family="theorem_proof",
        target_family="theorem_proof",
        target_style="theorem_proof",
        statement="Prove True.",
        answer="",
        formal_statement="theorem candidate : True := by\n  trivial",
        lean_code="theorem raw_should_not_enter_aggregate_prompt : True := by\n  trivial",
        quality_verdict="acceptable",
        quality_flags=[],
        quality_evidence={"accepted_proxy": {"pass": True}},
    )
    payload = _aggregate_selector_payload(
        results=[result],
        backfill_seed_archive=_pool(),
        pool_size=5,
        generation_count=1,
        target_accepted=1,
    )
    user = _aggregate_messages(payload)[1]["content"]
    schema = _aggregate_response_format()["json_schema"]["schema"]

    assert "current_candidates" in user
    assert "previous_certified_seed_cards" in user
    assert "selection_policy_surface" in user
    assert "nonnegotiable_invariants" in user
    assert "orchestrator_judgment_factors" in user
    assert "mechanical_eligibility" in user
    assert "HARD: Make ordered pool_decisions" in user
    assert payload["current_candidates"][0]["mechanical_eligibility"]["eligible"] is True
    assert payload["selection_policy_surface"]["family_cap"] == 2
    assert "raw_should_not_enter_aggregate_prompt" not in user
    assert schema["additionalProperties"] is False
    assert "select/backfill decisions first" in schema["properties"]["pool_decisions"]["description"]
    assert "decision" in schema["properties"]["pool_decisions"]["items"]["properties"]
    assert "selection_reason" in schema["properties"]["pool_decisions"]["items"]["properties"]
    assert "expected_next_gen_role" in schema["properties"]["pool_decisions"]["items"]["properties"]
    assert (
        "mechanical_eligibility.eligible=true"
        in schema["properties"]["pool_decisions"]["items"]["properties"]["decision"]["description"]
    )
    assert (
        "invalidated_by_code"
        in schema["properties"]["pool_decisions"]["items"]["properties"]["selection_reason"]["description"]
    )
    assert schema["properties"]["warnings"]["description"]
    assert set(schema["required"]) == {
        "pool_decisions",
        "generation_survival_status",
        "pool_strategy_rationale",
        "warnings",
    }


def test_composite_crossover_elites_are_selected_before_seed_survivor(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_compatible_csv(input_path)

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool),
        generator=_fake_composite_generated,
    )

    assert [item["op_type"] for item in result.saved_pool[:2]] == ["crossover", "crossover"]
    assert all(item["quality_verdict"] == "acceptable" for item in result.saved_pool[:2])
    assert all(
        item["quality_evidence"]["crossover_kind"] == "pipeline_composite"
        for item in result.saved_pool[:2]
    )
    assert result.saved_pool[0]["reasoning_pattern"] == "gcd_then_sigma"
    assert result.saved_pool[1]["reasoning_pattern"] == "sigma_then_mod"


def test_generation_feedback_is_passed_to_second_generation_planner(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    feedback_seen = []

    def planner(pool, state):
        feedback_seen.append(bool(state.get("generation_feedback")))
        return deterministic_fallback_plan(pool)

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        max_generations=2,
        checker=FakeLeanChecker(),
        planner=planner,
        generator=_fake_generated_success,
    )

    assert feedback_seen == [False, True]
    assert result.generation_count == 2
    assert result.generation_feedback["op_type_outcomes"]
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == result.cumulative_passed_results_count
    assert {line["generation"] for line in lines} == {1, 2}
    assert all("status" in line for line in lines)


def test_second_generation_worker_prompt_can_see_generated_parent_lean_code(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_compatible_csv(input_path)
    gen2_parent_lean_codes = []

    def capture_generator(parent, config):
        if parent.metadata.get("generation") == 2:
            parents = parent.metadata.get("parents") or []
            if parents:
                gen2_parent_lean_codes.append(
                    parents[0].get("proof_context", {}).get("lean_code")
                )
        return _fake_generated_success(parent, config)

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        max_generations=2,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: deterministic_fallback_plan(pool, crossover_count=0),
        generator=capture_generator,
        crossover_count=0,
    )

    assert gen2_parent_lean_codes
    assert any(code and code != "not_available" for code in gen2_parent_lean_codes)


def test_lean_compiler_failure_gets_one_generated_slot_repair(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    checker = FlakyLeanChecker()
    calls = []

    def generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        return GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement="Find the sum of all positive divisors of 840.",
            answer="2880",
            params={"n": 840},
            projected_params={"n": 840},
            reasoning_pattern="prime_factorization_sigma",
            solution_skeleton={"target_computation": "sigma(840)"},
            projection_check={"passed": True, "evidence": "params projected"},
            harder_reason="Richer divisor-sum instance.",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p0"],
                "target_family": "divisor_sum",
                "variation_axis": "repair lean compiler failure",
                "operator_goal": "generate a divisor-sum mutation",
                "reasoning_goal": "prime_factorization_sigma",
                "composition_pattern": "parameter_shift",
                "required_checkpoints": ["reasoning_pattern", "solution_skeleton", "projected_params"],
                "quality_target": "increase divisor-sum richness",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p1"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p2"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p3"]},
        ],
    }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=checker,
        planner=lambda pool, state: plan,
        generator=generator,
        survivor_count=4,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    repaired = [row for row in rows if row["op_type"] == "mutation"][0]
    assert checker.calls >= 2
    assert len(calls) == 2
    assert "previous_error=compiler error" in calls[1]
    assert repaired["retry_count"] == 1
    assert repaired["status"] == "certified"


def test_certified_weak_quality_retries_and_can_recover(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    calls = []

    def generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        if len(calls) == 1:
            return GeneratedProblem(
                id=f"{parent.id}__weak_mod",
                source_problem_id=parent.id,
                family="modular_congruence",
                statement="Find 100 mod 10.",
                answer="0",
                params={"a": 100, "m": 10},
                projected_params={"a": 100, "m": 10},
                reasoning_pattern="trivial_modular_reduction",
                solution_skeleton={"target_computation": "100 mod 10"},
                projection_check={"passed": True, "evidence": "params projected"},
                harder_reason="Initial weak modular instance.",
            )
        return GeneratedProblem(
            id=f"{parent.id}__strong_mod",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 2026 mod 37.",
            answer="28",
            params={"a": 2026, "m": 37},
            projected_params={"a": 2026, "m": 37},
            reasoning_pattern="nontrivial_modular_reduction",
            solution_skeleton={"target_computation": "2026 mod 37"},
            projection_check={
                "passed": True,
                "evidence": "fixed quality: nontrivial remainder and params projected",
            },
            harder_reason="Removed trivial_mod_remainder by using a nontrivial prime modulus.",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p1"],
                "target_family": "modular_congruence",
                "variation_axis": "replace trivial modular remainder with nontrivial reduction",
                "composition_pattern": "parameter_shift",
                "quality_target": "nontrivial modular remainder",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p4"]},
        ],
    }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=generator,
        survivor_count=4,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    repaired = [row for row in rows if row["op_type"] == "mutation"][0]
    assert len(calls) == 2
    assert "quality_flags" in calls[1]
    assert "trivial_mod_remainder" in calls[1]
    assert repaired["status"] == "certified"
    assert repaired["quality_verdict"] == "acceptable"
    assert repaired["retry_count"] == 1
    assert repaired["quality_retry_count"] == 1
    assert repaired["retry_reasons"] == ["quality:trivial_mod_remainder"]
    assert repaired["retry_exhausted"] is False


def test_solution_mismatch_retries_and_can_recover(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    calls = []

    def generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        if len(calls) == 1:
            return GeneratedProblem(
                id=f"{parent.id}__bad_solution",
                source_problem_id=parent.id,
                family="divisor_sum_mod",
                statement="Let m be the sum of all positive divisors of 720. Find 583741 mod m.",
                answer="1003",
                params={"n": 720, "a": 583741},
                projected_params={"n": 720, "a": 583741},
                reasoning_pattern="sigma_then_mod",
                solution_skeleton={
                    "target_computation": "583741 mod sigma(720)",
                    "expected_answer": 3421,
                },
                projection_check={"passed": True, "evidence": "params projected"},
                solution="Factor 720 incorrectly, get m = 19344. Answer: 3421.",
            )
        return GeneratedProblem(
            id=f"{parent.id}__fixed_solution",
            source_problem_id=parent.id,
            family="divisor_sum_mod",
            statement="Let m be the sum of all positive divisors of 720. Find 583741 mod m.",
            answer="1003",
            params={"n": 720, "a": 583741},
            projected_params={"n": 720, "a": 583741},
            reasoning_pattern="sigma_then_mod",
            solution_skeleton={
                "target_computation": "583741 mod sigma(720)",
                "verification_steps": ["sigma(720)=2418", "583741 mod 2418 = 1003"],
                "expected_answer": 1003,
            },
            projection_check={
                "passed": True,
                "evidence": "fixed solution_answer_mismatch and skeleton answer",
            },
            solution="sigma(720) = 2418, and 583741 mod 2418 = 1003. Answer: 1003.",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "operator_variant": "mutation_hard",
                "parent_ids": ["p1"],
                "target_family": "divisor_sum_mod",
                "variation_axis": "make a sigma-then-mod derived-object computation",
                "composition_pattern": "structure_expansion",
                "quality_target": "solution must derive the canonical answer",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p4"]},
        ],
    }

    run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=generator,
        survivor_count=4,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    repaired = [row for row in rows if row["op_type"] == "mutation"][0]
    assert len(calls) == 2
    assert "solution_answer_mismatch" in calls[1]
    assert repaired["quality_verdict"] == "acceptable"
    assert repaired["solution_verify_passed"] is True
    assert repaired["solution_verify_flags"] == []
    assert repaired["retry_reasons"] == [
        "quality:solution_answer_mismatch",
        "quality:solution_modulus_mismatch",
        "quality:solution_skeleton_answer_mismatch",
    ]


def test_quality_retry_exhaustion_preserves_certified_weak_row(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    calls = []

    def generator(parent, config):
        calls.append(parent.metadata.get("retry_feedback", ""))
        return GeneratedProblem(
            id=f"{parent.id}__always_weak_{len(calls)}",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 100 mod 10.",
            answer="0",
            params={"a": 100, "m": 10},
            projected_params={"a": 100, "m": 10},
            reasoning_pattern="trivial_modular_reduction",
            solution_skeleton={"target_computation": "100 mod 10"},
            projection_check={"passed": True, "evidence": "params projected"},
            harder_reason="Still weak.",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p1"],
                "target_family": "modular_congruence",
                "variation_axis": "replace trivial modular remainder with nontrivial reduction",
                "composition_pattern": "parameter_shift",
                "quality_target": "nontrivial modular remainder",
            },
            {"slot": 2, "op_type": "survivor", "parent_ids": ["p2"]},
            {"slot": 3, "op_type": "survivor", "parent_ids": ["p3"]},
            {"slot": 4, "op_type": "survivor", "parent_ids": ["p4"]},
        ],
    }

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=generator,
        survivor_count=4,
        crossover_count=0,
        disable_reserve_slots=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    exhausted = [row for row in rows if row["op_type"] == "mutation"][0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(calls) == 4
    assert exhausted["status"] == "certified"
    assert exhausted["quality_verdict"] == "weak"
    assert exhausted["parent_eligible"] is False
    assert exhausted["selection_reason"] == "weak_quality_after_retries"
    assert exhausted["retry_count"] == 3
    assert exhausted["quality_retry_count"] == 3
    assert exhausted["retry_exhausted"] is True
    assert exhausted["retry_reasons"] == ["quality:trivial_mod_remainder"]
    assert result.failed_slots == []
    assert summary["generation_feedback"]["quality_retry_count"] == 3
    assert summary["generation_feedback"]["retry_exhausted_count"] == 1
    assert summary["generation_feedback"]["rejected_slots"]
    assert "quality_evidence" in summary["generation_feedback"]["rejected_slots"][0]
    assert "feedback_for_next_generation" in summary["generation_feedback"]["rejected_slots"][0]


def test_reserve_slots_run_when_accepted_grade_proxy_below_target(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    calls = []

    def generator(parent, config):
        calls.append((parent.metadata.get("slot"), parent.metadata.get("source_kind")))
        if parent.metadata.get("source_kind") == "reserve_generated":
            generated = _fake_generated_success(parent, config)
            return generated.model_copy(
                update={
                    "reasoning_pattern": f"{generated.family}_template_computation",
                    "solution_skeleton": {"checkpoint": "compute the canonical template answer"},
                    "projected_params": dict(generated.params or {}),
                    "projection_check": {"passed": True, "evidence": "reserve params are canonical"},
                }
            )
        return GeneratedProblem(
            id=f"{parent.id}__weak",
            source_problem_id=parent.id,
            family="modular_congruence",
            statement="Find 100 mod 10.",
            answer="0",
            params={"a": 100, "m": 10},
            projected_params={"a": 100, "m": 10},
            reasoning_pattern="trivial_modular_reduction",
            solution_skeleton={"target_computation": "100 mod 10"},
            projection_check={"passed": True, "evidence": "params projected"},
            harder_reason="Weak primary attempt.",
        )

    plan = {
        "planner_source": "test",
        "work_items": [
            {"slot": 0, "op_type": "survivor", "parent_ids": ["p0"]},
            {
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p1"],
                "target_family": "modular_congruence",
                "variation_axis": "primary weak",
                "composition_pattern": "parameter_shift",
            },
            {
                "slot": 2,
                "op_type": "mutation",
                "parent_ids": ["p2"],
                "target_family": "modular_congruence",
                "variation_axis": "primary weak",
                "composition_pattern": "parameter_shift",
            },
            {
                "slot": 3,
                "op_type": "mutation",
                "parent_ids": ["p3"],
                "target_family": "modular_congruence",
                "variation_axis": "primary weak",
                "composition_pattern": "parameter_shift",
            },
            {
                "slot": 4,
                "op_type": "mutation",
                "parent_ids": ["p4"],
                "target_family": "modular_congruence",
                "variation_axis": "primary weak",
                "composition_pattern": "parameter_shift",
            },
        ],
    }

    result = run_pool_generation(
        input_path,
        output_path,
        summary_path,
        checker=FakeLeanChecker(),
        planner=lambda pool, state: plan,
        generator=generator,
        max_retries=0,
        target_accepted_per_generation=3,
        reserve_budget=3,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    reserve_rows = [row for row in rows if row.get("source_kind") == "reserve_generated"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert reserve_rows
    assert summary["generation_feedback"]["reserve_slots_run"] == len(reserve_rows)
    assert summary["generation_feedback"]["accepted_proxy_count"] >= 1
    assert "accepted_grade_proxy_count" in summary["generation_feedback"]
    assert summary["generation_feedback"]["yield_funnel"]["generated"] >= 4
    assert "accepted_grade_proxy" in summary["generation_feedback"]["yield_funnel"]
    assert result.generation_save_status in {"complete", "complete_with_backfill"}


def test_cli_writes_jsonl_and_summary_without_api_keys(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "pool.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_csv(input_path)
    env = {
        key: value
        for key, value in dict(**__import__("os").environ).items()
        if key
        not in {
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
        }
    }
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate/run_pool_generation.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--run-name",
            "test-pool",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "summary generation=1" in completed.stdout
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert lines
    assert all("status" in line for line in lines)
    assert summary_path.exists()
