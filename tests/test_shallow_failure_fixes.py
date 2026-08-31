"""Tests for the shallow-Lean-failure fixes.

Covers the three dominant non-mathematical failure sources measured in
data/certified/ (3,209 slots): goal-structure mismatch feedback, deprecated
Mathlib syntax, and transient LLM transport errors, plus the advisory-only
proof-surface policy.
"""

from __future__ import annotations

import asyncio

import pytest

from src.certification.certifier import CertificationResult
from src.certification.generation import (
    _is_transient_llm_error,
    _with_transport_retry_async,
    _with_transport_retry_sync,
)
from src.orchestration.pool_generation import (
    _failure_class,
    _normalize_theorem_lean_code,
    _operator_card,
    _prelint_lean_syntax,
    _retryable_generation_failure,
    _theorem_patch_instructions,
)


# ---------------------------------------------------------------------------
# Deprecated big-operator binder syntax pre-lint
# ---------------------------------------------------------------------------


def test_prelint_rewrites_deprecated_sum_binder():
    code = (
        "theorem explicit_even_index_sum (a : ℚ) :\n"
        "    (∑ k in Finset.range 49, (a + ((2 * k.succ : ℕ) : ℚ))) = 49 * a + 2450 := by\n"
        "  simp\n"
    )
    fixed = _prelint_lean_syntax(code)
    assert "∑ k ∈ Finset.range 49," in fixed
    assert " in Finset.range" not in fixed


def test_prelint_covers_prod_and_union_binders():
    assert _prelint_lean_syntax("∏ p in s.primeFactors, p") == (
        "∏ p ∈ s.primeFactors, p"
    )
    assert _prelint_lean_syntax("⋃ i in t, f i") == "⋃ i ∈ t, f i"


def test_prelint_leaves_modern_syntax_and_let_alone():
    modern = "∑ k ∈ Finset.range 3, k"
    assert _prelint_lean_syntax(modern) == modern
    let_expr = "let x := 3\nx + 1"
    assert _prelint_lean_syntax(let_expr) == let_expr


def test_normalize_theorem_lean_code_applies_prelint():
    header, lean_code = _normalize_theorem_lean_code(
        lean_code=(
            "import Mathlib\n"
            "theorem t : (∑ k in Finset.range 3, k) = 3 := by decide\n"
        ),
        lean_header="import Mathlib",
        formal_statement="theorem t : (∑ k in Finset.range 3, k) = 3",
    )
    assert "∑ k ∈ Finset.range 3," in lean_code
    assert " in Finset.range" not in lean_code


# ---------------------------------------------------------------------------
# Goal-structure mismatch patch instructions
# ---------------------------------------------------------------------------


def _result_with_error(error: str) -> CertificationResult:
    return CertificationResult(
        problem_id="p1",
        status="proof_failed",
        error=error,
        lean_code="theorem t : True := by trivial",
    )


def test_no_goals_error_gets_branch_structure_instruction():
    instructions = _theorem_patch_instructions(
        _result_with_error("error: No goals to be solved")
    )
    joined = " ".join(instructions)
    assert "more" in joined and "goals" in joined
    assert "delete" in joined.lower() or "Delete" in joined
    # The generic fallback must not be the only guidance for the #1 diagnostic.
    assert not any("First try a smaller formal_statement" in i for i in instructions)


def test_unsolved_goals_instruction_keeps_statement_first():
    instructions = _theorem_patch_instructions(
        _result_with_error("error: unsolved goals\n⊢ n + 0 = n")
    )
    joined = " ".join(instructions)
    assert "Keep formal_statement unchanged" in joined


def test_no_goals_and_unsolved_goals_are_distinct_branches():
    no_goals = " ".join(
        _theorem_patch_instructions(_result_with_error("No goals to be solved"))
    )
    unsolved = " ".join(
        _theorem_patch_instructions(_result_with_error("unsolved goals"))
    )
    assert no_goals != unsolved


# ---------------------------------------------------------------------------
# Transport-error classification and retry
# ---------------------------------------------------------------------------


def test_failure_class_transport_error_detected():
    result = CertificationResult(
        problem_id="p1",
        status="generation_failed",
        error="RateLimitError: Error code: 429 - Too Many Requests",
    )
    assert _failure_class(result) == "llm_transport_error"
    assert _retryable_generation_failure(result) is True


def test_failure_class_plain_generation_failure_unchanged():
    result = CertificationResult(
        problem_id="p1",
        status="generation_failed",
        error="LLM output missing params object",
    )
    assert _failure_class(result) == "generation_failed"


def test_is_transient_llm_error_markers():
    assert _is_transient_llm_error(RuntimeError("429 Too Many Requests"))
    assert _is_transient_llm_error(RuntimeError("codex exec timed out after 240s"))
    assert _is_transient_llm_error(asyncio.TimeoutError())
    assert not _is_transient_llm_error(RuntimeError("invalid api key"))


def test_transport_retry_async_recovers_after_transient_failures(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_RETRIES", "3")
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_BACKOFF", "0")
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    assert asyncio.run(_with_transport_retry_async(flaky)) == "ok"
    assert calls["n"] == 3


def test_transport_retry_async_does_not_retry_permanent_errors(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_RETRIES", "3")
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_BACKOFF", "0")
    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise RuntimeError("invalid request: model not found")

    with pytest.raises(RuntimeError, match="model not found"):
        asyncio.run(_with_transport_retry_async(broken))
    assert calls["n"] == 1


def test_transport_retry_sync_exhausts_budget_then_raises(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("GENERATION_LLM_TRANSPORT_BACKOFF", "0")
    calls = {"n": 0}

    def always_rate_limited():
        calls["n"] += 1
        raise RuntimeError("rate limit exceeded")

    with pytest.raises(RuntimeError, match="rate limit"):
        _with_transport_retry_sync(always_rate_limited)
    assert calls["n"] == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# Advisory-only proof surfaces
# ---------------------------------------------------------------------------


def test_operator_card_marks_proof_surfaces_advisory():
    card = _operator_card(
        {
            "op_type": "mutation",
            "slot": 1,
            "target_style": "theorem_proof",
            "parent_context_cards": [],
        }
    )
    assert card["theorem_proof_surfaces_policy"] == "advisory_preference_only"
    assert card["theorem_proof_surfaces"]
