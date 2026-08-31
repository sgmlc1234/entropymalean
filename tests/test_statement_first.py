"""Tests for statement-first theorem certification.

Gate order under POOL_STATEMENT_FIRST (default on):
  1. formal_statement alone must type-check with a sorry body (autoImplicit off)
  2. NL<->Lean alignment is judged BEFORE any proof effort
  3. the full proof is verified; failures get repair turns against the frozen
     statement instead of discarding the slot
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.certification.certifier import CertificationResult
from src.certification.generation import GenerationConfig
from src.certification.certifier import CertificationInput
from src.evaluation.lean_verifier import LeanVerifyResult
from src.orchestration.pool_generation import (
    THEOREM_CANONICAL_HEADER,
    TheoremAlignmentResult,
    TheoremGeneratedProblem,
    _certify_theorem_child,
    _failure_class,
    _proof_repair_turns,
    _retryable_theorem_failure,
    _statement_first_enabled,
    _statement_sorry_code,
)

STATEMENT = "theorem child_thm (n : Nat) (h : 0 < n) : n + 0 = n"
GOOD_PROOF = (
    f"{THEOREM_CANONICAL_HEADER}\n\n"
    f"{STATEMENT} := by\n  exact Nat.add_zero n\n"
)


def _generated(**overrides) -> TheoremGeneratedProblem:
    payload = {
        "id": "parent0__theorem_gen1",
        "source_problem_id": "parent0",
        "statement": "Show that n + 0 = n for every positive natural number n.",
        "formal_statement": STATEMENT,
        "lean_code": GOOD_PROOF,
        "statement_chunks": ["n + 0 = n for positive n"],
    }
    payload.update(overrides)
    return TheoremGeneratedProblem.model_validate(payload)


def _item() -> dict:
    return {
        "slot": 1,
        "op_type": "mutation",
        "operator_variant": "mutation_easy",
        "parent_ids": ["parent0"],
        "target_style": "theorem_proof",
        "target_family": "theorem_proof",
        "leansearch_enabled": False,
    }


def _parent() -> CertificationInput:
    return CertificationInput(
        id="parent0",
        statement="Prove n + 0 = n.",
        answer="",
        metadata={"formal_statement": STATEMENT, "lean_header": "import Mathlib"},
    )


def _aligned(aligned: bool = True) -> TheoremAlignmentResult:
    return TheoremAlignmentResult.model_validate(
        {"aligned": aligned, "rationale": "test", "unsupported_claims": [], "missing_claims": []}
    )


def _certify(*, verifier, alignment_verifier, repairer=None, generated=None):
    async def generator(parent_input, config):
        return generated or _generated()

    return asyncio.run(
        _certify_theorem_child(
            parent_input=_parent(),
            item=_item(),
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=generator,
            theorem_verifier=verifier,
            theorem_alignment_verifier=alignment_verifier,
            theorem_proof_repairer=repairer,
        )
    )


# ---------------------------------------------------------------------------
# _statement_sorry_code
# ---------------------------------------------------------------------------


def test_statement_sorry_code_strips_body_and_forces_auto_implicit_false():
    code = _statement_sorry_code(
        "import Mathlib",
        f"{STATEMENT} := by\n  exact Nat.add_zero n",
    )
    assert code.endswith(":= by\n  sorry")
    assert "exact Nat.add_zero" not in code
    assert "set_option autoImplicit false" in code
    assert STATEMENT in code


def test_statement_sorry_code_handles_bare_statement():
    code = _statement_sorry_code("import Mathlib", STATEMENT)
    assert code.count("sorry") == 1
    assert f"{STATEMENT} := by" in code


# ---------------------------------------------------------------------------
# Stage 1: statement type-check gate
# ---------------------------------------------------------------------------


def test_statement_typecheck_failure_short_circuits_before_alignment_and_proof():
    calls: List[str] = []

    async def verifier(code, timeout=300.0):
        calls.append("verify")
        assert "sorry" in code  # only the statement probe should ever run
        return LeanVerifyResult(
            ok=False, complete=False, verify_time=0.1,
        )

    async def alignment_verifier(generated, *, item, config):
        calls.append("align")
        return _aligned()

    result = _certify(verifier=verifier, alignment_verifier=alignment_verifier)

    assert result.status == "statement_failed"
    assert calls == ["verify"]
    assert "statement_typecheck_failed" in result.quality_flags
    assert _failure_class(result) == "statement_typecheck_failed"
    assert _retryable_theorem_failure(result, "mutation") is True
    # The failing statement is preserved for retry feedback.
    assert result.formal_statement == STATEMENT


# ---------------------------------------------------------------------------
# Stage 2: alignment before proof
# ---------------------------------------------------------------------------


def test_misaligned_statement_never_reaches_proof_verification():
    verify_codes: List[str] = []

    async def verifier(code, timeout=300.0):
        verify_codes.append(code)
        return LeanVerifyResult(ok=True, complete=False, verify_time=0.1)

    async def alignment_verifier(generated, *, item, config):
        return _aligned(False)

    result = _certify(verifier=verifier, alignment_verifier=alignment_verifier)

    assert result.status == "alignment_failed"
    # Exactly one verifier call: the sorry-statement probe. No proof budget spent.
    assert len(verify_codes) == 1
    assert "sorry" in verify_codes[0]


# ---------------------------------------------------------------------------
# Stage 3: proof verification with repair loop
# ---------------------------------------------------------------------------


def test_failing_proof_is_repaired_against_frozen_statement():
    repaired_code = GOOD_PROOF.replace("exact Nat.add_zero n", "simp")
    verify_log: List[str] = []

    async def verifier(code, timeout=300.0):
        verify_log.append(code)
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True, verify_time=0.1,
                raw_stdout="'child_thm' depends on axioms: [propext, Classical.choice]",
            )
        if "sorry" in code:
            return LeanVerifyResult(ok=True, complete=False, verify_time=0.1)
        if "simp" in code:
            return LeanVerifyResult(ok=True, complete=True, verify_time=0.1)
        return LeanVerifyResult(
            ok=False,
            complete=False,
            verify_time=0.1,
            system_error=None,
        )

    async def alignment_verifier(generated, *, item, config):
        return _aligned()

    async def repairer(generated, *, item, config, diagnostics, turn):
        return repaired_code

    broken = _generated(
        lean_code=GOOD_PROOF.replace("exact Nat.add_zero n", "linarith")
    )
    result = _certify(
        verifier=verifier,
        alignment_verifier=alignment_verifier,
        repairer=repairer,
        generated=broken,
    )

    assert result.status == "certified"
    assert result.lean_code == repaired_code
    repair = (result.quality_evidence or {}).get("proof_repair") or {}
    assert repair.get("repaired") is True
    assert repair.get("turns_used") == 1
    assert repair.get("attempts") == [{"turn": 1, "outcome": "proved"}]
    # statement probe + failing proof + repaired proof (+ axiom probe)
    assert len([c for c in verify_log if "#print axioms" not in c]) == 3


def test_repair_budget_exhaustion_returns_proof_failed(monkeypatch):
    monkeypatch.setenv("POOL_PROOF_REPAIR_TURNS", "2")
    repair_calls: List[int] = []

    async def verifier(code, timeout=300.0):
        if "sorry" in code:
            return LeanVerifyResult(ok=True, complete=False, verify_time=0.1)
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)

    async def alignment_verifier(generated, *, item, config):
        return _aligned()

    async def repairer(generated, *, item, config, diagnostics, turn):
        repair_calls.append(turn)
        return GOOD_PROOF  # still judged failing by the fake verifier

    result = _certify(
        verifier=verifier, alignment_verifier=alignment_verifier, repairer=repairer
    )

    assert result.status == "proof_failed"
    assert repair_calls == [1, 2]
    repair = (result.quality_evidence or {}).get("proof_repair") or {}
    assert repair.get("repaired") is False
    assert repair.get("turns_used") == 2


def test_repairer_returning_none_counts_turn_without_verification():
    verify_log: List[str] = []

    async def verifier(code, timeout=300.0):
        verify_log.append(code)
        if "sorry" in code:
            return LeanVerifyResult(ok=True, complete=False, verify_time=0.1)
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)

    async def alignment_verifier(generated, *, item, config):
        return _aligned()

    async def repairer(generated, *, item, config, diagnostics, turn):
        return None  # e.g. the model rewrote the statement — invariant guard

    result = _certify(
        verifier=verifier, alignment_verifier=alignment_verifier, repairer=repairer
    )

    assert result.status == "proof_failed"
    repair = (result.quality_evidence or {}).get("proof_repair") or {}
    assert all(a["outcome"] == "no_valid_candidate" for a in repair.get("attempts", []))
    # statement probe + initial proof attempt only — no repair verifications
    assert len(verify_log) == 2


# ---------------------------------------------------------------------------
# Legacy mode toggle
# ---------------------------------------------------------------------------


def test_legacy_mode_restores_proof_then_alignment_order(monkeypatch):
    monkeypatch.setenv("POOL_STATEMENT_FIRST", "0")
    assert _statement_first_enabled() is False
    calls: List[str] = []

    async def verifier(code, timeout=300.0):
        calls.append("verify")
        assert "sorry" not in code  # no statement probe in legacy mode
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)

    async def alignment_verifier(generated, *, item, config):
        calls.append("align")
        return _aligned()

    result = _certify(verifier=verifier, alignment_verifier=alignment_verifier)

    assert result.status == "proof_failed"
    assert calls == ["verify"]  # alignment never ran on a failing proof


def test_statement_first_default_and_repair_turns_env(monkeypatch):
    monkeypatch.delenv("POOL_STATEMENT_FIRST", raising=False)
    assert _statement_first_enabled() is True
    monkeypatch.setenv("POOL_PROOF_REPAIR_TURNS", "not-a-number")
    assert _proof_repair_turns() == 2
    monkeypatch.setenv("POOL_PROOF_REPAIR_TURNS", "0")
    assert _proof_repair_turns() == 0
