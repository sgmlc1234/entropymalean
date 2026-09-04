"""Tests for the certificate tier system (replacement for ad-hoc L1/L2/L3)."""

from __future__ import annotations

import asyncio

import pytest
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.certification.certifier import CertificationInput
from src.certification.generation import GenerationConfig
from src.certification import levels
from src.certification.levels import (
    LEVEL_NONE,
    LEVEL_PROOF,
    LEVEL_KERNEL,
    LEVEL_STATEMENT,
    PERMITTED_AXIOMS,
    axiom_audit,
    build_certificate_record,
    derive_certificate_level,
    legacy_lean_level_for,
    level_at_least,
    upgrade_to_kernel_replayed,
)
from src.evaluation.lean_verifier import LeanVerifyResult, parse_lean_messages
from src.orchestration.pool_generation import (
    THEOREM_CANONICAL_HEADER,
    TheoremAlignmentResult,
    TheoremGeneratedProblem,
    _certify_theorem_child,
)

STATEMENT = "theorem child_thm (n : Nat) (h : 0 < n) : n + 0 = n"
GOOD_PROOF = (
    f"{THEOREM_CANONICAL_HEADER}\n\n"
    f"{STATEMENT} := by\n  exact Nat.add_zero n\n"
)


def test_tier_derivation_is_monotone():
    assert derive_certificate_level(
        statement_checked=False, proof_checked=False
    ) == LEVEL_NONE
    assert derive_certificate_level(
        statement_checked=True, proof_checked=False
    ) == LEVEL_STATEMENT
    assert derive_certificate_level(
        statement_checked=True, proof_checked=True
    ) == LEVEL_PROOF
    assert derive_certificate_level(
        statement_checked=True, proof_checked=True, kernel_replayed=True
    ) == LEVEL_KERNEL


def test_tier_ordering_helper():
    assert level_at_least(LEVEL_KERNEL, LEVEL_PROOF)
    assert level_at_least(LEVEL_PROOF, LEVEL_STATEMENT)
    assert not level_at_least(LEVEL_STATEMENT, LEVEL_PROOF)
    assert level_at_least("garbage", LEVEL_NONE)  # unknown maps to 0


def test_certificate_record_carries_pins_and_audit_fields(tmp_path: Path):
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.30.0-rc2\n")
    (tmp_path / "lake-manifest.json").write_text(
        '{"packages": [{"name": "mathlib", "rev": "abc123"}]}'
    )
    record = build_certificate_record(
        statement_checked=True,
        proof_checked=True,
        axiom_closure=list(PERMITTED_AXIOMS),
        proof_method="tactic_proof",
        faithfulness="faithful",
        alignment_method="llm_judge",
        verifier="lake_env_lean",
        repo_root=tmp_path,
    )
    assert record["level"] == LEVEL_PROOF
    assert record["axiom_audit"]["passed"] is True
    assert record["lean_toolchain"] == "leanprover/lean4:v4.30.0-rc2"
    assert record["mathlib_revision"] == "abc123"
    assert record["faithfulness"] == "faithful"
    # Faithfulness is an audit field, never a certificate component.
    downgraded = build_certificate_record(
        statement_checked=True,
        proof_checked=True,
        axiom_closure=list(PERMITTED_AXIOMS),
        faithfulness="incomparable",
        repo_root=tmp_path,
    )
    assert downgraded["level"] == LEVEL_PROOF


def test_kernel_upgrade_records_comparator_pin():
    base = build_certificate_record(
        statement_checked=True,
        proof_checked=True,
        axiom_closure=list(PERMITTED_AXIOMS),
    )
    upgraded = upgrade_to_kernel_replayed(
        base,
        comparator_revision="71b52ec29e06d4b7d882726553b1ceb99a2499e0",
        permitted_axioms=["propext", "Quot.sound", "Classical.choice"],
        runner="lima-eml-linux",
    )
    assert upgraded["level"] == LEVEL_KERNEL
    assert upgraded["kernel_replayed"] is True
    assert upgraded["comparator_revision"].startswith("71b52ec")
    # Original record untouched.
    assert base["level"] == LEVEL_PROOF


def test_legacy_lean_level_mapping():
    assert legacy_lean_level_for(LEVEL_NONE) == 0
    assert legacy_lean_level_for(LEVEL_STATEMENT) == 1
    assert legacy_lean_level_for(LEVEL_PROOF) == 3
    assert legacy_lean_level_for(LEVEL_KERNEL) == 3


# ---------------------------------------------------------------------------
# Theorem-route integration: tiers reflect the statement-first gate outcomes
# ---------------------------------------------------------------------------


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


def _certify(*, verifier, aligned=True):
    async def generator(parent_input, config):
        return _generated()

    async def alignment_verifier(generated, *, item, config):
        return TheoremAlignmentResult.model_validate(
            {"aligned": aligned, "rationale": "test"}
        )

    async def repairer(generated, **kwargs):
        return None

    return asyncio.run(
        _certify_theorem_child(
            parent_input=CertificationInput(
                id="parent0", statement="Prove n + 0 = n.", answer="",
                metadata={"formal_statement": STATEMENT},
            ),
            item={
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["parent0"],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "leansearch_enabled": False,
            },
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=generator,
            theorem_verifier=verifier,
            theorem_alignment_verifier=alignment_verifier,
            theorem_proof_repairer=repairer,
        )
    )


async def _clean_axiom_probe_fn(code: str, timeout: float = 300.0) -> LeanVerifyResult:
    if ": False" in code:  # the vacuity probe; an all-accepting fake would file the row vacuous
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)
    return _clean_axiom_probe(code)


def _clean_axiom_probe(code: str) -> LeanVerifyResult:
    return LeanVerifyResult(
        ok=True,
        complete=True,
        verify_time=0.1,
        raw_stdout="'child_thm' depends on axioms: [propext, Classical.choice]",
    )


def test_certified_row_gets_proof_checked_certificate(monkeypatch):
    from src.orchestration import pool_generation as pg

    monkeypatch.setattr(pg, "verify_lean_proof", _clean_axiom_probe_fn)

    async def verifier(code, timeout=300.0):
        # A verifier that accepts everything would also "prove" the vacuity
        # probe (`example ... : False`) and the row would be filed vacuous.
        if ": False" in code:
            return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)
        return LeanVerifyResult(ok=True, complete="sorry" not in code, verify_time=0.1)

    result = _certify(verifier=verifier)
    assert result.status == "certified"
    assert result.certificate["level"] == LEVEL_PROOF
    assert result.certificate["statement_checked"] is True
    assert result.certificate["proof_accepted"] is True
    assert result.certificate["axiom_audit"]["passed"] is True
    assert result.certificate["kernel_replayed"] is False
    assert result.certificate["faithfulness"] == "faithful"
    # Legacy field unchanged for certified rows.
    assert result.lean_level == 3


def test_smuggled_axiom_blocks_proof_checked(monkeypatch):
    """An elaborator-accepted proof resting on a planted axiom must not
    reach proof_checked — the failure mode demonstrated on 2026-07-28."""
    from src.orchestration import pool_generation as pg

    async def file_probe(code, timeout=300.0):
        assert "#print axioms" in code  # the probe must not reuse the REPL
        return LeanVerifyResult(
            ok=True,
            complete=True,
            verify_time=0.1,
            raw_stdout="'child_thm' depends on axioms: [propext, convenient]",
        )

    monkeypatch.setattr(pg, "verify_lean_proof", file_probe)

    async def verifier(code, timeout=300.0):
        return LeanVerifyResult(ok=True, complete="sorry" not in code, verify_time=0.1)

    result = _certify(verifier=verifier)
    assert result.status == "proof_failed"
    assert "axiom_audit_failed" in str(result.error)
    assert result.certificate["level"] == LEVEL_STATEMENT
    assert result.certificate["axiom_audit"]["disallowed"] == ["convenient"]


def test_proof_failed_after_statement_pass_keeps_statement_checked():
    """The family-expansion payoff: a hard-to-prove but well-formed statement
    retains a real statement-level certificate instead of losing everything."""

    async def verifier(code, timeout=300.0):
        if "sorry" in code:
            return LeanVerifyResult(ok=True, complete=False, verify_time=0.1)
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)

    result = _certify(verifier=verifier)
    assert result.status == "proof_failed"
    assert result.certificate["level"] == LEVEL_STATEMENT
    assert result.certificate["statement_checked"] is True
    assert result.certificate["proof_accepted"] is False


def test_statement_failed_row_has_no_certificate_level():
    async def verifier(code, timeout=300.0):
        return LeanVerifyResult(ok=False, complete=False, verify_time=0.1)

    result = _certify(verifier=verifier)
    assert result.status == "statement_failed"
    assert result.certificate["level"] == LEVEL_NONE
    assert result.certificate["statement_checked"] is False


def test_new_lean_diag_format_is_parsed():
    """Toolchain v4.30 tags diagnostics as `error(name):` — the parser must
    capture them so statement failures carry real diagnostics."""
    payload = (
        "/tmp/x.lean:6:107: error(lean.synthInstanceFailed): failed to "
        "synthesize instance of type class\n  Fintype {a | a ≠ 0}\n"
    )
    messages = parse_lean_messages(payload)
    assert len(messages) == 1
    assert messages[0].severity == "error"
    assert "failed to synthesize" in messages[0].body


def test_reproducible_requires_the_kernel_replay_underneath() -> None:
    """Reproducing a weaker check widely is not a stronger check.

    A row that compiled on two platforms but was never replayed through an
    independent kernel has more evidence of the same kind, not evidence of a
    different kind, so it must stay at proof_checked.
    """
    assert (
        levels.derive_certificate_level(
            statement_checked=True, proof_checked=True,
            kernel_replayed=False, reproducible=True,
        )
        == levels.LEVEL_PROOF
    )
    assert (
        levels.derive_certificate_level(
            statement_checked=True, proof_checked=True,
            kernel_replayed=True, reproducible=True,
        )
        == levels.LEVEL_REPRODUCIBLE
    )


def test_reproducible_upgrade_refuses_a_single_platform() -> None:
    """One machine agreeing with itself is the failure mode being guarded."""
    certificate = levels.build_certificate_record(
        statement_checked=True, proof_checked=True, axiom_closure=list(levels.PERMITTED_AXIOMS)
    )
    replayed = levels.upgrade_to_kernel_replayed(
        certificate, comparator_revision="abc123", runner="linux-aarch64"
    )
    with pytest.raises(ValueError, match=">= 2 distinct platforms"):
        levels.upgrade_to_reproducible(replayed, platforms=["linux-aarch64"])
    with pytest.raises(ValueError, match=">= 2 distinct platforms"):
        levels.upgrade_to_reproducible(
            replayed, platforms=["linux-aarch64", "linux-aarch64"]
        )
    upgraded = levels.upgrade_to_reproducible(
        replayed, platforms=["linux-aarch64", "macos-aarch64"], export_digest="0558270e"
    )
    assert upgraded["level"] == levels.LEVEL_REPRODUCIBLE
    assert upgraded["platforms_verified"] == ["linux-aarch64", "macos-aarch64"]


def test_pins_record_every_package_not_just_mathlib() -> None:
    """One revision out of ten makes the pin look tighter than it is."""
    pins = levels.runtime_pins(str(REPO_ROOT))
    assert len(pins["package_revisions"]) >= 5
    assert pins["package_revisions"]["mathlib"] == pins["mathlib_revision"]
    assert len(pins["manifest_digest"]) == 16
