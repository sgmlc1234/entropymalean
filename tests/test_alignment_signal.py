"""Tests for the goal_roundtrip alignment signal."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

from src.certification.alignment import (
    build_goal_probe_code,
    elaborated_goal_alignment,
    extract_elaborated_goal,
    strip_proof_body,
)
from src.certification.certifier import CertificationInput
from src.certification.generation import GenerationConfig
from src.evaluation.lean_verifier import LeanVerifyResult, parse_lean_messages

STATEMENT = "theorem probe_target (n : ℕ) (h : 0 < n) : n + 0 = n"
EXTRACT_OUTPUT = (
    "theorem probe_target.extracted_1_1 (n : ℕ) (h : 0 < n) : n + 0 = n := sorry\n"
    "/tmp/goal_probe.lean:6:8: warning: declaration uses `sorry`\n"
)


def test_strip_proof_body():
    assert strip_proof_body(f"{STATEMENT} := by\n  simp") == STATEMENT
    assert strip_proof_body(STATEMENT) == STATEMENT


def test_build_goal_probe_code_contains_extract_goal_and_strict_options():
    code = build_goal_probe_code("import Mathlib", f"{STATEMENT} := by simp")
    assert "extract_goal" in code
    assert code.rstrip().endswith("sorry")
    assert "set_option autoImplicit false" in code
    assert "simp" not in code


def test_extract_elaborated_goal_parses_real_format():
    goal = extract_elaborated_goal(EXTRACT_OUTPUT)
    assert goal is not None
    assert goal.startswith("theorem probe_target.extracted_1_1")
    assert ":= sorry" not in goal
    assert "declaration uses" not in goal


def test_extract_elaborated_goal_handles_multiline_goal():
    raw = (
        "theorem long.extracted_1_1 (f : ℂ → ℂ) (Ω : Set ℂ)\n"
        "    (hΩ : IsOpen Ω) :\n"
        "    Set.EqOn f (fun _ => f 0) Ω := sorry\n"
        "/tmp/x.lean:9:2: warning: declaration uses `sorry`\n"
    )
    goal = extract_elaborated_goal(raw)
    assert goal is not None
    assert "IsOpen Ω" in goal
    assert goal.endswith("Set.EqOn f (fun _ => f 0) Ω")


def test_extract_elaborated_goal_none_when_absent():
    assert extract_elaborated_goal("/tmp/x.lean:1:0: error: nope") is None
    assert extract_elaborated_goal("") is None


def test_parse_lean_messages_folds_multiline_bodies():
    payload = (
        "/tmp/x.lean:6:2: error: unsolved goals\n"
        "n : ℕ\n"
        "⊢ n + 0 = n\n"
        "/tmp/x.lean:9:0: warning: declaration uses `sorry`\n"
    )
    messages = parse_lean_messages(payload)
    assert len(messages) == 2
    assert "⊢ n + 0 = n" in messages[0].body
    assert messages[1].severity == "warning"


# ---------------------------------------------------------------------------
# End-to-end signal with role-separated fakes
# ---------------------------------------------------------------------------


def _fake_verifier(*, ok=True, stdout=EXTRACT_OUTPUT):
    async def verifier(code, timeout=300.0):
        assert "extract_goal" in code
        return LeanVerifyResult(
            ok=ok, complete=False, verify_time=0.1, raw_stdout=stdout if ok else ""
        )

    return verifier


def _signal(**overrides):
    seen: dict = {"informalizer_input": None, "judge_input": None}

    async def informalizer(goal, *, config):
        seen["informalizer_input"] = goal
        return "For every natural number n with 0 < n, n + 0 equals n."

    async def judge(original, roundtrip, *, config):
        seen["judge_input"] = (original, roundtrip)
        return {"equivalent": True, "mismatches": [], "rationale": "same claim"}

    kwargs = dict(
        statement_nl="Show that n + 0 = n for every positive natural number n.",
        formal_statement=STATEMENT,
        lean_header="import Mathlib",
        config=GenerationConfig(model="test-model"),
        verifier=_fake_verifier(),
        informalizer=informalizer,
        judge=judge,
    )
    kwargs.update(overrides)
    return asyncio.run(elaborated_goal_alignment(**kwargs)), seen


def test_signal_ok_path_separates_roles():
    signal, seen = _signal()
    assert signal["status"] == "ok"
    assert signal["equivalent"] is True
    # Informalizer saw only the elaborated goal (never the original NL).
    assert seen["informalizer_input"].startswith("theorem probe_target.extracted")
    # Judge saw only the two prose statements (never any Lean).
    original, roundtrip = seen["judge_input"]
    assert "theorem" not in original and "theorem" not in roundtrip


def test_signal_records_mismatches():
    async def judge(original, roundtrip, *, config):
        return {
            "equivalent": False,
            "mismatches": ["hypothesis 0 < n dropped"],
            "rationale": "missing positivity hypothesis",
        }

    signal, _ = _signal(judge=judge)
    assert signal["equivalent"] is False
    assert signal["mismatches"] == ["hypothesis 0 < n dropped"]


def test_signal_statement_error_short_circuits():
    signal, seen = _signal(verifier=_fake_verifier(ok=False))
    assert signal["status"] == "statement_error"
    assert signal["equivalent"] is None
    assert seen["informalizer_input"] is None


def test_signal_no_goal_extracted():
    signal, _ = _signal(verifier=_fake_verifier(stdout="no goal here"))
    assert signal["status"] == "no_goal_extracted"


def test_signal_error_is_contained():
    async def broken_informalizer(goal, *, config):
        raise RuntimeError("429 Too Many Requests")

    signal, _ = _signal(informalizer=broken_informalizer)
    assert signal["status"] == "signal_error"
    assert "429" in signal["rationale"]


# ---------------------------------------------------------------------------
# Inline annotation hook in the theorem route
# ---------------------------------------------------------------------------


def test_inline_hook_annotates_quality_evidence(monkeypatch):
    from src.orchestration import pool_generation as pg

    monkeypatch.setenv("POOL_ALIGNMENT_GOAL_AUDIT", "1")

    async def fake_signal(**kwargs):
        assert kwargs["formal_statement"] == STATEMENT
        return {
            "source": "elaborated_goal_informalization",
            "status": "ok",
            "equivalent": True,
            "mismatches": [],
        }

    monkeypatch.setattr(pg, "elaborated_goal_alignment", fake_signal)

    async def generator(parent_input, config):
        return pg.TheoremGeneratedProblem.model_validate(
            {
                "id": "p0__theorem_gen1",
                "source_problem_id": "p0",
                "statement": "Show that n + 0 = n for positive n.",
                "formal_statement": STATEMENT,
                "lean_code": (
                    f"{pg.THEOREM_CANONICAL_HEADER}\n\n{STATEMENT} := by\n"
                    "  exact Nat.add_zero n\n"
                ),
            }
        )

    async def verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True, verify_time=0.1,
                raw_stdout="'probe_target' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(
            ok=True, complete="sorry" not in code, verify_time=0.1
        )

    async def alignment_verifier(generated, *, item, config):
        return pg.TheoremAlignmentResult.model_validate(
            {"aligned": True, "rationale": "test"}
        )

    result = asyncio.run(
        pg._certify_theorem_child(
            parent_input=CertificationInput(
                id="p0", statement="Prove n + 0 = n.", answer="",
                metadata={"formal_statement": STATEMENT},
            ),
            item={
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p0"],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "leansearch_enabled": False,
            },
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=generator,
            theorem_verifier=verifier,
            theorem_alignment_verifier=alignment_verifier,
        )
    )

    assert result.status == "certified"
    evidence = (result.quality_evidence or {}).get("alignment_evidence") or {}
    assert evidence.get("equivalent") is True
    assert evidence.get("source") == "elaborated_goal_informalization"


def test_inline_hook_off_by_default(monkeypatch):
    from src.orchestration import pool_generation as pg

    monkeypatch.delenv("POOL_ALIGNMENT_GOAL_AUDIT", raising=False)

    async def exploding_signal(**kwargs):  # must never be called
        raise AssertionError("A2 signal ran without opt-in")

    monkeypatch.setattr(pg, "elaborated_goal_alignment", exploding_signal)

    async def generator(parent_input, config):
        return pg.TheoremGeneratedProblem.model_validate(
            {
                "id": "p0__theorem_gen1",
                "source_problem_id": "p0",
                "statement": "Show that n + 0 = n for positive n.",
                "formal_statement": STATEMENT,
                "lean_code": (
                    f"{pg.THEOREM_CANONICAL_HEADER}\n\n{STATEMENT} := by\n"
                    "  exact Nat.add_zero n\n"
                ),
            }
        )

    async def verifier(code, timeout=300.0):
        if "#print axioms" in code:
            return LeanVerifyResult(
                ok=True, complete=True, verify_time=0.1,
                raw_stdout="'probe_target' depends on axioms: [propext, Classical.choice]",
            )
        return LeanVerifyResult(ok=True, complete="sorry" not in code, verify_time=0.1)

    async def alignment_verifier(generated, *, item, config):
        return pg.TheoremAlignmentResult.model_validate(
            {"aligned": True, "rationale": "test"}
        )

    result = asyncio.run(
        pg._certify_theorem_child(
            parent_input=CertificationInput(
                id="p0", statement="Prove n + 0 = n.", answer="",
                metadata={"formal_statement": STATEMENT},
            ),
            item={
                "slot": 1,
                "op_type": "mutation",
                "parent_ids": ["p0"],
                "target_style": "theorem_proof",
                "target_family": "theorem_proof",
                "leansearch_enabled": False,
            },
            generation_count=1,
            config=GenerationConfig(model="test-model"),
            theorem_generator=generator,
            theorem_verifier=verifier,
            theorem_alignment_verifier=alignment_verifier,
        )
    )
    assert result.status == "certified"
    assert (result.quality_evidence or {}).get("alignment_evidence") == {}


# ---------------------------------------------------------------------------
# Real-Lean integration (repo toolchain; ~30s)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("lake") is None, reason="lake toolchain not on PATH"
)
def test_goal_probe_extracts_real_elaborated_goal():
    repo = Path(__file__).resolve().parents[1]
    code = build_goal_probe_code(
        "import Mathlib\nimport Aesop\nset_option maxHeartbeats 400000",
        "theorem t_probe (n : ℕ) (h : 0 < n) : n + 0 = n := by\n  simp",
    )
    probe_path = repo / "tmp" / "alignment_probe_test.lean"
    probe_path.parent.mkdir(exist_ok=True)
    probe_path.write_text(code, encoding="utf-8")
    proc = subprocess.run(
        ["lake", "env", "lean", str(probe_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=300,
    )
    goal = extract_elaborated_goal(proc.stdout + "\n" + proc.stderr)
    assert goal is not None, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "(n : ℕ)" in goal and "n + 0 = n" in goal
