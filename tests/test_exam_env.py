"""Tests for the game-style Lean exam environment."""

from __future__ import annotations

import asyncio
import shutil
from typing import Dict, List

import pytest

from src.evaluation.lean_verifier import LeanMessage, LeanVerifyResult
from src.exam_env.environment import LeanExamEnv
from src.exam_env.palette import (
    TACTIC_DOCS,
    build_check_probe,
    build_palette,
    candidate_theorem_names,
    parse_check_probe_output,
    tactics_in_proof,
)

STATEMENT = "theorem exam_t (n : ℕ) (h : 0 < n) : n + 0 = n ∧ 0 < n"

GOALS_BOTH = (
    "unsolved goals\n"
    "case left\nn : ℕ\nh : 0 < n\n⊢ n + 0 = n\n\n"
    "case right\nn : ℕ\nh : 0 < n\n⊢ 0 < n"
)
GOAL_RIGHT = "unsolved goals\ncase right\nn : ℕ\nh : 0 < n\n⊢ 0 < n"
GOAL_INITIAL = "unsolved goals\nn : ℕ\nh : 0 < n\n⊢ n + 0 = n ∧ 0 < n"


def _unsolved(body: str) -> LeanVerifyResult:
    return LeanVerifyResult(
        ok=False,
        complete=False,
        errors=[LeanMessage(severity="error", line=6, column=0, body=body)],
        verify_time=0.05,
    )


def _rejecting(body: str) -> LeanVerifyResult:
    return LeanVerifyResult(
        ok=False,
        complete=False,
        errors=[LeanMessage(severity="error", line=7, column=2, body=body)],
        verify_time=0.05,
    )


def _script_verifier(script: Dict[str, LeanVerifyResult]):
    """Map the tactic suffix of the assembled code to a canned verdict."""
    calls: List[str] = []

    async def verifier(code: str, timeout: float = 300.0):
        calls.append(code)
        body = code.split(":= by\n", 1)[1]
        key = body.strip()
        if key in script:
            return script[key]
        raise AssertionError(f"unexpected assembled body: {key!r}")

    verifier.calls = calls  # type: ignore[attr-defined]
    return verifier


def _env(script, palette=None) -> LeanExamEnv:
    return LeanExamEnv(
        formal_statement=f"{STATEMENT} := by\n  exact ⟨Nat.add_zero n, h⟩",
        lean_header="import Mathlib",
        palette=palette,
        verifier=_script_verifier(script),
    )


BASE_SCRIPT = {
    "skip": _unsolved(GOAL_INITIAL),
    "constructor": _unsolved(GOALS_BOTH),
    "constructor\n  exact Nat.add_zero n": _unsolved(GOAL_RIGHT),
    "constructor\n  exact Nat.add_zero n\n  exact h": LeanVerifyResult(
        ok=True, complete=True, verify_time=0.05
    ),
    "constructor\n  linarith": _rejecting(
        "linarith failed to find a contradiction"
    ),
}


def test_reset_presents_exam_start_state():
    env = _env(BASE_SCRIPT)
    observation = asyncio.run(env.reset())
    assert observation.status == "active"
    assert len(observation.goals) == 1
    assert "⊢ n + 0 = n ∧ 0 < n" in observation.goals[0]
    # The proof body of the GT is stripped — the solver starts from scratch.
    assert env.statement == STATEMENT


def test_accepted_tactic_advances_and_shows_case_goals():
    env = _env(BASE_SCRIPT)

    async def play():
        await env.reset()
        return await env.step({"type": "tactic", "tactic": "constructor"})

    observation = asyncio.run(play())
    assert observation.status == "accepted"
    assert len(observation.goals) == 2
    assert any("case left" in goal for goal in observation.goals)
    assert env.steps == ["constructor"]


def test_rejected_tactic_leaves_state_unchanged():
    env = _env(BASE_SCRIPT)

    async def play():
        await env.reset()
        await env.step({"type": "tactic", "tactic": "constructor"})
        return await env.step({"type": "tactic", "tactic": "linarith"})

    observation = asyncio.run(play())
    assert observation.status == "rejected"
    assert "linarith failed" in observation.message
    assert env.steps == ["constructor"]  # unchanged — the game's Retry
    assert len(observation.goals) == 2  # still showing the pre-failure goals


def test_full_episode_solves_and_records_code():
    env = _env(BASE_SCRIPT)

    async def play():
        await env.reset()
        await env.step({"type": "tactic", "tactic": "constructor"})
        await env.step({"type": "tactic", "tactic": "exact Nat.add_zero n"})
        return await env.step({"type": "tactic", "tactic": "exact h"})

    observation = asyncio.run(play())
    assert observation.status == "solved"
    assert env.success and env.done
    solved = env.solved_code()
    assert solved is not None and solved.endswith("  exact h")
    assert len(env.transcript) == 4  # reset + 3 actions


def test_rollback_restores_earlier_goals_without_verification():
    env = _env(BASE_SCRIPT)

    async def play():
        await env.reset()
        await env.step({"type": "tactic", "tactic": "constructor"})
        await env.step({"type": "tactic", "tactic": "exact Nat.add_zero n"})
        before = len(env.verifier.calls)
        rollback = await env.step({"type": "rollback", "to_step": 1})
        return rollback, before, len(env.verifier.calls)

    rollback, before, after = asyncio.run(play())
    assert rollback.status == "active"
    assert env.steps == ["constructor"]
    assert len(rollback.goals) == 2  # goals restored from history
    assert before == after  # rollback is free — no Lean call


def test_inspect_returns_palette_card_without_lean_call():
    palette = {
        "tactics": {"constructor": TACTIC_DOCS["constructor"]},
        "theorems": {"Nat.add_zero": "∀ (n : ℕ), n + 0 = n"},
    }
    env = _env(BASE_SCRIPT, palette=palette)

    async def play():
        await env.reset()
        before = len(env.verifier.calls)
        card = await env.step({"type": "inspect", "name": "Nat.add_zero"})
        return card, before, len(env.verifier.calls)

    card, before, after = asyncio.run(play())
    assert card.card == {
        "name": "Nat.add_zero",
        "kind": "theorem",
        "doc": "∀ (n : ℕ), n + 0 = n",
    }
    assert before == after


# ---------------------------------------------------------------------------
# Palette extraction
# ---------------------------------------------------------------------------


def test_candidate_theorem_names_filters_locals_and_tactics():
    proof = (
        "  intro n hn\n"
        "  have h2 : 0 < n := by omega\n"
        "  exact ⟨Nat.add_zero n, mul_comm a b ▸ h2⟩\n"
    )
    names = candidate_theorem_names(proof)
    assert "Nat.add_zero" in names
    assert "mul_comm" in names
    assert "omega" not in names  # tactic, not theorem
    assert "n" not in names and "hn" not in names


def test_tactics_in_proof_reads_line_heads():
    proof = "  intro n\n  · exact h\n  norm_num\n"
    assert tactics_in_proof(proof) == ["intro", "exact", "norm_num"]


def test_parse_check_probe_output_real_format():
    raw = (
        "Nat.add_zero : ∀ (n : ℕ), n + 0 = n\n"
        "Nat.gcd_self : ∀ (n : ℕ), n.gcd n = n\n"
        "/tmp/x.lean:6:8: error(lean.unknownIdentifier): Unknown identifier `Fake.name`\n"
    )
    names = ["Nat.add_zero", "Nat.gcd_self", "Fake.name"]
    signatures = parse_check_probe_output(raw, names)
    assert signatures == {
        "Nat.add_zero": "∀ (n : ℕ), n + 0 = n",
        "Nat.gcd_self": "∀ (n : ℕ), n.gcd n = n",
    }
    assert "#check @Fake.name" in build_check_probe("import Mathlib", ["Fake.name"])


def test_parse_check_probe_output_at_prefixed_multiline_signature():
    """Lemmas with implicit binders print as `@name : sig` with wrapping —
    the exact shape that made the first real episode's palette come up empty."""
    raw = (
        "@AnalyticOnNhd.eq_const_of_re_eq_const : ∀ {f : ℂ → ℂ} {U : Set ℂ} {c₀ : ℝ},\n"
        "  AnalyticOnNhd ℂ f U → (∀ x ∈ U, (f x).re = c₀) → IsOpen U → ∃ c, ∀ x ∈ U, f x = c\n"
        "/tmp/x.lean:6:8: error(lean.unknownIdentifier): Unknown identifier `a.property`\n"
    )
    signatures = parse_check_probe_output(
        raw, ["AnalyticOnNhd.eq_const_of_re_eq_const", "a.property"]
    )
    assert list(signatures) == ["AnalyticOnNhd.eq_const_of_re_eq_const"]
    assert "IsOpen U" in signatures["AnalyticOnNhd.eq_const_of_re_eq_const"]


def test_strict_steps_rejects_semicolon_chains():
    env = LeanExamEnv(
        formal_statement=f"{STATEMENT} := by\n  exact ⟨Nat.add_zero n, h⟩",
        lean_header="import Mathlib",
        verifier=_script_verifier(BASE_SCRIPT),
        strict_steps=True,
    )

    async def play():
        await env.reset()
        chained = await env.step(
            {"type": "tactic", "tactic": "constructor; exact Nat.add_zero n"}
        )
        combinator = await env.step({"type": "tactic", "tactic": "constructor"})
        return chained, combinator

    chained, combinator = asyncio.run(play())
    assert chained.status == "rejected" and "strict mode" in chained.message
    assert combinator.status == "accepted"  # single tactics still fine


def test_build_palette_with_fake_runner():
    async def runner(code: str) -> str:
        assert "#check @Nat.add_zero" in code
        return "Nat.add_zero : ∀ (n : ℕ), n + 0 = n\n"

    palette = asyncio.run(
        build_palette(
            lean_code=(
                "import Mathlib\n\n"
                f"{STATEMENT} := by\n"
                "  constructor\n"
                "  · exact Nat.add_zero n\n"
                "  · exact h\n"
            ),
            formal_statement=STATEMENT,
            lean_header="import Mathlib",
            check_runner=runner,
        )
    )
    assert palette["theorems"] == {"Nat.add_zero": "∀ (n : ℕ), n + 0 = n"}
    assert "constructor" in palette["tactics"]
    assert "exact" in palette["tactics"]
    # Core tactics are always offered.
    assert "intro" in palette["tactics"]


# ---------------------------------------------------------------------------
# Real-Lean integration (repo toolchain)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
def test_real_lean_exam_episode():
    env = LeanExamEnv(
        formal_statement=f"{STATEMENT} := by\n  exact ⟨Nat.add_zero n, h⟩",
        lean_header="import Mathlib",
        palette={"tactics": dict(TACTIC_DOCS), "theorems": {}},
    )

    async def play():
        first = await env.reset()
        split = await env.step({"type": "tactic", "tactic": "constructor"})
        bad = await env.step({"type": "tactic", "tactic": "exact h"})  # wrong goal
        left = await env.step(
            {"type": "tactic", "tactic": "exact Nat.add_zero n"}
        )
        return first, split, bad, left, await env.step(
            {"type": "tactic", "tactic": "exact h"}
        )

    first, split, bad, left, final = asyncio.run(play())
    assert first.status == "active" and "n + 0 = n ∧ 0 < n" in first.goals[0]
    assert split.status == "accepted" and len(split.goals) == 2
    assert bad.status == "rejected"  # exact h cannot close `n + 0 = n`
    assert left.status == "accepted" and len(left.goals) == 1
    assert final.status == "solved" and env.success
