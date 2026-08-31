"""Tests for the BFS-Prover exam player policy."""

from __future__ import annotations

import asyncio
from typing import Dict, List

from src.evaluation.lean_verifier import LeanMessage, LeanVerifyResult
from src.exam_env.bfs_player import (
    BFSExamPlayer,
    build_goal_state_prompt,
    rank_candidates,
)
from src.exam_env.environment import LeanExamEnv

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
    )


def _reject(body: str) -> LeanVerifyResult:
    return LeanVerifyResult(
        ok=False,
        complete=False,
        errors=[LeanMessage(severity="error", line=7, column=2, body=body)],
    )


_DONE = LeanVerifyResult(ok=True, complete=True)


def _make_env(script: Dict[str, LeanVerifyResult]) -> LeanExamEnv:
    async def verifier(code: str, timeout: float = 300.0):
        key = code.split(":= by\n", 1)[1].strip()
        if key in script:
            return script[key]
        raise AssertionError(f"unexpected body: {key!r}")

    return LeanExamEnv(
        formal_statement=STATEMENT,
        lean_header="import Mathlib",
        palette={"tactics": {}, "theorems": {"Nat.add_zero": "∀ (n : ℕ), n + 0 = n"}},
        verifier=verifier,
    )


def _scripted_sampler(batches: List[List[str]]):
    calls: List[str] = []

    async def sampler(prompt: str, n: int):
        calls.append(prompt)
        return batches.pop(0) if batches else []

    sampler.calls = calls  # type: ignore[attr-defined]
    return sampler


def test_prompt_carries_only_the_goal_state():
    """BFS-Prover is a completion model; anything but the state is
    off-distribution, so the palette must never enter the prompt."""
    goals = ["case left\nn : ℕ\n⊢ n + 0 = n"]
    prompt = build_goal_state_prompt(goals)
    assert prompt.endswith(":::")
    assert "⊢ n + 0 = n" in prompt
    assert "useful lemmas" not in prompt and "--" not in prompt


def test_rank_candidates_filters_dedupes_and_prefers_palette():
    sampled = [
        "  simp  ",
        "simp",                      # duplicate after strip
        "",                          # empty -> buggy
        "sorry",                     # banned by the reference filter
        "exact Nat.add_zero n",
    ]
    plain = rank_candidates(sampled, None)
    # `exact?`-style exploration is NOT filtered — the reference allows it.
    assert plain == ["simp", "exact Nat.add_zero n"]
    ranked = rank_candidates(sampled, ["Nat.add_zero"])
    assert ranked[0] == "exact Nat.add_zero n"  # palette lemma tried first


def test_player_tries_candidates_in_order_and_solves():
    script = {
        "skip": _unsolved(GOAL_INITIAL),
        "linarith": _reject("linarith failed"),
        "constructor": _unsolved(GOALS_BOTH),
        "constructor\n  exact Nat.add_zero n": _unsolved(GOAL_RIGHT),
        "constructor\n  exact Nat.add_zero n\n  exact h": _DONE,
    }
    env = _make_env(script)
    sampler = _scripted_sampler(
        [
            ["linarith", "constructor"],
            ["exact Nat.add_zero n"],
            ["exact h"],
        ]
    )
    player = BFSExamPlayer(sampler, n_per_step=2, resample_rounds=0)
    result = asyncio.run(player.play(env))
    assert result["success"] is True
    assert result["steps"] == ["constructor", "exact Nat.add_zero n", "exact h"]
    assert result["rejected"] == 1
    assert result["rollbacks"] == 0
    # The live goal state (not the theorem prefix) was prompted, one goal at
    # a time and without its `case` header — the reference's pp1 form.
    assert "⊢ n + 0 = n ∧ 0 < n" in sampler.calls[0]
    assert "⊢ n + 0 = n" in sampler.calls[1]
    assert "case" not in sampler.calls[1]


def test_player_rolls_back_when_stuck_and_avoids_dead_branch():
    script = {
        "skip": _unsolved(GOAL_INITIAL),
        # Branch A: refine gets accepted but leads to a dead end.
        "refine ⟨?_, ?_⟩": _unsolved(GOALS_BOTH),
        "refine ⟨?_, ?_⟩\n  norm_num": _reject("norm_num made no progress"),
        # Branch B after rollback: the direct term proof closes it.
        "exact ⟨Nat.add_zero n, h⟩": _DONE,
    }
    env = _make_env(script)
    sampler = _scripted_sampler(
        [
            ["refine ⟨?_, ?_⟩"],          # depth 0 → accepted
            ["norm_num"],                  # depth 1 → rejected, batch exhausted
            ["refine ⟨?_, ?_⟩", "exact ⟨Nat.add_zero n, h⟩"],  # depth 0 again
        ]
    )
    player = BFSExamPlayer(
        sampler, n_per_step=2, resample_rounds=0, max_rollbacks=2
    )
    result = asyncio.run(player.play(env))
    assert result["success"] is True
    assert result["rollbacks"] == 1
    # After rollback the dead first choice is excluded; only the fresh
    # candidate is played, so the solved proof is the one-liner.
    assert result["steps"] == ["exact ⟨Nat.add_zero n, h⟩"]


def test_player_gives_up_at_root_without_rollback_budget_abuse():
    script = {
        "skip": _unsolved(GOAL_INITIAL),
        "bad1": _reject("nope"),
        "bad2": _reject("nope"),
    }
    env = _make_env(script)
    sampler = _scripted_sampler([["bad1"], ["bad2"], [], []])
    player = BFSExamPlayer(sampler, n_per_step=1, resample_rounds=1)
    result = asyncio.run(player.play(env))
    assert result["success"] is False
    assert result["rollbacks"] == 0  # nothing to roll back at the root
    assert result["rejected"] == 2
