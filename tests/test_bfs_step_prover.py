"""Unit tests for the BFS-Prover-V2 aligned tactic-step prover."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.evaluation.bfs_step_prover import (
    assemble_proof,
    build_state_prompt,
    is_buggy_tactic,
    parse_tactic,
    prove_bfs_step,
)
from src.evaluation.lean_verifier import LeanVerifyResult
from src.evaluation.model_runner import ModelConfig


# ---------- prompt / parser primitives ----------


def test_build_state_prompt_empty_history_uses_separator():
    prompt = build_state_prompt("theorem t : 1 + 1 = 2 := by", [])
    assert prompt.endswith(":::")
    assert "theorem t" in prompt
    assert "\n  " not in prompt  # no extra tactic block when history is empty


def test_build_state_prompt_appends_history_with_two_space_indent():
    prompt = build_state_prompt("theorem t : True := by", ["intro", "exact trivial"])
    assert prompt == "theorem t : True := by\n  intro\n  exact trivial:::"


def test_parse_tactic_strips_whitespace():
    assert parse_tactic("  norm_num  \n") == "norm_num"
    assert parse_tactic("") == ""
    assert parse_tactic(None) == ""  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tactic, expected",
    [
        ("", True),
        ("exact sorry", True),
        ("admit", True),
        ("native_decide", True),
        ("rcases h with ⟨a, ?_⟩", True),
        ("cases' h with a ?_", True),
        ("simpa using ?_", True),
        ("simpa using _", True),
        ("simpa [foo, _]", True),
        ("norm_num", False),
        ("rfl", False),
        ("rcases h with ⟨a, b⟩", False),
        ("simpa using h", False),
    ],
)
def test_is_buggy_tactic(tactic, expected):
    assert is_buggy_tactic(tactic) is expected


def test_assemble_proof_inserts_by_when_missing():
    out = assemble_proof("theorem t : True", ["trivial"], header="import Foo")
    assert "theorem t : True := by\n  trivial" in out
    assert out.startswith("import Foo")


def test_assemble_proof_keeps_existing_by():
    out = assemble_proof("theorem t : True := by", ["trivial"], header="")
    assert "theorem t : True := by\n  trivial" in out


# ---------- end-to-end with mocked sampler + verifier ----------


def _cfg(label="bfs", paradigm="completion") -> ModelConfig:
    return ModelConfig(
        label=label,
        provider_slug="local-bfs",
        backend="lm_studio",
        paradigm=paradigm,
        temperature=0.7,
        max_tokens=64,
        stop=[":::", "\n\n"],
    )


def _verify_factory(verdicts):
    """Build an async verifier that returns the next verdict in `verdicts`."""
    idx = {"i": 0}

    async def _verify(code: str, *, timeout=None):
        i = idx["i"]
        v = verdicts[min(i, len(verdicts) - 1)]
        idx["i"] = i + 1
        return v

    return _verify


def _sampler_factory(per_step_candidates):
    """Build a TacticSampler that yields fixed candidates per step."""
    calls = {"i": 0}

    async def _sample(config, prompt, n):
        i = calls["i"]
        calls["i"] = i + 1
        return list(per_step_candidates[min(i, len(per_step_candidates) - 1)])

    return _sample


def test_prove_bfs_step_first_candidate_closes():
    sampler = _sampler_factory([["norm_num", "rfl"]])  # first candidate wins
    verifier = _verify_factory(
        [LeanVerifyResult(ok=True, complete=True)]
    )
    record = asyncio.run(
        prove_bfs_step(
            benchmark="miniF2F",
            arm="control",
            problem_id="t1",
            statement="trivial",
            formal_prefix="theorem t : True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=2,
            S_max=3,
            n_per_step=2,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is True
    assert record.min_steps_to_success == 1
    assert len(record.attempts) == 1  # second attempt skipped after pass


def test_prove_bfs_step_falls_back_to_first_typing_candidate():
    # Step 1: candidates [bad, partial]; partial type-checks but does not close.
    # Step 2: candidate [closer] closes the proof.
    sampler = _sampler_factory([["bad_tactic", "intro h"], ["exact h"]])
    verifier = _verify_factory(
        [
            LeanVerifyResult(ok=False, complete=False),  # bad_tactic
            LeanVerifyResult(ok=True, complete=False),   # intro h
            LeanVerifyResult(ok=True, complete=True),    # exact h
        ]
    )
    record = asyncio.run(
        prove_bfs_step(
            benchmark="miniF2F",
            arm="control",
            problem_id="t2",
            statement="implication",
            formal_prefix="theorem t : True → True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            S_max=3,
            n_per_step=2,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is True
    assert record.min_steps_to_success == 2
    step1 = record.attempts[0].steps[0]
    assert step1.chosen_tactic == "intro h"
    assert step1.candidates[0].accepted is False
    assert step1.candidates[1].accepted is True


def test_prove_bfs_step_marks_failure_when_no_candidate_types():
    sampler = _sampler_factory([["bad1", "bad2"]])
    verifier = _verify_factory(
        [LeanVerifyResult(ok=False, complete=False)]
    )
    record = asyncio.run(
        prove_bfs_step(
            benchmark="miniF2F",
            arm="control",
            problem_id="t3",
            statement="hard",
            formal_prefix="theorem t : False := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            S_max=2,
            n_per_step=2,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is False
    assert record.attempts[0].success is False
    # Both candidates were checked before the attempt was abandoned.
    assert len(record.attempts[0].steps[0].candidates) == 2


def test_prove_bfs_step_to_summary_exposes_compat_aliases():
    sampler = _sampler_factory([["norm_num"]])
    verifier = _verify_factory(
        [LeanVerifyResult(ok=True, complete=True)]
    )
    record = asyncio.run(
        prove_bfs_step(
            benchmark="miniF2F",
            arm="control",
            problem_id="t4",
            statement="trivial",
            formal_prefix="theorem t : True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            S_max=2,
            n_per_step=1,
            lean_timeout=10.0,
        )
    )
    summary = record.to_summary()
    assert summary["paradigm"] == "step_level"
    assert summary["pass_at_k"] is True
    # `summarize_proof_jsonl` reads `min_turns_to_success`; we alias the
    # step-level count to that key so the aggregator works uniformly.
    assert summary["min_steps_to_success"] == 1
    assert summary["min_turns_to_success"] == 1
