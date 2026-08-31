"""Unit tests for the BFS tree-search prover."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.evaluation.bfs_tree_prover import (
    ProofState,
    prove_bfs_tree,
)
from src.evaluation.lean_verifier import LeanVerifyResult
from src.evaluation.model_runner import ModelConfig


def _cfg() -> ModelConfig:
    return ModelConfig(
        label="bfs-test",
        provider_slug="local-bfs",
        backend="lm_studio",
        paradigm="completion",
        temperature=0.7,
        max_tokens=64,
        stop=[":::", "\n\n"],
    )


def _sampler_factory(per_state_candidates):
    """Map a ProofState's tactic tuple → list of next tactic candidates.

    Lets tests stage a specific tree shape: pass a dict keyed by tuple of
    accepted tactics; missing keys yield no candidates (dead end).
    """
    async def _sample(config, prompt, n):
        # Recover tactics tuple from prompt (state is encoded as
        # ``formal_prefix\n  tac1\n  tac2:::`` by build_state_prompt).
        # We cheat: tests stash the tuple via a side channel below.
        return list(per_state_candidates.get(tuple(_sample.current_state), []))[:n]
    _sample.current_state = ()
    _sample.per_state_candidates = per_state_candidates
    return _sample


def _verify_factory(verdicts):
    """Verify each proof_code in order; tests pre-stage verdicts."""
    idx = {"i": 0}

    async def _verify(code, *, timeout=None):
        i = idx["i"]
        idx["i"] = i + 1
        v = verdicts[min(i, len(verdicts) - 1)]
        return v

    return _verify


def test_root_state_is_empty():
    r = ProofState.root()
    assert r.tactics == ()
    assert r.depth == 0


def test_extend_creates_child():
    r = ProofState.root().extend("intro h")
    assert r.tactics == ("intro h",)
    assert r.depth == 1


def test_tree_search_picks_closing_candidate_at_root():
    """If the root expansion includes a closing tactic, return immediately
    without growing the tree."""

    async def sampler(config, prompt, n):
        # All n candidates at the root; one of them closes.
        return ["norm_num", "rfl", "decide"][:n]

    verifier = _verify_factory([
        LeanVerifyResult(ok=False, complete=False),  # norm_num fails
        LeanVerifyResult(ok=True, complete=True),    # rfl closes
        LeanVerifyResult(ok=False, complete=False),  # never reached
    ])

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="miniF2F",
            arm="control",
            problem_id="t1",
            statement="trivial",
            formal_prefix="theorem t : True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            n_per_step=3,
            max_nodes=16,
            timeout_per_attempt_s=10.0,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is True
    assert record.min_steps_to_success == 1
    a = record.attempts[0]
    assert a.success is True
    assert a.terminated_reason == "proved"
    assert a.success_tactics == ["rfl"]


def test_tree_search_finds_proof_after_root_branching():
    """Root expansion enqueues two siblings; both type but only the
    deeper-explored branch yields a closing tactic. Verifies the queue
    actually visits multiple nodes (not just the first sibling)."""

    async def sampler(config, prompt, n):
        # Root: branch into two typing children. Either branch's child
        # closes the proof — the test asserts SOME success, not a
        # specific path (priority order is depth-first FIFO, which is
        # an implementation choice).
        if "intro" not in prompt:
            return ["intro a", "intro b"]
        # Both branches' grandchildren close.
        return ["exact h"]

    verifier = _verify_factory([
        LeanVerifyResult(ok=True, complete=False),   # intro a types
        LeanVerifyResult(ok=True, complete=False),   # intro b types
        LeanVerifyResult(ok=True, complete=True),    # exact h closes
    ])

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="miniF2F",
            arm="control",
            problem_id="t2",
            statement="implication",
            formal_prefix="theorem t : True → True → True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            n_per_step=2,
            max_nodes=16,
            timeout_per_attempt_s=10.0,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is True
    a = record.attempts[0]
    # We don't pin the exact path (priority is impl-defined), but the
    # successful path must have exactly 2 tactics — the root branch +
    # the closing tactic.
    assert a.success_tactics is not None
    assert len(a.success_tactics) == 2
    assert a.success_tactics[-1] == "exact h"
    assert a.nodes_explored >= 2


def test_tree_search_terminates_on_budget_exhaustion():
    """When no closing tactic ever appears and the queue keeps growing,
    the loop should terminate at max_nodes with terminated_reason='budget'."""

    async def sampler(config, prompt, n):
        # Always emit one typing child — tree grows linearly.
        return ["typing"]

    async def verifier(code, *, timeout=None):
        return LeanVerifyResult(ok=True, complete=False)  # always types, never closes

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="miniF2F",
            arm="control",
            problem_id="t3",
            statement="loop",
            formal_prefix="theorem t : True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            n_per_step=1,
            max_nodes=5,
            timeout_per_attempt_s=30.0,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is False
    a = record.attempts[0]
    assert a.terminated_reason == "budget"
    assert a.nodes_explored == 5


def _sorry_aware_verify_factory(closing_tactics):
    """Mock verifier that mimics Lean's real behavior on `partial proof + sorry`.

    Regression mock for the partial-proof bug: the production
    ``verify_lean_proof`` returns ``ok=False`` on a partial proof with
    no ``sorry``, which used to trap the tree-search loop at depth 1.
    The fix in ``bfs_tree_prover`` switched to appending ``sorry``
    before verifying — this mock simulates the three Lean outcomes that
    fix relies on:

      * ``proof body + sorry`` where the body already closed →
        ``ok=False`` with ``"no goals to be solved"`` error
      * ``proof body + sorry`` where the body is incomplete but
        type-checks → ``ok=True, complete=False``
      * ``proof body + sorry`` with garbage tactics → ``ok=False``
        with some other error.

    ``closing_tactics`` is a set of tactics that close the proof when
    appearing as the LAST tactic in the body.
    """
    from src.evaluation.lean_verifier import LeanMessage

    def _err(body):
        return LeanMessage(severity="error", line=None, column=None, body=body)

    async def _verify(code, *, timeout=None):
        body = code.strip()
        with_sorry = body.endswith("sorry")
        no_sorry = body[: body.rfind("sorry")].rstrip() if with_sorry else body
        last_line = no_sorry.splitlines()[-1].strip() if no_sorry else ""
        if last_line in closing_tactics:
            if not with_sorry:
                return LeanVerifyResult(ok=True, complete=True)
            return LeanVerifyResult(
                ok=False,
                complete=False,
                errors=[_err("error: no goals to be solved")],
            )
        if not with_sorry:
            return LeanVerifyResult(
                ok=False,
                complete=False,
                errors=[_err("error: unsolved goals")],
            )
        if last_line.startswith("intro") or last_line.startswith("have"):
            # Treat these as typing partials.
            return LeanVerifyResult(ok=True, complete=False)
        return LeanVerifyResult(
            ok=False,
            complete=False,
            errors=[_err("error: unknown identifier")],
        )

    return _verify


def test_tree_search_handles_real_partial_proof_semantics():
    """Regression: production ``verify_lean_proof`` rejects partial
    proofs lacking ``sorry``. Pre-fix this trapped the search at depth
    1 because no candidate was ever classified as 'typing'. The new
    prover appends ``sorry`` before verifying and detects closure via
    the post-sorry "no goals to be solved" error. This test uses a
    sorry-aware mock that fails the OLD verify path and exercises the
    NEW one — it must observe the tree growing past depth 1."""

    async def sampler(config, prompt, n):
        if "intro a" not in prompt:
            # Root expansion: only typing candidates (no closing tactic
            # at root, forcing the loop to enqueue children and expand
            # again at depth 1).
            return ["intro a", "intro b"][:n]
        # Depth-1 expansion: one closing candidate.
        return ["rfl"][:n]

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="miniF2F",
            arm="control",
            problem_id="t_partial",
            statement="partial",
            formal_prefix="theorem t : True → True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=_sorry_aware_verify_factory(closing_tactics={"rfl"}),
            K=1,
            n_per_step=2,
            max_nodes=16,
            timeout_per_attempt_s=10.0,
            lean_timeout=10.0,
        )
    )
    assert record.pass_at_k is True
    a = record.attempts[0]
    # Tree must have grown past depth 1 — proof of fix: nodes_explored>=2.
    assert a.nodes_explored >= 2, (
        f"expected tree to grow past root, got nodes_explored={a.nodes_explored} "
        f"(reason={a.terminated_reason}); this means the with-sorry verify path "
        f"never triggered and the regression is back"
    )
    assert a.success_tactics is not None
    assert a.success_tactics[-1] == "rfl"
    assert len(a.success_tactics) == 2


def test_tree_search_does_not_promote_no_goals_with_hard_errors():
    """Regression: a with-sorry probe can contain both "no goals" and
    unrelated hard errors. The prover must validate the actual no-sorry
    proof before reporting success."""

    from src.evaluation.lean_verifier import LeanMessage

    def _err(body):
        return LeanMessage(severity="error", line=None, column=None, body=body)

    async def sampler(config, prompt, n):
        return ["apply MissingLemma"][:n]

    async def verifier(code, *, timeout=None):
        if code.strip().endswith("sorry"):
            return LeanVerifyResult(
                ok=False,
                complete=False,
                errors=[
                    _err("Unknown identifier `MissingLemma`"),
                    _err("No goals to be solved"),
                ],
            )
        return LeanVerifyResult(
            ok=False,
            complete=False,
            errors=[_err("Unknown identifier `MissingLemma`")],
        )

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="proofnet",
            arm="treatment",
            problem_id="false_positive",
            statement="false positive",
            formal_prefix="theorem t : True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            n_per_step=1,
            max_nodes=4,
            timeout_per_attempt_s=10.0,
            lean_timeout=10.0,
        )
    )

    assert record.pass_at_k is False
    a = record.attempts[0]
    assert a.success is False
    assert a.terminated_reason == "empty_queue"


def test_tree_search_deduplicates_repeated_states():
    """If the sampler emits the same tactic twice, only one child should
    be enqueued."""

    async def sampler(config, prompt, n):
        # Same tactic repeated — dedup should drop one.
        return ["intro h", "intro h"]

    async def verifier(code, *, timeout=None):
        return LeanVerifyResult(ok=True, complete=False)

    record = asyncio.run(
        prove_bfs_tree(
            benchmark="miniF2F",
            arm="control",
            problem_id="t4",
            statement="dedup",
            formal_prefix="theorem t : True → True := by",
            header="",
            model_config=_cfg(),
            tactic_sampler=sampler,
            verifier=verifier,
            K=1,
            n_per_step=2,
            max_nodes=3,
            timeout_per_attempt_s=30.0,
            lean_timeout=10.0,
        )
    )
    a = record.attempts[0]
    # Only one unique tactic was kept → only one child pushed per expansion.
    assert all(len(e.candidates) == 1 for e in a.expansions)
