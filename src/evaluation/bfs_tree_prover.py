"""BFS-Prover-V2 tree-search prover, at this panel's proof budget, no LeanDojo.

This is the *full* port of ByteDance-Seed/BFS-Prover-V2/src/search semantics
for a setup that doesn't have LeanDojo's per-tactic REPL. Compared to
``bfs_step_prover.py`` (greedy first-typing wins), this prover does what
the reference paper actually does: maintain a priority queue of partial
proof states, expand the best-scoring state, branch via N candidate
tactics per expansion, and **backtrack** through alternative siblings
when a path dead-ends.

Key differences from the BFS-V2 reference:
* The reference scores nodes by vLLM-emitted ``cumulative_logprob``. LM
  Studio's llama.cpp backend doesn't expose logprobs on
  ``/v1/chat/completions`` or ``/v1/completions``, so we use **depth as
  the tiebreaker** (deeper-first, FIFO among siblings) — this is what
  the reference reduces to with ``depth_reward=1.0`` and uniform
  logprob.
* Per-step verification calls ``lake env lean`` on the whole file rather
  than ``Dojo.run_tac`` on the live state. Slower per call but
  semantically equivalent for our budget (~5s warm per verify).

Search loop (per attempt):
    seen = {root}
    queue = [(priority(root), root)]
    while queue and budget remaining:
        state = queue.pop_highest_priority()
        candidates = sampler(state, n=n_per_step)
        verify each candidate in parallel
        for tac, verdict in candidates:
            if verdict.complete: return success(state + [tac])
            if verdict.ok and not seen:
                push child to queue
        expanded_nodes += 1
"""

from __future__ import annotations

import asyncio
import heapq
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import re

from src.evaluation.bfs_step_prover import (
    BFS_SEPARATOR,
    TacticSampler,
    assemble_partial_proof,
    assemble_proof,
    build_state_prompt,
    is_buggy_tactic,
    parse_tactic,
)
from src.evaluation.lean_verifier import LeanVerifyResult, verify_lean_proof
from src.evaluation.model_runner import ModelConfig
from src.evaluation.multi_turn_prover import DEFAULT_HEADER


# Matches Lean 4's "No goals to be solved" error, emitted when the
# proof body has already closed the goal and the trailing ``sorry`` we
# append has nothing to discharge. Match is case-insensitive and
# tolerates whitespace variation.
_NO_GOALS_RE = re.compile(r"no\s+goals(?:\s+to\s+be\s+solved)?", re.IGNORECASE)


def _is_no_goals_error(verdict: LeanVerifyResult) -> bool:
    """True iff ``verdict`` reports the post-sorry "no goals" error.

    A partial proof's appended ``sorry`` triggers this error only when
    the tactics preceding ``sorry`` have already discharged the goal.
    The tree-search verifier uses this as a one-shot signal for proof
    closure (avoids a second verify call on the no-sorry version).
    """
    if verdict.ok:
        return False
    return any(_NO_GOALS_RE.search(err.body or "") for err in verdict.errors)


# ---------------------------------------------------------------------------
#  Search state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofState:
    """Immutable, hashable node in the BFS search tree.

    ``tactics`` is the sequence of accepted tactics from the root, in
    application order. Two states are equal iff their tactic sequences
    match — we use this for the ``seen`` set to suppress cycles and
    duplicate expansion of the same partial proof.
    """

    tactics: Tuple[str, ...]
    depth: int

    @classmethod
    def root(cls) -> "ProofState":
        return cls(tactics=(), depth=0)

    def extend(self, tactic: str) -> "ProofState":
        return ProofState(tactics=self.tactics + (tactic,), depth=self.depth + 1)


@dataclass
class ExpansionRecord:
    """Per-expansion telemetry, for the JSONL summary."""

    expansion_index: int
    parent_depth: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class TreeAttemptRecord:
    """One independent BFS attempt (one seed of the priority queue)."""

    repeat_index: int
    expansions: List[ExpansionRecord] = field(default_factory=list)
    success: bool = False
    success_tactics: Optional[List[str]] = None
    final_proof: Optional[str] = None
    nodes_explored: int = 0
    nodes_queued_total: int = 0
    elapsed_seconds: float = 0.0
    terminated_reason: Optional[str] = None  # "proved" | "budget" | "empty_queue" | "timeout"


@dataclass
class TreeProblemAttempt:
    benchmark: str
    arm: str
    problem_id: str
    model_label: str
    provider_slug: str
    statement: str
    formal_prefix: str
    attempts: List[TreeAttemptRecord] = field(default_factory=list)
    pass_at_k: bool = False
    min_steps_to_success: Optional[int] = None
    total_elapsed_seconds: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "arm": self.arm,
            "problem_id": self.problem_id,
            "model": self.model_label,
            "provider_slug": self.provider_slug,
            "statement": self.statement,
            "formal_prefix": self.formal_prefix,
            "paradigm": "tree_search",
            "pass_at_k": self.pass_at_k,
            "min_steps_to_success": self.min_steps_to_success,
            "min_turns_to_success": self.min_steps_to_success,  # alias for aggregator
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "attempts": [
                {
                    "repeat_index": a.repeat_index,
                    "success": a.success,
                    "success_tactics": a.success_tactics,
                    "final_proof": a.final_proof,
                    "nodes_explored": a.nodes_explored,
                    "nodes_queued_total": a.nodes_queued_total,
                    "elapsed_seconds": a.elapsed_seconds,
                    "terminated_reason": a.terminated_reason,
                    "expansion_diagnostics": [
                        {
                            "expansion_index": e.expansion_index,
                            "parent_depth": e.parent_depth,
                            "elapsed_seconds": e.elapsed_seconds,
                            "candidates": e.candidates,
                        }
                        for e in a.expansions
                    ],
                }
                for a in self.attempts
            ],
        }


# ---------------------------------------------------------------------------
#  Best-first search loop
# ---------------------------------------------------------------------------


def _priority_key(state: ProofState, expansion_seq: int) -> Tuple[int, int]:
    """Heap priority: prefer deeper states first (negative depth makes
    Python's min-heap behave as max-heap on depth); FIFO break ties by
    insertion sequence so a smaller ``expansion_seq`` wins among nodes
    of the same depth.

    Without logprob signals from LM Studio we can't reproduce BFS-V2's
    ``cumulative_logprob`` priority exactly; depth-first with FIFO
    siblings is the reference's behaviour with ``depth_reward=1`` and
    uniform logprob (see prover_manager.py:578-584).
    """
    return (-state.depth, expansion_seq)


# Caps the number of concurrent ``lake env lean`` subprocesses launched
# per expansion. Each Lean process maps in ~20 GB of Mathlib OLEAN data
# (mostly shared/file-backed, but resident memory still grows with the
# typing/elaboration workload). Without a cap, an expansion with n_per_step=8
# would spin up 8 lean processes simultaneously and push the machine into
# memory pressure — empirically we lost three campaigns at 14:36 on
# 2026-05-20 to macOS SIGTERMing the python parent under VM pressure with
# 60+ GB of stuck-lean ghosts. Override via env if you have headroom.
_LEAN_PARALLEL_PER_EXPANSION = max(
    1, int(os.getenv("LEAN_CONCURRENCY_PER_EXPANSION", "3"))
)


async def _expand_state(
    *,
    state: ProofState,
    formal_prefix: str,
    header: str,
    model_config: ModelConfig,
    sampler: TacticSampler,
    verifier: Callable[..., Awaitable[LeanVerifyResult]],
    n_per_step: int,
    lean_timeout: float,
    seen: Set[Tuple[str, ...]],
) -> Tuple[ExpansionRecord, Optional[Tuple[ProofState, str]], List[ProofState]]:
    """Sample tactic candidates for ``state``, verify each in parallel,
    and return:
      - a populated ``ExpansionRecord`` for telemetry,
      - either ``(final_state, final_proof)`` when a candidate closes
        the proof (the caller short-circuits the search), or ``None``,
      - a list of new (typed-but-not-closed) child states to enqueue.
    """
    started = time.monotonic()
    prompt = build_state_prompt(formal_prefix, state.tactics)
    candidate_tactics = await sampler(model_config, prompt, n_per_step)

    expansion = ExpansionRecord(
        expansion_index=-1,  # caller fills in
        parent_depth=state.depth,
    )

    async def _verify_candidate(tactic: str) -> Tuple[str, LeanVerifyResult, str]:
        # Verify the partial proof with `sorry` appended so Lean accepts
        # typing-but-not-closed bodies (ok=True, complete=False with a
        # `declaration uses 'sorry'` warning). When the tactic sequence
        # has *already* closed the goal, the appended sorry instead
        # triggers an "no goals to be solved" error — we detect that
        # below and re-tag the verdict as a closing proof.
        no_sorry_proof = assemble_proof(
            formal_prefix, list(state.tactics) + [tactic], header
        )
        probe_proof = assemble_partial_proof(
            formal_prefix, list(state.tactics) + [tactic], header
        )
        try:
            verdict = await verifier(probe_proof, timeout=lean_timeout)
        except asyncio.TimeoutError:
            verdict = LeanVerifyResult(
                ok=False, complete=False,
                system_error=f"verify_timeout after {lean_timeout}s",
            )
            return tactic, verdict, no_sorry_proof
        except Exception as exc:  # pragma: no cover
            verdict = LeanVerifyResult(
                ok=False, complete=False,
                system_error=f"verify_exception: {type(exc).__name__}: {exc}",
            )
            return tactic, verdict, no_sorry_proof

        if _is_no_goals_error(verdict):
            # The appended `sorry` can report "no goals to be solved" when
            # the candidate has already closed the proof. It can also be
            # accompanied by unrelated hard errors (unknown identifiers,
            # type mismatches, etc.). Never promote that probe verdict
            # directly; validate the actual no-sorry proof before declaring
            # success.
            try:
                closed_verdict = await verifier(no_sorry_proof, timeout=lean_timeout)
            except asyncio.TimeoutError:
                closed_verdict = LeanVerifyResult(
                    ok=False,
                    complete=False,
                    system_error=f"verify_timeout after {lean_timeout}s",
                )
            except Exception as exc:  # pragma: no cover
                closed_verdict = LeanVerifyResult(
                    ok=False,
                    complete=False,
                    system_error=f"verify_exception: {type(exc).__name__}: {exc}",
                )
            return tactic, closed_verdict, no_sorry_proof

        # Typing partial (ok=True, complete=False thanks to the sorry
        # warning) or a true dead-end (ok=False with unrelated errors).
        # Either way return the no-sorry assembled proof for the
        # candidate-record's `final_proof` field so a downstream
        # consumer never sees the probe-only sorry suffix.
        return tactic, verdict, no_sorry_proof

    # Deduplicate child states before verifying — saves wasted Lean calls.
    unique_tactics: List[str] = []
    for tac in candidate_tactics:
        key = state.tactics + (tac,)
        if key in seen or tac in unique_tactics:
            continue
        unique_tactics.append(tac)

    # Cap concurrent Lean processes — see _LEAN_PARALLEL_PER_EXPANSION above
    # for the memory-pressure rationale.
    sem = asyncio.Semaphore(_LEAN_PARALLEL_PER_EXPANSION)

    async def _bounded(t):
        async with sem:
            return await _verify_candidate(t)

    verifications = await asyncio.gather(*[_bounded(t) for t in unique_tactics])

    closing: Optional[Tuple[ProofState, str]] = None
    new_children: List[ProofState] = []
    for tac, verdict, proof_code in verifications:
        child_tactics = state.tactics + (tac,)
        seen.add(child_tactics)
        expansion.candidates.append({
            "tactic": tac,
            "ok": verdict.ok,
            "complete": verdict.complete,
            "summary": verdict.summary(max_errors=2),
            "system_error": verdict.system_error,
            "verify_time": verdict.verify_time,
        })
        if verdict.complete:
            closing = (state.extend(tac), proof_code)
            # Short-circuit further evaluation but keep the loop running
            # so the rest of the expansion record is populated.
            break
        if verdict.ok:
            new_children.append(state.extend(tac))

    expansion.elapsed_seconds = time.monotonic() - started
    return expansion, closing, new_children


async def _run_one_attempt(
    *,
    repeat_index: int,
    formal_prefix: str,
    header: str,
    model_config: ModelConfig,
    sampler: TacticSampler,
    verifier: Callable[..., Awaitable[LeanVerifyResult]],
    n_per_step: int,
    max_nodes: int,
    timeout_s: float,
    lean_timeout: float,
) -> TreeAttemptRecord:
    """Run a single best-first search attempt."""
    started = time.monotonic()
    record = TreeAttemptRecord(repeat_index=repeat_index)

    root = ProofState.root()
    seen: Set[Tuple[str, ...]] = {root.tactics}
    queue: List[Tuple[Tuple[int, int], int, ProofState]] = []
    expansion_seq = 0
    heapq.heappush(queue, (_priority_key(root, expansion_seq), expansion_seq, root))
    record.nodes_queued_total = 1

    while queue and record.nodes_explored < max_nodes:
        if time.monotonic() - started > timeout_s:
            record.terminated_reason = "timeout"
            break

        _, _, state = heapq.heappop(queue)
        record.nodes_explored += 1

        expansion, closing, new_children = await _expand_state(
            state=state,
            formal_prefix=formal_prefix,
            header=header,
            model_config=model_config,
            sampler=sampler,
            verifier=verifier,
            n_per_step=n_per_step,
            lean_timeout=lean_timeout,
            seen=seen,
        )
        expansion.expansion_index = record.nodes_explored
        record.expansions.append(expansion)

        if closing is not None:
            final_state, final_proof = closing
            record.success = True
            record.success_tactics = list(final_state.tactics)
            record.final_proof = final_proof
            record.terminated_reason = "proved"
            break

        for child in new_children:
            expansion_seq += 1
            heapq.heappush(queue, (_priority_key(child, expansion_seq), expansion_seq, child))
            record.nodes_queued_total += 1

    if record.terminated_reason is None:
        record.terminated_reason = "budget" if record.nodes_explored >= max_nodes else "empty_queue"

    record.elapsed_seconds = time.monotonic() - started
    return record


async def prove_bfs_tree(
    *,
    benchmark: str,
    arm: str,
    problem_id: str,
    statement: str,
    formal_prefix: str,
    header: str = DEFAULT_HEADER,
    model_config: ModelConfig,
    tactic_sampler: TacticSampler,
    verifier: Optional[Callable[..., Awaitable[LeanVerifyResult]]] = None,
    K: int = 3,
    n_per_step: int = 16,
    max_nodes: int = 64,
    timeout_per_attempt_s: float = 600.0,
    lean_timeout: float = 90.0,
) -> TreeProblemAttempt:
    """Run ``K`` independent best-first search attempts.

    Each attempt seeds a fresh priority queue and explores up to
    ``max_nodes`` proof states (or ``timeout_per_attempt_s`` seconds).
    Stops the campaign cell as soon as ANY attempt finds a closing
    proof.
    """
    verify_fn = verifier or verify_lean_proof
    record = TreeProblemAttempt(
        benchmark=benchmark,
        arm=arm,
        problem_id=problem_id,
        model_label=model_config.label,
        provider_slug=model_config.provider_slug,
        statement=statement,
        formal_prefix=formal_prefix,
    )
    started = time.monotonic()

    for k in range(1, K + 1):
        attempt = await _run_one_attempt(
            repeat_index=k,
            formal_prefix=formal_prefix,
            header=header,
            model_config=model_config,
            sampler=tactic_sampler,
            verifier=verify_fn,
            n_per_step=n_per_step,
            max_nodes=max_nodes,
            timeout_s=timeout_per_attempt_s,
            lean_timeout=lean_timeout,
        )
        record.attempts.append(attempt)
        if attempt.success:
            record.pass_at_k = True
            steps_used = len(attempt.success_tactics or [])
            if (
                record.min_steps_to_success is None
                or steps_used < record.min_steps_to_success
            ):
                record.min_steps_to_success = steps_used
            break

    record.total_elapsed_seconds = time.monotonic() - started
    return record
