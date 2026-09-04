"""BFS-Prover-V2 aligned tactic-step prover, at this panel's proof budget.

The conventions mirror ByteDance-Seed/BFS-Prover-V2/src/search:
- Prompt format ``f"{tactic_state}:::"`` (prover_manager.py:166).
- Sampling temperature 0.7, top_p 1.0, ``n_sampling_search`` candidates per
  expansion (launcher.py defaults).
- Tactic filters identical to the reference (prover_manager.py:174-184):
  drop ``sorry``/``admit``/``native_decide``; drop ``rcases``/``cases'``/
  ``simpa`` combined with ``?_``; drop ``simpa`` with a bare ``_``.

Adaptations forced by that budget:
- No LeanDojo REPL on this machine, so we verify each step by re-typing
  the whole file with ``lake env lean`` instead of running the tactic in
  a live ``Dojo`` session.
- No logprob-driven priority queue; we expand greedily, picking the first
  candidate that closes the proof, else the first that types without
  errors, else abandoning the attempt. This matches what BFS-V2 reduces
  to once the priority queue is replaced by a single-trace greedy walk.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.evaluation.lean_verifier import LeanVerifyResult, verify_lean_proof
from src.evaluation.model_runner import ModelConfig
from src.evaluation.multi_turn_prover import DEFAULT_HEADER

BFS_SEPARATOR = ":::"
BANNED_SUBSTRINGS = ("sorry", "admit", "native_decide")
BUGGY_KEYWORDS = ("rcases", "cases'", "simpa")


# ---------------------------------------------------------------------------
#  Prompt + parsing
# ---------------------------------------------------------------------------


def build_state_prompt(formal_prefix: str, tactics: Sequence[str]) -> str:
    """Build the BFS-V2 prompt ``{state}:::``.

    Without LeanDojo we cannot show the live proof state, so we
    approximate it by concatenating the theorem prefix and the tactics
    accepted so far. BFS-Prover-V2 is trained on a similar text form, so
    this is the substitute that fits the budget.
    """
    body = formal_prefix.rstrip()
    if tactics:
        body = body + "\n  " + "\n  ".join(t.strip() for t in tactics if t.strip())
    return f"{body}{BFS_SEPARATOR}"


def parse_tactic(raw: str) -> str:
    """Trim whitespace; the LM Studio ``stop`` list already cut the tail."""
    return (raw or "").strip()


def is_buggy_tactic(tactic: str) -> bool:
    """Match the BFS-V2 reference filter (prover_manager.py:174-184)."""
    if not tactic:
        return True
    if any(s in tactic for s in BANNED_SUBSTRINGS):
        return True
    if any(k in tactic for k in BUGGY_KEYWORDS) and "?_" in tactic:
        return True
    if "simpa" in tactic and (
        " _" in tactic or "_ " in tactic or "_," in tactic or ",_" in tactic
    ):
        return True
    return False


def assemble_proof(formal_prefix: str, tactics: Sequence[str], header: str) -> str:
    """Bundle (header, theorem prefix, tactics) into a complete ``.lean`` source."""
    body = formal_prefix.rstrip()
    if not body.endswith("by") and ":= by" not in body:
        body = body + " := by"
    indented = "\n  ".join(t.strip() for t in tactics if t.strip())
    return f"{header.rstrip()}\n\n{body}\n  {indented}\n"


def assemble_partial_proof(
    formal_prefix: str, tactics: Sequence[str], header: str
) -> str:
    """Variant of :func:`assemble_proof` that appends ``sorry`` so a partial
    proof body type-checks under Lean's whole-file verifier.

    Whole-file ``lake env lean`` rejects an incomplete proof body with
    ``error: unsolved goals``; appending ``sorry`` turns the same body
    into a typing-but-uncertified proof (``ok=True, complete=False``
    with a ``declaration uses 'sorry'`` warning). The tree-search prover
    uses this to distinguish *typing* partials (enqueue as children)
    from *dead* partials (drop) without an extra round trip.

    When the appended ``sorry`` produces ``error: No goals to be solved``
    instead, that means the partial body has ALREADY closed the proof —
    callers should treat that signal as "complete" and reconstruct the
    final source via :func:`assemble_proof`.
    """
    base = assemble_proof(formal_prefix, tactics, header)
    return base.rstrip() + "\n  sorry\n"


# ---------------------------------------------------------------------------
#  Tactic-sampler protocol
# ---------------------------------------------------------------------------

#: A function that, given (model, prompt, n), returns up to ``n`` candidate
#: tactic strings already deduped and filtered. Constructed by the
#: orchestrator with a closure over the LM Studio client so the prover
#: stays backend-agnostic.
TacticSampler = Callable[[ModelConfig, str, int], Awaitable[List[str]]]


async def default_tactic_sampler(
    config: ModelConfig,
    client,  # AsyncOpenAI
    prompt: str,
    n: int,
    *,
    timeout_seconds: float = 60.0,
) -> List[str]:
    """Sample up to ``n`` candidate tactics, deduped and filtered.

    The BFS-V2 reference uses vLLM's ``SamplingParams(n=...)`` which
    returns multiple sequences from one engine invocation. LM Studio's
    llama.cpp backend ignores the OpenAI-spec ``n`` parameter and only
    ever returns a single choice, so we issue ``n`` parallel completion
    requests with the parallel-slot count the server was loaded with and
    aggregate the unique tactics.
    """
    base_request: Dict[str, Any] = {
        "model": config.provider_slug,
        "prompt": prompt,
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        base_request["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        base_request["top_p"] = config.top_p
    if config.seed is not None:
        base_request["seed"] = config.seed
    if config.stop:
        base_request["stop"] = list(config.stop)

    async def _one(sample_index: int) -> Optional[str]:
        request = dict(base_request)
        if config.seed is not None:
            request["seed"] = config.seed + sample_index
        try:
            response = await asyncio.wait_for(
                client.completions.create(**request),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        return parse_tactic(getattr(choices[0], "text", "") or "")

    results = await asyncio.gather(*[_one(i) for i in range(n)])
    seen: set = set()
    out: List[str] = []
    for text in results:
        if not text or text in seen or is_buggy_tactic(text):
            continue
        seen.add(text)
        out.append(text)
    return out


# ---------------------------------------------------------------------------
#  Per-step / per-attempt records
# ---------------------------------------------------------------------------


@dataclass
class StepCandidate:
    tactic: str
    verify: LeanVerifyResult
    elapsed_seconds: float
    accepted: bool = False  # True if this candidate was carried into the next step


@dataclass
class StepRecord:
    step_index: int  # 1-based within an attempt
    chosen_tactic: Optional[str]
    cumulative_proof: Optional[str]
    candidates: List[StepCandidate] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class StepAttemptRecord:
    repeat_index: int  # 1-based
    steps: List[StepRecord] = field(default_factory=list)
    success: bool = False
    success_step: Optional[int] = None
    final_proof: Optional[str] = None


@dataclass
class StepProblemAttempt:
    benchmark: str
    arm: str
    problem_id: str
    model_label: str
    provider_slug: str
    statement: str
    formal_prefix: str
    attempts: List[StepAttemptRecord] = field(default_factory=list)
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
            "paradigm": "step_level",
            "pass_at_k": self.pass_at_k,
            "min_steps_to_success": self.min_steps_to_success,
            # Alias so ``proof_orchestrator.summarize_proof_jsonl`` can fold
            # step-level and whole-proof rows into one Pass@K table.
            "min_turns_to_success": self.min_steps_to_success,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "attempts": [
                {
                    "repeat_index": a.repeat_index,
                    "success": a.success,
                    "success_step": a.success_step,
                    "steps_used": len(a.steps),
                    "final_proof": a.final_proof,
                    "step_diagnostics": [
                        {
                            "step_index": s.step_index,
                            "chosen_tactic": s.chosen_tactic,
                            "candidates": [
                                {
                                    "tactic": c.tactic,
                                    "ok": c.verify.ok,
                                    "complete": c.verify.complete,
                                    "summary": c.verify.summary(max_errors=2),
                                    "system_error": c.verify.system_error,
                                    "verify_time": c.verify.verify_time,
                                    "elapsed_seconds": c.elapsed_seconds,
                                    "accepted": c.accepted,
                                }
                                for c in s.candidates
                            ],
                            "elapsed_seconds": s.elapsed_seconds,
                        }
                        for s in a.steps
                    ],
                }
                for a in self.attempts
            ],
        }


# ---------------------------------------------------------------------------
#  The K × S_max × n-candidate search loop
# ---------------------------------------------------------------------------


async def prove_bfs_step(
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
    S_max: int = 6,
    n_per_step: int = 8,
    lean_timeout: float = 120.0,
) -> StepProblemAttempt:
    """Run ``K`` independent attempts, each up to ``S_max`` greedy steps.

    At every step we sample ``n_per_step`` candidate tactics in a single
    multi-sample call, filter using the BFS-V2 reference rules, and
    type-check each via the Lean verifier. The first candidate that
    closes the proof wins immediately; if none close, we keep the first
    that types and continue to the next step.
    """
    verify_fn = verifier or verify_lean_proof
    record = StepProblemAttempt(
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
        attempt = StepAttemptRecord(repeat_index=k)
        tactics: List[str] = []

        for s in range(1, S_max + 1):
            step_started = time.monotonic()
            prompt = build_state_prompt(formal_prefix, tactics)
            candidates_raw = await tactic_sampler(model_config, prompt, n_per_step)

            if not candidates_raw:
                attempt.steps.append(
                    StepRecord(
                        step_index=s,
                        chosen_tactic=None,
                        cumulative_proof=None,
                        elapsed_seconds=time.monotonic() - step_started,
                    )
                )
                break

            candidates: List[StepCandidate] = []
            chosen_tactic: Optional[str] = None
            chosen_proof: Optional[str] = None
            chosen_idx: Optional[int] = None
            closed_now = False

            for tactic in candidates_raw:
                v_started = time.monotonic()
                proof = assemble_proof(formal_prefix, tactics + [tactic], header)
                verdict = await verify_fn(proof, timeout=lean_timeout)
                candidates.append(
                    StepCandidate(
                        tactic=tactic,
                        verify=verdict,
                        elapsed_seconds=time.monotonic() - v_started,
                    )
                )
                if verdict.complete:
                    # Proof closure beats partial progress — short-circuit.
                    chosen_tactic = tactic
                    chosen_proof = proof
                    chosen_idx = len(candidates) - 1
                    closed_now = True
                    break
                if chosen_idx is None and verdict.ok:
                    # Remember the first candidate that at least type-checks;
                    # keep scanning in case a later candidate closes the goal.
                    chosen_tactic = tactic
                    chosen_proof = proof
                    chosen_idx = len(candidates) - 1

            if chosen_idx is not None:
                candidates[chosen_idx].accepted = True

            attempt.steps.append(
                StepRecord(
                    step_index=s,
                    chosen_tactic=chosen_tactic,
                    cumulative_proof=chosen_proof,
                    candidates=candidates,
                    elapsed_seconds=time.monotonic() - step_started,
                )
            )

            if closed_now:
                attempt.success = True
                attempt.success_step = s
                attempt.final_proof = chosen_proof
                break
            if chosen_tactic is None:
                break  # no candidate even types — abandon this attempt
            tactics.append(chosen_tactic)

        record.attempts.append(attempt)
        if attempt.success:
            record.pass_at_k = True
            steps_used = attempt.success_step or len(attempt.steps)
            if (
                record.min_steps_to_success is None
                or steps_used < record.min_steps_to_success
            ):
                record.min_steps_to_success = steps_used
            break  # Pass@K satisfied; remaining attempts skipped.

    record.total_elapsed_seconds = time.monotonic() - started
    return record
