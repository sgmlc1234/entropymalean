"""Multi-turn proof-verification evaluator.

Protocol (cf. references/eval_refs/eval_design.md):

    for k in 1..K:                                    # K independent attempts
        proof_1 = model.generate(statement, header)
        result_1 = verify_lean(proof_1)
        if result_1.complete: break

        for t in 2..T_max:
            proof_t = model.refine(statement, header,
                                   prior_proofs=[(p, errors)])
            result_t = verify_lean(proof_t)
            if result_t.complete: break

    Pass@K = any of K attempts reached complete proof within T_max turns.

The implementation is API-friendly (every turn is a chat completion) and
treats the Lean verifier as the loop oracle. Each turn is logged so
``orchestrator.py`` can compute turn distributions and refinement-yield
statistics.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional

import langsmith as ls

from src.evaluation.lean_verifier import (
    LeanVerifyResult,
    contains_sorry,
    extract_lean_block,
    verify_lean_proof,
)
from src.evaluation.model_runner import ModelConfig


# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a Lean 4 theorem prover. Given a natural-language problem "
    "statement and a Lean 4 theorem prefix, produce a complete proof. "
    "Reply with exactly one fenced code block tagged ```lean4 ... ``` "
    "containing the full theorem statement together with the proof body. "
    "Do not include explanatory prose outside the code block. Do not use "
    "the `sorry` tactic. Do not change the theorem statement unless the "
    "Lean diagnostics show a syntax/header-only mismatch."
)


INITIAL_USER_TEMPLATE = (
    "Problem (natural language):\n{statement}\n\n"
    "Lean 4 setup (use as-is; you may add `import` lines if strictly "
    "necessary):\n```lean4\n{header}\n```\n\n"
    "Lean 4 theorem to prove (complete the `:= by ...` body):\n"
    "```lean4\n{formal_prefix}\n```\n\n"
    "{proof_context_block}"
    "Reply with a single ```lean4 ... ``` block containing the full "
    "theorem and a closing proof."
)


REFINE_USER_TEMPLATE = (
    "Your previous Lean 4 attempt failed verification.\n\n"
    "Previous proof:\n```lean4\n{previous_proof}\n```\n\n"
    "Lean reported the following diagnostics (truncated to the first "
    "{max_errors} errors):\n{diagnostics}\n\n"
    "The theorem statement is fixed. Prefer changing imports, local proof "
    "terms, tactics, or intermediate rewrites before changing the statement.\n\n"
    "{premise_context_block}"
    "Repair the proof. Reply with a single ```lean4 ... ``` block "
    "containing the corrected full theorem and a closing proof. Do not "
    "use `sorry`."
)


# ---------------------------------------------------------------------------
#  Goedel-Prover-V2 prompt dialect
# ---------------------------------------------------------------------------
#
# The Goedel-V2 HF model card prescribes a specific user prompt and no
# system message. The model is trained to emit a "detailed proof plan"
# before the Lean code block; suppressing that preamble drops Pass@K.
# We therefore use these templates whenever the panel resolves a model
# with ``prompt_style="goedel_v2"``.

GOEDEL_V2_SYSTEM_PROMPT = ""

GOEDEL_V2_INITIAL_USER_TEMPLATE = (
    # Mirror Goedel-Prover-V2 reference (`src/utils.py:DeepSeekCoTHandler`):
    # the formal statement is presented with a trailing ``sorry`` so the
    # model — trained on ``... := by sorry`` shells — fills in the proof
    # in the slot it expects. No system message, no closing "Conclude
    # with..." instruction (the model was not trained on that suffix).
    # The ``formal_prefix`` is normalized upstream by
    # :func:`_normalize_formal_prefix_for_goedel_v2` to guarantee the
    # ``:= ... sorry`` suffix regardless of dataset convention.
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n{header}\n\n{formal_prefix}\n```\n\n"
    "Before producing the Lean 4 code to formally prove the given "
    "theorem, provide a detailed proof plan outlining the main proof "
    "steps and strategies. The plan should highlight key ideas, "
    "intermediate lemmas, and proof structures that will guide the "
    "construction of the final formal proof.\n\n"
    "{proof_context_block}"
)


def _normalize_formal_prefix_for_goedel_v2(prefix: str) -> str:
    """Goedel-V2 prompts expect ``theorem ... := by sorry`` (or ``:= sorry``).

    The dataset conventions differ:
      - miniF2F treatments end with ``:= by`` (we need to add ``sorry``)
      - PutnamBench rows end with ``:=\\nsorry`` (already there — leave it)
      - some EMG-2 rows end with a complete proof (no sorry, no := by) —
        we still want a sorry slot for the model to overwrite.

    Returns ``prefix`` with a trailing ``sorry`` ensured, exactly once.
    """
    s = prefix.rstrip()
    if not s:
        return s
    # Already ends with ``sorry`` (any form): keep as-is.
    if s.endswith("sorry"):
        return s
    if s.endswith(":= by"):
        return s + " sorry"
    if s.endswith(":="):
        return s + " sorry"
    # Statement-only rows need an explicit proof slot.
    return s + " := by sorry"


def _extract_fenced_lean_block(text: str) -> Optional[str]:
    """Goedel reference extraction: only fenced Lean blocks are accepted.

    ``InferenceHandler.extrac_code`` in the Goedel-Prover-V2 repo returns
    ``None`` when no `````lean4`` block is present.  Falling back to raw text
    is unsafe for CoT models because proof plans are ordinary markdown.
    """
    if not text:
        return None
    matches = list(_GOEDEL_FENCE_RE.finditer(text))
    if matches:
        return matches[-1].group("body").rstrip()
    return None


_GOEDEL_FENCE_RE = re.compile(
    r"```lean4?\s*\n(?P<body>.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)

_LEAN_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
_DECL_WITH_BY_RE = re.compile(
    r"((?:^|\s)(?:theorem|lemma|def)\s+.*?:=\s*by\b)",
    flags=re.DOTALL,
)
_DECL_WITH_SORRY_SLOT_RE = re.compile(
    r"((?:^|\s)(?:theorem|lemma|def)\s+.*?:=\s*by\s*sorry\b)",
    flags=re.DOTALL,
)


def _remove_lean_comments(text: str) -> str:
    """Port of the Goedel utility used before statement replacement."""
    text = _LEAN_BLOCK_COMMENT_RE.sub("", text or "")
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines()).strip()


def _goedel_sorry_statement(prefix: str) -> str:
    """Return a Goedel-style original statement ending in ``:= by sorry``."""
    s = (prefix or "").strip()
    if not s:
        return ""
    if re.search(r":=\s*by\s*sorry\s*$", s):
        return s
    if re.search(r":=\s*by\s*$", s):
        return re.sub(r":=\s*by\s*$", ":= by sorry", s)
    if re.search(r":=\s*$", s):
        return re.sub(r":=\s*$", ":= by sorry", s)
    return s + " := by sorry"


def _drop_trailing_sorry(statement: str) -> str:
    return re.sub(r"\s*sorry\s*$", "", statement.rstrip())


def _goedel_replace_statement_in_proof(
    original_prefix: str,
    generated_code: str,
) -> tuple[Optional[str], Optional[str]]:
    """Use the model's proof body but the benchmark's original statement.

    This mirrors ``DeepSeekCoTHandler.problem_check`` in the Goedel-Prover-V2
    reference: generated theorem headers are not trusted.  The verifier sees
    ``original theorem ... := by`` plus whatever body the model wrote after
    its own declaration's ``:= by``.
    """
    forbidden = [
        token
        for token in ("apply?", "exact?", "native_decide")
        if token in generated_code
    ]
    if forbidden:
        return None, (
            "model used forbidden evaluation tactic(s): "
            + ", ".join(f"`{token}`" for token in forbidden)
        )

    statement = _remove_lean_comments(_goedel_sorry_statement(original_prefix))
    statement_match = _DECL_WITH_SORRY_SLOT_RE.search(statement)
    if statement_match is None:
        return None, "could not locate original declaration ending in `:= by sorry`"

    proof = _remove_lean_comments(generated_code)
    proof_match = _DECL_WITH_BY_RE.search(proof)
    if proof_match is None:
        stripped = proof.strip()
        if stripped.startswith("by"):
            body = stripped[len("by") :]
        else:
            return None, "could not locate generated declaration ending in `:= by`"
    else:
        body = proof[proof_match.end() :]

    original_decl = _drop_trailing_sorry(statement[: statement_match.end()])
    return original_decl + body, None

GOEDEL_V2_REFINE_USER_TEMPLATE = (
    "Your previous Lean 4 attempt failed verification.\n\n"
    "Previous proof:\n```lean4\n{previous_proof}\n```\n\n"
    "Lean reported the following diagnostics (truncated to the first "
    "{max_errors} errors):\n{diagnostics}\n\n"
    "Briefly explain what went wrong and revise the proof plan, then "
    "conclude with a single ```lean4 ... ``` block containing the "
    "corrected full theorem and a closing proof. The theorem statement "
    "is fixed; do not use `sorry`.\n\n"
    "{premise_context_block}"
)


# ---------------------------------------------------------------------------
#  Goedel-Prover-V1 / DSP-V1.5 prompt dialect
# ---------------------------------------------------------------------------
#
# Goedel-V1 SFT + DeepSeek-Prover-V1.5 share a minimalist prompt: no
# system message, no reasoning preamble, just "Complete the following
# Lean 4 code". The model emits the proof body directly. Best-of-N
# sampling (n=32 in the official Goedel V1 eval) compensates for the
# narrower decoding budget (max_tokens=2048). Use this style when the
# V2 reasoning preamble overflows the budget or drives the 8B Q4
# checkpoint into sorry-fallbacks.

GOEDEL_V1_SYSTEM_PROMPT = ""

GOEDEL_V1_INITIAL_USER_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n{header}\n\n{formal_prefix}\n```\n\n"
    "Return exactly one fenced ```lean4 ... ``` block containing the "
    "complete theorem proof. Do not include proof plans, explanations, "
    "markdown outside the code block, or `sorry`."
)

GOEDEL_V1_REFINE_USER_TEMPLATE = (
    "Your previous Lean 4 attempt failed verification:\n\n"
    "```lean4\n{previous_proof}\n```\n\n"
    "Lean diagnostics (first {max_errors} errors):\n{diagnostics}\n\n"
    "{premise_context_block}"
    "Provide a corrected complete Lean 4 proof inside a single "
    "```lean4 ... ``` block. Do not use `sorry`."
)


def _select_templates(prompt_style: str):
    """Return ``(system_prompt, initial_template, refine_template)`` for the dialect."""
    if prompt_style == "goedel_v2":
        return (
            GOEDEL_V2_SYSTEM_PROMPT,
            GOEDEL_V2_INITIAL_USER_TEMPLATE,
            GOEDEL_V2_REFINE_USER_TEMPLATE,
        )
    if prompt_style == "goedel_v1":
        return (
            GOEDEL_V1_SYSTEM_PROMPT,
            GOEDEL_V1_INITIAL_USER_TEMPLATE,
            GOEDEL_V1_REFINE_USER_TEMPLATE,
        )
    return SYSTEM_PROMPT, INITIAL_USER_TEMPLATE, REFINE_USER_TEMPLATE


DEFAULT_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 400000\n"
    "set_option autoImplicit false\n"
    "open BigOperators Real Nat Topology Rat"
)


# ---------------------------------------------------------------------------
#  Per-turn / per-attempt records
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    turn_index: int                  # 1-based within an attempt
    proof_code: str                  # what we sent to Lean
    raw_model_text: str              # full model response (for trace)
    verify: LeanVerifyResult
    elapsed_seconds: float           # full turn including model call
    premise_pack: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    #: Replies thrown away before this turn was scored, with why. Kept so the
    #: cost of degenerate sampling stays measurable instead of disappearing
    #: into a turn count that looks healthy.
    discarded_responses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AttemptRecord:
    repeat_index: int                # 1-based outer attempt
    turns: List[TurnRecord] = field(default_factory=list)
    success: bool = False
    success_turn: Optional[int] = None
    final_proof: Optional[str] = None
    terminated_reason: Optional[str] = None


@dataclass
class ProblemAttempt:
    benchmark: str
    arm: str
    problem_id: str
    model_label: str
    provider_slug: str
    statement: str
    formal_prefix: str
    attempts: List[AttemptRecord] = field(default_factory=list)
    pass_at_k: bool = False
    min_turns_to_success: Optional[int] = None
    total_elapsed_seconds: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        def turn_outcome(turn: TurnRecord) -> str:
            if turn.verify.complete:
                return "valid_no_sorry"
            system_error = turn.verify.system_error or ""
            if system_error.startswith("model_timeout:"):
                return "model_timeout"
            if system_error.startswith("model_exception:"):
                return "model_error"
            if system_error.startswith("model returned no fenced lean code"):
                return "extraction_error"
            if system_error.startswith("model used forbidden"):
                return "policy_rejection"
            if system_error:
                return "verifier_system_error"
            if turn.verify.ok and contains_sorry(turn.proof_code):
                return "valid_with_sorry"
            if turn.verify.ok:
                return "valid_but_incomplete"
            return "lean_error"

        return {
            "benchmark": self.benchmark,
            "arm": self.arm,
            "problem_id": self.problem_id,
            "model": self.model_label,
            "provider_slug": self.provider_slug,
            "statement": self.statement,
            "formal_prefix": self.formal_prefix,
            "pass_at_k": self.pass_at_k,
            "min_turns_to_success": self.min_turns_to_success,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "attempts": [
                {
                    "repeat_index": a.repeat_index,
                    "success": a.success,
                    "success_turn": a.success_turn,
                    "turns_used": len(a.turns),
                    # Turns are the budget; resamples are what the budget nearly
                    # leaked to. Reported next to each other so a healthy-looking
                    # turn count cannot hide a run that mostly sampled nothing.
                    "discarded_response_count": sum(
                        len(t.discarded_responses) for t in a.turns
                    ),
                    "final_proof": a.final_proof,
                    "terminated_reason": a.terminated_reason,
                    "turn_diagnostics": [
                        {
                            "turn_index": t.turn_index,
                            "candidate_proof": t.proof_code,
                            "raw_model_text": t.raw_model_text,
                            "finish_reason": t.finish_reason,
                            "discarded_responses": t.discarded_responses,
                            "outcome": turn_outcome(t),
                            "ok": t.verify.ok,
                            "complete": t.verify.complete,
                            "summary": t.verify.summary(),
                            "system_error": t.verify.system_error,
                            "verify_time": t.verify.verify_time,
                            "elapsed_seconds": t.elapsed_seconds,
                            "premise_pack": t.premise_pack or {},
                        }
                        for t in a.turns
                    ],
                }
                for a in self.attempts
            ],
        }


# ---------------------------------------------------------------------------
#  Model interface
# ---------------------------------------------------------------------------


ModelCallable = Callable[
    [ModelConfig, str, str, float, Optional[int]],  # config, system, user, temperature, max_tokens
    Awaitable["ModelTurnResponse"],
]

PremiseContextProvider = Callable[..., Awaitable[Dict[str, Any]]]


@dataclass
class ModelTurnResponse:
    raw_text: str
    finish_reason: Optional[str] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


async def _default_model_call(
    config: ModelConfig,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    *,
    client,
    timeout_seconds: float,
) -> ModelTurnResponse:
    """Thin wrapper over the OpenRouter client used by ``model_runner.py``.

    Kept here so unit tests can pass a mock callable while production calls
    flow through the same panel that ``model_runner.run_model_panel`` uses.
    """
    started = time.monotonic()
    try:
        if getattr(config, "paradigm", "chat") == "completion":
            # Tactic-step provers (BFS-Prover V2 etc.) expect a raw text prompt
            # and would be corrupted by chat-template wrapping.
            request: Dict[str, Any] = {
                "model": config.provider_slug,
                "prompt": user,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            if config.top_p is not None:
                request["top_p"] = config.top_p
            if config.seed is not None:
                request["seed"] = config.seed
            if getattr(config, "stop", None):
                request["stop"] = list(config.stop)
            response = await asyncio.wait_for(
                client.completions.create(**request, timeout=timeout_seconds),
                timeout=timeout_seconds,
            )
            choice = response.choices[0]
            return ModelTurnResponse(
                raw_text=getattr(choice, "text", "") or "",
                finish_reason=getattr(choice, "finish_reason", None),
                elapsed_seconds=time.monotonic() - started,
            )

        request = {
            "model": config.provider_slug,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.seed is not None:
            request["seed"] = config.seed
        if getattr(config, "stop", None):
            request["stop"] = list(config.stop)
        response = await asyncio.wait_for(
            client.chat.completions.create(**request, timeout=timeout_seconds),
            timeout=timeout_seconds,
        )
        choice = response.choices[0]
        return ModelTurnResponse(
            raw_text=choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None),
            elapsed_seconds=time.monotonic() - started,
        )
    except asyncio.TimeoutError:
        return ModelTurnResponse(
            raw_text="",
            finish_reason="timeout",
            elapsed_seconds=time.monotonic() - started,
            error=f"timeout after {timeout_seconds}s",
        )
    except Exception as exc:
        return ModelTurnResponse(
            raw_text="",
            finish_reason="error",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
#  The multi-turn loop
# ---------------------------------------------------------------------------


async def prove_with_refinement(
    *,
    benchmark: str,
    arm: str,
    problem_id: str,
    statement: str,
    formal_prefix: str,
    header: str = DEFAULT_HEADER,
    model_config: ModelConfig,
    model_call: Optional[Callable[..., Awaitable[ModelTurnResponse]]] = None,
    verifier: Optional[Callable[..., Awaitable[LeanVerifyResult]]] = None,
    K: int = 3,
    T_max: int = 4,
    n_parallel: int = 1,
    first_temperature: float = 1.0,
    refine_temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    lean_timeout: float = 300.0,
    max_errors_in_prompt: int = 5,
    #: Extra model calls allowed per turn when the reply carries no Lean at all.
    max_resamples: int = 3,
    proof_context: str = "",
    trace_run_prefix: Optional[str] = None,
    premise_context_provider: Optional[PremiseContextProvider] = None,
) -> ProblemAttempt:
    """Run K independent attempts, each up to T_max verifier-driven turns.

    ``n_parallel`` activates best-of-N sampling within each (attempt,
    turn) cell: ``n_parallel`` independent completions are generated in
    parallel, each fed to the Lean verifier, and the first ``complete``
    candidate wins. If no candidate closes, the first ``ok`` (types but
    not closed) is recorded so the refine turn can build on it.
    Matches the Goedel-Prover-V1 / DSP-V1.5 best-of-N protocol.
    """
    if model_call is None:  # pragma: no cover - production path needs a client
        raise ValueError("model_call must be provided (a wrapper over the chat client)")
    verify_fn = verifier or verify_lean_proof

    record = ProblemAttempt(
        benchmark=benchmark,
        arm=arm,
        problem_id=problem_id,
        model_label=model_config.label,
        provider_slug=model_config.provider_slug,
        statement=statement,
        formal_prefix=formal_prefix,
    )
    started_problem = time.monotonic()

    def _ensure_header(proof_code: str) -> str:
        """Goedel/DSP-style chat models emit the proof inside a fenced
        ``lean4`` block that often omits the imports/header (since they
        were already in the prompt). Lean's CLI cannot resolve Mathlib
        symbols without those imports, so we prepend the row header
        whenever the extracted code doesn't already start with one.
        """
        stripped = proof_code.lstrip()
        if stripped.startswith("import") or stripped.startswith("--"):
            return proof_code
        return f"{header.rstrip()}\n\n{proof_code}"

    async def run_verifier(proof_code: str) -> LeanVerifyResult:
        proof_code = _ensure_header(proof_code)
        try:
            return await verify_fn(proof_code, timeout=lean_timeout)
        except asyncio.TimeoutError:
            return LeanVerifyResult(
                ok=False,
                complete=False,
                system_error=f"lean_timeout: verifier timed out after {lean_timeout}s",
            )
        except Exception as exc:
            return LeanVerifyResult(
                ok=False,
                complete=False,
                system_error=f"lean_verify_exception: {type(exc).__name__}: {exc}",
            )

    async def premise_context_block(
        *,
        phase: str,
        attempt_index: int,
        turn_index: int,
        diagnostics: str = "",
        previous_proof: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        if premise_context_provider is None:
            return "", {}
        try:
            payload = await premise_context_provider(
                phase=phase,
                attempt=attempt_index,
                turn=turn_index,
                statement=statement,
                formal_prefix=formal_prefix,
                diagnostics=diagnostics,
                previous_proof=previous_proof,
            )
        except Exception as exc:
            payload = {
                "prompt_block": "",
                "digest": {
                    "retrieval_error": f"premise_context_provider: {type(exc).__name__}: {exc}"
                },
            }
        block = str(payload.get("prompt_block") or "")
        digest = payload.get("digest") if isinstance(payload.get("digest"), dict) else {}
        if block.strip():
            block = (
                "Additional validated premise context. Treat this as hints only; "
                "the Lean verifier is authoritative:\n"
                f"{block.strip()}\n\n"
            )
        return block, dict(digest)

    style = getattr(model_config, "prompt_style", "default")
    system_prompt, initial_tmpl, refine_tmpl = _select_templates(style)
    # Goedel's reference pipeline does not trust the theorem declaration
    # generated by the model. Its `problem_check` path replaces the generated
    # statement with the original target statement and keeps only the proof
    # body. Keep that guard for both our CoT (`goedel_v2`) and code-only
    # (`goedel_v1`) prompt dialects; otherwise an unrelated easy theorem that
    # type-checks would be counted as a target pass.
    goedel_reference_mode = style in {"goedel_v1", "goedel_v2"}

    def failed_call_reason(resp) -> Optional[str]:
        """Whether the provider failed to answer at all, as opposed to answering badly.

        A turn is a unit of mathematical effort, and the budget should only be
        charged for one. A traced run showed 8 of 29 calls returning zero
        characters with ``codex exec exited with code 1`` — the CLI crashed —
        and each of those still consumed a turn, so the tail of two attempts
        went to nothing at all. Those get resampled.

        The line is drawn at the provider's own verdict, not at the length of
        the reply. Short answers (52-185 chars were common in the same run)
        come back with no error and ``finish_reason == "stop"``: the model did
        answer, and Lean rejecting it is a real result about a real attempt.
        Resampling those would be discarding evidence, and would also let a
        prover quietly reroll every response it did not like.

        Timeouts are excluded for the same reason. A timeout means the model
        was working and ran out of clock, which says something about the
        prompt; the loop already feeds that back as `model_timeout` so the next
        turn can ask for something shorter. A crash says only that the call
        never happened.
        """
        if not resp.error:
            return None
        if resp.finish_reason == "timeout" or "timeout" in resp.error.lower():
            return None
        return f"failed_call: {resp.error[:140]}"

    def extract_candidate_code(raw_text: str) -> tuple[str, Optional[str]]:
        if goedel_reference_mode:
            raw_code = _extract_fenced_lean_block(raw_text)
            if raw_code is None:
                return "", "model returned no fenced lean code"
            code, error = _goedel_replace_statement_in_proof(formal_prefix, raw_code)
            return code or "", error
        return extract_lean_block(raw_text), None

    cfg_first = getattr(model_config, "first_temperature", None)
    if cfg_first is not None:
        first_temperature = cfg_first
    cfg_refine = getattr(model_config, "refine_temperature", None)
    if cfg_refine is not None:
        refine_temperature = cfg_refine
    sampling_config = (
        model_config
        if model_config.top_p is not None
        else replace(model_config, top_p=top_p)
    )

    for k in range(1, K + 1):
        attempt = AttemptRecord(repeat_index=k)
        history: List[Dict[str, str]] = []

        for t in range(1, T_max + 1):
            turn_started = time.monotonic()
            premise_digest: Dict[str, Any] = {}
            if t == 1:
                context_block = ""
                if proof_context.strip():
                    context_block = (
                        "Additional proof context. Treat this as hints only; "
                        "the Lean verifier is authoritative:\n"
                        f"{proof_context.strip()}\n\n"
                    )
                premise_block, premise_digest = await premise_context_block(
                    phase="initial",
                    attempt_index=k,
                    turn_index=t,
                )
                context_block += premise_block
                # Goedel-V2 prompt expects a ``:= ... sorry`` slot;
                # other styles use the prefix verbatim.
                prefix_for_prompt = (
                    _normalize_formal_prefix_for_goedel_v2(formal_prefix.strip())
                    if style == "goedel_v2"
                    else formal_prefix.strip()
                )
                user_prompt = initial_tmpl.format(
                    statement=statement.strip(),
                    header=header,
                    formal_prefix=prefix_for_prompt,
                    proof_context_block=context_block,
                )
                temperature = first_temperature
            else:
                prior = attempt.turns[-1]
                diagnostics = prior.verify.summary(max_errors_in_prompt)
                premise_block, premise_digest = await premise_context_block(
                    phase="reflection",
                    attempt_index=k,
                    turn_index=t,
                    diagnostics=diagnostics,
                    previous_proof=prior.proof_code,
                )
                user_prompt = refine_tmpl.format(
                    previous_proof=prior.proof_code,
                    diagnostics=diagnostics,
                    max_errors=max_errors_in_prompt,
                    premise_context_block=premise_block,
                )
                temperature = refine_temperature

            if n_parallel > 1:
                # Best-of-N: issue n_parallel completions, verify each in
                # parallel, return the first that closes the proof (or the
                # first that types as a fallback for the next refine turn).
                async def _one_sample(_idx: int) -> ModelTurnResponse:
                    return await model_call(
                        sampling_config,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_tokens,
                    )

                responses = await asyncio.gather(*[_one_sample(i) for i in range(n_parallel)])

                async def _verify_sample(resp: ModelTurnResponse) -> LeanVerifyResult:
                    if resp.error:
                        return LeanVerifyResult(
                            ok=False, complete=False,
                            system_error=f"model_exception: {resp.error}",
                        )
                    code, extraction_error = extract_candidate_code(resp.raw_text)
                    if not code.strip():
                        return LeanVerifyResult(
                            ok=False, complete=False,
                            system_error=extraction_error or "model returned no parseable lean code",
                        )
                    return await run_verifier(code)

                verifications = await asyncio.gather(
                    *[_verify_sample(r) for r in responses]
                )

                model_response = responses[0]
                verify_result = verifications[0]
                proof_code, extraction_error = extract_candidate_code(model_response.raw_text)
                for r, v in zip(responses, verifications):
                    if v.complete:
                        model_response, verify_result = r, v
                        proof_code, extraction_error = extract_candidate_code(r.raw_text)
                        break
                else:
                    for r, v in zip(responses, verifications):
                        if v.ok:
                            model_response, verify_result = r, v
                            proof_code, extraction_error = extract_candidate_code(r.raw_text)
                            break
            elif trace_run_prefix:
                with ls.trace(
                    name=f"{trace_run_prefix}.attempt_{k}.turn_{t}.llm",
                    run_type="llm",
                    inputs={
                        "problem_id": problem_id,
                        "attempt": k,
                        "turn": t,
                        "model": model_config.provider_slug,
                    },
                ) as llm_run:
                    model_response = await model_call(
                        sampling_config,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_tokens,
                    )
                    llm_run.end(
                        outputs={
                            "finish_reason": model_response.finish_reason,
                            "error": model_response.error,
                            "elapsed_seconds": model_response.elapsed_seconds,
                            "raw_text_chars": len(model_response.raw_text or ""),
                        }
                    )
                proof_code, extraction_error = extract_candidate_code(model_response.raw_text)
                verify_result = None  # computed below
            else:
                model_response = await model_call(
                    sampling_config,
                    system_prompt,
                    user_prompt,
                    temperature,
                    max_tokens,
                )
                proof_code, extraction_error = extract_candidate_code(model_response.raw_text)
                verify_result = None  # computed below

            # Resample a degenerate reply rather than spending the turn on it.
            # Bounded, because a model that answers nothing three times in a row
            # is not going to answer on the fourth, and the per-seed clock is
            # still running.
            discarded: List[Dict[str, Any]] = []
            if verify_result is None:
                reason = failed_call_reason(model_response)
                while reason and len(discarded) < max_resamples:
                    discarded.append(
                        {
                            "reason": reason,
                            "chars": len(model_response.raw_text or ""),
                            "finish_reason": model_response.finish_reason,
                        }
                    )
                    # A crashed CLI that is retried instantly tends to crash
                    # again; give it a moment before asking twice.
                    await asyncio.sleep(2.0 * len(discarded))
                    model_response = await model_call(
                        sampling_config,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_tokens,
                    )
                    proof_code, extraction_error = extract_candidate_code(
                        model_response.raw_text
                    )
                    reason = failed_call_reason(model_response)

            if verify_result is not None:
                # best-of-N branch already verified; skip the single-call
                # verify below.
                pass
            elif model_response.error:
                error_kind = (
                    "model_timeout"
                    if model_response.finish_reason == "timeout" or "timeout" in model_response.error.lower()
                    else "model_exception"
                )
                verify_result = LeanVerifyResult(
                    ok=False,
                    complete=False,
                    system_error=f"{error_kind}: {model_response.error}",
                )
            elif extraction_error:
                verify_result = LeanVerifyResult(
                    ok=False,
                    complete=False,
                    system_error=extraction_error,
                )
            elif not proof_code.strip():
                verify_result = LeanVerifyResult(
                    ok=False,
                    complete=False,
                    system_error="model returned no parseable lean code",
                )
            else:
                if trace_run_prefix:
                    with ls.trace(
                        name=f"{trace_run_prefix}.attempt_{k}.turn_{t}.lean_verify",
                        run_type="tool",
                        inputs={
                            "problem_id": problem_id,
                            "attempt": k,
                            "turn": t,
                            "lean_code_chars": len(proof_code),
                            "timeout_seconds": lean_timeout,
                        },
                    ) as verify_run:
                        verify_result = await run_verifier(proof_code)
                        verify_run.end(
                            outputs={
                                "ok": verify_result.ok,
                                "complete": verify_result.complete,
                                "summary": verify_result.summary(),
                                "verify_time": verify_result.verify_time,
                            }
                        )
                else:
                    verify_result = await run_verifier(proof_code)

            elapsed = time.monotonic() - turn_started
            attempt.turns.append(
                TurnRecord(
                    turn_index=t,
                    proof_code=proof_code,
                    raw_model_text=model_response.raw_text,
                    verify=verify_result,
                    elapsed_seconds=elapsed,
                    premise_pack=premise_digest,
                    finish_reason=model_response.finish_reason,
                    discarded_responses=discarded,
                )
            )
            history.append({"role": "assistant", "content": model_response.raw_text})

            if verify_result.complete:
                attempt.success = True
                attempt.success_turn = t
                attempt.final_proof = proof_code
                attempt.terminated_reason = "proved"
                break

        if attempt.terminated_reason is None:
            attempt.terminated_reason = "turn_budget_exhausted"
        record.attempts.append(attempt)
        if attempt.success:
            record.pass_at_k = True
            turns_used = attempt.success_turn or len(attempt.turns)
            if (
                record.min_turns_to_success is None
                or turns_used < record.min_turns_to_success
            ):
                record.min_turns_to_success = turns_used
            # K is "any of K"; we can break early on success per Pass@K
            # convention, but we still record this attempt's turns.
            break

    record.total_elapsed_seconds = time.monotonic() - started_problem
    return record
