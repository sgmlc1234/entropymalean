"""Play the exam environment with a whole-proof model.

`BFSExamPlayer` assumes the prover's unit of action is a tactic: it reads a
goal state and answers with one line. Goedel-V2 does not work that way — it is
prompted with `theorem … := by sorry` and answers with the entire proof. Asking
it for single tactics would take it off the distribution it was trained on, and
scoring it that way would measure the mismatch rather than the model.

So the episode has a different clock. The state a whole-proof model can act on
is not the goal after k tactics; it is *its own previous attempt plus what Lean
said about it*, which is exactly what Goedel-V2 was trained to consume. One
episode is therefore a sequence of attempts under diagnostic feedback:

    attempt → Lean → diagnostics → revised attempt → …

Both players are episodic; they differ in what an episode is made of.

The budget is what makes the two comparable. Counting actions favours the
whole-proof model absurdly (one action for a whole proof against forty for a
search), and counting attempts favours the tactic model just as absurdly. What
both spend is generated tokens, so that is the currency: give each the same
token budget and ask who proves more. Under the previous action-based budget
the BFS arm was in fact spending roughly four times the tokens.

Environment affordances arrive differently here, and that difference is the
point rather than a compromise:

  palette   goes in the prompt. Injecting lemma names ruined the BFS arm — a
            completion model has no slot for advice — but Goedel-V2 is a chat
            model with a premise-context slot already in its reference prompt.
  rollback  has no goal state to return to. Its analogue is choosing *which*
            earlier attempt to revise from, rather than always the most recent.
  hints     are consumed directly. A model asked to "provide a detailed proof
            plan" can be handed the level-2 outline as that plan.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

_SORRY = re.compile(r"(?<![A-Za-z_])(sorry|admit)(?![A-Za-z_])")

from src.evaluation.multi_turn_prover import (
    _extract_fenced_lean_block,
    _goedel_replace_statement_in_proof,
    _normalize_formal_prefix_for_goedel_v2,
    _select_templates,
)
from src.exam_env.environment import LeanExamEnv

#: ``(messages, max_tokens) -> {"text": str, "completion_tokens": int}``
ProofGenerator = Callable[[List[Dict[str, str]], int], Awaitable[Dict[str, Any]]]

_STATEMENT_LINE = re.compile(r"^\s*(theorem|lemma)\b")
_FENCE = re.compile(r"```")


def lean_text_of(raw: str) -> str:
    """The Lean part of a reply, whether or not the model fenced it.

    Goedel-V2 answers in markdown — prose, then ```lean4 blocks — and its
    reference extractor looks for those fences. Pythagoras-Prover does not use
    them: it emits the file directly, opening on `import Mathlib`. Requiring a
    fence would score every one of its replies as "no code produced", which is
    the same shape of mistake as reading a truncated sketch as a finished
    proof: a harness assumption misreported as a model failure.
    """
    fenced = _extract_fenced_lean_block(raw)
    if fenced:
        return fenced
    if _FENCE.search(raw or ""):
        return ""          # fences exist but none held Lean — trust that
    lines = str(raw or "").splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"^\s*(import|open|set_option|theorem|lemma)\b", line)
        ),
        None,
    )
    if start is None:
        return ""
    # Stop where Lean stops. Running to the end of the reply instead swept up
    # trailing markdown, and the environment reported it as `unexpected token
    # '#'` — a syntax error that describes our slicing, not the proof.
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"^\s*(#{1,6}\s|```|\*\*)", lines[i])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def proof_body_of(code: str, formal_prefix: str) -> str:
    """The tactic block from a generated file, with the statement discarded.

    Goedel's own evaluation replaces the model's theorem line with the target
    statement and keeps only what follows, so a model that quietly restates an
    easier theorem cannot be scored as having proved this one. The environment
    owns the statement, so here the whole prefix is dropped rather than
    substituted.
    """
    replaced, _ = _goedel_replace_statement_in_proof(formal_prefix, code)
    text = replaced or code
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _STATEMENT_LINE.match(line):
            rest = "\n".join(lines[index:])
            match = re.search(r":=\s*by\b", rest)
            if match:
                body = rest[match.end():]
                return "\n".join(
                    line[2:] if line.startswith("  ") else line
                    for line in body.splitlines()
                ).strip("\n")
            return ""
    return text.strip("\n")


#: Characters per token, measured on this panel's own traffic. Lean is dense in
#: non-ASCII (``ℝ``, ``∀``, ``₀``) and tokenises at 2.97; English prose runs
#: near 4.0. The two rates are only used to *split* one reply, never to replace
#: the provider's own count of it.
_CHARS_PER_TOKEN_LEAN = 2.97
_CHARS_PER_TOKEN_PROSE = 4.0


def _split_proof_tokens(raw: str, content_tokens: int) -> tuple:
    """Divide a reply's content tokens into the proof and the plan before it.

    The budget is the proof the model writes. A reasoning model's deliberation
    is already outside the count because the endpoint reports it separately, but
    a prover whose official prompt asks for a written plan emits the same
    deliberation as ordinary content, and charging it means the channel a model
    happens to think in decides how much proof it is allowed to write. Measured
    on Pythagoras-Prover-4B, whose prompt demands a plan before the Lean: 81% of
    a *successful* attempt's tokens were the plan, about 2,620 of the 8,192-token
    episode budget, against a Goedel cell that skips the plan by prefill and
    spends the whole budget on Lean.

    Everything before the first code fence is the plan. Both parts are estimated
    at their own character rate and the provider's content count is then split in
    that proportion, so the total still comes from the endpoint and only the
    boundary comes from us. A reply with no fence produced no proof and costs
    nothing; ``max_attempts`` is what bounds that case.

    Declared per cell, never inferred. A prefilled cell opens the fence in the
    prompt and LM Studio returns only the continuation, so its reply carries a
    *closing* fence and no opening one -- split here, almost the whole proof
    would land on the wrong side of the boundary and the budget would stop
    binding. The split belongs only to cells whose official prompt asks for a
    written plan and which therefore run without a prefill.
    """
    fence = raw.find("```")
    if fence < 0:
        return 0, content_tokens
    plan_chars = len(raw[:fence])
    code_chars = len(raw[fence:])
    est_plan = plan_chars / _CHARS_PER_TOKEN_PROSE
    est_code = code_chars / _CHARS_PER_TOKEN_LEAN
    if est_plan + est_code <= 0:
        return content_tokens, 0
    proof = int(round(content_tokens * est_code / (est_plan + est_code)))
    proof = max(0, min(proof, content_tokens))
    return proof, content_tokens - proof


class GoedelExamPlayer:
    def __init__(
        self,
        generate: ProofGenerator,
        *,
        max_tokens: int = 2048,
        token_budget: int = 8192,
        max_attempts: int = 8,
        max_errors_in_prompt: int = 5,
        palette_names: Optional[Sequence[str]] = None,
        hint_outline: Optional[Sequence[str]] = None,
        prefill: str = "",
        prompt_style: str = "goedel_v2",
        prefill_prefix_flag: bool = False,
        reasoning_headroom: int = 0,
        plan_outside_budget: bool = False,
        nl_statement: str = "",
    ) -> None:
        self.generate = generate
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.max_attempts = max_attempts
        self.max_errors_in_prompt = max_errors_in_prompt
        self.palette_names = list(palette_names or [])
        self.hint_outline = list(hint_outline or [])
        #: Text the assistant turn is started with, so the model continues it
        #: rather than choosing how to open. Goedel-V2 answers a bare request
        #: with "### Detailed Proof" and can spend the whole window on analysis:
        #: across 814 control attempts, 44% never reached a Lean block, and
        #: those attempts spent a median 4096 tokens against 1458 for the ones
        #: that proved the theorem. Asking in the prompt for code is a request;
        #: opening the turn with a fence is a constraint. Empty disables it,
        #: which is what the control cells ran with.
        self.prefill = prefill
        #: Which prompt dialect to drive the model in. The paper's protocol is
        #: that each model gets its own official format inside the shared token
        #: budget, so this cannot stay pinned to Goedel's: a general instruction
        #: model handed a prover's dialect is being measured on someone else's
        #: prompt, and the drop that comes out carries that difference.
        #: `goedel_v2` reasons then emits a fence; `goedel_v1` is the minimalist
        #: "complete this Lean 4 code" shared with DeepSeek-Prover-V1.5; anything
        #: else falls through to the generic instruct template.
        self.prompt_style = prompt_style
        self.prefill_prefix_flag = prefill_prefix_flag
        self.reasoning_headroom = reasoning_headroom
        self.plan_outside_budget = plan_outside_budget
        #: The row's natural-language statement. Only the generic instruct
        #: template has a slot for it; the two Goedel dialects show the Lean
        #: prefix alone. It is carried here so that selecting `generic` cannot
        #: raise `KeyError: 'statement'` half-way through a cell — which is how
        #: it first surfaced, as three consecutive empty episodes that read
        #: exactly like a dead server.
        self.nl_statement = nl_statement

    def _context_block(self) -> str:
        parts = []
        if self.palette_names:
            parts.append(
                "Lemmas available in Mathlib that are likely relevant:\n"
                + "\n".join(f"- {name}" for name in self.palette_names)
            )
        if self.hint_outline:
            parts.append(
                "A proof plan for this theorem:\n"
                + "\n".join(f"{i}. {step}" for i, step in enumerate(self.hint_outline, 1))
            )
        return ("\n\n".join(parts) + "\n\n") if parts else ""

    async def play(self, env: LeanExamEnv) -> Dict[str, Any]:
        await env.reset()
        _, initial_template, refine_template = _select_templates(self.prompt_style)
        prefix = _normalize_formal_prefix_for_goedel_v2(env.statement)
        context = self._context_block()

        attempts: List[Dict[str, Any]] = []
        tokens_used = 0
        plan_used = 0
        reasoning_used = 0
        generator_empty = False
        generator_error = ""
        prompt = initial_template.format(
            header=env.header, formal_prefix=prefix, proof_context_block=context,
            statement=self.nl_statement or env.statement,
        )

        while (
            not env.done
            and len(attempts) < self.max_attempts
            and tokens_used < self.token_budget
        ):
            # `max_tokens` is this cell's *proof* allowance per attempt, and the
            # budget is counted in proof tokens too, so the smaller of the two is
            # what the model is allowed to write. What has to be *requested* is
            # larger wherever the endpoint counts thinking inside the same cap:
            # ask for exactly the allowance there and reasoning eats it, leaving
            # a few dozen tokens of Lean. `reasoning_headroom` is that difference,
            # declared per cell because it is a property of the serving stack --
            # zero for xAI, which caps the visible answer, and zero for a model
            # with no reasoning channel at all.
            allowance = min(self.max_tokens, self.token_budget - tokens_used)
            remaining = allowance + self.reasoning_headroom
            messages = [{"role": "user", "content": prompt}]
            if self.prefill:
                turn = {"role": "assistant", "content": self.prefill}
                # Mistral refuses a trailing assistant turn outright --- "Expected
                # last role User or Tool (or Assistant with prefix True)" --- and
                # continues it when the flag is set. llama.cpp ignores the extra
                # key, so the same message shape serves both.
                if self.prefill_prefix_flag:
                    turn["prefix"] = True
                messages.append(turn)
            reply = await self.generate(messages, remaining)
            # The budget is the proof the model writes, not the thinking it does
            # on the way. A reasoning model's `<think>` block is generated
            # tokens, but charging it here would mean a prover that answers
            # directly and a model that deliberates for four thousand tokens
            # before writing the same proof are given different amounts of
            # proof to write. Providers also disagree about whether `max_tokens`
            # covers thinking at all, so a total-token budget is not even the
            # same quantity across the panel.
            spent = int(reply.get("completion_tokens") or 0)
            thought = int(reply.get("reasoning_tokens") or 0)
            reasoning_used += thought
            raw = str(reply.get("text") or "")
            # Same principle one step further: a plan written as content is
            # deliberation too, so only the fenced proof is charged.
            content = max(spent - thought, 0)
            if self.plan_outside_budget:
                proof_tokens, plan_tokens = _split_proof_tokens(raw, content)
            else:
                proof_tokens, plan_tokens = content, 0
            plan_used += plan_tokens
            tokens_used += proof_tokens
            # A reply with no text and no tokens is the serving stack, not an
            # answer. Left to fall through, the prefill repair below turns it
            # into `prefill + "" + "```"` — a body that is not empty, so the
            # `no_code` guard does not catch it — and Lean rejects the fragment
            # with `unexpected end of input`, which is then filed as the model
            # failing to prove the theorem. One 342-episode run lost 266
            # episodes that way after the server died at episode 76 and the run
            # played on to completion. Stop the episode and say who failed.
            if not raw.strip() and not int(reply.get("completion_tokens") or 0):
                generator_empty = True
                # The transport's own account of what went wrong. `generate`
                # already catches the exception and returns it here; dropping it
                # leaves the episode saying only "empty", which is the symptom
                # every cause shares — a timeout, a refused connection, and a
                # 400 are indistinguishable without it, and one 98-episode run
                # aborted with no way to tell which had happened.
                generator_error = str(reply.get("error") or "")[:200] or "empty reply, no error reported"
                break
            # Whether the prefill comes back is a server convention: llama.cpp
            # echoes it, LM Studio returns only the new tokens. Restoring it
            # naively made things worse — the prefill opens a ```lean4 fence
            # that the continuation never closes, and the extractor, seeing a
            # fence, trusts the fenced path and finds nothing. Across one
            # 342-episode run that turned 89% of attempts into "no code" on a
            # model that was in fact answering correctly.
            #
            # So close what we open. The unfenced continuation is already
            # handled by the extractor's fallback, and a balanced fence is
            # handled by its primary path; an unbalanced one is the only shape
            # neither reads.
            if self.prefill and not raw.lstrip().startswith(self.prefill.strip()[:8]):
                raw = self.prefill + raw
                if raw.count("```") % 2 == 1:
                    raw = raw.rstrip() + "\n```"
            code = lean_text_of(raw)
            # No `or raw` fallback. An empty extraction is a verdict — the
            # reply held no Lean — and substituting the whole reply overrides
            # it, which is how 300-line "proofs" of markdown reached Lean and
            # came back as `unexpected token '#'`.
            body = proof_body_of(code, prefix) if code else ""
            truncated = str(reply.get("finish_reason") or "") == "length"
            # Goedel-V2 answers with a sketch block, then prose, then the real
            # proof. Cut the generation short and only the sketch survives —
            # `sorry` placeholders and a dangling fence — which the environment
            # then rejects for a syntax error that says nothing about the model.
            # Both failures are recorded as themselves so a truncated run is
            # never read as a model that could not prove the theorem.
            if body.strip() and _SORRY.search(body):
                attempts.append(
                    {
                        "status": "sketch_only",
                        "tokens": tokens_used,
                        "truncated": truncated,
                        "body_lines": len(body.splitlines()),
                    }
                )
                prompt = refine_template.format(
                    previous_proof=body[:1500],
                    diagnostics=(
                        "the proof was cut off before it was finished"
                        if truncated
                        else "the proof still contains `sorry`"
                    ),
                    max_errors=self.max_errors_in_prompt,
                    premise_context_block=context,
                    statement=self.nl_statement or env.statement,
                )
                continue
            if not body.strip():
                # Record `truncated` here too. Without it the one field that
                # separates "the model stopped and wrote no Lean" from "the
                # reply was cut off before it got to any" is absent on exactly
                # the outcome where the question arises, and a cell reading 72%
                # no_code cannot be diagnosed from its own records.
                attempts.append(
                    {
                        "status": "no_code",
                        "tokens": tokens_used,
                        "truncated": truncated,
                        "reply_chars": len(raw),
                    }
                )
                prompt = refine_template.format(
                    previous_proof=(code or raw)[:1200],
                    diagnostics="no Lean code block was produced",
                    max_errors=self.max_errors_in_prompt,
                    premise_context_block=context,
                    statement=self.nl_statement or env.statement,
                )
                continue

            # One whole proof, one action: the environment indents and verifies
            # the block exactly as it would a tactic, so nothing special is
            # needed to submit it.
            # The request ceiling the provider sees is `allowance +
            # reasoning_headroom`, and it cannot be told which part is which. A
            # model that writes content where the headroom expected thinking
            # spends the whole of it on the proof, so a single attempt can be
            # charged past the episode budget -- measured at up to 2.1x on the
            # reasoning SLMs, whose headroom is the largest. The loop's own
            # check runs at the top and therefore only ever notices afterwards.
            # A proof that cost more than the budget allows is not a solve at
            # this budget, so it is not offered to the verifier as one. No
            # episode in the panel was decided this way -- every success on
            # record was already inside the budget when it landed -- which is
            # what makes the guard a statement of the invariant rather than a
            # change to it.
            if tokens_used > self.token_budget:
                attempts.append(
                    {
                        "status": "over_budget",
                        "tokens": tokens_used,
                        "body_lines": len(body.splitlines()),
                        "truncated": truncated,
                    }
                )
                break
            observation = await env.step({"type": "tactic", "tactic": body})
            attempts.append(
                {
                    "status": observation.status,
                    "tokens": tokens_used,
                    "body_lines": len(body.splitlines()),
                    "message": (observation.message or "")[:400],
                }
            )
            if observation.status == "solved":
                break
            # Rollback's analogue: revise from the attempt that got furthest,
            # not simply the last one. Lean's message length is a poor proxy for
            # progress, so "fewest reported errors" is used instead.
            best = min(
                (a for a in attempts if a.get("status") in {"rejected", "error"}),
                key=lambda a: (a.get("message") or "").count("error"),
                default=attempts[-1],
            )
            prompt = refine_template.format(
                previous_proof=body[:1500],
                diagnostics=(best.get("message") or "")[:1200],
                max_errors=self.max_errors_in_prompt,
                premise_context_block=context,
            )

        return {
            "success": env.success,
            "steps": list(env.steps),
            "generator_empty": generator_empty,
            "generator_error": generator_error,
            "attempts": len(attempts),
            "attempt_log": attempts,
            "tokens_used": tokens_used,
            "reasoning_tokens": reasoning_used,
            "plan_tokens": plan_used,
            "token_budget": self.token_budget,
            "rejected": sum(1 for a in attempts if a.get("status") != "solved"),
            "truncated_attempts": sum(1 for a in attempts if a.get("truncated")),
            "sketch_only_attempts": sum(
                1 for a in attempts if a.get("status") == "sketch_only"
            ),
            "solved_code": env.solved_code() or "",
            "used_palette": bool(self.palette_names),
            "used_hint_outline": bool(self.hint_outline),
        }
