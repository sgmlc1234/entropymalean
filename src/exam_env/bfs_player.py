"""BFS-Prover completion-model player for the Lean exam environment.

BFS-Prover-V2 is a ``{state}:::{tactic}`` completion model. The historical
evaluation harness could not show it a live proof state ("without LeanDojo we
cannot show the live proof state" — bfs_step_prover.build_state_prompt) and
approximated the state with theorem-prefix + accepted tactics. The exam
environment HAS the live goal state, so this player prompts BFS-Prover with
the actual goals — matching its training distribution for the first time in
this repo.

Policy (the game translated into an algorithm):
  - each turn, sample ``n_per_step`` tactic candidates for the current state;
  - try them in order through ``env.step`` — a rejection costs nothing but a
    verifier call and leaves the state unchanged (the game's Retry);
  - if no candidate is accepted after ``resample_rounds`` fresh batches, the
    player rolls back — by default to the most promising OTHER state it has
    seen, not merely one step up.

The reference implementation runs a best-first search over a proof tree,
ordering nodes by cumulative log-probability (`proof_tree.py:197`). LM Studio
does not return logprobs on its completion endpoint (verified 2026-07-29), so
this player keeps a frontier of visited states and, lacking a score, revisits
the shallowest one with untried candidates. That is weaker than the
reference's ordering but strictly stronger than climbing one step at a time.
"""

from __future__ import annotations

import asyncio
import re

from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.bfs_step_prover import is_buggy_tactic
from src.exam_env.environment import ExamObservation, LeanExamEnv

BFS_SEPARATOR = ":::"


TacticSampler = Callable[[str, int], Awaitable[Sequence[str]]]


_CASE_HEADER_RE = re.compile(r"(?m)^\s*case .*\n?")
_TRIVIAL_THIS_RE = re.compile(r"(?m)^\s*this.*? : True\n?")


def first_goal(goals: Sequence[str]) -> str:
    """The reference's ``pp1``: one goal, without its case header.

    BFS-Prover-V2 is prompted with a single tactic state
    (`prover_manager.py:54-60` takes the head of the pretty-printed state,
    then strips `case …` and `this … : True` lines). Concatenating every open
    goal — which is what an `unsolved goals` diagnostic gives us — is a shape
    the model never saw in training.
    """
    for goal in goals:
        head = str(goal or "").split("\n\n")[0]
        head = _CASE_HEADER_RE.sub("", head)
        head = _TRIVIAL_THIS_RE.sub("", head)
        head = head.strip()
        if head:
            return head
    return ""


def build_goal_state_prompt(goals: Sequence[str]) -> str:
    """Live goal-state prompt for BFS-Prover (``state:::`` completion form).

    Only the first goal goes in, and nothing else: a prepended comment line
    (we tried ``-- useful lemmas: …``) is off-distribution and measurably cost
    solve rate, so palette knowledge is applied after sampling instead — see
    ``rank_candidates``.
    """
    return f"{first_goal(goals)}{BFS_SEPARATOR}"


class SampledTactics(list):
    """A candidate list that also knows what it cost to produce."""

    tokens_used: int = 0


def rank_candidates(
    candidates: Sequence[Any],
    palette_theorems: Optional[Sequence[str]] = None,
) -> List[str]:
    """Order sampled tactics by log-probability, then float palette hits to the front.

    Accepts plain strings or ``(tactic, score)`` pairs. With scores present the
    base order is the reference's — highest mean token log-probability first
    (`prover_manager.py:170`).

    The palette is applied as a *rerank*, never injected into the prompt: this
    is a completion model, and listing lemmas in the prompt puts it off
    distribution — an earlier run scored 9/18 that way against 11/18 without.

    Note what the rerank costs. Candidates naming a palette lemma go ahead of
    all others, so a low-probability candidate can be tried before the model's
    own favourite; within each group the log-probability order is kept. That is
    a real trade and it does not always pay — in a paired run it rescued four
    seeds and lost one. Making log-probability primary instead would make the
    palette almost inert, since continuous scores rarely tie.
    """
    scores: Dict[str, float] = {}
    unique: List[str] = []
    for candidate in candidates:
        if isinstance(candidate, tuple):
            raw, score = candidate[0], float(candidate[1])
        else:
            raw, score = candidate, None
        tactic = (raw or "").strip()
        if score is not None and (
            tactic not in scores or score > scores[tactic]
        ):
            scores[tactic] = score
        # Filter parity with the reference implementation
        # (prover_manager.py:175-183): sorry / admit / native_decide, `?_`
        # with rcases|cases'|simpa, and ambiguous `simpa _`. Exploratory
        # tactics are NOT banned — the reference allows them.
        if tactic and not is_buggy_tactic(tactic) and tactic not in unique:
            unique.append(tactic)
    if scores:
        unique.sort(key=lambda t: scores.get(t, float("-inf")), reverse=True)
    if not palette_theorems:
        return unique
    names = list(palette_theorems)
    # One compound key rather than a second sort leaning on stability: the
    # precedence is then visible here instead of depending on a property of
    # `list.sort` that a future edit could quietly break.
    unique.sort(
        key=lambda t: (
            sum(1 for name in names if name in t),
            scores.get(t, float("-inf")),
        ),
        reverse=True,
    )
    return unique


class BFSExamPlayer:
    def __init__(
        self,
        sampler: TacticSampler,
        *,
        n_per_step: int = 6,
        resample_rounds: int = 2,
        max_rollbacks: int = 3,
        use_palette: bool = False,
        seed_tactic: str = "",
        token_budget: int = 0,
    ) -> None:
        self.sampler = sampler
        self.n_per_step = n_per_step
        self.resample_rounds = resample_rounds
        self.max_rollbacks = max_rollbacks
        self.use_palette = use_palette
        #: A level-3 hint: the first tactic of a known proof, played before the
        #: search begins. It is not fed to the model — a completion prover has
        #: no channel for advice — so the only way to deliver it is to take the
        #: step on the player's behalf and let the search continue from there.
        self.seed_tactic = (seed_tactic or "").strip()
        #: Generated-token ceiling, 0 for uncapped. Actions are not a currency a
        #: whole-proof prover can be billed in — one proof is one action — so a
        #: comparison across player types has to meter what both actually spend.
        #: Under the action budget this search was drawing roughly four times
        #: the tokens of a whole-proof attempt.
        self.token_budget = int(token_budget or 0)

    async def play(
        self, env: LeanExamEnv, *, max_actions: int = 120
    ) -> Dict[str, Any]:
        observation = await env.reset()
        palette_names = (
            sorted((env.palette.get("theorems") or {}).keys())
            if self.use_palette
            else None
        )
        tried_at_depth: Dict[int, set] = {}
        # depth -> steps prefix, so the player can return to any visited state
        frontier: Dict[int, List[str]] = {0: []}
        rollbacks_used = 0
        actions = 0
        rejected = 0
        filtered = 0
        sampled_batches = 0
        tokens_used = 0
        hint_played = False
        # Why Lean turned a candidate down, not just how often. The count alone
        # cannot distinguish a prover that has never heard of a lemma from one
        # that was handed the name and transcribed it wrong — and those two
        # call for different environment affordances. Goedel's player kept
        # these all along; this one was throwing them away.
        rejections: List[Dict[str, str]] = []

        if self.seed_tactic and not env.done:
            # Charged as an action, and recorded, so an arm that is handed its
            # opening move cannot look cheaper than one that found it.
            observation = await env.step(
                {"type": "tactic", "tactic": self.seed_tactic}
            )
            actions += 1
            if observation.status in {"accepted", "solved"}:
                hint_played = True
                frontier[len(env.steps)] = list(env.steps)
            else:
                rejected += 1
                rejections.append(
                    {"tactic": self.seed_tactic[:120],
                     "message": (observation.message or "")[:300]}
                )

        def out_of_tokens() -> bool:
            return bool(self.token_budget) and tokens_used >= self.token_budget

        while not env.done and actions < max_actions and not out_of_tokens():
            depth = len(env.steps)
            tried = tried_at_depth.setdefault(depth, set())
            prompt = build_goal_state_prompt(observation.goals)
            accepted = False
            for _ in range(self.resample_rounds + 1):
                batch = await self.sampler(prompt, self.n_per_step)
                tokens_used += int(getattr(batch, "tokens_used", 0) or 0)
                sampled = list(batch)
                sampled_batches += 1
                candidates = rank_candidates(sampled, palette_names)
                filtered += len(sampled) - len(candidates)
                fresh = [c for c in candidates if c not in tried]
                for candidate in fresh:
                    tried.add(candidate)
                    observation = await env.step(
                        {"type": "tactic", "tactic": candidate}
                    )
                    actions += 1
                    if observation.status in {"accepted", "solved"}:
                        accepted = True
                        break
                    rejected += 1
                    if len(rejections) < 60:   # bounded: episodes can reject a lot
                        rejections.append(
                            {"tactic": candidate[:120],
                             "message": (observation.message or "")[:300]}
                        )
                    if env.done or actions >= max_actions or out_of_tokens():
                        break
                if accepted or env.done or actions >= max_actions or out_of_tokens():
                    break
            if accepted or env.done:
                frontier[len(env.steps)] = list(env.steps)
                continue
            # Stuck here — return to another visited state (the game's
            # "retry from an earlier point", generalized to a frontier).
            if depth > 0 and rollbacks_used < self.max_rollbacks:
                if env.steps:
                    tried_at_depth.setdefault(depth - 1, set()).add(env.steps[-1])
                tried_at_depth.pop(depth, None)
                frontier.pop(depth, None)
                target = min(
                    (d for d in frontier if d < depth),
                    default=None,
                )
                if target is None:
                    break
                rollbacks_used += 1
                observation = await env.step({"type": "rollback", "to_step": target})
                actions += 1
                continue
            break  # rollback budget spent

        return {
            "success": env.success,
            "steps": list(env.steps),
            "actions": actions,
            "rejected": rejected,
            "rollbacks": rollbacks_used,
            "sampled_batches": sampled_batches,
            "tokens_used": tokens_used,
            "token_budget": self.token_budget,
            "hint_tactic": self.seed_tactic,
            "hint_accepted": hint_played,
            "filtered_candidates": filtered,
            "rejections": rejections,
            "solved_code": env.solved_code(),
        }


# ---------------------------------------------------------------------------
# llama.cpp sampler with log-probabilities
# ---------------------------------------------------------------------------

#: A sampled tactic with the reference's score: mean token log-probability
#: (`prover_manager.py:170` divides the cumulative logprob by the token count).
ScoredTactic = Tuple[str, float]


def parse_scored_choices(payload: Dict[str, Any]) -> List[ScoredTactic]:
    """Extract (tactic, mean-token-logprob) from a llama.cpp completion body.

    llama.cpp emits the newer OpenAI shape — ``logprobs.content`` is a list of
    per-token records — rather than the legacy ``token_logprobs`` array. A
    choice without log-probabilities scores 0.0 so ordering degrades to the
    sampled order instead of failing.
    """
    scored: List[ScoredTactic] = []
    for choice in payload.get("choices") or []:
        text = str(choice.get("text") or "").strip()
        if not text:
            continue
        content = ((choice.get("logprobs") or {}).get("content")) or []
        values = [
            float(item["logprob"])
            for item in content
            if isinstance(item, dict) and item.get("logprob") is not None
        ]
        scored.append((text, sum(values) / len(values) if values else 0.0))
    return scored


async def llama_cpp_scored_sampler(
    prompt: str,
    n: int,
    *,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Sequence[str] = (":::", "\n\n"),
    timeout: float = 180.0,
) -> List[ScoredTactic]:
    """Sample ``n`` tactics from a llama.cpp server, keeping log-probabilities.

    Requests are issued one at a time: llama.cpp's ``n`` returns a single
    choice, so independent draws are what actually produce a candidate set.
    """
    import httpx

    url = base_url.rstrip("/") + "/completions"
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": 1,
        "stop": list(stop),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def one() -> List[ScoredTactic]:
            try:
                response = await client.post(url, json=body)
                response.raise_for_status()
                payload = response.json()
                # Tokens are the currency a tactic prover and a whole-proof
                # prover can both be billed in; actions and attempts are not
                # comparable across the two. Recorded here because only the
                # server knows the true count.
                spent = int((payload.get("usage") or {}).get("completion_tokens") or 0)
                return parse_scored_choices(payload), spent
            except Exception:
                return [], 0

        batches = await asyncio.gather(*(one() for _ in range(n)))
    items = SampledTactics(item for batch, _ in batches for item in batch)
    # Carried on the list itself so the caller can bill tokens without every
    # player having to learn a new sampler protocol. A plain list cannot hold
    # the attribute, hence the subclass.
    items.tokens_used = sum(spent for _, spent in batches)
    return items
