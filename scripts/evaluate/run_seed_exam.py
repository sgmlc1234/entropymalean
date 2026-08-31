#!/usr/bin/env python3
"""Play the exam environment over a seed set with the BFS prover.

`run_exam_grid.py` predates the certified seed sets: it harvests miniF2F rows
from a fixed glob and knows two arms. This takes an exam-rows file — either
benchmark, or both concatenated — so a run is defined by the seed set it names
rather than by what the script happens to find on disk.

What it records is deliberately more than pass/fail. An episode that ends
without a proof can end that way for reasons that are not the prover's fault:
Lean refused every candidate, the sampler returned nothing usable, the goal
state outgrew the model's 4096-token window, or the action budget ran out with
progress still being made. Averaging those together reports a model score for
what may be an environment limit, so each is counted separately.

Usage:
  set -a; source .env; set +a
  python scripts/evaluate/run_seed_exam.py --rows /tmp/smoke6.jsonl \
    --output data/evaluation/exam/smoke1 --arm official_parity \
    --llama-cpp http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.exam_env.bfs_player import (  # noqa: E402
    BFSExamPlayer,
    llama_cpp_scored_sampler,
)
from src.exam_env.environment import LeanExamEnv  # noqa: E402
from src.exam_env.goedel_player import GoedelExamPlayer  # noqa: E402
from src.utils.codex_cli import call_codex_cli  # noqa: E402

#: Outcomes that mean the episode measured nothing. The two players report the
#: same event under different names — a whole-proof loop gets no reply
#: (`generator_empty`), a tactic search gets no candidates (`sampler_empty`) —
#: and a guard that watches only one leaves the other player able to record a
#: full-length run of a dead server as its own failures.
UNMEASURED = {"generator_empty", "sampler_empty", "verifier_error"}

#: Consecutive unmeasured episodes that end the run. Both events occur
#: legitimately now and then — one row's goal state yields no usable candidate —
#: but a dead server yields none forever, and three in a row separates the two
#: at a cost of at most three wasted episodes.
EMPTY_RUN_ABORT = 3

#: The aid axis of the 2x2x2 design, plus the finer arms kept for the
#: affordance study. `closed_book` and `open_book` are the two the campaign
#: uses; the rest isolate single affordances on the seed set.
#:
#: `open_book` offers lemma names and a proof plan. What actually reaches the
#: model differs by architecture and cannot not differ: a whole-proof chat
#: model reads both from its prompt, while a completion prover trained on
#: `{state}:::{tactic}` has no channel for prose — injecting it measurably hurt
#: it (9/18 against 11/18), so its lemmas arrive as a rerank and the plan does
#: not arrive at all.
#:
#: That asymmetry does not damage the measurement, because the reported
#: quantity is the control-minus-treatment drop *within* one model. The aid has
#: to be identical across the two arms of a model, which it is; it does not
#: have to be identical across models, which it cannot be. Each episode records
#: what it actually received.
ARMS = {
    "closed_book":     {"rollback": 0, "palette": False, "plan": False},
    "open_book":       {"rollback": 0, "palette": True,  "plan": True},
    # affordance study on the seeds — one variable at a time
    "official_parity": {"rollback": 0, "palette": False, "plan": False},
    "rollback":        {"rollback": 3, "palette": False, "plan": False},
    "palette":         {"rollback": 3, "palette": True,  "plan": False},
    "hints":           {"rollback": 3, "palette": True,  "plan": True},
}


def outcome_of(result: Dict[str, Any], max_actions: int) -> str:
    """Why the episode ended, in terms that separate model from environment.

    The two players run out in different ways, so the reason has to be read
    off whichever fields the player actually reports — a tactic search stops on
    actions and candidates, a whole-proof loop stops on tokens and attempts.
    Reading BFS's fields off a Goedel result would file every episode under
    `sampler_empty`.
    """
    if result.get("success"):
        return "solved"
    # The environment refused the statement at reset, so no player ever moved.
    # This has to outrank everything: a row that will not elaborate produces
    # exactly the shape of a dead server — no tokens, no actions, zero seconds —
    # and would otherwise be filed as `sampler_empty` or `generator_empty`,
    # which says the serving stack failed. It also has to be distinguishable
    # from one, because three unplayable rows in a row would abort the run.
    if result.get("statement_invalid"):
        return "statement_invalid"
    # Lean never judged the row. Unlike a bad statement this is worth retrying,
    # so it joins the unmeasured set: dropped and replayed rather than counted.
    if result.get("verifier_failed"):
        return "verifier_error"
    # The player stopped because the serving stack returned nothing. This has
    # to outrank the budget and attempt tests: an episode that dies on its
    # third revision has attempts on the board and would otherwise be filed
    # under `attempts_exhausted`, which reads as the model running out of ideas.
    # An empty reply only means the serving stack failed if the episode got
    # nothing else either. A generator that answers three times, is rejected
    # three times, and then returns empty on the fourth has been measured — the
    # three rejections are the model's, and filing the episode under
    # `generator_empty` both discards them and, worse, feeds a streak counter
    # whose whole job is to notice a server that stopped answering. Two such
    # episodes in a row nearly ended a healthy 921-episode run.
    if result.get("generator_empty") and not result.get("attempts") and not result.get("tokens_used"):
        return "generator_empty"
    budget = result.get("token_budget") or 0
    if budget and result.get("tokens_used", 0) >= budget:
        return "token_budget"
    if "attempts" in result:                       # whole-proof player
        if not result.get("attempts"):
            return "generator_empty"
        return "attempts_exhausted"
    if not result.get("sampled_batches"):
        return "sampler_empty"
    if result.get("actions", 0) >= max_actions:
        return "action_budget"
    if not (result.get("steps") or []):
        return "no_accepted_step"
    return "search_exhausted"


#: Status codes worth coming back for. A 4xx other than 429 is the request
#: being wrong, and retrying it only delays the error that should be read.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, in seconds."""
    return (2 ** attempt) * 5 + random.uniform(0, 3)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="official_parity", choices=sorted(ARMS))
    parser.add_argument("--llama-cpp", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--n-per-step", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lean-timeout", type=float, default=180.0)
    parser.add_argument(
        "--generate-timeout", type=float, default=0.0,
        help="seconds to wait on one generation request; 0 keeps the old "
             "lean-timeout*2. Mistral with reasoning effort high routinely "
             "passes 360s on ProofNet rows and the cell recorded those as empty",
    )
    parser.add_argument(
        "--generate-retries", type=int, default=3,
        help="retries for transient generation failures -- 429, 5xx, read "
             "timeouts -- with backoff. A 4xx other than 429 is not retried",
    )
    parser.add_argument(
        "--episode-concurrency", type=int, default=1,
        help="episodes played at once within this cell. 1 is the old serial "
             "behaviour and the right value for a local llama.cpp started with "
             "--parallel 1. A hosted endpoint is the opposite case: it batches, "
             "and one request at a time leaves it idle. Raise LEAN_REPL_POOL_SIZE "
             "to match, or Lean becomes the new queue -- each REPL holds its own "
             "Mathlib environment in memory, so size it to the machine",
    )
    parser.add_argument(
        "--skip-generator-check", action="store_true",
        help="do not probe the generator before the first episode",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "independent plays per seed. Pass@k needs k>1: a single play at "
            "temperature 1 confounds the arm with the luck of one sample"
        ),
    )
    parser.add_argument(
        "--player",
        default="bfs",
        choices=("bfs", "whole_proof", "codex", "genai"),
        help=(
            "bfs plays one tactic per action from the goal state; goedel emits "
            "a whole proof and revises it from Lean's diagnostics. Different "
            "episodes, same token budget"
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="generated-token ceiling per episode; 0 leaves the player uncapped",
    )
    parser.add_argument(
        "--whole-proof-url",
        default="http://127.0.0.1:8081/v1",
        help="chat endpoint for the whole-proof player (goedel :8081, pythagoras :8082)",
    )
    parser.add_argument(
        "--whole-proof-model",
        default="",
        help=(
            "model id sent to that endpoint. Required when it serves more than "
            "one (e.g. the Mac Studio's LM Studio on :5678); leave empty for a "
            "single-model llama.cpp server, which ignores the field"
        ),
    )
    parser.add_argument(
        "--model-label",
        default="",
        help="name recorded on each episode; defaults to the player",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help=(
            "whole-proof revisions per episode. The baseline saw no episode "
            "succeed after attempt 2 in 24 tries, so a deeper cap buys little "
            "and costs most of the wall clock"
        ),
    )
    parser.add_argument(
        "--plans",
        type=Path,
        default=None,
        help=(
            "machine-written proof plans keyed by row name. Both arms must draw "
            "from one procedure: hand-written plans exist only for the seeds, so "
            "using them would confound problem difficulty with plan authorship"
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--codex-model",
        default="gpt-5.6-luna",
        help="hosted model driven through `codex exec` when --player codex",
    )
    parser.add_argument("--codex-timeout", type=float, default=300.0)
    parser.add_argument(
        "--prefill",
        default="",
        help=(
            "text the whole-proof model's turn is opened with, e.g. "
            "'```lean4\\nimport Mathlib\\n'. Empty reproduces the control cells"
        ),
    )
    parser.add_argument(
        "--provider",
        default="",
        help=(
            "JSON passed through as the request's `provider` block, for "
            "OpenAI-compatible routers that accept one. e.g. "
            "'{\"only\":[\"DeepInfra\"],\"quantizations\":[\"fp8\"],"
            "\"allow_fallbacks\":false}'"
        ),
    )
    parser.add_argument(
        "--reasoning-headroom", type=int, default=0,
        help="extra tokens to request beyond the proof allowance, for endpoints "
             "that count thinking inside max_tokens",
    )
    parser.add_argument(
        "--plan-outside-budget", action="store_true",
        help="charge only the fenced proof against the budget, not the plan written "
             "before it; for cells whose official prompt asks for a plan and which "
             "therefore run without a prefill (a prefilled reply carries a closing "
             "fence and no opening one, so the split would misread it)",
    )
    parser.add_argument(
        "--prefill-prefix", action="store_true",
        help="mark the prefill turn with `prefix: true` (required by Mistral)",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="environment variable holding this cell's credential, e.g. MISTRAL_API",
    )
    parser.add_argument(
        "--extra-body",
        default="",
        help="JSON merged into the request body, for per-model serving options",
    )
    parser.add_argument(
        "--prompt-style",
        default="goedel_v2",
        help=(
            "prompt dialect: goedel_v2 (reason, then a fenced block), goedel_v1 "
            "(minimalist 'complete this Lean 4 code', shared with "
            "DeepSeek-Prover-V1.5), or anything else for the generic instruct "
            "template. The protocol is that each model is driven in its own "
            "official format inside the shared budget; a model handed another "
            "model's dialect is measured on someone else's prompt, and the drop "
            "that comes out carries that difference"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep episodes already in the output JSONL and play only what is missing",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    arm = ARMS[args.arm]
    plans: Dict[str, Any] = (
        json.loads(args.plans.read_text(encoding="utf-8"))
        if args.plans and args.plans.is_file()
        else {}
    )
    if arm["plan"] and not plans:
        raise SystemExit(
            f"arm {args.arm!r} offers a proof plan but --plans was not supplied"
        )

    if arm["plan"]:
        missing = [
            str(r.get("name")) for r in rows
            if not (plans.get(str(r.get("name"))) or {}).get("plan")
        ]
        if missing:
            # Silent degradation is the failure to guard against: the arm would
            # still run, hand out lemmas only, and be recorded as `open_book`.
            raise SystemExit(
                f"{len(missing)} of {len(rows)} rows have no plan "
                f"(e.g. {', '.join(missing[:3])}) — the arm would quietly become "
                f"lemmas-only for them"
            )

    async def goedel_generate(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        import httpx

        url = args.whole_proof_url.rstrip("/") + "/chat/completions"
        body = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
        }
        # Omitted when the endpoint serves one model, which is why the local
        # llama.cpp servers never needed it. A host serving several — the Mac
        # Studio has both provers loaded on one port — either errors out or
        # picks for us, and picking wrong means a tactic-completion prover is
        # handed whole-proof prompts and its garbage is recorded as this
        # model's failures.
        if args.whole_proof_model:
            body["model"] = args.whole_proof_model
        # OpenRouter routes to whichever provider is cheapest unless told
        # otherwise, and providers serve the same weights at different
        # quantisations. Left free, one cell could be answered by an fp8 host on
        # Monday and a bf16 host on Tuesday, and the difference would sit inside
        # the drop. Pin the provider and the quantisation, and refuse fallbacks.
        if args.provider:
            body["provider"] = json.loads(args.provider)
        # Whatever else this model needs to be driven the way the cell declares
        # it -- `{"reasoning": {"enabled": true}}` for a model that reasons only
        # when asked, for instance. Kept general so a per-model quirk does not
        # become a per-model flag.
        if args.extra_body:
            body.update(json.loads(args.extra_body))
        # A local llama.cpp needs no credential; a hosted OpenAI-compatible
        # endpoint refuses without one, and the refusal arrives as an empty
        # reply that reads exactly like a dead server. Send the key whenever the
        # endpoint is not on this machine.
        headers = {}
        if not any(host in url for host in ("127.0.0.1", "localhost", "0.0.0.0")):
            # The cell names which variable holds its credential, because one
            # panel now spans three hosts with three different keys. Falling
            # back to a fixed name would send OpenRouter's key to Mistral, and
            # the refusal arrives as an empty reply that reads like a dead
            # server rather than a wrong credential.
            names = [args.api_key_env] if args.api_key_env else []
            names += ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
            key = next((os.environ[n] for n in names if n and os.environ.get(n)), "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        # A hosted endpoint fails in two quite different ways and the transport
        # used to treat them alike: a 400 means the request is wrong and will
        # stay wrong, while a 429, a 502 or a read timeout means come back in a
        # moment. Without a retry the second kind became `generator_empty` -- an
        # episode with no attempts, scored as the model failing to prove the row
        # -- and three in a row abort the cell. Across this panel that was 14
        # episodes on Mistral (read timeouts at the 360s ceiling), 10 on Venice
        # (seven 429s and three 502s), and two aborted cells. Retry the
        # transient ones; surface everything else immediately.
        timeout = args.generate_timeout or args.lean_timeout * 2
        payload = None
        last = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(args.generate_retries + 1):
                final = attempt == args.generate_retries
                try:
                    response = await client.post(url, json=body, headers=headers or None)
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    last = str(error)[:160]
                except Exception as error:
                    return {"text": "", "completion_tokens": 0, "error": str(error)[:160]}
                else:
                    if response.status_code not in _RETRY_STATUS:
                        try:
                            response.raise_for_status()
                        except Exception as error:
                            return {"text": "", "completion_tokens": 0,
                                    "error": str(error)[:160]}
                        payload = response.json()
                        break
                    last = f"HTTP {response.status_code}"
                    if not final:
                        # A 429 carries the wait the provider wants; honour it
                        # rather than guessing. The jitter keeps parallel cells
                        # from retrying in lockstep against the same host.
                        hinted = (response.headers.get("retry-after") or "").strip()
                        wait = float(hinted) if hinted.replace(".", "", 1).isdigit() else 0.0
                        await asyncio.sleep(max(wait, _backoff(attempt)))
                        continue
                if final:
                    break
                await asyncio.sleep(_backoff(attempt))
        if payload is None:
            return {"text": "", "completion_tokens": 0,
                    "error": f"{last or 'no response'} after {args.generate_retries} retries"}
        choices = payload.get("choices") or [{}]
        usage = payload.get("usage") or {}
        message = choices[0].get("message") or {}
        # Mistral returns a *list* of typed blocks once reasoning is switched on
        # -- `{"type": "thinking", ...}` then `{"type": "text", ...}` -- where
        # every other endpoint returns a string. Read as a string that is an
        # empty answer with a nonzero token count, which reads as a model that
        # produced nothing rather than a schema this code did not expect.
        raw_content = message.get("content")
        thinking_chars = 0
        if isinstance(raw_content, list):
            parts = []
            for block in raw_content:
                if not isinstance(block, dict):
                    parts.append(str(block)); continue
                kind = block.get("type")
                if kind == "text":
                    parts.append(str(block.get("text") or ""))
                elif kind == "thinking":
                    # The payload nests differently depending on how the turn was
                    # opened: a list of `{"type": "text", ...}` pieces in one
                    # shape, a bare string in another. Walk whatever is there.
                    inner = block.get("thinking")
                    if isinstance(inner, str):
                        thinking_chars += len(inner)
                    else:
                        for piece in (inner or []):
                            if isinstance(piece, dict):
                                thinking_chars += len(str(piece.get("text") or ""))
                            else:
                                thinking_chars += len(str(piece))
            text = "".join(parts)
        else:
            text = str(raw_content or "")
        # A reasoning model's thinking is generated but is not the artifact
        # being budgeted: the budget is what the model spends *writing the
        # proof*. xAI caps `max_tokens` against the visible content and lets
        # thinking run past it, while vLLM-style hosts cap the total, so the
        # same field means different things by provider. Reporting both keeps
        # the accounting the harness does independent of that difference.
        reasoning = int(
            ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
        )
        # Mistral reports one `completion_tokens` with thinking folded into it
        # and no separate count, so the budget cannot be charged by subtraction
        # there. Count the answer instead: the proof text is what the budget is
        # for, and it is the smaller and better-behaved of the two strings.
        # 2.97 characters per token measured on this model's own Lean output;
        # the 4.0 rule of thumb under-counts Unicode-dense Lean by a quarter.
        # NVIDIA's NIM endpoints put the thinking in a sibling field of the
        # message rather than in `usage` -- `reasoning_content`, alongside
        # `content` -- so a reader that only looks at
        # `completion_tokens_details` sees a reasoning model with no reasoning
        # and charges every thought against the proof budget. Count it here at
        # the prose rate; it is prose, not Lean.
        if not reasoning:
            side = message.get("reasoning_content")
            if isinstance(side, str) and side.strip():
                reasoning = max(1, round(len(side) / 4.0))
        if thinking_chars and not reasoning:
            answer_tokens = max(1, round(len(text) / 2.97)) if text.strip() else 0
            reasoning = max(int(usage.get("completion_tokens") or 0) - answer_tokens, 0)
        return {
            "text": text,
            "finish_reason": choices[0].get("finish_reason") or "",
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": reasoning,
        }

    async def genai_generate(
        messages: List[Dict[str, str]], max_tokens: int
    ) -> Dict[str, Any]:
        """Drive a Gemini model through google-genai, in the same shape.

        Vertex speaks its own protocol rather than the OpenAI one every other
        hosted cell uses, so the mapping is done here and the player above sees
        no difference: `r.text` is the content, and `thoughts_token_count` is
        the thinking. That last field is why this cell can be budgeted at all --
        Google bills thinking at the output rate but reports it separately, so
        the proof budget can charge the proof alone while the invoice charges
        both.
        """
        from google import genai
        from google.genai import types

        key = os.environ.get(args.api_key_env or "GEMINI_API_KEY", "")
        client = genai.Client(vertexai=True, api_key=key)
        prompt = "\n\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        )
        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
        )
        last = ""
        for attempt in range(args.generate_retries + 1):
            try:
                # Bound the call. The httpx path above carries a timeout and
                # this one did not: a request that never returned left the
                # episode waiting forever, and the cell sat at the same count
                # for sixty-nine minutes with a live process writing nothing.
                # `to_thread` cannot cancel the underlying call, so the thread
                # may linger, but the episode gives up and the retry proceeds.
                reply = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=args.whole_proof_model or "gemini-3.7-flash",
                        contents=prompt,
                        config=cfg,
                    ),
                    timeout=args.generate_timeout or args.lean_timeout * 2,
                )
            except asyncio.TimeoutError:
                last = f"no response within {args.generate_timeout or args.lean_timeout * 2:.0f}s"
                if attempt == args.generate_retries:
                    break
                await asyncio.sleep(_backoff(attempt))
                continue
            except Exception as error:
                last = str(error)[:160]
                if attempt == args.generate_retries:
                    break
                await asyncio.sleep(_backoff(attempt))
                continue
            usage = reply.usage_metadata
            content = reply.text or ""
            thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
            answer = int(getattr(usage, "candidates_token_count", 0) or 0)
            return {
                "text": content,
                "finish_reason": str(
                    reply.candidates[0].finish_reason if reply.candidates else ""
                ),
                "completion_tokens": answer + thoughts,
                "reasoning_tokens": thoughts,
            }
        return {"text": "", "completion_tokens": 0,
                "error": f"{last or 'no response'} after {args.generate_retries} retries"}

    async def codex_generate(
        messages: List[Dict[str, str]], max_tokens: int
    ) -> Dict[str, Any]:
        """Drive a hosted model through `codex exec`, in the same shape.

        This player is a reference ceiling rather than a fourth entry in the
        budgeted comparison. `codex exec` reports no usage, so the generated-
        token budget that makes BFS and Goedel commensurable cannot be applied
        here, and it takes no assistant turn, so the prefill that cut Goedel's
        wasted output by five-sixths is unavailable too. The episode therefore
        ends on attempts alone. That is sound for the question being asked —
        the reported quantity is the control-minus-treatment drop *within* a
        model — but it does not make this model's absolute score comparable to
        the local two, and the runs record `token_budget: 0` to say so.
        """
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"),
            "You are an expert in Lean 4 and Mathlib.",
        )
        user = "\n\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        )
        reply = await call_codex_cli(
            model=args.codex_model,
            system=system,
            user=user,
            timeout_seconds=args.codex_timeout,
        )
        return {
            "text": reply.raw_text,
            "finish_reason": reply.finish_reason,
            # Nothing to report: `codex exec` does not return usage. Zero keeps
            # the player's budget test permanently false rather than faking a number.
            "completion_tokens": 0,
        }

    async def sampler(prompt: str, n: int) -> List[Any]:
        return await llama_cpp_scored_sampler(
            prompt, n,
            base_url=args.llama_cpp,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    label = args.model_label or args.player
    episodes = args.output / f"episodes_{label}_{args.arm}.jsonl"

    # A cell is 300 episodes at minutes apiece, so a run that is interrupted
    # partway — a reboot, a wedged server, a decision to move the work to
    # another machine — should not cost the episodes already played. Each one
    # is appended as it finishes, so the file itself is the record of what is
    # done; resuming reads back the (seed, attempt) pairs and skips them.
    records: List[Dict[str, Any]] = []
    done: set = set()
    if args.resume and episodes.is_file():
        for line in episodes.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records.append(record)
                done.add((str(record.get("seed")), int(record.get("attempt") or 0)))
        print(f"resuming: {len(done)} episodes already played", flush=True)
    else:
        episodes.write_text("", encoding="utf-8")

    # Ask the generator one cheap question before opening the floodgates. The
    # three-consecutive-empties guard below cannot help here: it fires on
    # results, and at concurrency 128 a hundred and twenty-eight requests are
    # already in flight before the first result comes back. That is not
    # hypothetical -- a FriendliAI endpoint that had scaled itself down to zero
    # answered 503 to the first 129 episodes of a cell, every one of them
    # recorded as the model failing to prove its row, before anything could
    # stop it. One request costs a second and settles it.
    if args.player in ("whole_proof", "genai") and not args.skip_generator_check:
        # Probe the generator this cell actually uses. Calling `goedel_generate`
        # unconditionally sent the check to `--whole-proof-url`, which a genai
        # cell never touches: the Gemini cell refused to start against a local
        # llama.cpp that was not running and had nothing to do with it.
        probe_fn = genai_generate if args.player == "genai" else goedel_generate
        probe = await probe_fn(
            [{"role": "user", "content": "Reply with the single word: ok"}], 64
        )
        if not str(probe.get("text") or "").strip() and not probe.get("completion_tokens"):
            raise SystemExit(
                f"refusing to start: {args.whole_proof_url} did not answer a "
                f"one-token probe ({probe.get('error') or 'empty reply'}). A "
                f"dedicated endpoint may have scaled to zero and need a moment; "
                f"a shared one may be down. Check it, then re-run with --resume."
            )
        print("generator OK", flush=True)

    consecutive_empty = 0
    # One episode is one network round trip after another, and the loop used to
    # wait through all of them before starting the next episode. That is the
    # whole cost on a hosted cell: the H100 endpoint accepts a batch of 128 and
    # was being fed one request at a time, and OpenRouter the same. The Lean
    # side already supports the concurrency -- `LEAN_REPL_POOL_SIZE` keeps N
    # independent REPLs, each behind its own lock -- so the only thing missing
    # was letting episodes overlap. Ordering is not part of the measurement:
    # every episode is independent, `--resume` keys on (seed, attempt), and the
    # file is appended under the same single-threaded event loop.
    gate = asyncio.Semaphore(max(1, args.episode_concurrency))
    write_lock = asyncio.Lock()
    aborted: List[str] = []

    async def play(attempt: int, index: int, row: Dict[str, Any]) -> None:
        nonlocal consecutive_empty
        async with gate:
            if aborted:
                return
            name = str(row.get("name") or row.get("stem"))
            if (name, attempt) in done:
                return
            palette = row.get("palette") or {"tactics": {}, "theorems": {}}
            env = LeanExamEnv(
                formal_statement=str(row.get("formal_statement") or ""),
                lean_header=str(row.get("lean_header") or "import Mathlib"),
                palette=palette if arm["palette"] else {"tactics": {}, "theorems": {}},
                verifier=verify_lean_proof_repl,
                max_steps=args.max_steps,
                lean_timeout=args.lean_timeout,
                strict_steps=False,  # BFS emits one tactic at a time natively
            )
            # The hints arm plays the row's level-3 rung — the first tactic of
            # the known proof — as its opening move. Rows flagged
            # `single_line_proof` are excluded from that arm's headline: for
            # them the hint *is* the proof, so a pass measures the hint rather
            # than the prover.
            ladder = {h.get("level"): h for h in (row.get("hint_ladder") or [])}
            started = time.monotonic()
            try:
                if args.player in ("whole_proof", "codex", "genai"):
                    # The affordances arrive where this model can use them: the
                    # palette as prompt context, the outline as the "detailed
                    # proof plan" its reference prompt already asks for.
                    result = await GoedelExamPlayer(
                        codex_generate if args.player == "codex"
                        else genai_generate if args.player == "genai"
                        else goedel_generate,
                        max_tokens=args.max_tokens,
                        token_budget=args.token_budget or 8192,
                        max_attempts=args.max_attempts,
                        palette_names=(
                            sorted((row.get("palette") or {}).get("theorems") or {})
                            if arm["palette"] else None
                        ),
                        hint_outline=(
                            (plans.get(name) or {}).get("plan")
                            if arm["plan"] else None
                        ),
                        prefill=args.prefill,
                        prompt_style=args.prompt_style,
                        prefill_prefix_flag=args.prefill_prefix,
                        reasoning_headroom=args.reasoning_headroom,
                        plan_outside_budget=args.plan_outside_budget,
                        nl_statement=str(row.get("statement_nl") or ""),
                    ).play(env)
                else:
                    result = await BFSExamPlayer(
                        sampler,
                        n_per_step=args.n_per_step,
                        resample_rounds=1,
                        max_rollbacks=arm["rollback"],
                        use_palette=arm["palette"],
                        # The opening move is deliberately not part of
                        # `open_book`: it hands over a piece of the answer, and
                        # on the 16 single-line rows it *is* the answer.
                        seed_tactic=(
                            str((ladder.get(3) or {}).get("content") or "")
                            if args.arm == "hints" else ""
                        ),
                        token_budget=args.token_budget,
                    ).play(env, max_actions=args.max_actions)
            except Exception as error:
                result = {
                    "success": False, "steps": [], "actions": 0, "rejected": 0,
                    "filtered": 0, "rollbacks": 0, "sampled_batches": 0,
                    "solved_code": "", "error": f"{type(error).__name__}: {error}"[:200],
                }
            # Read off the environment rather than the player: both players hit
            # this the same way — reset refuses the statement, `done` is already
            # true, and their loop never runs — so asking the environment once
            # here covers both without either needing to know about it.
            opening = (env.transcript[0]["observation"] if env.transcript else {}) or {}
            if opening.get("status") == "error":
                # Which of the two refusals it was decides whether the row is
                # out of the corpus or merely needs replaying: a rejected
                # statement is a permanent property of the row, a REPL that
                # timed out or returned unparseable JSON is not.
                if env.reset_error_kind == "verifier":
                    result["verifier_failed"] = True
                else:
                    result["statement_invalid"] = True
                result["statement_error"] = str(opening.get("message") or "")[:300]
            record = {
                "seed": name,
                "attempt": attempt,
                "benchmark": row.get("benchmark") or "proofnet_verified",
                "arm": args.arm,
                "tag": args.tag,
                "gt_step_count": row.get("gt_step_count"),
                "hint_caveat": sorted(
                    k for k, v in (row.get("hint_degeneracy") or {}).items() if v
                ),
                "outcome": outcome_of(result, args.max_actions),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "player": args.player,
                "model": args.model_label or args.player,
                # What this episode was actually handed, so the per-model
                # asymmetry is visible in the data rather than only in the code.
                # Every entry tests the aid's *contents*, not the arm's config
                # flag. An open-book arm over rows with empty palettes is a
                # closed-book run, and recording the flag would label it as
                # aided — which is how a corpus with no palettes would produce
                # a clean "aid does not help here" result with evidence
                # attached.
                "aid_delivered": sorted(
                    k for k, v in {
                        "lemmas": arm["palette"] and bool(palette.get("theorems")),
                        "plan": arm["plan"] and args.player == "whole_proof"
                                and bool((plans.get(name) or {}).get("plan")),
                        "opening": args.arm == "hints" and args.player == "bfs"
                                   and bool((ladder.get(3) or {}).get("content")),
                    }.items() if v
                ),
                # The whole budget, on every episode. Three of these used to
                # live only in the summary JSON, and `max_actions` was one of
                # them — which is how a control cell at 200 actions was compared
                # against a treatment cell at 40 for a full campaign without the
                # difference being visible anywhere the comparison was made.
                # A number that decides a result belongs beside the result.
                "n_per_step": args.n_per_step,
                "max_tokens": args.max_tokens,
                "max_actions": args.max_actions,
                "max_attempts": args.max_attempts,
                "max_steps": args.max_steps,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "prefill_used": bool(args.prefill),
                "prompt_style": args.prompt_style,
                "reasoning_headroom": args.reasoning_headroom,
                "plan_outside_budget": args.plan_outside_budget,
                "provider": args.provider,
                "extra_body": args.extra_body,
                **{k: v for k, v in result.items() if k != "solved_code"},
                "solved_code": result.get("solved_code", ""),
            }
            async with write_lock:
                records.append(record)
                with episodes.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # A dead generator does not recover on its own, and a run that
            # plays on against one produces a full-length file of episodes that
            # look like model failures. Stop while the file is still short
            # enough to be obviously partial, and resumable.
            consecutive_empty = (
                consecutive_empty + 1
                if record["outcome"] in UNMEASURED else 0
            )
            if consecutive_empty >= EMPTY_RUN_ABORT:
                # Name the generator this run actually used. The codex player
                # never touches `--whole-proof-url`, and printing it sends
                # whoever reads the abort to a healthy llama.cpp server while
                # the real fault — a revoked codex session — goes unlooked-at.
                where = (
                    f"`codex exec --model {args.codex_model}`"
                    if args.player == "codex"
                    else f"the server at {args.whole_proof_url}"
                    if args.player == "whole_proof"
                    else f"the sampler at {args.llama_cpp}"
                )
                aborted.append(
                    f"aborting: {consecutive_empty} consecutive episodes came back "
                    f"empty — {where} is not answering. Check it directly before "
                    f"resuming. {len(records)} episodes kept in {episodes}; "
                    f"restart with --resume once it is back, after clearing the "
                    f"empty rows with scripts/evaluate/drop_unmeasured_episodes.py."
                )
                return
            print(
                f"[{record['outcome']:16s}] a{attempt} {index:2d}/{len(rows)} {name[:34]:34s} "
                f"steps={len(result.get('steps') or [])} "
                f"{'att=' + str(result.get('attempts')) if 'attempts' in result else 'act=' + str(result.get('actions'))} "
                f"rej={result.get('rejected')} tok={result.get('tokens_used', 0)} "
                f"{record['elapsed_seconds']:.0f}s",
                flush=True,
            )
    try:
        for attempt in range(1, args.attempts + 1):
            pending = [
                play(attempt, index, row)
                for index, row in enumerate(rows, 1)
                if (str(row.get("name") or row.get("stem")), attempt) not in done
            ]
            await asyncio.gather(*pending)
            if aborted:
                break
    finally:
        await close_global_repl_verifier()
    if aborted:
        raise SystemExit(aborted[0])

    by_seed: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(record["seed"], []).append(record)
    solved_any = sum(
        1 for plays in by_seed.values() if any(p["outcome"] == "solved" for p in plays)
    )
    solved_first = sum(
        1
        for plays in by_seed.values()
        if any(p["outcome"] == "solved" for p in plays if p["attempt"] == 1)
    )
    summary = {
        "seeds": len(by_seed),
        "episodes": len(records),
        "attempts": args.attempts,
        "arm": args.arm,
        "model": label,
        f"pass_at_{args.attempts}": solved_any,
        "pass_at_1": solved_first,
        "solved_episodes": sum(1 for r in records if r["outcome"] == "solved"),
        "outcomes": dict(Counter(r["outcome"] for r in records).most_common()),
        # Rejections are the higher-information signal: a seed either passes or
        # not, but how many candidates Lean threw out says how far the search
        # was from the goal, and it moves even when the verdict does not.
        "rejected_total": sum(int(r.get("rejected") or 0) for r in records),
        "rejected_median_per_episode": sorted(
            int(r.get("rejected") or 0) for r in records
        )[len(records) // 2] if records else 0,
        # The summary carried `max_actions` but not `token_budget`; the episodes
        # carried `token_budget` but not `max_actions`. Neither file alone could
        # answer "did these two cells run under the same budget?", which is how
        # a 200-vs-40 action gap survived a whole campaign. Both are complete now.
        "config": {
            "n_per_step": args.n_per_step, "max_tokens": args.max_tokens,
            "max_steps": args.max_steps, "max_actions": args.max_actions,
            "max_attempts": args.max_attempts, "token_budget": args.token_budget,
            "temperature": args.temperature, "top_p": args.top_p,
            "prefill_used": bool(args.prefill), "prompt_style": args.prompt_style,
            "player": args.player,
            "rows": str(args.rows),
        },
        "median_seconds": sorted(r["elapsed_seconds"] for r in records)[len(records) // 2]
        if records else 0,
    }
    (args.output / f"summary_{label}_{args.arm}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
    # The episode data and the summary are both on disk by the time `main`
    # returns, but the process has repeatedly failed to exit after that —
    # something in the verifier teardown holds the loop open with no work left
    # and no REPL alive. A run that has written its results is finished, so it
    # leaves rather than waiting to be killed, which is what stalled the queue
    # for six hours after the first cell completed.
    os._exit(0)
