"""OpenRouter-backed model panel for direct no-tool evaluation.

Three canonical models, matching EntropyMaG-1:
- GPT-5.4-mini    (openai/gpt-5.4-mini)
- Claude Haiku 4.5 (anthropic/claude-haiku-4.5)
- Gemini 3.1 Flash Lite (google/gemini-3.1-flash-lite)

All traffic is routed through OpenRouter so a single OPENROUTER_API_KEY drives
the panel. Override slugs via environment variables if a provider renames a
model between runs.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - import-time guard
    AsyncOpenAI = None  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelConfig:
    """Identification + decoding parameters for one panel member.

    `backend`:    "openrouter" (default) | "lm_studio" | "codex_cli"
    `paradigm`:   "chat" (whole-proof + refinement) | "completion" (raw
                  text-completion API used by tactic-step provers like
                  BFS-Prover-V2)
    `stop`:       optional stop strings forwarded to the provider when the
                  paradigm needs them (e.g. tactic-step provers stop at the
                  ``:::`` separator or a newline).
    `prompt_style`: chat-paradigm prompt dialect.
                  ``"default"`` (EML-1 system + INITIAL_USER_TEMPLATE) is
                  used for OpenRouter generalist models. ``"goedel_v2"``
                  matches the Goedel-Prover-V2 HF model card recipe —
                  empty system message + a user prompt that explicitly
                  asks the model to emit a proof plan *before* the
                  ```lean4 ... ``` block (V2 is trained to refuse to
                  skip the plan, so suppressing it costs Pass@K).
    `first_temperature` / `refine_temperature`: optional per-model
                  overrides of the multi-turn loop's sampling temperature
                  for the first attempt vs. verifier-driven refinement
                  turns. Leave ``None`` to inherit
                  ``prove_with_refinement``'s defaults (1.0 / 0.0).
    `temperature` / `max_tokens` are per-cell decoding overrides; both
    backends honour them. For chat-paradigm models ``temperature`` is
    effectively dead — use ``first_temperature``/``refine_temperature``
    instead.
    `top_p` / `seed`: optional provider decoding controls. Evaluation
                  orchestration derives a distinct deterministic seed for
                  each model call from this base seed.
    """

    label: str
    provider_slug: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    backend: str = "openrouter"
    paradigm: str = "chat"
    stop: Optional[List[str]] = None
    prompt_style: str = "default"
    first_temperature: Optional[float] = None
    refine_temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None


def _env_slug(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)


def _default_panel() -> List[ModelConfig]:
    """Pick the default panel based on env config.

    - If `EVAL_PANEL=local` (or `LM_STUDIO_BASE_URL` is set without
      `EVAL_PANEL=openrouter`), serve the local LM-Studio panel with one
      whole-proof Lean prover (Goedel-Prover-V2-8B, chat paradigm) and
      one tactic-step Lean prover (BFS-Prover-V2-7B, completion
      paradigm). Either slot can be disabled by setting its
      ``EVAL_*_SLUG`` env var to the literal string ``"disabled"``.
    - Otherwise, serve the OpenRouter chat panel previously used by
      EntropyMaG-1 (3 closed-source generalist LLMs).
    """
    panel_mode = os.getenv("EVAL_PANEL", "").lower()
    if panel_mode == "local" or (panel_mode != "openrouter" and os.getenv("LM_STUDIO_BASE_URL")):
        panel: List[ModelConfig] = []
        goedel_slug = _env_slug("EVAL_GOEDEL_V2_SLUG", "goedel-prover-eval")
        if goedel_slug and goedel_slug.lower() != "disabled":
            panel.append(
                ModelConfig(
                    label="Goedel-Prover-V2-8B",
                    provider_slug=goedel_slug,
                    backend="lm_studio",
                    paradigm="chat",
                    # Goedel-Prover-V2's official pipeline is CoT-first:
                    # generate a proof plan, extract only the fenced Lean
                    # block, then splice the proof body onto the original
                    # statement before verifier scoring.  Our evaluator now
                    # mirrors that contract in ``goedel_v2`` mode.
                    prompt_style=os.getenv("GOEDEL_PROMPT_STYLE", "goedel_v2"),
                    first_temperature=float(os.getenv("GOEDEL_TEMPERATURE", "1.0")),
                    top_p=float(os.getenv("GOEDEL_TOP_P", "0.95")),
                    max_tokens=int(os.getenv("GOEDEL_MAX_TOKENS", "6144")),
                    seed=int(os.getenv("GOEDEL_SEED", "30")),
                )
            )
        bfs_slug = _env_slug("EVAL_BFS_PROVER_SLUG", "bfs-prover-eval")
        if bfs_slug and bfs_slug.lower() != "disabled":
            panel.append(
                ModelConfig(
                    label="BFS-Prover-V2-7B",
                    provider_slug=bfs_slug,
                    backend="lm_studio",
                    paradigm="completion",
                    temperature=float(os.getenv("BFS_TEMPERATURE", "0.7")),
                    top_p=float(os.getenv("BFS_TOP_P", "1.0")),
                    max_tokens=int(os.getenv("BFS_MAX_TOKENS", "256")),
                    seed=int(os.getenv("BFS_SEED", "30")),
                    stop=[":::", "\n\n"],
                )
            )
        return panel
    return [
        ModelConfig(
            label="GPT-5.4-mini",
            provider_slug=_env_slug("EVAL_GPT_SLUG", "openai/gpt-5.4-mini"),
        ),
        ModelConfig(
            label="Claude Haiku 4.5",
            provider_slug=_env_slug("EVAL_CLAUDE_SLUG", "anthropic/claude-haiku-4.5"),
        ),
        ModelConfig(
            label="Gemini 3.1 Flash Lite",
            provider_slug=_env_slug(
                "EVAL_GEMINI_SLUG", "google/gemini-3.1-flash-lite"
            ),
        ),
    ]


MODEL_PANEL: List[ModelConfig] = _default_panel()


@dataclass
class ModelResponse:
    """Single (model, problem, repeat) outcome before grading."""

    model_label: str
    provider_slug: str
    problem_id: str
    repeat_index: int
    raw_text: str
    extracted_answer: Optional[str]
    finish_reason: Optional[str]
    elapsed_seconds: float
    error: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


def _client(backend: str = "openrouter") -> "AsyncOpenAI":
    """Return an AsyncOpenAI client wired to the chosen backend.

    - ``"openrouter"`` reads ``OPENROUTER_API_KEY`` (falls back to
      ``OPENAI_API_KEY``) and ``OPENROUTER_BASE_URL``.
    - ``"lm_studio"`` reads ``LM_STUDIO_BASE_URL`` (default
      ``http://127.0.0.1:1234/v1``) and uses a no-op api_key because
      LM Studio's local server does not authenticate.
    """
    if AsyncOpenAI is None:
        raise RuntimeError(
            "openai package not available; install openai>=1.0.0 for evaluation"
        )
    if backend == "lm_studio":
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        # LM Studio ignores the key but the SDK requires a non-empty value.
        return AsyncOpenAI(api_key="lm-studio", base_url=base_url)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY (or OPENAI_API_KEY) is required for evaluation"
        )
    base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _client_for(config: ModelConfig) -> "AsyncOpenAI":
    """Per-ModelConfig client cache. Always returns a fresh client because
    AsyncOpenAI is cheap to instantiate; the orchestrator decides when to
    close them."""
    return _client(config.backend)


async def _call_one(
    client: "AsyncOpenAI",
    config: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_seconds: float,
) -> Dict[str, object]:
    """Call the configured backend with one of two paradigms.

    Paradigm = ``"chat"`` → ``/v1/chat/completions`` with a
    system+user messages array (DeepSeek-Prover-V1.5's quick_start.py
    convention; the default for closed-source frontier models).

    Paradigm = ``"completion"`` → ``/v1/completions`` with a raw prompt.
    Used by tactic-step provers (BFS-Prover-V2) that expect a single
    string of the form ``"{state}:::"`` and return ``"{state}::: {tactic}"``.
    """
    started = time.monotonic()
    try:
        if config.paradigm == "completion":
            request: Dict[str, object] = {
                "model": config.provider_slug,
                "prompt": user_prompt,  # system prompt ignored in completion mode
                "temperature": config.temperature,
            }
            if config.max_tokens is not None:
                request["max_tokens"] = config.max_tokens
            if config.top_p is not None:
                request["top_p"] = config.top_p
            if config.seed is not None:
                request["seed"] = config.seed
            if config.stop:
                request["stop"] = list(config.stop)
            response = await asyncio.wait_for(
                client.completions.create(**request, timeout=timeout_seconds),
                timeout=timeout_seconds,
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return {
                "raw_text": getattr(choice, "text", "") or "",
                "finish_reason": getattr(choice, "finish_reason", None),
                "elapsed": time.monotonic() - started,
                "error": None,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
            }

        # default chat paradigm
        request = {
            "model": config.provider_slug,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.seed is not None:
            request["seed"] = config.seed
        if config.stop:
            request["stop"] = list(config.stop)
        response = await asyncio.wait_for(
            client.chat.completions.create(**request, timeout=timeout_seconds),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return {
            "raw_text": "",
            "finish_reason": "timeout",
            "elapsed": time.monotonic() - started,
            "error": f"timeout after {timeout_seconds}s",
            "usage": {},
        }
    except Exception as exc:
        return {
            "raw_text": "",
            "finish_reason": "error",
            "elapsed": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": {},
        }
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    return {
        "raw_text": choice.message.content or "",
        "finish_reason": getattr(choice, "finish_reason", None),
        "elapsed": time.monotonic() - started,
        "error": None,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        },
    }


async def run_model_panel(
    problem_id: str,
    statement: str,
    *,
    repeats: int = 3,
    models: Optional[List[ModelConfig]] = None,
    timeout_seconds: float = 180.0,
    answer_extractor: Optional[Callable[[str], Optional[str]]] = None,
    client: Optional["AsyncOpenAI"] = None,
    prompt_builder: Optional[Callable[[str], "tuple[str, str]"]] = None,
) -> List[ModelResponse]:
    """Run the full panel (models × repeats) for a single problem.

    Returns one ``ModelResponse`` per (model, repeat) cell. The caller is
    responsible for downstream grading and aggregation.
    """
    from src.evaluation.answer_grader import extract_boxed_answer
    from src.evaluation.prompts import build_direct_prompt

    extract = answer_extractor or extract_boxed_answer
    build = prompt_builder or build_direct_prompt
    panel = list(models or MODEL_PANEL)
    system_prompt, user_prompt = build(statement)
    own_client = client is None
    cli = client or _client()

    async def _one(cfg: ModelConfig, repeat: int) -> ModelResponse:
        result = await _call_one(
            cli,
            cfg,
            system_prompt,
            user_prompt,
            timeout_seconds=timeout_seconds,
        )
        raw_text = str(result["raw_text"] or "")
        return ModelResponse(
            model_label=cfg.label,
            provider_slug=cfg.provider_slug,
            problem_id=problem_id,
            repeat_index=repeat,
            raw_text=raw_text,
            extracted_answer=extract(raw_text),
            finish_reason=str(result["finish_reason"]) if result["finish_reason"] else None,
            elapsed_seconds=float(result["elapsed"]),
            error=str(result["error"]) if result["error"] else None,
            usage=dict(result["usage"] or {}),
        )

    coros = [_one(cfg, repeat) for cfg in panel for repeat in range(repeats)]
    try:
        return await asyncio.gather(*coros)
    finally:
        if own_client:
            close = getattr(cli, "close", None)
            if close is not None:
                maybe: Awaitable[None] | None = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
