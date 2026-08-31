"""LLM-assisted harder problem generation for Lean-certifiable families."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import langsmith as ls
from pydantic import BaseModel, Field

from src.no_go_policy import build_no_go_policy_pack, format_no_go_policy_pack
from src.utils.lean_templates import SUPPORTED_FAMILIES, detect_family
from src.utils.codex_cli import call_codex_cli, call_codex_cli_sync

if TYPE_CHECKING:
    from src.certification.certifier import CertificationInput


DEFAULT_GENERATION_MODEL = "moonshotai/kimi-k2.6"
SUPPORTED_FAMILY_NAMES = sorted(SUPPORTED_FAMILIES)
DERIVED_PARAM_KEYS = {
    "gcd_divisor_sum": {"gcd"},
    "divisor_sum_mod": {"modulus"},
}
FAMILY_INPUT_KEYS = {
    "gcd": ["a", "b"],
    "gcd_divisor_sum": ["a", "b"],
    "units_digit": ["base", "exp"],
    "divisor_sum": ["n"],
    "divisor_sum_mod": ["n", "a"],
    "stars_and_bars": ["vars", "sum"],
    "arithmetic_series": ["n_terms", "first", "diff"],
    "modular_congruence": ["a", "m"],
}
PARAM_DESCRIPTIONS = {
    "a": "GCD input, modular dividend, or divisor_sum_mod dividend.",
    "b": "Second GCD input.",
    "n": "Divisor-sum input.",
    "base": "Units digit base.",
    "exp": "Units digit exponent.",
    "vars": "Stars-and-bars variable count.",
    "sum": "Stars-and-bars total sum.",
    "n_terms": "Arithmetic series term count.",
    "first": "Arithmetic series first term.",
    "diff": "Arithmetic series common difference.",
    "m": "Modular congruence modulus.",
}


class GeneratedProblem(BaseModel):
    """A generated child problem after canonicalization."""

    id: str
    source_problem_id: str
    family: str
    statement: str
    answer: str
    solution: str = ""
    difficulty_label: str = "medium"
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning_pattern: str = ""
    solution_skeleton: Dict[str, Any] = Field(default_factory=dict)
    projected_params: Dict[str, Any] = Field(default_factory=dict)
    projection_check: Dict[str, Any] = Field(default_factory=dict)
    harder_reason: str = ""
    axis_applied: str = ""
    axis_aligned: bool = False
    raw_llm_output: Dict[str, Any] = Field(default_factory=dict)


class PlannerContractError(ValueError):
    """Generated problem did not follow the orchestrator slot contract."""


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for one LLM generation call."""

    model: str = DEFAULT_GENERATION_MODEL
    temperature: float = 0.3


def default_generation_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> GenerationConfig:
    return GenerationConfig(
        model=(
            model
            or os.getenv("GENERATION_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_GENERATION_MODEL
        ),
        temperature=float(
            temperature if temperature is not None else os.getenv("GENERATION_TEMPERATURE", "0.3")
        ),
    )


def verification_config(config: GenerationConfig) -> GenerationConfig:
    """Config for judging roles, on a different model than the generator.

    A verifier that shares the generator's weights is grading a sibling of its
    own output. ``VERIFICATION_MODEL`` names the model used for statement/Lean
    alignment and for the goal-roundtrip roles; when unset the generator's own
    model is reused and the separation is only informational.
    """
    model = os.getenv("VERIFICATION_MODEL", "").strip()
    if not model or model == config.model:
        return config
    return GenerationConfig(model=model, temperature=config.temperature)


def orchestrator_config(config: GenerationConfig) -> GenerationConfig:
    """Config for planning roles, which need not be the generator's model.

    Planning a slot and filling it are different jobs. The planner reads the
    pool, the lesson memory, and the failure history to decide which parents
    meet and what the child should attempt; the worker writes one artifact
    against that decision. The planner is the smaller number of calls and the
    larger share of the outcome, so it is worth spending a stronger model there
    even when the worker stays cheap.

    ``ORCHESTRATOR_MODEL`` names it, and covers slot planning, replanning, pool
    selection, and generation-zero planning. Unset, the generator's model is
    reused and the two roles are the same model, as they were before.
    """
    model = os.getenv("ORCHESTRATOR_MODEL", "").strip()
    if not model or model == config.model:
        return config
    return GenerationConfig(model=model, temperature=config.temperature)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:80] or "generated"


def _int_param(params: Dict[str, Any], name: str, *, low: int, high: int) -> int:
    if name not in params:
        raise ValueError(f"param {name} missing from params")
    value = int(params[name])
    if value < low or value > high:
        raise ValueError(f"param {name}={value} outside [{low}, {high}]")
    return value


def _sum_of_divisors(n: int) -> int:
    return sum(d for d in range(1, n + 1) if n % d == 0)


def _factorization_text(n: int) -> str:
    value = int(n)
    factors: list[str] = []
    d = 2
    while d * d <= value:
        exponent = 0
        while value % d == 0:
            exponent += 1
            value //= d
        if exponent:
            factors.append(f"{d}^{exponent}" if exponent > 1 else str(d))
        d += 1 if d == 2 else 2
    if value > 1:
        factors.append(str(value))
    return " * ".join(factors) if factors else str(n)


def _sigma_prime_power_factor(p: int, exponent: int) -> int:
    return sum(p**i for i in range(exponent + 1))


def _sigma_factor_terms(n: int) -> tuple[str, int]:
    value = int(n)
    terms: list[str] = []
    total = 1
    d = 2
    while d * d <= value:
        exponent = 0
        while value % d == 0:
            exponent += 1
            value //= d
        if exponent:
            term = _sigma_prime_power_factor(d, exponent)
            terms.append(f"sigma({d}^{exponent})={term}" if exponent > 1 else f"sigma({d})={term}")
            total *= term
        d += 1 if d == 2 else 2
    if value > 1:
        term = value + 1
        terms.append(f"sigma({value})={term}")
        total *= term
    return ", ".join(terms), total


def _deterministic_solution(family: str, params: Dict[str, Any], answer: str) -> str:
    """Write a compact solution from canonical params for supported templates."""
    try:
        if family == "gcd":
            a = int(params["a"])
            b = int(params["b"])
            return f"Using the Euclidean algorithm or prime factorization, GCD({a}, {b}) = {answer}. Answer: {answer}."
        if family == "gcd_divisor_sum":
            a = int(params["a"])
            b = int(params["b"])
            g = math.gcd(a, b)
            factor_text = _factorization_text(g)
            sigma_terms, sigma_value = _sigma_factor_terms(g)
            return (
                f"First compute GCD({a}, {b}) = {g}. Factor n = {g} = {factor_text}. "
                f"Then {sigma_terms}, so the sum of positive divisors of n is {sigma_value}. "
                f"Answer: {answer}."
            )
        if family == "units_digit":
            base = int(params["base"])
            exp = int(params["exp"])
            return (
                f"The units digits of powers of {base} repeat modulo 10. "
                f"Computing {base}^{exp} mod 10 gives {answer}. Answer: {answer}."
            )
        if family == "divisor_sum":
            n = int(params["n"])
            factor_text = _factorization_text(n)
            sigma_terms, sigma_value = _sigma_factor_terms(n)
            return (
                f"Factor {n} = {factor_text}. Using multiplicativity of sigma, "
                f"{sigma_terms}, so sigma({n}) = {sigma_value}. Answer: {answer}."
            )
        if family == "divisor_sum_mod":
            n = int(params["n"])
            a = int(params["a"])
            factor_text = _factorization_text(n)
            sigma_terms, modulus = _sigma_factor_terms(n)
            remainder = a % modulus
            quotient = a // modulus
            return (
                f"Factor {n} = {factor_text}. Using multiplicativity of sigma, "
                f"{sigma_terms}, so sigma({n}) = {modulus}. Thus m = {modulus}. "
                f"Since {a} = {quotient} * {modulus} + {remainder}, {a} mod m = {remainder}. "
                f"Answer: {answer}."
            )
        if family == "stars_and_bars":
            vars_count = int(params["vars"])
            total = int(params["sum"])
            return (
                f"By stars and bars, the number of non-negative solutions is "
                f"C({total}+{vars_count}-1, {vars_count}-1) = {answer}. Answer: {answer}."
            )
        if family == "arithmetic_series":
            n_terms = int(params["n_terms"])
            first = int(params["first"])
            diff = int(params["diff"])
            last = first + diff * (n_terms - 1)
            return (
                f"The last term is {first} + ({n_terms}-1)*{diff} = {last}. "
                f"The arithmetic sum is {n_terms}*({first}+{last})/2 = {answer}. Answer: {answer}."
            )
        if family == "modular_congruence":
            a = int(params["a"])
            m = int(params["m"])
            q = a // m
            r = a % m
            return f"Divide {a} by {m}: {a} = {q} * {m} + {r}. Therefore {a} mod {m} = {r}. Answer: {answer}."
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ""
    return ""


def _canonical_problem_from_params(
    *,
    source_problem_id: str,
    family: str,
    params: Dict[str, Any],
    raw: Dict[str, Any],
) -> GeneratedProblem:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported generated family: {family}")

    difficulty_label = str(raw.get("difficulty_label") or "medium")
    harder_reason = str(raw.get("harder_reason") or "")
    raw_solution = str(raw.get("solution") or "")
    axis_applied = str(raw.get("axis_applied") or "")
    axis_alignment = raw.get("axis_alignment") or {}
    axis_aligned = bool(axis_alignment.get("passed")) if isinstance(axis_alignment, dict) else False
    reasoning_pattern = str(raw.get("reasoning_pattern") or "")
    solution_skeleton = raw.get("solution_skeleton") or {}
    projected_params = raw.get("projected_params") or params
    projection_check = raw.get("projection_check") or {}
    if not isinstance(solution_skeleton, dict):
        solution_skeleton = {"raw": str(solution_skeleton)}
    if not isinstance(projected_params, dict):
        projected_params = {}
    if not isinstance(projection_check, dict):
        projection_check = {}

    if family == "gcd":
        a = _int_param(params, "a", low=2, high=50000)
        b = _int_param(params, "b", low=2, high=50000)
        answer = str(math.gcd(a, b))
        statement = f"Find GCD({a}, {b})."
        canonical_params = {"a": a, "b": b}
    elif family == "gcd_divisor_sum":
        a = _int_param(params, "a", low=2, high=50000)
        b = _int_param(params, "b", low=2, high=50000)
        g = math.gcd(a, b)
        answer = str(_sum_of_divisors(g))
        statement = (
            f"Let n = GCD({a}, {b}). Find the sum of all positive divisors of n."
        )
        canonical_params = {"a": a, "b": b, "gcd": g}
    elif family == "units_digit":
        base = _int_param(params, "base", low=2, high=99)
        exp = _int_param(params, "exp", low=2, high=5000)
        answer = str(pow(base, exp, 10))
        statement = f"Find the units digit of {base}^{{{exp}}}."
        canonical_params = {"base": base, "exp": exp}
    elif family == "divisor_sum":
        n = _int_param(params, "n", low=2, high=2000)
        answer = str(_sum_of_divisors(n))
        statement = f"Find the sum of all positive divisors of {n}."
        canonical_params = {"n": n}
    elif family == "divisor_sum_mod":
        n = _int_param(params, "n", low=2, high=2000)
        a = _int_param(params, "a", low=1, high=1_000_000)
        modulus = _sum_of_divisors(n)
        answer = str(a % modulus)
        statement = (
            f"Let m be the sum of all positive divisors of {n}. Find {a} mod m."
        )
        canonical_params = {"n": n, "a": a, "modulus": modulus}
    elif family == "stars_and_bars":
        var_count = _int_param(params, "vars", low=2, high=6)
        total = _int_param(params, "sum", low=1, high=30)
        answer = str(math.comb(total + var_count - 1, var_count - 1))
        variables = " + ".join(f"x_{i}" for i in range(1, var_count + 1))
        statement = (
            "Count the number of non-negative integer solutions to " f"{variables} = {total}."
        )
        canonical_params = {"vars": var_count, "sum": total}
    elif family == "arithmetic_series":
        n_terms = _int_param(params, "n_terms", low=2, high=100)
        first = _int_param(params, "first", low=0, high=200)
        diff = _int_param(params, "diff", low=1, high=100)
        answer = str(sum(first + diff * i for i in range(n_terms)))
        terms = [first + diff * i for i in range(4)]
        statement = (
            f"Find the sum of the first {n_terms} terms of the arithmetic sequence "
            f"{terms[0]}, {terms[1]}, {terms[2]}, {terms[3]}, ..."
        )
        canonical_params = {"n_terms": n_terms, "first": first, "diff": diff}
    else:
        a = _int_param(params, "a", low=1, high=1_000_000)
        m = _int_param(params, "m", low=2, high=10000)
        answer = str(a % m)
        statement = f"Find {a} mod {m}."
        canonical_params = {"a": a, "m": m}

    solution_skeleton = dict(solution_skeleton)
    solution_skeleton["expected_answer"] = answer

    return GeneratedProblem(
        id=f"{source_problem_id}__gen1_{_slug(family)}",
        source_problem_id=source_problem_id,
        family=family,
        statement=statement,
        answer=answer,
        solution=_deterministic_solution(family, canonical_params, answer) or raw_solution,
        difficulty_label=difficulty_label,
        params=canonical_params,
        reasoning_pattern=reasoning_pattern,
        solution_skeleton=solution_skeleton,
        projected_params=projected_params,
        projection_check=projection_check,
        harder_reason=harder_reason,
        axis_applied=axis_applied,
        axis_aligned=axis_aligned,
        raw_llm_output=raw,
    )


def _same_contract_value(actual: Any, expected: Any) -> bool:
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def validate_generated_contract(
    parent: "CertificationInput",
    generated: GeneratedProblem,
) -> None:
    """Enforce the orchestrator contract before Lean certification."""
    target_family = parent.metadata.get("target_family")
    if target_family and generated.family != target_family:
        raise PlannerContractError(
            f"generated family {generated.family!r} did not match target_family {target_family!r}"
        )

    required_params = parent.metadata.get("required_params") or {}
    if not isinstance(required_params, dict):
        raise PlannerContractError("required_params must be an object")
    for key, expected in required_params.items():
        if key not in generated.params:
            raise PlannerContractError(f"generated params missing required key {key!r}")
        if not _same_contract_value(generated.params[key], expected):
            raise PlannerContractError(
                f"generated param {key}={generated.params[key]!r} did not match required {expected!r}"
            )

    projected_params = generated.projected_params or {}
    if not isinstance(projected_params, dict):
        raise PlannerContractError("projected_params must be an object")
    if projected_params:
        derived_keys = DERIVED_PARAM_KEYS.get(generated.family, set())
        for key, value in generated.params.items():
            if key in derived_keys:
                continue
            if key not in projected_params:
                raise PlannerContractError(f"projected_params missing canonical key {key!r}")
            if not _same_contract_value(projected_params[key], value):
                raise PlannerContractError(
                    f"projected_params[{key}]={projected_params[key]!r} did not match params[{key}]={value!r}"
                )

    if parent.metadata.get("op_type") == "crossover" and generated.raw_llm_output:
        parent_ids = [str(parent_id) for parent_id in parent.metadata.get("parent_ids") or []]
        evidence = generated.raw_llm_output.get("parent_contribution_evidence") or {}
        skeleton = generated.solution_skeleton or {}
        skeleton_evidence = (
            skeleton.get("parent_contributions") if isinstance(skeleton, dict) else {}
        ) or {}
        if not isinstance(evidence, dict):
            evidence = {}
        if not isinstance(skeleton_evidence, dict):
            skeleton_evidence = {}
        missing_evidence = [
            parent_id
            for parent_id in parent_ids
            if not str(evidence.get(parent_id) or "").strip()
        ]
        missing_skeleton = [
            parent_id
            for parent_id in parent_ids
            if not str(skeleton_evidence.get(parent_id) or "").strip()
        ]
        if missing_evidence or missing_skeleton:
            raise PlannerContractError(
                "crossover missing direct parent contribution evidence: "
                f"parent_contribution_evidence missing {missing_evidence}; "
                f"solution_skeleton.parent_contributions missing {missing_skeleton}"
            )


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _schema_response_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": False,
            "schema": schema,
        },
    }


def _structured_schema_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "response_format" in text
        and ("json_schema" in text or "unsupported" in text)
    ) or any(
        marker in text
        for marker in [
            "too many optional parameters",
            "grammar compilation",
            "tool schemas",
            "json schema",
        ]
    )


def _use_codex_cli() -> bool:
    return os.getenv("GENERATION_PROVIDER", "").lower() == "codex_cli"


def _schema_from_response_format(response_format: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, dict) else None


def _messages_to_system_user(messages: list[Dict[str, str]]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


# ~25% of hard generation failures in data/certified/ were provider 429s,
# timeouts, and connection drops recorded as if they were mathematical
# failures. Retry transient transport errors with exponential backoff before
# they ever reach the certification funnel.
_TRANSIENT_LLM_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "connection error",
    "connection reset",
    "timed out",
    "timeout",
    "service unavailable",
    "server overloaded",
    "internal server error",
    "bad gateway",
    "temporarily unavailable",
    # OpenRouter/upstream failures sometimes surface as a 200 with an error
    # payload and no choices array.
    "response had no choices",
)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        detail = getattr(response, "error", None)
        raise RuntimeError(
            f"llm response had no choices: {str(detail)[:300] if detail else 'no error detail'}"
        )
    return choices[0].message.content or "{}"


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_LLM_ERROR_MARKERS)


def _transport_retry_budget() -> tuple[int, float]:
    try:
        retries = max(0, int(os.getenv("GENERATION_LLM_TRANSPORT_RETRIES", "3")))
    except ValueError:
        retries = 3
    try:
        base_delay = max(
            0.0, float(os.getenv("GENERATION_LLM_TRANSPORT_BACKOFF", "2"))
        )
    except ValueError:
        base_delay = 2.0
    return retries, base_delay


def _transport_backoff_delay(base_delay: float, attempt: int) -> float:
    return base_delay * (2**attempt) + random.uniform(0.0, 0.5)


async def _with_transport_retry_async(invoke):
    retries, base_delay = _transport_retry_budget()
    for attempt in range(retries + 1):
        try:
            return await invoke()
        except Exception as exc:
            if attempt >= retries or not _is_transient_llm_error(exc):
                raise
            await asyncio.sleep(_transport_backoff_delay(base_delay, attempt))
    raise RuntimeError("unreachable")  # pragma: no cover


def _with_transport_retry_sync(invoke):
    retries, base_delay = _transport_retry_budget()
    for attempt in range(retries + 1):
        try:
            return invoke()
        except Exception as exc:
            if attempt >= retries or not _is_transient_llm_error(exc):
                raise
            time.sleep(_transport_backoff_delay(base_delay, attempt))
    raise RuntimeError("unreachable")  # pragma: no cover


async def _chat_completion_text_async(
    *,
    model: str,
    messages: list[Dict[str, str]],
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    """Return assistant text from the configured generation provider."""
    if _use_codex_cli():
        system, user = _messages_to_system_user(messages)
        with ls.trace(
            name="codex_cli_call",
            run_type="llm",
            inputs={
                "model": model,
                "system": system,
                # The assembled prompt, not its length. The operator card, the
                # NoGoPolicyPack, the memory delta contract and the parent
                # blocks are all composed here and nowhere else; without the
                # text there is no way to ask afterwards whether a bad child
                # came from a bad instruction or from the model ignoring a good
                # one.
                "prompt": user,
                "system_chars": len(system),
                "user_chars": len(user),
                "schema_enabled": bool(_schema_from_response_format(response_format)),
                "async": True,
            },
            tags=["codex-cli", "llm"],
        ) as codex_run:
            retries, base_delay = _transport_retry_budget()
            for attempt in range(retries + 1):
                response = await call_codex_cli(
                    model=model,
                    system=system,
                    user=user,
                    timeout_seconds=timeout_seconds or float(os.getenv("GENERATION_LLM_TIMEOUT", "240")),
                    cwd=Path.cwd(),
                    output_schema=_schema_from_response_format(response_format),
                )
                if not (
                    response.error
                    and attempt < retries
                    and _is_transient_llm_error(RuntimeError(response.error))
                ):
                    break
                await asyncio.sleep(_transport_backoff_delay(base_delay, attempt))
            codex_run.end(
                outputs={
                    "finish_reason": response.finish_reason,
                    "elapsed_seconds": response.elapsed_seconds,
                    "stdout_chars": len(response.raw_text or ""),
                    "stderr_tail": (response.stderr or "")[-1000:],
                    "stdout_tail": (response.stdout or "")[-1000:],
                    "error": response.error,
                }
            )
        if response.error:
            raise RuntimeError(response.error)
        return response.raw_text

    client = _openai_client(model)
    request: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        request["response_format"] = response_format
    extra_body = _openrouter_extra_body(model)
    if extra_body:
        request["extra_body"] = extra_body
    request_timeout = timeout_seconds or float(os.getenv("GENERATION_LLM_TIMEOUT", "180"))

    async def _invoke_once():
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    **request,
                ),
                timeout=request_timeout,
            )
        except Exception as exc:
            if not response_format or not _structured_schema_unsupported(exc):
                raise
            fallback_request = {**request, "response_format": {"type": "json_object"}}
            return await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    **fallback_request,
                ),
                timeout=request_timeout,
            )

    response = await _with_transport_retry_async(_invoke_once)
    return _response_text(response)


def _chat_completion_text_sync(
    *,
    model: str,
    messages: list[Dict[str, str]],
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    """Synchronous variant used by the current planner node."""
    if _use_codex_cli():
        system, user = _messages_to_system_user(messages)
        with ls.trace(
            name="codex_cli_call",
            run_type="llm",
            inputs={
                "model": model,
                "system": system,
                # The assembled prompt, not its length. The operator card, the
                # NoGoPolicyPack, the memory delta contract and the parent
                # blocks are all composed here and nowhere else; without the
                # text there is no way to ask afterwards whether a bad child
                # came from a bad instruction or from the model ignoring a good
                # one.
                "prompt": user,
                "system_chars": len(system),
                "user_chars": len(user),
                "schema_enabled": bool(_schema_from_response_format(response_format)),
                "async": False,
            },
            tags=["codex-cli", "llm"],
        ) as codex_run:
            retries, base_delay = _transport_retry_budget()
            for attempt in range(retries + 1):
                response = call_codex_cli_sync(
                    model=model,
                    system=system,
                    user=user,
                    timeout_seconds=timeout_seconds or float(os.getenv("GENERATION_LLM_TIMEOUT", "240")),
                    cwd=Path.cwd(),
                    output_schema=_schema_from_response_format(response_format),
                )
                if not (
                    response.error
                    and attempt < retries
                    and _is_transient_llm_error(RuntimeError(response.error))
                ):
                    break
                time.sleep(_transport_backoff_delay(base_delay, attempt))
            codex_run.end(
                outputs={
                    "finish_reason": response.finish_reason,
                    "elapsed_seconds": response.elapsed_seconds,
                    "stdout_chars": len(response.raw_text or ""),
                    "stderr_tail": (response.stderr or "")[-1000:],
                    "stdout_tail": (response.stdout or "")[-1000:],
                    "error": response.error,
                }
            )
        if response.error:
            raise RuntimeError(response.error)
        return response.raw_text

    client = _openai_client(model)
    request: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        request["response_format"] = response_format
    extra_body = _openrouter_extra_body(model)
    if extra_body:
        request["extra_body"] = extra_body
    def _invoke_once():
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:
            if not response_format or not _structured_schema_unsupported(exc):
                raise
            fallback_request = {**request, "response_format": {"type": "json_object"}}
            return client.chat.completions.create(**fallback_request)

    response = _with_transport_retry_sync(_invoke_once)
    return _response_text(response)


def _param_schema_for_family(family: Optional[str]) -> Dict[str, Any]:
    keys = FAMILY_INPUT_KEYS.get(str(family or ""))
    if not keys:
        return {
            "type": "object",
            "description": "Canonical integer params for the selected family.",
            "additionalProperties": {"type": "integer"},
        }
    return {
        "type": "object",
        "description": f"Canonical integer params for {family}. Must include exactly {keys}.",
        "additionalProperties": False,
        "required": keys,
        "properties": {
            key: {"type": "integer", "description": PARAM_DESCRIPTIONS[key]} for key in keys
        },
    }


def _parent_contribution_schema(parent_ids: Optional[list[str]] = None) -> Dict[str, Any]:
    ids = [str(parent_id) for parent_id in parent_ids or [] if str(parent_id)]
    if not ids:
        return {
            "type": "object",
            "description": (
                "For each parent id, name the visible param, solution step, "
                "proof_context artifact, or supported composite semantics it contributes."
            ),
            "additionalProperties": {"type": "string"},
        }
    return {
        "type": "object",
        "description": (
            "CRITICAL for crossover: include every parent id as a key. "
            f"Required keys: {ids}. Empty object is invalid. Each value must name "
            "the generated field affected by that parent, such as params.<key>, "
            "solution_skeleton.derived_quantities, solution_skeleton.verification_steps, "
            "axis_applied, or solution."
        ),
        "required": ids,
        "additionalProperties": {"type": "string"},
        "properties": {
            parent_id: {
                "type": "string",
                "description": (
                    f"Required non-empty contribution for parent {parent_id}. "
                    "Name the concrete generated field this parent changed."
                ),
            }
            for parent_id in ids
        },
    }


def _generation_response_schema(
    target_family: Optional[str] = None,
    *,
    op_type: Optional[str] = None,
    parent_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    family_enum = [target_family] if target_family in SUPPORTED_FAMILIES else SUPPORTED_FAMILY_NAMES
    param_schema = _param_schema_for_family(target_family)
    return {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "status",
            "family",
            "params",
            "answer",
            "statement",
            "solution",
            "reason",
        ],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["generated", "axis_failed", "cannot_execute"],
                "description": "generated only when family and params are executable.",
            },
            "contract_status": {
                "type": "string",
                "enum": ["generated", "axis_failed", "cannot_execute"],
                "description": "Backward-compatible alias for status.",
            },
            "family": {
                "type": "string",
                "enum": family_enum,
                "description": "Supported canonical family.",
            },
            "params": param_schema,
            "answer": {"type": ["string", "integer"], "description": "Proposed numeric answer; canonicalizer recomputes it."},
            "statement": {
                "type": "string",
                "description": (
                    "Proposed public math statement; canonicalizer rebuilds it from params. "
                    "Do not include workflow terms such as checkpoint, parent, certified, "
                    "generated, mutation, crossover, pipeline, operator, proof obligation, Lean, or formal."
                ),
            },
            "solution": {"type": "string", "description": "Short derivation of the numeric answer from params."},
            "reason": {"type": "string", "description": "Short explanation or cannot_execute reason."},
            "reasoning_pattern": {
                "type": "string",
                "description": "Optional verifier hint.",
            },
            "solution_skeleton": {
                "type": "object",
                "description": "Optional verifier hint.",
                "additionalProperties": True,
            },
            "projected_params": param_schema,
            "projection_check": {
                "type": "object",
                "description": "Optional verifier hint.",
                "additionalProperties": True,
            },
            "axis_applied": {
                "type": "string",
                "description": "Optional backward-compatible transformation note.",
            },
            "axis_alignment": {
                "type": "object",
                "additionalProperties": True,
            },
            "parent_contribution_evidence": {"type": "object", "additionalProperties": {"type": "string"}},
            "parent_usage": {"type": "object", "additionalProperties": {"type": "string"}},
            "difficulty_label": {"type": "string", "enum": ["easy", "medium", "hard", "superhard"]},
            "harder_reason": {"type": "string", "description": "Optional reason this is harder."},
        },
    }


def _generation_response_format(
    target_family: Optional[str] = None,
    *,
    op_type: Optional[str] = None,
    parent_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return _schema_response_format(
        "generated_problem_contract",
        _generation_response_schema(target_family, op_type=op_type, parent_ids=parent_ids),
    )


def _coerce_string_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        for key, val in value.items()
        if str(key).strip()
    }


def _normalize_generation_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept compact Codex artifacts and expand legacy verifier hints."""
    normalized = dict(raw)
    if not normalized.get("contract_status") and normalized.get("status"):
        normalized["contract_status"] = normalized.get("status")
    status = str(normalized.get("contract_status") or "").strip().lower()
    if status in {"", "ok", "success", "complete", "completed", "certified"}:
        normalized["contract_status"] = "generated"
    if status in {"cannot_execute", "cannot-execute", "not_executable"}:
        normalized["contract_status"] = "axis_failed"
    if not normalized.get("harder_reason") and normalized.get("reason"):
        normalized["harder_reason"] = normalized.get("reason")
    if not normalized.get("axis_applied") and normalized.get("reason"):
        normalized["axis_applied"] = normalized.get("reason")
    if not normalized.get("reasoning_pattern") and normalized.get("reason"):
        normalized["reasoning_pattern"] = normalized.get("reason")
    if not normalized.get("projected_params") and isinstance(normalized.get("params"), dict):
        normalized["projected_params"] = normalized.get("params")
    if not isinstance(normalized.get("projection_check"), dict):
        normalized["projection_check"] = {"passed": True, "evidence": "minimal artifact accepted"}
    if not isinstance(normalized.get("axis_alignment"), dict):
        normalized["axis_alignment"] = {"passed": True, "evidence": "minimal artifact accepted"}
    if not isinstance(normalized.get("solution_skeleton"), dict):
        normalized["solution_skeleton"] = {
            "target_computation": normalized.get("reason") or normalized.get("solution") or ""
        }
    if not normalized.get("parent_contribution_evidence") and normalized.get("parent_usage"):
        normalized["parent_contribution_evidence"] = normalized.get("parent_usage")
    normalized["parent_contribution_evidence"] = _coerce_string_dict(
        normalized.get("parent_contribution_evidence")
    )
    skeleton = normalized.get("solution_skeleton")
    if isinstance(skeleton, dict) and "parent_contributions" not in skeleton:
        skeleton["parent_contributions"] = dict(normalized["parent_contribution_evidence"])
    return normalized


def _family_param_contract_lines() -> str:
    return "\n".join(
        [
            "- gcd params exactly: {\"a\": int 2..50000, \"b\": int 2..50000}",
            "- gcd_divisor_sum params exactly: {\"a\": int 2..50000, \"b\": int 2..50000}",
            "- units_digit params exactly: {\"base\": int 2..99, \"exp\": int 2..5000}",
            "- divisor_sum params exactly: {\"n\": int 2..2000}",
            "- divisor_sum_mod params exactly: {\"n\": int 2..2000, \"a\": int 1..1000000}",
            "- stars_and_bars params exactly: {\"vars\": int 2..6, \"sum\": int 1..30}",
            "- arithmetic_series params exactly: {\"n_terms\": int 2..100, \"first\": int 0..200, \"diff\": int 1..100}",
            "- modular_congruence params exactly: {\"a\": int 1..1000000, \"m\": int 2..10000}",
        ]
    )


def _parent_blocks(parent: "CertificationInput") -> str:
    parents = parent.metadata.get("parents")
    if isinstance(parents, list) and parents:
        return json.dumps(parents, ensure_ascii=False, indent=2)
    return json.dumps(
        [
            {
                "id": parent.id,
                "family": detect_family(parent.statement) or "unsupported",
                "statement": parent.statement,
                "answer": parent.answer,
                "required_contribution": (parent.metadata.get("parent_contributions") or {}).get(parent.id, ""),
            }
        ],
        ensure_ascii=False,
        indent=2,
    )


def _parent_proof_context_docstring() -> str:
    return """
ParentProofContext contract:
- Each parent may include proof_context.solution, proof_context.verification_code, proof_context.lean_code, proof_context.solution_skeleton, and proof_context.quality_evidence.
- Treat proof artifacts as hints, not authority. The child must still use target_family canonical params and Lean-supported templates.
- verification_code.kind is one of lean, python, unknown, not_available. Never execute verification_code.
- If lean_code is available, use it only to understand the local Lean template footprint and proof style.
- If verification_code is available but lean_code is not, use it only as semantic computation guidance.
- Do not copy unsupported proof code into the child.
- If a parent proof artifact changes the child, name the affected generated field in parent_contribution_evidence.
- If a field is not_available, do not invent it.
"""


def _build_generation_messages(parent: "CertificationInput") -> list[dict[str, str]]:
    parent_family = detect_family(parent.statement) or "unsupported"
    op_type = str(parent.metadata.get("op_type") or "mutation")
    operator_variant = str(parent.metadata.get("operator_variant") or op_type)
    target_family = parent.metadata.get("target_family") or parent_family
    variation_axis = (
        parent.metadata.get("variation_axis") or "increase difficulty within the family"
    )
    reasoning_goal = str(parent.metadata.get("reasoning_goal") or variation_axis)
    required_params = parent.metadata.get("required_params") or {}
    composition_pattern = str(parent.metadata.get("composition_pattern") or "")
    parent_contributions = parent.metadata.get("parent_contributions") or {}
    avoid_patterns = parent.metadata.get("avoid_patterns") or []
    quality_target = str(parent.metadata.get("quality_target") or "")
    retry_feedback = str(parent.metadata.get("retry_feedback") or "")
    attempt_history = parent.metadata.get("attempt_history") or []
    operator_card = parent.metadata.get("operator_card") or {
        "op_type": op_type,
        "operator_variant": operator_variant,
        "target_family": target_family,
        "operator_goal": reasoning_goal,
        "composition_pattern": composition_pattern,
        "parent_ids": parent.metadata.get("parent_ids") or [parent.id],
        "required_checkpoints": parent.metadata.get("required_checkpoints") or [],
        "avoid_signatures": parent.metadata.get("avoid_signatures") or [],
    }
    memory_delta_contract = operator_card.get("memory_delta_contract")
    if not isinstance(memory_delta_contract, dict):
        metadata_contract = parent.metadata.get("memory_delta_contract")
        memory_delta_contract = metadata_contract if isinstance(metadata_contract, dict) else {}
    recent_flags = list(operator_card.get("avoid") or []) + list(operator_card.get("avoid_signatures") or [])
    if isinstance(attempt_history, list):
        for attempt in attempt_history[-4:]:
            if isinstance(attempt, dict):
                recent_flags.extend(str(flag) for flag in attempt.get("quality_flags") or [])
    no_go_pack = build_no_go_policy_pack(
        op_type=op_type,
        target_style="numeric_answer",
        target_family=str(target_family),
        operator_variant=operator_variant,
        recent_failure_flags=recent_flags,
        limit=6,
    )
    if op_type == "crossover":
        fusion_contract = operator_card.get("fusion_contract") or {}
        crossover_intensity = (
            "- crossover_easy: a conservative bridge is acceptable, but both parents must "
            "change skeleton semantics or params."
            if operator_variant == "crossover_easy"
            else "- crossover_hard: require distinct semantic roles; true fusion is best, but a real pipeline composite is acceptable."
        )
        op_rules = """
Crossover rules:
- The final child must still be exactly one supported target_family.
- Use both parents when possible. Parent usage should be visible in params, solution, reason, or parent_usage.
- The final statement is canonicalized from family + params, so unsupported free-form wording is ignored.
- Parent B should change a generated parameter, the explicit solution strategy, or the chosen supported family.
- If only a numeric parameter is transferred from parent B, the child must still be harder than the target-family parent.
- For cross-family bridges, prefer supported composite families when the second parent contributes an operation rather than a raw number.
- Do not use vague "inspired by", capped, or downscaled parent usage unless the contribution appears in params or solution.
- Do not merely mention a second parent in prose if it does not affect the generated child.
- Pipeline composite is allowed: one parent may provide a checkpoint/object/result that becomes an input to the other parent's target. Make that dependency observable in statement, formal_statement, proof_plan, solution, or parent_usage.
- Do not produce side-by-side conjunctions where parent A and parent B are proved independently.
- For gcd_divisor_sum, one parent supplies GCD structure and one supplies divisor-sum operation semantics.
- For divisor_sum_mod, one parent supplies divisor-sum-derived modulus semantics and one supplies modular reduction structure.
- If fusion_goal or parent_roles are present, reflect them in parent_usage and one concrete generated surface.
""" + crossover_intensity + "\n"
        if fusion_contract:
            op_rules += (
                "\nFusion contract to implement directly:\n"
                f"{json.dumps(fusion_contract, ensure_ascii=False, indent=2)}\n"
            )
    else:
        if operator_variant == "mutation_easy":
            intensity_rule = (
                "- mutation_easy: build a stable Lean-template bridge first; prefer clear "
                "solution consistency over aggressive difficulty."
            )
        elif operator_variant == "mutation_hard":
            intensity_rule = (
                "- mutation_hard: keep the family certifiable but add a real reasoning "
                "checkpoint or family-specific complexity increase."
            )
        else:
            intensity_rule = "- mutation: keep the family certifiable and make the harder step checkable."
        op_rules = """
Mutation rules:
- Stay within the selected target_family.
- Treat required_params as optional hard hints only when provided.
- Make the harder step concrete and checkable from the reasoning skeleton and params, not just larger-looking.
- bounded_generalization is allowed for numeric templates: change one canonical parameter or derived object within supported ranges, keep the answer deterministic, and explain the generalized reasoning in solution/reason.
- Do not use arbitrary broad generalization or unsupported symbolic answers; all generalization must still project to executable canonical params.
""" + intensity_rule + "\n"
    system = (
        "You are a slot worker. Generate exactly one harder math problem for a "
        "deterministic Lean template. Follow the operator_card and return a compact "
        "executable artifact. Return JSON only."
    )
    user = f"""
Hard priority order:
1. If retry_feedback is non-empty, revise the previous child to fix that failure first.
2. Satisfy target_family exactly.
3. Choose canonical params for that family.
4. Provide a short solution derivation.
5. If the contract cannot be satisfied, return status="cannot_execute".

Operator card:
{json.dumps(operator_card, ensure_ascii=False, indent=2)}

{format_no_go_policy_pack(no_go_pack, title="Worker NoGoPolicyPack")}

MemoryDeltaContract:
{json.dumps(memory_delta_contract, ensure_ascii=False, indent=2) if memory_delta_contract else "not_available"}
- This is compact novelty context, not a parent source.
- Avoid same family+params, same final target semantics, and parameter-shift-only variants from similar_card_ids.
- If the required_distinguishing_delta cannot be satisfied, return status="cannot_execute" rather than a near-copy.

Backward-compatible slot fields:
- variation_axis: {variation_axis}
- reasoning_goal: {reasoning_goal}
- required_params: {json.dumps(required_params, ensure_ascii=False)}
- parent_contributions: {json.dumps(parent_contributions, ensure_ascii=False)}
- avoid_patterns: {json.dumps(avoid_patterns, ensure_ascii=False)}
- quality_target: {quality_target}
- operator_variant: {operator_variant}
- retry_feedback: {retry_feedback}
- attempt_history: {json.dumps(attempt_history[-4:] if isinstance(attempt_history, list) else [], ensure_ascii=False)[:2500]}

Retry mode:
- If retry_feedback is non-empty, do not ignore it and do not simply restate the same child.
- For quality retry, keep target_family but revise params, solution, reason, and parent_usage so the listed quality_flags are removed.
- For solution verification retry, keep canonical params/answer when possible and repair solution text so it derives the canonical answer.
- For Lean retry, keep target_family and repair params/answer/solution consistency.
- For contract retry, repair the required JSON fields before changing anything else.

Parents:
{_parent_blocks(parent)}

{_parent_proof_context_docstring()}

{op_rules}

Execution surface:
- The system ignores any free-form problem statement you might intend and rebuilds it from family + params.
- Therefore, all difficulty and crossover evidence must be encoded in params, solution, reason, or parent_usage.
- Supported composite families are the only valid way to create derived-object crossover statements.
- If the slot contract cannot be satisfied on that surface, return status="cannot_execute".
- Public statement hygiene: if you write statement text, make it read like a math contest/theorem statement only.
- Do not put workflow or lineage words in statement: checkpoint, parent, certified, generated, mutation, crossover, pipeline, operator, proof obligation, Lean, formal.
- Put process evidence in reason, solution, parent_usage, or proof_plan-like fields instead of statement.

Generate-one policy:
- Return exactly one generated child, not multiple candidates.
- Do not include alternatives, rankings, or rejected candidates.

Before returning generated, self-check:
- target_family matches family exactly.
- every required_params value appears exactly in params when required_params is non-empty.
- solution text agrees with the answer implied by params.
- for crossover, parent_usage names how each parent affected params, solution, or family choice.
- avoid_patterns are not repeated.

Supported families and required params:
{_family_param_contract_lines()}

Params rule:
- The params object must use the exact keys for target_family from the table above.
- Never output placeholder keys such as "required_param" or "param_name".
- If target_family is gcd, params must contain "a" and "b".
- If target_family is divisor_sum_mod, params must contain "n" and "a".

Return exactly this JSON shape:
{{
  "status": "generated",
  "family": "target_family exactly",
  "params": {{"a": 123, "b": 456}},
  "answer": "numeric answer",
  "statement": "natural language statement",
  "solution": "short derivation of the numeric answer",
  "reason": "short reason this follows the operator card",
  "parent_usage": {{"parent_id": "short usage note"}}
}}

Do not include markdown. Do not include unsupported families. Do not use symbolic answers.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _openai_client(model: str):
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai

    use_openrouter = os.getenv("GENERATION_PROVIDER", "").lower() == "openrouter" or "/" in model
    timeout = float(os.getenv("GENERATION_LLM_TIMEOUT", "180"))

    if use_openrouter and os.getenv("OPENROUTER_API_KEY"):
        return wrap_openai(
            OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout=timeout,
            )
        )

    if os.getenv("OPENAI_API_KEY"):
        return wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=timeout))

    if os.getenv("OPENROUTER_API_KEY"):
        return wrap_openai(
            OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout=timeout,
            )
        )

    raise EnvironmentError("Set OPENAI_API_KEY or OPENROUTER_API_KEY to generate problems")


def _openrouter_extra_body(model: str) -> Optional[Dict[str, Any]]:
    """Small OpenRouter-specific request config from env.

    This keeps provider routing/reasoning operational knobs out of prompts and
    avoids adding CLI surface for every provider-specific experiment.
    """
    use_openrouter = os.getenv("GENERATION_PROVIDER", "").lower() == "openrouter" or "/" in model
    if not use_openrouter:
        return None

    extra: Dict[str, Any] = {}
    effort = (os.getenv("GENERATION_REASONING_EFFORT") or "").strip()
    if effort:
        extra["reasoning"] = {
            "effort": effort,
            "exclude": os.getenv("GENERATION_REASONING_EXCLUDE", "true").lower()
            not in {"0", "false", "no"},
        }

    provider_order = [
        provider.strip()
        for provider in (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").split(",")
        if provider.strip()
    ]
    if provider_order:
        provider: Dict[str, Any] = {"order": provider_order}
        provider["allow_fallbacks"] = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        if os.getenv("OPENROUTER_REQUIRE_PARAMETERS"):
            provider["require_parameters"] = os.getenv("OPENROUTER_REQUIRE_PARAMETERS", "").lower() not in {
                "0",
                "false",
                "no",
            }
        extra["provider"] = provider

    return extra or None


async def generate_harder_problem(
    parent: "CertificationInput",
    config: Optional[GenerationConfig] = None,
) -> GeneratedProblem:
    """Generate one harder child problem with an LLM and canonicalize it."""
    config = config or default_generation_config()
    content = await _chat_completion_text_async(
        model=config.model,
        messages=_build_generation_messages(parent),
        temperature=config.temperature,
        response_format=_generation_response_format(
            parent.metadata.get("target_family"),
            op_type=parent.metadata.get("op_type"),
            parent_ids=parent.metadata.get("parent_ids"),
        ),
    )
    raw = _normalize_generation_raw(_parse_json_object(content))
    if raw.get("contract_status") == "axis_failed":
        raise PlannerContractError(
            str(raw.get("harder_reason") or "worker could not satisfy slot contract")
        )
    family = str(raw.get("family") or "").strip()
    params = raw.get("params")
    if not isinstance(params, dict):
        raise ValueError("LLM output missing params object")
    return _canonical_problem_from_params(
        source_problem_id=parent.id,
        family=family,
        params=params,
        raw=raw,
    )
