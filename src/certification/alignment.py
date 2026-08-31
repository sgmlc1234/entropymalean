"""goal_roundtrip alignment signal (docs/semantic_alignment_plan.md).

The trusted semantic object is not the surface ``formal_statement`` string but
the goal Lean actually elaborates from it. This module:

  1. probes the statement with ``extract_goal; sorry`` so Lean prints the
     elaborated goal as a standalone theorem (binders, coercions, and
     instances resolved by the parser — not by the author);
  2. has an *informalizer* model translate that goal back to natural language
     while seeing ONLY the Lean goal (never the original prose);
  3. has a separate *judge* model compare the round-tripped prose against the
     original natural-language statement while seeing ONLY the two prose
     texts (never the Lean).

Role separation means no single model both produces and grades an alignment
claim. The output is an evidence record, not a verdict: alignment can only be
falsified, never certified, so downstream gates decide what to do with a
mismatch (release-gate rejection, audit queue, …).
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.certification.generation import (
    GenerationConfig,
    verification_config,
    _chat_completion_text_async,
    _parse_json_object,
    _schema_response_format,
)
from src.evaluation.lean_verifier import LeanVerifyResult, verify_lean_proof

ALIGNMENT_SIGNAL_SOURCE = "elaborated_goal_informalization"

_DIAG_PREFIX_RE = re.compile(r"^[^:\n]+:\d+:\d+:\s*(?:error|warning|info)")
_EXTRACTED_THEOREM_RE = re.compile(
    r"(?m)^theorem\s+\S*extracted\S*\s"
)


def strip_proof_body(formal_statement: str) -> str:
    """Statement prefix with any ``:= …`` proof body removed."""
    statement = str(formal_statement or "").strip()
    match = re.search(r":=", statement)
    if match:
        statement = statement[: match.start()].rstrip()
    return statement


def build_goal_probe_code(lean_header: str, formal_statement: str) -> str:
    """Probe file whose only output is the elaborated goal plus a sorry."""
    header = (lean_header or "").rstrip()
    if "set_option autoImplicit false" not in header:
        header = f"{header}\nset_option autoImplicit false"
    statement = strip_proof_body(formal_statement)
    return f"{header}\n\n{statement} := by\n  extract_goal\n  sorry"


def extract_elaborated_goal(raw_output: str) -> Optional[str]:
    """Pull the ``extract_goal`` block out of the Lean process output.

    ``extract_goal`` prints the goal as a standalone ``theorem …extracted…``
    declaration on plain (non-diagnostic) lines. Accumulate from that line
    until the next diagnostic-shaped line.
    """
    if not raw_output:
        return None
    lines = raw_output.splitlines()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        if _EXTRACTED_THEOREM_RE.match(line):
            start = index
            break
    if start is None:
        return None
    block: List[str] = []
    for line in lines[start:]:
        if _DIAG_PREFIX_RE.match(line):
            break
        block.append(line)
    goal = "\n".join(block).strip()
    # Drop the trailing `:= sorry` body — the goal is the statement itself.
    goal = re.sub(r":=\s*(?:by\s+)?sorry\s*$", "", goal).strip()
    return goal or None


def _informalizer_response_format() -> Dict[str, Any]:
    return _schema_response_format(
        "goal_informalization",
        {
            "type": "object",
            "additionalProperties": True,
            "required": ["informal_statement"],
            "properties": {
                "informal_statement": {
                    "type": "string",
                    "description": (
                        "Faithful natural-language rendering of the Lean goal, "
                        "including every hypothesis and the exact conclusion."
                    ),
                },
            },
        },
    )


def _judge_response_format() -> Dict[str, Any]:
    return _schema_response_format(
        "nl_equivalence_verdict",
        {
            "type": "object",
            "additionalProperties": True,
            "required": ["equivalent", "mismatches", "rationale"],
            "properties": {
                "equivalent": {"type": "boolean"},
                "mismatches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Every semantic difference: missing hypotheses, extra "
                        "assumptions, changed quantifiers, different "
                        "conclusions. Empty only when equivalent."
                    ),
                },
                "rationale": {"type": "string"},
            },
        },
    )


async def informalize_elaborated_goal(
    goal: str, *, config: GenerationConfig
) -> str:
    """Model B: sees ONLY the elaborated Lean goal.

    Runs on ``VERIFICATION_MODEL`` when one is configured, so the model that
    reads the Lean is not the model that wrote it.
    """
    config = verification_config(config)
    messages = [
        {
            "role": "system",
            "content": (
                "You translate Lean 4 theorem statements into precise natural-"
                "language mathematics. You see ONLY the Lean statement — you "
                "have no access to any original problem text. State every "
                "hypothesis and the exact conclusion; do not simplify, "
                "strengthen, or weaken anything."
            ),
        },
        {
            "role": "user",
            "content": (
                "Render this elaborated Lean goal as a natural-language "
                f"mathematical statement:\n\n{goal}\n\n"
                "Respond with ONLY a JSON object of the exact shape "
                '{"informal_statement": "<your natural-language statement>"} '
                "and nothing else."
            ),
        },
    ]
    content = await _chat_completion_text_async(
        model=config.model,
        messages=messages,
        temperature=0,
        response_format=_informalizer_response_format(),
    )
    parsed = _parse_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("informalizer returned non-object JSON")
    informal = str(parsed.get("informal_statement") or "").strip()
    if not informal:
        raise ValueError(
            "informalizer returned no informal_statement "
            f"(keys={sorted(parsed.keys())[:6]})"
        )
    return informal


async def judge_nl_equivalence(
    original_nl: str, roundtrip_nl: str, *, config: GenerationConfig
) -> Dict[str, Any]:
    """Model C: sees ONLY the two prose statements, never any Lean."""
    messages = [
        {
            "role": "system",
            "content": (
                "You compare two natural-language mathematical statements for "
                "semantic equivalence. Statement B was produced by translating "
                "a formal statement back to prose; your job is to FALSIFY the "
                "claim that it matches Statement A. Hunt for missing or added "
                "hypotheses, changed quantifier structure, weakened or "
                "strengthened conclusions, and changed objects.\n\n"
                "Judge equivalence MODULO the standard background theory of "
                "formalized mathematics: standard definitions, library "
                "conventions, and encoding idioms are NOT differences. In "
                "particular, do not report any of the following as a mismatch:\n"
                "- a function on a set encoded as a total function together "
                "with the set as a hypothesis or restriction;\n"
                "- library notation for a named object (e.g. a formal symbol "
                "for √2 or for a quotient ring) instead of its prose name;\n"
                "- an operation inlined by its defining expression instead of "
                "being given a symbol;\n"
                "- typing/elaboration detail that the prose leaves implicit "
                "(coercions, index types, instance arguments, currying);\n"
                "- rephrasing, ordering, or naming of variables.\n\n"
                "Report a mismatch only when the two statements would have "
                "different truth values, or when one is provable and the other "
                "is not, under that background theory. Judge meaning, not "
                "encoding."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Statement A (original problem):\n{original_nl}\n\n"
                f"Statement B (round-tripped from the formal version):\n{roundtrip_nl}\n\n"
                "Are A and B semantically equivalent as mathematical claims?\n"
                "Respond with ONLY a JSON object of the exact shape "
                '{"equivalent": true|false, "mismatches": ["<each semantic '
                'difference>"], "rationale": "<one short paragraph>"} '
                "and nothing else."
            ),
        },
    ]
    content = await _chat_completion_text_async(
        model=config.model,
        messages=messages,
        temperature=0,
        response_format=_judge_response_format(),
    )
    parsed = _parse_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("equivalence judge returned non-object JSON")
    if "equivalent" not in parsed:
        raise ValueError(
            "equivalence judge response missing 'equivalent' "
            f"(keys={sorted(parsed.keys())[:6]})"
        )
    return {
        "equivalent": bool(parsed.get("equivalent")),
        "mismatches": [str(item) for item in parsed.get("mismatches") or []],
        "rationale": str(parsed.get("rationale") or "")[:800],
    }


GoalVerifier = Callable[..., Awaitable[LeanVerifyResult]]
Informalizer = Callable[..., Awaitable[str]]
EquivalenceJudge = Callable[..., Awaitable[Dict[str, Any]]]


async def elaborated_goal_alignment(
    *,
    statement_nl: str,
    formal_statement: str,
    lean_header: str,
    config: GenerationConfig,
    verifier: Optional[GoalVerifier] = None,
    informalizer: Optional[Informalizer] = None,
    judge: Optional[EquivalenceJudge] = None,
    lean_timeout: float = 300.0,
) -> Dict[str, Any]:
    """Full goal_roundtrip signal for one row. Returns an evidence record.

    ``status`` values: ``ok`` (signal computed), ``statement_error`` (the
    statement itself failed to elaborate — a T1 failure, not an alignment
    verdict), ``no_goal_extracted``, ``signal_error`` (LLM/transport failure).
    """
    signal: Dict[str, Any] = {
        "source": ALIGNMENT_SIGNAL_SOURCE,
        "status": "ok",
        "equivalent": None,
        "mismatches": [],
        "elaborated_goal": None,
        "informalized_statement": None,
        "rationale": None,
    }
    probe = build_goal_probe_code(lean_header, formal_statement)
    verify_fn = verifier or verify_lean_proof
    try:
        verdict = await verify_fn(probe, timeout=lean_timeout)
    except TypeError:
        verdict = await verify_fn(probe)
    if not isinstance(verdict, LeanVerifyResult):
        verdict = LeanVerifyResult(**dict(verdict))
    if not verdict.ok:
        signal["status"] = "statement_error"
        signal["rationale"] = verdict.summary()[:500]
        return signal
    goal = extract_elaborated_goal(
        "\n".join(part for part in (verdict.raw_stdout, verdict.raw_stderr) if part)
    )
    if not goal:
        signal["status"] = "no_goal_extracted"
        return signal
    signal["elaborated_goal"] = goal
    try:
        informal = await (
            informalizer(goal, config=config)
            if informalizer is not None
            else informalize_elaborated_goal(goal, config=config)
        )
        signal["informalized_statement"] = informal
        verdict_record = await (
            judge(statement_nl, informal, config=config)
            if judge is not None
            else judge_nl_equivalence(statement_nl, informal, config=config)
        )
    except Exception as exc:
        signal["status"] = "signal_error"
        signal["rationale"] = f"{type(exc).__name__}: {exc}"[:500]
        return signal
    signal["equivalent"] = bool(verdict_record.get("equivalent"))
    signal["mismatches"] = list(verdict_record.get("mismatches") or [])
    signal["rationale"] = verdict_record.get("rationale")
    return signal
