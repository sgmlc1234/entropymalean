"""LLM agent that plays the Lean exam environment via JSON actions."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.certification.generation import (
    GenerationConfig,
    _chat_completion_text_async,
    _parse_json_object,
    _schema_response_format,
)
from src.exam_env.environment import ExamObservation, LeanExamEnv

_ACTION_FORMAT = _schema_response_format(
    "exam_action",
    {
        "type": "object",
        "additionalProperties": True,
        "required": ["type"],
        "properties": {
            "type": {"type": "string", "enum": ["tactic", "inspect", "rollback"]},
            "tactic": {"type": "string"},
            "name": {"type": "string"},
            "to_step": {"type": "integer"},
            "reason": {"type": "string"},
        },
    },
)

_SYSTEM_PROMPT = (
    "You are taking a Lean 4 proof exam, played like a Lean game.\n"
    "Rules:\n"
    "- The theorem statement is fixed; you play ONE action per turn.\n"
    "- Actions (respond with ONLY one JSON object):\n"
    '  {\"type\": \"tactic\", \"tactic\": \"<one Lean tactic line>\"} — play a tactic.\n'
    '  {\"type\": \"inspect\", \"name\": \"<palette name>\"} — open a palette card '
    "(tactic usage or theorem signature) before committing.\n"
    '  {\"type\": \"rollback\", \"to_step\": <k>} — return to an earlier verified '
    "step when you believe a previous choice was wrong; prefer this over "
    "repeating a failing line.\n"
    "- A rejected tactic leaves the state unchanged; read the error and adapt.\n"
    "- Palette names are the tools that suffice for this exam. Inspect before "
    "guessing signatures.\n"
    "- Close every goal. Short, targeted tactics beat long speculative ones."
)


def _render_observation(
    observation: ExamObservation, palette: Dict[str, Dict[str, str]]
) -> str:
    lines: List[str] = []
    lines.append(f"status: {observation.status} — {observation.message}")
    if observation.card:
        lines.append(
            f"palette card [{observation.card['kind']}] {observation.card['name']}:\n"
            f"{observation.card['doc']}"
        )
    if observation.goals:
        lines.append("active goals:")
        for index, goal in enumerate(observation.goals, 1):
            lines.append(f"--- goal {index} ---\n{goal}")
    else:
        lines.append("active goals: (none reported)")
    if observation.steps:
        numbered = "\n".join(
            f"  {index}. {step}" for index, step in enumerate(observation.steps, 1)
        )
        lines.append(f"verified steps so far:\n{numbered}")
    lines.append(
        "palette tactics: " + ", ".join(sorted(palette.get("tactics") or {}))
    )
    lines.append(
        "palette theorems: " + ", ".join(sorted(palette.get("theorems") or {}))
    )
    return "\n".join(lines)


class ChatExamAgent:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    async def act(
        self,
        observation: ExamObservation,
        palette: Dict[str, Dict[str, str]],
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        user = (
            _render_observation(observation, palette)
            + "\n\nRespond with ONLY one JSON action object of the shape "
            '{"type": "tactic|inspect|rollback", ...} and nothing else.'
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user},
        ]
        content = await _chat_completion_text_async(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            response_format=_ACTION_FORMAT,
        )
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": content})
        parsed = _parse_json_object(content)
        if not isinstance(parsed, dict) or "type" not in parsed:
            return {"type": "tactic", "tactic": ""}  # rejected; agent sees error
        return parsed


async def run_exam_episode(
    env: LeanExamEnv,
    agent: ChatExamAgent,
    *,
    max_actions: int = 30,
) -> Dict[str, Any]:
    observation = await env.reset()
    history: List[Dict[str, str]] = []
    actions_taken = 0
    while not env.done and actions_taken < max_actions:
        action = await agent.act(observation, env.palette, history)
        observation = await env.step(action)
        actions_taken += 1
    return {
        "success": env.success,
        "actions": actions_taken,
        "steps": list(env.steps),
        "solved_code": env.solved_code(),
        "transcript": env.transcript,
    }
