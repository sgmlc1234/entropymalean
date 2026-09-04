"""Direct no-tool prompts.

The stress-test protocol these prompts implement:
- single-shot, no tools, no scratchpad augmentation
- explicit \\boxed{...} answer contract
- temperature 0, no local completion cap by default
"""

from __future__ import annotations

DIRECT_NO_TOOL_SYSTEM = (
    "You are a mathematical reasoning model. "
    "Solve the problem step by step, then state the final answer "
    "inside a single \\boxed{...} expression at the very end. "
    "Do not call external tools or write executable code; reason in text only."
)


DIRECT_NO_TOOL_USER_TEMPLATE = (
    "Problem:\n{statement}\n\n"
    "Provide concise reasoning and end with the final answer in \\boxed{{...}}."
)


def build_direct_prompt(statement: str) -> tuple[str, str]:
    """Return (system, user) prompts for the direct no-tool protocol."""
    user = DIRECT_NO_TOOL_USER_TEMPLATE.format(statement=statement.strip())
    return DIRECT_NO_TOOL_SYSTEM, user
