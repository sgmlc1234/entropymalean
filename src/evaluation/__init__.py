"""Direct no-tool model evaluation pipeline for EntropyMaG-2.

Mirrors the EntropyMaG-1 stress-test protocol:
- 3 benchmarks (miniF2F, PutnamBench, ProofNet)
- 2 arms per benchmark (control seeds vs. quality-gated treatment)
- 3 models (GPT-5.4-mini, Claude Haiku 4.5, Gemini 3.1 Flash Lite)
- 3 repeats per problem (temperature 0, no local completion cap by default)
- Bootstrap CI on dataset row indices
- Pass@3 and per-generation slope diagnostics
"""

from src.evaluation.answer_grader import (
    extract_boxed_answer,
    grade_answer,
    normalize_answer,
)
from src.evaluation.bootstrap_ci import bootstrap_ci, bootstrap_drop_ci
from src.evaluation.model_runner import (
    ModelConfig,
    ModelResponse,
    MODEL_PANEL,
    run_model_panel,
)
from src.evaluation.prompts import build_direct_prompt

__all__ = [
    "ModelConfig",
    "ModelResponse",
    "MODEL_PANEL",
    "run_model_panel",
    "build_direct_prompt",
    "extract_boxed_answer",
    "grade_answer",
    "normalize_answer",
    "bootstrap_ci",
    "bootstrap_drop_ci",
]
