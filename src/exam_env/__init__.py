"""Game-style Lean exam environment (docs/semantic_alignment_plan.md 참조 아님 — 자체 실험)."""

from src.exam_env.environment import ExamObservation, LeanExamEnv
from src.exam_env.palette import TACTIC_DOCS, build_palette

__all__ = ["ExamObservation", "LeanExamEnv", "TACTIC_DOCS", "build_palette"]
