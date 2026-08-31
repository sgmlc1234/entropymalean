"""Unit tests for the evaluation pipeline that do not call any model API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.answer_grader import (
    extract_boxed_answer,
    grade_answer,
    normalize_answer,
)
from src.evaluation.bootstrap_ci import bootstrap_ci, bootstrap_drop_ci
from src.evaluation.dataset import (
    EvalRow,
    cap_rows,
    ensure_auto_implicit_false,
    lean_header_from_source,
    lean_theorem_prefix,
    load_control_rows,
    load_treatment_rows,
)
from src.evaluation.orchestrator import generation_slope_table, summarize_jsonl


# ---------- answer grader ----------


def test_extract_boxed_picks_last():
    assert extract_boxed_answer("foo \\boxed{1} bar \\boxed{42}") == "42"


def test_extract_boxed_nested_braces():
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_missing():
    assert extract_boxed_answer("no answer") is None


def test_normalize_answer_strips_commas_and_dollars():
    assert normalize_answer("$+1,234.50$") == "1234.5"


def test_grade_answer_exact():
    assert grade_answer("360", "360") is True
    assert grade_answer("360 ", " 360") is True
    assert grade_answer("361", "360") is False


def test_grade_answer_missing_prediction():
    assert grade_answer(None, "5") is False


# ---------- bootstrap ----------


def test_bootstrap_ci_constant():
    mean, lo, hi = bootstrap_ci([1.0] * 10, iterations=100, seed=0)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0


def test_bootstrap_drop_ci_directional():
    drop, lo, hi = bootstrap_drop_ci(
        [1.0] * 30, [0.0] * 30, iterations=200, seed=0
    )
    assert drop == pytest.approx(100.0)
    assert lo > 0 and hi > 0


# ---------- dataset ----------


def test_cap_rows_dedups_and_caps():
    rows = [
        EvalRow("miniF2F", "treatment", "a", "s", "1"),
        EvalRow("miniF2F", "treatment", "a", "s", "1"),  # dup
        EvalRow("miniF2F", "treatment", "b", "s", "2"),
        EvalRow("miniF2F", "treatment", "c", "s", "3"),
    ]
    out = cap_rows(rows, cap=2)
    assert [r.problem_id for r in out] == ["a", "b"]


def test_cap_rows_nonpositive_cap_keeps_all_deduped_rows():
    rows = [
        EvalRow("miniF2F", "treatment", "a", "s", "1"),
        EvalRow("miniF2F", "treatment", "a", "s", "1"),
        EvalRow("miniF2F", "treatment", "b", "s", "2"),
        EvalRow("miniF2F", "treatment", "c", "s", "3"),
    ]
    out = cap_rows(rows, cap=0)
    assert [r.problem_id for r in out] == ["a", "b", "c"]


def test_load_treatment_rows_filters_uncertified(tmp_path: Path):
    path = tmp_path / "trt.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "problem_id": "p1",
                    "statement": "Find x",
                    "answer": "1",
                    "status": "certified",
                    "generation": 3,
                    "formal_statement": "theorem p1 : True := by trivial",
                    "lean_header": "import Mathlib",
                    "family": "gcd",
                    "lean_level": 3,
                },
                {
                    "problem_id": "p2",
                    "statement": "Find y",
                    "answer": "",
                    "status": "certified",  # dropped: empty answer
                },
                {
                    "problem_id": "p3",
                    "statement": "Find z",
                    "answer": "5",
                    "status": "failed",  # dropped: not certified
                },
            ]
        )
    )
    rows = load_treatment_rows(path, "miniF2F")
    assert [r.problem_id for r in rows] == ["p1"]
    assert rows[0].generation == 3
    assert rows[0].formal_statement == "theorem p1 : True := by"
    assert rows[0].lean_header == "import Mathlib"
    assert rows[0].family == "gcd"
    assert rows[0].lean_level == 3


def test_load_treatment_rows_prefers_eval_problem_id(tmp_path: Path):
    path = tmp_path / "trt.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "lineage_id",
                "eval_problem_id": "lineage_id__eval_deadbeef00",
                "benchmark": "minif2f",
                "statement": "Show x",
                "certificate": {"status": "certified"},
                "formal_statement": "theorem row_level : True := by trivial",
            }
        )
        + "\n"
    )

    rows = load_treatment_rows(path, "miniF2F")

    assert [r.problem_id for r in rows] == ["lineage_id__eval_deadbeef00"]


def test_load_treatment_rows_strips_released_proof_body(tmp_path: Path):
    path = tmp_path / "trt.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "leaky",
                "benchmark": "minif2f",
                "statement": "Show a divisor sum",
                "certificate": {"status": "certified"},
                "formal_statement": "theorem divisor_sum_720 : True := by native_decide",
                "lean_code": "theorem divisor_sum_720 : True := by\n  trivial",
            }
        )
        + "\n"
    )

    rows = load_treatment_rows(path, "miniF2F")

    assert rows[0].formal_statement == "theorem divisor_sum_720 : True := by"


def test_load_treatment_rows_recovers_header_from_full_lean_code(tmp_path: Path):
    path = tmp_path / "trt.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "limit",
                "benchmark": "proofnet",
                "statement": "Show the limit.",
                "certificate": {"status": "certified"},
                "formal_statement": (
                    "theorem limit : Tendsto f atTop (𝓝 0) := by"
                ),
                "lean_code": (
                    "import Mathlib\n"
                    "open Filter Real\n"
                    "open scoped Topology\n\n"
                    "theorem limit : Tendsto f atTop (𝓝 0) := by\n"
                    "  sorry\n"
                ),
            }
        )
        + "\n"
    )

    rows = load_treatment_rows(path, "proofnet")

    assert rows[0].lean_header == (
        "import Mathlib\nopen Filter Real\nopen scoped Topology"
    )


def test_lean_header_from_source_keeps_trusted_setup_only():
    source = (
        "import Mathlib\n"
        "set_option maxHeartbeats 2000000\n"
        "open Filter Real\n\n"
        "theorem target : True := by\n"
        "  trivial\n"
    )
    assert lean_header_from_source(source) == (
        "import Mathlib\n"
        "set_option maxHeartbeats 2000000\n"
        "open Filter Real"
    )


def test_ensure_auto_implicit_false_is_idempotent():
    header = "import Mathlib\nset_option autoImplicit false\nopen Filter"
    assert ensure_auto_implicit_false(header) == header
    assert ensure_auto_implicit_false("import Mathlib").endswith(
        "set_option autoImplicit false"
    )


def test_load_control_rows_strips_released_proof_body(tmp_path: Path):
    path = tmp_path / "control.csv"
    path.write_text(
        "id,statement,answer,formal_statement\n"
        'c1,Show true,proof,"theorem c1 : True := by trivial"\n'
    )

    rows = load_control_rows(path, "proofnet")

    assert rows[0].formal_statement == "theorem c1 : True := by"


def test_lean_theorem_prefix_handles_assignment_without_by():
    assert lean_theorem_prefix("theorem t : True :=") == "theorem t : True := by"


# ---------- summarizer ----------


def _write_eval_jsonl(path: Path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


def test_summarize_jsonl_drop_and_pass_at_3(tmp_path: Path):
    records = []
    # control: 5 problems, all 3 repeats correct under both models
    for pid in range(5):
        for model in ("Claude Haiku 4.5", "GPT-5.4-mini"):
            for rep in range(3):
                records.append(
                    {
                        "benchmark": "miniF2F",
                        "arm": "control",
                        "problem_id": f"ctl_{pid}",
                        "generation": 0,
                        "model": model,
                        "provider_slug": "x",
                        "repeat_index": rep,
                        "is_correct": True,
                    }
                )
    # treatment: 10 problems, half correct under Claude, all wrong under GPT
    for pid in range(10):
        for rep in range(3):
            records.append(
                {
                    "benchmark": "miniF2F",
                    "arm": "treatment",
                    "problem_id": f"trt_{pid}",
                    "generation": 1 + pid % 3,
                    "model": "Claude Haiku 4.5",
                    "provider_slug": "x",
                    "repeat_index": rep,
                    "is_correct": pid < 5,
                }
            )
            records.append(
                {
                    "benchmark": "miniF2F",
                    "arm": "treatment",
                    "problem_id": f"trt_{pid}",
                    "generation": 1 + pid % 3,
                    "model": "GPT-5.4-mini",
                    "provider_slug": "x",
                    "repeat_index": rep,
                    "is_correct": False,
                }
            )

    eval_path = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_eval_jsonl(eval_path, records)

    payload = summarize_jsonl(eval_path, summary_path, bootstrap_iterations=200)
    cells = {(c["benchmark"], c["model"]): c for c in payload["cells"]}

    claude = cells[("miniF2F", "Claude Haiku 4.5")]
    assert claude["control_accuracy"] == 1.0
    assert claude["treatment_accuracy"] == 0.5
    assert claude["drop_pp"] == pytest.approx(50.0)
    assert claude["control_pass_at_3"] == 1.0
    assert claude["treatment_pass_at_3"] == 0.5

    gpt = cells[("miniF2F", "GPT-5.4-mini")]
    assert gpt["drop_pp"] == pytest.approx(100.0)


def test_generation_slope_decreasing(tmp_path: Path):
    records = []
    # Gen 1: 90% correct, Gen 5: 50%, Gen 10: 10% -> negative slope
    pattern = {1: 0.9, 5: 0.5, 10: 0.1}
    pid = 0
    for gen, acc in pattern.items():
        for _ in range(10):
            for rep in range(3):
                records.append(
                    {
                        "benchmark": "putnambench",
                        "arm": "treatment",
                        "problem_id": f"trt_{pid}_{rep}",
                        "generation": gen,
                        "model": "Claude Haiku 4.5",
                        "is_correct": pid < 10 * acc,
                        "repeat_index": rep,
                    }
                )
            pid += 1
        pid = 0  # restart pid pool inside generation for simplicity

    path = tmp_path / "eval.jsonl"
    _write_eval_jsonl(path, records)
    slopes = generation_slope_table(path)
    assert slopes
    row = slopes[0]
    assert row["slope_pp_per_gen"] < 0
