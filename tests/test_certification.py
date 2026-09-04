import asyncio
import pytest
from tests._lean_available import mathlib_built, SKIP_REASON
import csv
import json
import subprocess
import sys
from pathlib import Path

from src.certification import (
    CertificationInput,
    GeneratedProblem,
    build_certification_graph,
    certify_csv,
    certify_problem,
)
from src.utils.lean_templates import (
    SUPPORTED_FAMILIES,
    detect_family,
    generate_lean_template,
    is_trivial_stub,
)
from src.utils.lean_interface import LeanChecker


class FakeLeanChecker:
    def __init__(self, *, available=True, type_ok=True, error="lean failed"):
        self.available = available
        self.type_ok = type_ok
        self.error = error

    async def health_check(self):
        return self.available

    async def type_check(self, lean_code):
        if self.type_ok:
            return True, ""
        return False, self.error


def test_supported_family_detection_and_templates():
    cases = [
        ("Find GCD(2026, 1234).", "2", "gcd"),
        (
            "Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
            "96",
            "gcd_divisor_sum",
        ),
        ("Find the units digit of 7^{2026}.", "9", "units_digit"),
        ("Find the sum of all positive divisors of 120.", "360", "divisor_sum"),
        (
            "Let m be the sum of all positive divisors of 120. Find 2026 mod m.",
            "226",
            "divisor_sum_mod",
        ),
        ("Sum of 20-term arithmetic series, first=3, diff=4.", "820", "arithmetic_series"),
        (
            "Find the sum of the first 20 terms of the arithmetic sequence 3, 7, 11, 15, ...",
            "820",
            "arithmetic_series",
        ),
        ("Find 2026 mod 7.", "3", "modular_congruence"),
        ("How many nonneg integer solutions does x₁ + x₂ + x₃ = 10 have?", "66", "stars_and_bars"),
    ]
    assert SUPPORTED_FAMILIES == {
            "gcd",
            "gcd_divisor_sum",
            "units_digit",
            "divisor_sum",
            "divisor_sum_mod",
            "stars_and_bars",
        "arithmetic_series",
        "modular_congruence",
    }
    for statement, answer, family in cases:
        assert detect_family(statement) == family
        template = generate_lean_template(statement, answer)
        assert template is not None
        assert template.family == family
        assert template.proof_method == "native_decide"
        assert not is_trivial_stub(template.lean_code)


def test_anti_stub_rejects_true_trivial():
    assert is_trivial_stub("theorem problem_statement : True := by trivial")
    assert not is_trivial_stub("theorem gcd_2026_1234 : Nat.gcd 2026 1234 = 2 := by native_decide")


def test_unsupported_family_returns_unsupported():
    result = certify_problem(
        CertificationInput(
            id="putnam_real_analysis",
            statement="Find the maximum value of f(x) = x/(1+x^4) for x >= 0.",
            answer="3^(3/4)/4",
        ),
        checker=FakeLeanChecker(),
    )
    assert result.status == "unsupported"
    assert result.lean_level == 0
    assert result.lean_code is None


def test_certified_when_template_and_lean_succeed():
    result = certify_problem(
        CertificationInput(
            id="divisor_sum_120",
            statement="Find the sum of all positive divisors of 120.",
            answer="360",
        ),
        checker=FakeLeanChecker(type_ok=True),
    )
    assert result.status == "certified"
    assert result.lean_level >= 2
    assert result.anti_stub_passed is True
    assert result.llm_used is False
    assert result.llm_model is None
    assert result.lean_code and "native_decide" in result.lean_code


def test_wrong_answer_failure_preserves_lean_error():
    result = certify_problem(
        CertificationInput(
            id="bad_units_digit",
            statement="Find the units digit of 7^{2026}.",
            answer="3",
        ),
        checker=FakeLeanChecker(type_ok=False, error="native_decide failed"),
    )
    assert result.status == "failed"
    assert result.error == "native_decide failed"
    assert "% 10 = 3" in result.lean_code


def test_missing_lean_binary_is_row_level_status():
    result = certify_problem(
        CertificationInput(
            id="gcd",
            statement="Find GCD(2026, 1234).",
            answer="2",
        ),
        checker=FakeLeanChecker(available=False),
    )
    assert result.status == "lean_unavailable"
    assert result.error == "Lean binary unavailable"


def test_lean_health_does_not_cache_negative_result(monkeypatch):
    calls = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"Lean (version test)", b""

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise FileNotFoundError("lean missing")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    checker = LeanChecker()
    assert asyncio.run(checker.health_check()) is False
    assert checker._health_check_cache is None
    assert asyncio.run(checker.health_check()) is True
    assert len(calls) == 2


def test_langgraph_certification_graph_runs_one_problem():
    graph = build_certification_graph(checker=FakeLeanChecker())
    final_state = asyncio.run(
        graph.ainvoke(
            {
                "input": CertificationInput(
                    id="mod",
                    statement="Find 2026 mod 7.",
                    answer="3",
                ),
                "errors": [],
            }
        )
    )
    assert final_state["result"].status == "certified"


def test_langgraph_certification_graph_exposes_traceable_steps():
    graph = build_certification_graph(checker=FakeLeanChecker())
    graph_view = graph.get_graph()
    assert {
        "detect_family",
        "template",
        "anti_stub",
        "lean_health",
        "lean_check",
    }.issubset(set(graph_view.nodes))


def test_llm_generation_graph_adds_harder_problem_node():
    graph = build_certification_graph(
        checker=FakeLeanChecker(),
        generate_harder=True,
        generator=lambda parent, config: GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement="Find the sum of all positive divisors of 840.",
            answer="2880",
            difficulty_label="medium",
            params={"n": 840},
            harder_reason="Uses a larger highly composite number.",
        ),
    )
    graph_view = graph.get_graph()
    assert "generate_candidate" in set(graph_view.nodes)


def test_generated_harder_problem_is_certified_and_marks_llm_usage():
    result = certify_problem(
        CertificationInput(
            id="seed_divisor_sum",
            statement="Find the sum of all positive divisors of 120.",
            answer="360",
        ),
        checker=FakeLeanChecker(),
        generate_harder=True,
        generation_model="test-model",
        generator=lambda parent, config: GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement="Find the sum of all positive divisors of 840.",
            answer="2880",
            difficulty_label="medium",
            params={"n": 840},
            harder_reason="Uses a larger highly composite number.",
        ),
    )
    assert result.status == "certified"
    assert result.problem_id == "seed_divisor_sum__gen1"
    assert result.source_problem_id == "seed_divisor_sum"
    assert result.generation == 1
    assert result.statement == "Find the sum of all positive divisors of 840."
    assert result.answer == "2880"
    assert result.axis_aligned is True
    assert result.llm_used is True
    assert result.llm_model == "test-model"


def test_generated_problem_must_match_planner_required_params():
    result = certify_problem(
        CertificationInput(
            id="seed_divisor_sum",
            statement="Find the sum of all positive divisors of 120.",
            answer="360",
            metadata={
                "target_family": "divisor_sum",
                "variation_axis": "change n from 120 to 360",
                "required_params": {"n": 360},
            },
        ),
        checker=FakeLeanChecker(),
        generate_harder=True,
        generation_model="test-model",
        generator=lambda parent, config: GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement="Find the sum of all positive divisors of 840.",
            answer="2880",
            params={"n": 840},
            harder_reason="Off-contract generated value.",
        ),
    )
    assert result.status == "planner_axis_mismatch"
    assert "did not match required" in result.error


def test_generated_problem_projected_params_must_match_params():
    result = certify_problem(
        CertificationInput(
            id="seed_divisor_sum",
            statement="Find the sum of all positive divisors of 120.",
            answer="360",
            metadata={
                "target_family": "divisor_sum",
                "variation_axis": "derive a richer divisor-sum skeleton",
            },
        ),
        checker=FakeLeanChecker(),
        generate_harder=True,
        generation_model="test-model",
        generator=lambda parent, config: GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="divisor_sum",
            statement="Find the sum of all positive divisors of 840.",
            answer="2880",
            params={"n": 840},
            projected_params={"n": 360},
            reasoning_pattern="prime_factorization_sigma",
            solution_skeleton={"target_computation": "sigma(840)"},
            projection_check={"passed": False, "evidence": "intentional mismatch"},
            harder_reason="Off-contract projected value.",
        ),
    )
    assert result.status == "planner_axis_mismatch"
    assert "projected_params" in result.error


def test_composite_projection_allows_derived_canonical_params():
    result = certify_problem(
        CertificationInput(
            id="seed_gcd",
            statement="Find GCD(84, 126).",
            answer="42",
            metadata={
                "target_family": "gcd_divisor_sum",
                "variation_axis": "derive n by gcd and apply divisor-sum operation",
            },
        ),
        checker=FakeLeanChecker(),
        generate_harder=True,
        generation_model="test-model",
        generator=lambda parent, config: GeneratedProblem(
            id=f"{parent.id}__gen1",
            source_problem_id=parent.id,
            family="gcd_divisor_sum",
            statement="Let n = GCD(84, 126). Find the sum of all positive divisors of n.",
            answer="96",
            params={"a": 84, "b": 126, "gcd": 42},
            projected_params={"a": 84, "b": 126},
            reasoning_pattern="gcd_then_sigma",
            solution_skeleton={"target_computation": "sigma(gcd(84,126))"},
            projection_check={"passed": True, "evidence": "a,b define the derived gcd"},
            harder_reason="Two-step composite reasoning.",
        ),
    )
    assert result.status == "certified"
    assert result.generated_params["gcd"] == 42


def test_certify_csv_writes_one_jsonl_row_per_input(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.jsonl"
    rows = [
        {"id": "a", "statement": "Find GCD(2026, 1234).", "answer": "2"},
        {"id": "b", "statement": "Find the maximum of x/(1+x^4).", "answer": "1"},
    ]
    with input_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "statement", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    results = certify_csv(input_path, output_path, checker=FakeLeanChecker())
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(results) == 2
    assert len(lines) == 2
    assert lines[0]["status"] == "certified"
    assert lines[0]["llm_used"] is False
    assert lines[1]["status"] == "unsupported"


def test_certify_csv_preserves_extended_input_metadata(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "release_id,id,statement,answer,solution,verification_code,formal_statement,"
        "lean_header,formal_status,operation,difficulty,difficulty_label,generation,"
        "source_run,source_file,source_slot,parent_ids,ancestor_ids,statement_sha256,"
        "answer_sha256,formal_statement_sha256\n"
        'rel1,p1,"Find GCD(2026, 1234).",2,sol,,theorem native : True := by trivial,'
        "import Mathlib,native_seed,seed,0.5,easy,0,run1,source://file,slot_001,[],[],"
        "stmtsha,answersha,formalsha\n",
        encoding="utf-8",
    )

    certify_csv(input_path, output_path, checker=FakeLeanChecker())

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["problem_id"] == "p1"
    assert row["release_id"] == "rel1"
    assert row["source_run"] == "run1"
    assert row["source_file"] == "source://file"
    assert row["source_slot"] == "slot_001"
    assert row["operation"] == "seed"
    assert row["difficulty"] == "0.5"
    assert row["difficulty_label"] == "easy"
    assert row["statement_sha256"] == "stmtsha"
    assert row["answer_sha256"] == "answersha"
    assert row["formal_statement_sha256"] == "formalsha"
    assert row["parent_ids"] == []
    assert row["ancestor_ids"] == []
    assert row["input_metadata"]["formal_status"] == "native_seed"


def test_certify_csv_limit(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.jsonl"
    with input_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "statement", "answer"])
        writer.writeheader()
        writer.writerow({"id": "a", "statement": "Find 2026 mod 7.", "answer": "3"})
        writer.writerow({"id": "b", "statement": "Find GCD(12, 18).", "answer": "6"})

    results = certify_csv(input_path, output_path, limit=1, checker=FakeLeanChecker())
    assert len(results) == 1
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.skipif(not mathlib_built(), reason=SKIP_REASON)
def test_cli_summary_and_jsonl_for_unsupported_rows(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "id,statement,answer\n"
        "u1,Find the maximum of x/(1+x^4).,1\n"
        "u2,Find the maximum of sin(x)+cos(x).,1\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/archive/certify_csv.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "total=1" in completed.stdout
    assert "unsupported=1" in completed.stdout
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
