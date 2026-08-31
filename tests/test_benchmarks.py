"""Unit tests for the benchmark loader (no HF download required)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.evaluation.benchmarks import (
    BenchmarkCatalog,
    BenchmarkSchema,
    BenchmarkSpec,
    _extract_answer,
    load_benchmark_seeds,
    load_catalog,
)


# ---------- answer extraction ----------


def test_extract_answer_lean_theorem_rhs():
    rule = {"kind": "lean_theorem_rhs", "column": "formal_statement"}
    row = {
        "formal_statement": (
            "theorem amc12a_2015_p10 (x y : Int) "
            "(h0 : 0 < y) (h1 : y < x) "
            "(h2 : x + y + x * y = 80) : x = 26 := by"
        )
    }
    assert _extract_answer(row, rule) == "26"


def test_extract_answer_boxed_then_text_with_boxed():
    rule = {"kind": "boxed_then_text", "column": "reference_solution"}
    row = {"reference_solution": "Some prose then \\boxed{42} done."}
    assert _extract_answer(row, rule) == "42"


def test_extract_answer_boxed_then_text_with_textual_fallback():
    rule = {"kind": "boxed_then_text", "column": "informal_solution"}
    row = {"informal_solution": "The answer is 12."}
    assert _extract_answer(row, rule) == "12"


def test_extract_answer_returns_none_on_empty():
    rule = {"kind": "boxed_then_text", "column": "x"}
    assert _extract_answer({"x": ""}, rule) is None


# ---------- catalog parsing ----------


def test_load_catalog_round_trip(tmp_path: Path):
    yaml_text = textwrap.dedent(
        """\
        defaults:
          cache_root: data/benchmarks
          default_split: train
          control_cap: 20
        benchmarks:
          dummy:
            hf_repo: example/dummy
            revision: main
            local_dir: data/benchmarks/dummy
            license: MIT
            citation: dummy2026
            splits:
              train: train
            schema:
              problem_id_col: id
              statement_col: stmt
              answer_extraction:
                kind: literal
                column: ans
        """
    )
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml_text)
    catalog = load_catalog(p)
    assert set(catalog.benchmarks) == {"dummy"}
    spec = catalog.benchmarks["dummy"]
    assert spec.hf_repo == "example/dummy"
    assert spec.schema.statement_col == "stmt"
    assert spec.schema.answer_extraction == {"kind": "literal", "column": "ans"}


# ---------- end-to-end loader against synthetic parquet ----------


def _write_parquet(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, path)


def test_load_benchmark_seeds_with_quality_filter(tmp_path: Path):
    local_dir = tmp_path / "fake_proofnet"
    _write_parquet(
        local_dir / "data" / "train-00000-of-00001.parquet",
        [
            {
                "problem_id": "p1",
                "problem": "Compute 2+2",
                "reference_solution": "The answer is \\boxed{4}",
                "expert_rating": 7,
            },
            {
                "problem_id": "p2",
                "problem": "Hard",
                "reference_solution": "We get 12",
                "expert_rating": 2,  # filtered out by quality
            },
            {
                "problem_id": "p3",
                "problem": "",
                "reference_solution": "irrelevant",
                "expert_rating": 7,  # filtered out because statement empty
            },
        ],
    )
    spec = BenchmarkSpec(
        name="proofnet",
        hf_repo="x/y",
        revision="main",
        local_dir=local_dir,
        license="MIT",
        citation="x",
        splits={"train": "train"},
        schema=BenchmarkSchema(
            problem_id_col="problem_id",
            statement_col="problem",
            answer_extraction={"kind": "boxed_then_text", "column": "reference_solution"},
            quality_filter={"column": "expert_rating", "min": 5},
        ),
    )
    rows = load_benchmark_seeds(spec, limit=10)
    assert [r.problem_id for r in rows] == ["p1"]
    assert rows[0].gold_answer == "4"
    assert rows[0].benchmark == "proofnet"
    assert rows[0].arm == "control"


def test_load_benchmark_seeds_from_jsonl(tmp_path: Path):
    local_dir = tmp_path / "fake_minif2f"
    local_dir.mkdir()
    (local_dir / "minif2f.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "name": "n1",
                    "informal_prefix": "Compute x.",
                    "formal_statement": "theorem n1 : x = 7 := by",
                    "split": "valid",
                },
                {
                    "name": "n2",
                    "informal_prefix": "Compute y.",
                    "formal_statement": "theorem n2 : y = 99 := by",
                    "split": "train",  # filtered out by split_filter
                },
            ]
        )
    )
    spec = BenchmarkSpec(
        name="miniF2F",
        hf_repo="x/y",
        revision="main",
        local_dir=local_dir,
        license="MIT",
        citation="x",
        splits={"train": "train"},
        split_filter={"column": "split", "keep": ["valid", "test"]},
        schema=BenchmarkSchema(
            problem_id_col="name",
            statement_col="informal_prefix",
            answer_extraction={
                "kind": "lean_theorem_rhs",
                "column": "formal_statement",
            },
        ),
    )
    rows = load_benchmark_seeds(spec, limit=10)
    assert [r.problem_id for r in rows] == ["n1"]
    assert rows[0].gold_answer == "7"


def test_load_benchmark_seeds_missing_dir(tmp_path: Path):
    spec = BenchmarkSpec(
        name="missing",
        hf_repo="x/y",
        revision="main",
        local_dir=tmp_path / "does_not_exist",
        license="x",
        citation="x",
        splits={"train": "train"},
        schema=BenchmarkSchema(
            problem_id_col="id",
            statement_col="stmt",
            answer_extraction={"kind": "literal", "column": "ans"},
        ),
    )
    with pytest.raises(FileNotFoundError):
        load_benchmark_seeds(spec)
