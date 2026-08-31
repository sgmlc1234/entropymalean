"""YAML-driven benchmark loaders.

Three external benchmarks back the EntropyMaLean evaluation campaign:
miniF2F, PutnamBench, and ProofNet. Their Hugging Face schemas differ, so
this module turns each one into the pipeline's canonical ``EvalRow``
(``problem_id``, ``statement``, ``gold_answer``) using the catalog in
``config/benchmarks.yaml``.

The loader supports three answer-extraction strategies, declared per
benchmark in the YAML:

- ``lean_theorem_rhs``: pull ``= <ANS> := by`` from a Lean theorem statement;
  used for miniF2F's ``formal_statement`` column.
- ``boxed_then_text``: prefer the last ``\\boxed{...}``; fall back to a
  ``"The answer is X"``-style heuristic; used for the natural-language
  solutions in PutnamBench and (when relevant) ProofNet.
- ``literal``: take the column value verbatim.

If a row cannot be canonicalized into a non-empty ``gold_answer`` it is
silently dropped from the control arm, matching the EntropyMaG-1 control
construction protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from src.evaluation.answer_grader import extract_boxed_answer
from src.evaluation.dataset import EvalRow


# --------------------------------------------------------------------------
#  YAML schema
# --------------------------------------------------------------------------


@dataclass
class BenchmarkSchema:
    """How to project a benchmark's parquet row onto ``EvalRow``."""

    problem_id_col: str
    statement_col: str
    answer_extraction: Dict[str, Any]
    quality_filter: Optional[Dict[str, Any]] = None
    # Optional: where the native Lean 4 statement / mathlib header live.
    # Populated for miniF2F (`formal_statement`, `header`), PutnamBench
    # (`lean4_statement`), and ProofNet (`formal_statement`, `header`).
    formal_statement_col: Optional[str] = None
    lean_header_col: Optional[str] = None


@dataclass
class BenchmarkSpec:
    """One ``benchmarks.<name>`` entry from the YAML catalog."""

    name: str
    hf_repo: str
    revision: str
    local_dir: Path
    license: str
    citation: str
    splits: Dict[str, str]
    schema: BenchmarkSchema
    split_filter: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkCatalog:
    cache_root: Path
    default_split: str
    control_cap: int
    benchmarks: Dict[str, BenchmarkSpec] = field(default_factory=dict)


def load_catalog(path: Path) -> BenchmarkCatalog:
    """Read ``config/benchmarks.yaml`` into a typed ``BenchmarkCatalog``."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    catalog = BenchmarkCatalog(
        cache_root=Path(defaults.get("cache_root", "data/benchmarks")),
        default_split=defaults.get("default_split", "train"),
        control_cap=int(defaults.get("control_cap", 20)),
    )
    for name, entry in (raw.get("benchmarks") or {}).items():
        schema_raw = entry["schema"]
        spec = BenchmarkSpec(
            name=name,
            hf_repo=entry["hf_repo"],
            revision=entry.get("revision", "main"),
            local_dir=Path(entry["local_dir"]),
            license=entry.get("license", "unknown"),
            citation=entry.get("citation", ""),
            splits=dict(entry.get("splits") or {}),
            split_filter=entry.get("split_filter"),
            schema=BenchmarkSchema(
                problem_id_col=schema_raw["problem_id_col"],
                statement_col=schema_raw["statement_col"],
                answer_extraction=schema_raw["answer_extraction"],
                quality_filter=schema_raw.get("quality_filter"),
                formal_statement_col=schema_raw.get("formal_statement_col"),
                lean_header_col=schema_raw.get("lean_header_col"),
            ),
        )
        catalog.benchmarks[name] = spec
    return catalog


# --------------------------------------------------------------------------
#  Answer extraction
# --------------------------------------------------------------------------


_LEAN_RHS_RE = re.compile(
    r"=\s*(?P<rhs>[^:=]+?)\s*:=\s*by",
    flags=re.DOTALL,
)
_THE_ANSWER_RE = re.compile(
    r"(?:the\s+answer\s+(?:is|equals|=)|answer\s*[:=])\s*(?P<ans>[^.\n]+)",
    flags=re.IGNORECASE,
)


def _extract_lean_rhs(formal_statement: str) -> Optional[str]:
    """From a Lean theorem like ``... = 26 := by`` return ``"26"``."""
    if not formal_statement:
        return None
    match = _LEAN_RHS_RE.search(formal_statement)
    if not match:
        return None
    rhs = match.group("rhs").strip()
    # Drop trailing parentheses, by-clauses, leftover commas.
    rhs = rhs.strip("(),; ")
    return rhs or None


def _extract_boxed_then_text(text: str) -> Optional[str]:
    """Prefer the last ``\\boxed{...}``, else fall back to "The answer is X"."""
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    match = _THE_ANSWER_RE.search(text)
    if match:
        return match.group("ans").strip().strip(".,;: ")
    # Last resort: strip the whole string of leading boilerplate.
    stripped = text.strip()
    return stripped if 0 < len(stripped) <= 64 else None


def _extract_answer(row: Dict[str, Any], rule: Dict[str, Any]) -> Optional[str]:
    """Dispatch on the YAML-declared extraction strategy."""
    kind = rule.get("kind", "literal")
    column = rule.get("column")
    raw = row.get(column) if column else None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    if kind == "literal":
        return (str(raw).strip() if raw is not None else None) or None
    if kind == "lean_theorem_rhs":
        ans = _extract_lean_rhs(str(raw or ""))
        if ans:
            return ans
    if kind in ("boxed_then_text", "lean_theorem_rhs"):
        # boxed fallback either as primary kind or as the fallback for Lean
        ans = _extract_boxed_then_text(str(raw or ""))
        if ans:
            return ans
    # explicit fallback fields
    fb_kind = rule.get("fallback_kind")
    fb_col = rule.get("fallback_column")
    if fb_kind and fb_col:
        return _extract_answer(row, {"kind": fb_kind, "column": fb_col})
    return None


def _passes_quality_filter(row: Dict[str, Any], rule: Optional[Dict[str, Any]]) -> bool:
    """Return True if ``row`` passes the optional quality filter."""
    if not rule:
        return True
    val = row.get(rule.get("column"))
    if val is None:
        return False
    minimum = rule.get("min")
    if minimum is not None and float(val) < float(minimum):
        return False
    maximum = rule.get("max")
    if maximum is not None and float(val) > float(maximum):
        return False
    return True


def _passes_split_filter(row: Dict[str, Any], rule: Optional[Dict[str, Any]]) -> bool:
    if not rule:
        return True
    keep = set(rule.get("keep") or [])
    return str(row.get(rule.get("column"))) in keep


# --------------------------------------------------------------------------
#  Loading
# --------------------------------------------------------------------------


def _find_data_files(spec: BenchmarkSpec, split: str) -> List[Path]:
    """Locate the row-bearing files for ``split`` inside the local snapshot.

    Supports three Hugging Face layouts that we have seen in the wild:
    - ``data/<split>-NNNN-of-NNNN.parquet`` (canonical ProofBench layout)
    - ``minif2f.jsonl`` or ``putnam.csv`` (raw single-file dumps)
    - ``default/<split>/0000.parquet`` (refs/convert/parquet branch)
    """
    if not spec.local_dir.exists():
        raise FileNotFoundError(
            f"benchmark not downloaded: {spec.local_dir} missing. "
            f"Run scripts/download_benchmarks.py {spec.name} first."
        )
    candidates: List[Path] = []
    # canonical split-prefixed parquet (ProofBench-style)
    candidates.extend(sorted((spec.local_dir / "data").glob(f"{split}-*.parquet")))
    # auto-converted parquet branch
    candidates.extend(sorted(spec.local_dir.glob(f"default/{split}/*.parquet")))
    # raw record formats (miniF2F.jsonl, putnam.csv, etc.)
    candidates.extend(sorted(spec.local_dir.glob("*.jsonl")))
    candidates.extend(sorted(spec.local_dir.glob("*.csv")))
    candidates.extend(sorted(spec.local_dir.glob("data/*.jsonl")))
    candidates.extend(sorted(spec.local_dir.glob("data/*.csv")))
    # final fallback: any parquet anywhere
    if not candidates:
        candidates.extend(sorted(spec.local_dir.rglob("*.parquet")))
    if not candidates:
        raise FileNotFoundError(
            f"no data files under {spec.local_dir} for split={split}"
        )
    return candidates


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream rows from a parquet / jsonl / csv file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        for row in pq.read_table(path).to_pylist():
            yield row
        return
    if suffix == ".jsonl":
        import json

        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".csv":
        import csv

        with path.open() as f:
            for row in csv.DictReader(f):
                yield row
        return
    raise ValueError(f"unsupported benchmark file format: {path}")


def _iter_parquet(spec: BenchmarkSpec, split: str) -> Iterable[Dict[str, Any]]:
    """Stream rows from the local snapshot, auto-detecting the file format."""
    for p in _find_data_files(spec, split):
        yield from _read_rows(p)


def load_benchmark_seeds(
    spec: BenchmarkSpec,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[EvalRow]:
    """Load up to ``limit`` ``EvalRow`` records from the local snapshot."""
    split = split or next(iter(spec.splits.values()))
    rows: List[EvalRow] = []
    for raw in _iter_parquet(spec, split):
        if not _passes_split_filter(raw, spec.split_filter):
            continue
        if not _passes_quality_filter(raw, spec.schema.quality_filter):
            continue
        pid = raw.get(spec.schema.problem_id_col)
        statement = raw.get(spec.schema.statement_col)
        gold = _extract_answer(raw, spec.schema.answer_extraction)
        if not pid or not statement or not gold:
            continue
        formal_stmt = None
        if spec.schema.formal_statement_col:
            v = raw.get(spec.schema.formal_statement_col)
            formal_stmt = str(v).strip() if v else None
        lean_header = None
        if spec.schema.lean_header_col:
            v = raw.get(spec.schema.lean_header_col)
            lean_header = str(v).strip() if v else None
        formal_status = (
            f"native_{spec.name.lower()}" if formal_stmt else "none"
        )
        rows.append(
            EvalRow(
                benchmark=spec.name,
                arm="control",
                problem_id=str(pid),
                statement=str(statement).strip(),
                gold_answer=str(gold).strip(),
                generation=0,
                formal_statement=formal_stmt,
                lean_header=lean_header,
                formal_status=formal_status,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_benchmark_seeds_by_name(
    catalog: BenchmarkCatalog,
    name: str,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[EvalRow]:
    """Convenience wrapper: pick the spec by name, then load seeds."""
    if name not in catalog.benchmarks:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(catalog.benchmarks)}"
        )
    return load_benchmark_seeds(
        catalog.benchmarks[name],
        split=split,
        limit=limit if limit is not None else catalog.control_cap,
    )
