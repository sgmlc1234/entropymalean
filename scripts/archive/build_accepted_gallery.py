#!/usr/bin/env python3
"""Build a human-review gallery for the curated accepted problem ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolved one level short after the
    move -- to a directory that exists, so nothing raised."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
DEFAULT_INPUT = REPO_ROOT / "data/evaluation/treatment_inventory/final_curated/accepted.jsonl"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs/accepted_problem_gallery.md"
DEFAULT_OUTPUT_TEX = REPO_ROOT / "docs/accepted_problem_gallery.tex"
BENCHMARK_ORDER = ["minif2f", "putnambench", "proofnet"]
TEX_SYMBOL_REPLACEMENTS = {
    "ℕ": "Nat",
    "ℤ": "Int",
    "ℚ": "Rat",
    "ℝ": "Real",
    "ℂ": "Complex",
    "∀": "forall",
    "∃": "exists",
    "∈": " in ",
    "∉": " notin ",
    "∧": " /\\ ",
    "∨": " \\/ ",
    "¬": "not ",
    "→": " -> ",
    "↔": " <-> ",
    "⇒": " => ",
    "←": " <- ",
    "↦": " maps to ",
    "≤": " <= ",
    "≥": " >= ",
    "≠": " != ",
    "≡": " == ",
    "≃": " ~= ",
    "∣": " | ",
    "∑": "sum",
    "∏": "prod",
    "√": "sqrt",
    "∞": "infinity",
    "⋯": "...",
    "⊢": "|-",
    "⊤": "Top",
    "⟨": "<",
    "⟩": ">",
    "𝕜": "k",
    "⁻": "^-",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "ˣ": "^x",
    "ᶠ": "^f",
    "𝓝": "nhds",
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
    "ₜ": "_t",
}


@dataclass
class ParentContext:
    parent_id: str
    statement: str = ""
    formal_statement: str = ""
    proof_idea: str = ""
    source: str = ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compact(value: Any, *, limit: int = 900) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _first_text(*values: Any, limit: int = 900) -> str:
    for value in values:
        text = _compact(value, limit=limit)
        if text and text != "not_available":
            return text
    return ""


def _json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except Exception:
            pass
        return [value] if value else []
    return [str(value)]


def _markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _tex_normalize_symbols(text: str) -> str:
    for old, new in TEX_SYMBOL_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def _tex_escape_text(text: str) -> str:
    text = _tex_normalize_symbols(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _tex_statement(statement: str) -> str:
    """Return a conservative TeX display for a mostly-natural statement."""
    statement = _compact(statement, limit=1600)
    if not statement:
        return r"\text{Statement unavailable.}"
    parts = statement.split("$")
    rendered: List[str] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        if idx % 2 == 1:
            rendered.append(part)
        else:
            rendered.append(r"\text{" + _tex_escape_text(part) + "}")
    body = " ".join(rendered).strip()
    return r"\begin{gathered}" + body + r"\end{gathered}"


def _tex_rich_text(text: str, *, limit: int = 1800) -> str:
    """Escape prose while preserving simple `$...$` math spans."""
    text = _tex_normalize_symbols(_compact(text, limit=limit))
    if not text:
        return ""
    parts = text.split("$")
    rendered: List[str] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        if idx % 2 == 1:
            rendered.append("$" + part + "$")
        else:
            rendered.append(_tex_escape_text(part))
    return "".join(rendered)


MATH_EXPR_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[A-Za-z][A-Za-z0-9_]*|\d+)"
    r"(?:\^[A-Za-z0-9_()]+)"
    r"(?:\s*(?:<=|>=|=|[+\-*/%])\s*"
    r"(?:[A-Za-z][A-Za-z0-9_]*|\d+)"
    r"(?:\^[A-Za-z0-9_()]+)?)*"
)


def _tex_escape_inline_math(expr: str) -> str:
    expr = _tex_normalize_symbols(expr)
    expr = re.sub(r"([A-Za-z][A-Za-z0-9_]*|\d+)\^([A-Za-z0-9_()]+)", r"\1^{\2}", expr)
    expr = expr.replace("*", r"\cdot ")
    expr = expr.replace("%", r"\bmod ")
    expr = expr.replace("_", r"\_")
    return expr


def _tex_statement_prose(text: str, *, limit: int = 1800) -> str:
    """Render natural statements as readable prose with simple inline math."""
    text = _tex_normalize_symbols(_compact(text, limit=limit))
    if not text:
        return ""

    parts = text.split("$")
    rendered: List[str] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        if idx % 2 == 1:
            rendered.append("$" + part + "$")
            continue

        cursor = 0
        for match in MATH_EXPR_RE.finditer(part):
            rendered.append(_tex_escape_text(part[cursor : match.start()]))
            rendered.append("$" + _tex_escape_inline_math(match.group(0)) + "$")
            cursor = match.end()
        rendered.append(_tex_escape_text(part[cursor:]))
    return "".join(rendered)


def _tex_paragraph(text: str) -> str:
    return _tex_escape_text(_compact(text, limit=1800))


def _tex_path(value: str) -> str:
    value = value.replace("|", "/")
    return rf"\path|{value}|"


def _short_label(value: str, *, limit: int = 86) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    keep = max(12, (limit - 3) // 2)
    return value[:keep].rstrip("_-") + "..." + value[-keep:].lstrip("_-")


def _verbatim_safe(text: str) -> str:
    return _tex_normalize_symbols(text or "").replace(
        r"\end{Verbatim}", r"\textbackslash{}end\{Verbatim\}"
    )


def _theorem_name(row: Dict[str, Any]) -> str:
    text = _first_text(row.get("formal_statement"), row.get("lean_code"), limit=4000)
    match = re.search(r"\b(?:theorem|lemma|def)\s+([A-Za-z0-9_'.]+)", text)
    return match.group(1) if match else ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _statement_hash(row: Dict[str, Any]) -> str:
    hashes = row.get("hashes") if isinstance(row.get("hashes"), dict) else {}
    return str(row.get("statement_sha256") or hashes.get("statement_sha256") or _sha256_text(str(row.get("statement") or "")))


def _formal_hash(row: Dict[str, Any]) -> str:
    hashes = row.get("hashes") if isinstance(row.get("hashes"), dict) else {}
    formal = str(row.get("formal_statement") or row.get("lean_code") or "")
    return str(row.get("formal_statement_sha256") or hashes.get("formal_statement_sha256") or _sha256_text(formal))


def _benchmark(row: Dict[str, Any]) -> str:
    return str(row.get("_accepted_benchmark") or row.get("benchmark") or "unknown").lower()


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    original_index = {id(row): idx for idx, row in enumerate(rows)}
    bench_rank = {bench: idx for idx, bench in enumerate(BENCHMARK_ORDER)}

    def key(row: Dict[str, Any]) -> tuple[int, int, int]:
        op_rank = 0 if row.get("op_type") == "crossover" else 1
        return (bench_rank.get(_benchmark(row), 99), op_rank, original_index[id(row)])

    return sorted(rows, key=key)


def _row_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key in ("problem_id", "id", "name"):
            value = row.get(key)
            if value and value not in index:
                index[str(value)] = row
    return index


def _source_paths(rows: Iterable[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    for row in rows:
        for key in ("source_file", "_inventory_source_file"):
            value = row.get(key)
            if not value or "://" in str(value):
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.exists() and path.suffix == ".jsonl":
                paths.append(path)
    return sorted(set(paths))


def _load_source_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = list(rows)
    for path in _source_paths(rows):
        all_rows.extend(_read_jsonl(path))
    return _row_index(all_rows)


def _load_csv_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    patterns = [
        REPO_ROOT / "data/raw/*.csv",
        REPO_ROOT / "data/certified/*.gen0_seeds.csv",
    ]
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        row = dict(row)
                        row["_csv_source"] = str(path.relative_to(REPO_ROOT))
                        for key in ("problem_id", "id", "name"):
                            value = row.get(key)
                            if value and value not in index:
                                index[str(value)] = row
            except Exception:
                continue
    return index


def _context_from_parent_card(parent_id: str, card: Dict[str, Any]) -> ParentContext:
    proof_context = card.get("proof_context") if isinstance(card.get("proof_context"), dict) else {}
    decomposition = (
        card.get("theorem_decomposition")
        if isinstance(card.get("theorem_decomposition"), dict)
        else {}
    )
    statement = _first_text(card.get("statement_preview"), card.get("statement"), limit=900)
    formal = _first_text(
        card.get("formal_statement"),
        proof_context.get("formal_statement"),
        proof_context.get("lean_code"),
        decomposition.get("main_conclusion"),
        limit=1200,
    )
    proof_idea = _first_text(
        proof_context.get("solution"),
        card.get("answer_preview"),
        decomposition.get("proof_checkpoints"),
        proof_context.get("usable_proof_atoms"),
        limit=900,
    )
    return ParentContext(
        parent_id=parent_id,
        statement=statement,
        formal_statement=formal,
        proof_idea=proof_idea,
        source="operator_card.parent_cards",
    )


def _context_from_row(parent_id: str, row: Dict[str, Any], source: str) -> ParentContext:
    proof_context = row.get("proof_context") if isinstance(row.get("proof_context"), dict) else {}
    statement = _first_text(
        row.get("statement"),
        row.get("statement_preview"),
        row.get("informal_statement"),
        limit=900,
    )
    formal = _first_text(
        row.get("formal_statement"),
        row.get("lean_code"),
        proof_context.get("lean_code"),
        limit=1200,
    )
    proof_idea = _first_text(
        row.get("proof_plan"),
        row.get("solution"),
        row.get("generation_notes"),
        proof_context.get("solution"),
        row.get("answer"),
        limit=900,
    )
    return ParentContext(parent_id=parent_id, statement=statement, formal_statement=formal, proof_idea=proof_idea, source=source)


def _resolve_parents(
    row: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
    csv_index: Dict[str, Dict[str, Any]],
) -> List[ParentContext]:
    normalized_parents = row.get("parents")
    if isinstance(normalized_parents, list):
        contexts: List[ParentContext] = []
        for parent in normalized_parents:
            if not isinstance(parent, dict):
                continue
            parent_id = str(parent.get("parent_id") or parent.get("id") or "")
            if not parent_id:
                continue
            contexts.append(
                ParentContext(
                    parent_id=parent_id,
                    statement=_first_text(parent.get("statement"), limit=900),
                    formal_statement=_first_text(parent.get("formal_statement"), limit=1200),
                    proof_idea=_first_text(parent.get("proof_idea"), limit=900),
                    source=_first_text(parent.get("source"), limit=200),
                )
            )
        if contexts:
            return contexts

    parent_ids = _json_list(row.get("parent_ids"))
    if not parent_ids:
        operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
        parent_ids = _json_list(operator_card.get("parent_ids"))
    if not parent_ids:
        source_problem_id = row.get("source_problem_id")
        if source_problem_id and source_problem_id != row.get("problem_id"):
            parent_ids = [str(source_problem_id)]

    operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
    parent_cards = operator_card.get("parent_cards")
    parent_card_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(parent_cards, list):
        for card in parent_cards:
            if isinstance(card, dict) and card.get("id"):
                parent_card_index[str(card["id"])] = card

    contexts: List[ParentContext] = []
    for parent_id in parent_ids:
        if parent_id in parent_card_index:
            contexts.append(_context_from_parent_card(parent_id, parent_card_index[parent_id]))
        elif parent_id in source_index:
            contexts.append(_context_from_row(parent_id, source_index[parent_id], "accepted/source JSONL"))
        elif parent_id in csv_index:
            contexts.append(_context_from_row(parent_id, csv_index[parent_id], csv_index[parent_id].get("_csv_source", "CSV")))
        else:
            contexts.append(ParentContext(parent_id=parent_id, source="parent context unavailable"))
    return contexts


def _extension_idea(row: Dict[str, Any]) -> str:
    operator_card = row.get("operator_card") if isinstance(row.get("operator_card"), dict) else {}
    curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    return _first_text(
        curation.get("rationale"),
        row.get("proof_summary"),
        row.get("_manual_qa_rationale"),
        row.get("proof_plan"),
        row.get("solution"),
        operator_card.get("goal"),
        operator_card.get("operator_goal"),
        limit=1200,
    )


def _source_label(row: Dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return _first_text(
        provenance.get("source_file"),
        row.get("source_file"),
        row.get("_inventory_source_file"),
        row.get("_accepted_source"),
        limit=500,
    )


def _snapshot_label(row: Dict[str, Any]) -> str:
    curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    return _first_text(row.get("_accepted_snapshot"), curation.get("snapshot"), limit=300) or "unknown"


def _render_markdown_card(
    number: int,
    row: Dict[str, Any],
    parents: List[ParentContext],
) -> str:
    problem_id = str(row.get("problem_id") or "")
    theorem_name = _theorem_name(row)
    benchmark = _benchmark(row)
    op_type = row.get("op_type") or "unknown"
    generation = row.get("generation", "")
    slot = row.get("slot", "")
    curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    cluster = row.get("_manual_qa_cluster") or curation.get("cluster") or ""
    statement = _compact(row.get("statement"), limit=1800)
    formal = _first_text(row.get("formal_statement"), row.get("lean_code"), limit=1800)
    lean_code = str(row.get("lean_code") or "").strip()
    extension = _extension_idea(row)
    hashes = [
        f"`statement_sha256={_statement_hash(row)}`",
        f"`formal_statement_sha256={_formal_hash(row)}`",
    ]

    lines: List[str] = []
    title = f"### #{number:02d} `{problem_id}`"
    if theorem_name:
        title += f" · `{theorem_name}`"
    lines.append(title)
    lines.append("")
    lines.append(
        f"- **Benchmark / op:** `{benchmark}` / `{op_type}`"
        f" · **generation/slot:** `{generation}` / `{slot}`"
    )
    if cluster:
        lines.append(f"- **Manual QA cluster:** `{cluster}`")
    lines.append(f"- **Source:** `{_source_label(row)}`")
    lines.append("")
    lines.append("**Direct parents**")
    if parents:
        for parent in parents:
            lines.append(f"- `{parent.parent_id}` _({parent.source})_")
            if parent.statement:
                lines.append(f"  - Statement: {_markdown_escape(parent.statement)}")
            if parent.formal_statement:
                lines.append(f"  - Formal/proof surface: `{_compact(parent.formal_statement, limit=700)}`")
            if parent.proof_idea:
                lines.append(f"  - Proof idea: {_markdown_escape(parent.proof_idea)}")
            if not (parent.statement or parent.formal_statement or parent.proof_idea):
                lines.append("  - Parent context unavailable in local sources.")
    else:
        lines.append("- Parent context unavailable.")
    lines.append("")
    lines.append(f"**Extension idea.** {_markdown_escape(extension) if extension else 'No extension note available.'}")
    lines.append("")
    lines.append("**Generated statement.**")
    lines.append("")
    lines.append(f"> {_markdown_escape(statement)}")
    lines.append("")
    lines.append("**TeX-style statement.**")
    lines.append("")
    lines.append("$$")
    lines.append(_tex_statement(statement))
    lines.append("$$")
    lines.append("")
    lines.append("**Lean formal statement.**")
    lines.append("")
    lines.append("```lean")
    lines.append(formal)
    lines.append("```")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Lean certificate</summary>")
    lines.append("")
    lines.append("```lean")
    lines.append(lean_code or "-- Lean code unavailable")
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("**Audit metadata.** " + " · ".join(hashes))
    lines.append("")
    return "\n".join(lines)


def _render_markdown(rows: List[Dict[str, Any]], parents_by_id: Dict[str, List[ParentContext]]) -> str:
    total = len(rows)
    by_benchmark = Counter(_benchmark(row) for row in rows)
    by_op = Counter(str(row.get("op_type") or "unknown") for row in rows)
    snapshots = Counter(_snapshot_label(row) for row in rows)
    source_files = Counter(_source_label(row) for row in rows)

    lines: List[str] = [
        "# Accepted Problem Gallery",
        "",
        "This gallery is generated from the curated accepted ledger. It is intended for manual QA and paper-facing review of parent-to-child mathematical development.",
        "",
        "## Summary",
        "",
        f"- **Total accepted problems:** {total}",
        "- **By benchmark:** " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_benchmark.items())),
        "- **By op type:** " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_op.items())),
        "- **Source ledger:** `data/evaluation/treatment_inventory/final_curated/accepted.jsonl`",
        "",
        "## Snapshot Ledger",
        "",
    ]
    for snapshot, count in snapshots.most_common():
        lines.append(f"- `{snapshot}`: {count}")
    lines.extend(["", "## Source Files", ""])
    for source, count in source_files.most_common():
        lines.append(f"- `{source}`: {count}")

    sorted_rows = _sort_rows(rows)
    number_by_id = {id(row): idx + 1 for idx, row in enumerate(sorted_rows)}
    for benchmark in BENCHMARK_ORDER:
        bench_rows = [row for row in sorted_rows if _benchmark(row) == benchmark]
        if not bench_rows:
            continue
        lines.extend(["", f"## {benchmark}", ""])
        for row in bench_rows:
            lines.append(_render_markdown_card(number_by_id[id(row)], row, parents_by_id.get(str(row.get("problem_id")), [])))
    other_rows = [row for row in sorted_rows if _benchmark(row) not in BENCHMARK_ORDER]
    if other_rows:
        lines.extend(["", "## other", ""])
        for row in other_rows:
            lines.append(_render_markdown_card(number_by_id[id(row)], row, parents_by_id.get(str(row.get("problem_id")), [])))
    return "\n".join(lines).rstrip() + "\n"


def _render_tex_card(number: int, row: Dict[str, Any], parents: List[ParentContext]) -> str:
    problem_id = str(row.get("problem_id") or "")
    theorem_name = _theorem_name(row) or "unknown"
    benchmark = _benchmark(row)
    op_type = str(row.get("op_type") or "unknown")
    statement = _compact(row.get("statement"), limit=1800)
    formal = _first_text(row.get("formal_statement"), row.get("lean_code"), limit=2400)
    lean_code = str(row.get("lean_code") or "-- Lean certificate unavailable").strip()
    extension = _extension_idea(row)
    curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    cluster = str(row.get("_manual_qa_cluster") or curation.get("cluster") or "")
    title = _short_label(theorem_name, limit=84)
    toc_title = _short_label(f"#{number:02d} {theorem_name}", limit=58)

    lines = [
        rf"\subsection[{_tex_escape_text(toc_title)}]{{\#{number:02d} \texorpdfstring{{\texttt{{{_tex_escape_text(title)}}}}}{{{_tex_escape_text(title)}}}}}",
        r"\begin{theorembox}{Generated theorem}",
        r"\textbf{Theorem.} {\itshape " + _tex_statement_prose(statement, limit=1800) + r"}",
        r"\end{theorembox}",
        r"\begin{formalbox}{Lean formal statement}",
        _verbatim_safe(formal),
        r"\end{formalbox}",
        r"\begin{leanbox}{Lean certificate}",
        _verbatim_safe(lean_code),
        r"\end{leanbox}",
        r"\begin{ideabox}{How this extends the parents}",
        _tex_rich_text(extension or "No extension note available.", limit=1500),
        r"\end{ideabox}",
        "",
    ]
    if parents:
        for parent_index, parent in enumerate(parents, start=1):
            lines.append(rf"\begin{{parentthmbox}}{{Parent theorem {parent_index}}}")
            lines.append(
                rf"{{\footnotesize \textbf{{ID.}} {_tex_path(parent.parent_id)}\quad "
                rf"\textbf{{Source.}} {_tex_escape_text(parent.source)}\par}}"
            )
            if parent.statement:
                lines.append(rf"\medskip \textbf{{Theorem.}} {{\itshape {_tex_statement_prose(parent.statement, limit=1000)}}}")
            if parent.proof_idea:
                lines.append(r"\tcblower")
                lines.append(rf"\textbf{{Proof idea.}} {_tex_rich_text(parent.proof_idea, limit=900)}")
            lines.append(r"\end{parentthmbox}")
    else:
        lines.extend(
            [
                r"\begin{parentthmbox}{Parent theorem unavailable}",
                "Parent context unavailable.",
                r"\end{parentthmbox}",
            ]
        )
    lines.extend(
        [
            r"{\scriptsize",
            rf"\textbf{{Audit.}} {_tex_escape_text(benchmark)} / {_tex_escape_text(op_type)}"
            rf"\quad generation/slot {_tex_escape_text(str(row.get('generation', '')))} / {_tex_escape_text(str(row.get('slot', '')))}"
            rf"\quad cluster {_tex_path(cluster)}",
            r"}",
        ]
    )
    return "\n".join(lines)


def _render_tex(rows: List[Dict[str, Any]], parents_by_id: Dict[str, List[ParentContext]]) -> str:
    by_benchmark = Counter(_benchmark(row) for row in rows)
    by_op = Counter(str(row.get("op_type") or "unknown") for row in rows)
    sorted_rows = _sort_rows(rows)
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{amsmath}",
        r"\usepackage{unicode-math}",
        r"\IfFontExistsTF{STIX Two Text}{\setmainfont{STIX Two Text}}{\setmainfont{Palatino}}",
        r"\IfFontExistsTF{Helvetica Neue}{\setsansfont{Helvetica Neue}}{\setsansfont{Arial}}",
        r"\IfFontExistsTF{Menlo}{\setmonofont{Menlo}[Scale=0.82]}{\setmonofont{Latin Modern Mono}[Scale=0.82]}",
        r"\IfFontExistsTF{STIX Two Math}{\setmathfont{STIX Two Math}}{\setmathfont{Latin Modern Math}}",
        r"\IfFontExistsTF{Menlo}{\newfontfamily\leanfont{Menlo}[Scale=0.78]}{\newfontfamily\leanfont{Latin Modern Mono}[Scale=0.78]}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\usepackage{fvextra}",
        r"\usepackage[most]{tcolorbox}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!50!black}",
        r"\definecolor{emgblue}{HTML}{1E4E79}",
        r"\definecolor{emglightblue}{HTML}{EEF6FC}",
        r"\definecolor{emggreen}{HTML}{1F6F50}",
        r"\definecolor{emglightgreen}{HTML}{EFF8F1}",
        r"\definecolor{emggray}{HTML}{F7F7F7}",
        r"\definecolor{emgborder}{HTML}{CBD5E1}",
        r"\tcbset{boxrule=0.45pt, arc=1.5mm, left=1.3mm, right=1.3mm, top=1mm, bottom=1mm, before skip=0.55em, after skip=0.55em}",
        r"\newtcolorbox{reviewbox}[1]{enhanced, breakable, colback=emggray, colframe=emgborder, title={#1}, coltitle=black, fonttitle=\sffamily\bfseries}",
        r"\newtcolorbox{theorembox}[1]{enhanced, breakable, colback=white, colframe=black!55, title={#1}, coltitle=black, colbacktitle=black!8, fonttitle=\sffamily\bfseries, fontupper=\large}",
        r"\newtcolorbox{ideabox}[1]{enhanced, breakable, colback=emglightgreen, colframe=emggreen!65!black, title={#1}, coltitle=white, colbacktitle=emggreen!70!black, fonttitle=\sffamily\bfseries}",
        r"\newtcolorbox{parentbox}[1]{enhanced, breakable, colback=emggray, colframe=emgborder, title={#1}, coltitle=black, fonttitle=\sffamily\bfseries}",
        r"\newtcolorbox{parentthmbox}[1]{enhanced, breakable, colback=white, colframe=black!35, title={#1}, coltitle=black, colbacktitle=black!7, fonttitle=\sffamily\bfseries}",
        r"\newtcolorbox{metabox}[1]{enhanced, breakable, colback=black!1, colframe=black!20, title={#1}, coltitle=black, colbacktitle=black!12, fonttitle=\sffamily\bfseries}",
        r"\newenvironment{formalbox}[1]{\VerbatimEnvironment\begin{tcolorbox}[enhanced, breakable, colback=black!2, colframe=black!35, title={#1}, coltitle=black, colbacktitle=black!12, fonttitle=\sffamily\bfseries]\begin{Verbatim}[breaklines=true, breakanywhere=true, fontsize=\footnotesize, formatcom=\leanfont, tabsize=2]}{\end{Verbatim}\end{tcolorbox}}",
        r"\newenvironment{leanbox}[1]{\VerbatimEnvironment\begin{tcolorbox}[enhanced, breakable, colback=black!2, colframe=black!35, title={#1}, coltitle=black, fonttitle=\sffamily\bfseries]\begin{Verbatim}[breaklines=true, breakanywhere=true, fontsize=\footnotesize, formatcom=\leanfont, tabsize=2]}{\end{Verbatim}\end{tcolorbox}}",
        r"\setlist[itemize]{topsep=0.25em, parsep=0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\setlength{\parindent}{0pt}",
        r"\setcounter{tocdepth}{2}",
        r"\begin{document}",
        r"\title{Accepted Problem Gallery}",
        r"\date{}",
        r"\maketitle",
        r"\begin{reviewbox}{Corpus summary}",
        rf"\textbf{{Curated accepted problems.}} {len(rows)}\par",
        r"\textbf{By benchmark.} "
        + ", ".join(rf"\texttt{{{_tex_escape_text(k)}}}={v}" for k, v in sorted(by_benchmark.items()))
        + r"\par",
        r"\textbf{By op type.} "
        + ", ".join(rf"\texttt{{{_tex_escape_text(k)}}}={v}" for k, v in sorted(by_op.items()))
        + r"\par",
        r"\textbf{Canonical artifact.} Lean formal statements and certificates remain the source of truth; this TeX file is a review rendering.",
        r"\end{reviewbox}",
        r"\tableofcontents",
        r"\clearpage",
    ]
    for benchmark in BENCHMARK_ORDER:
        bench_rows = [row for row in sorted_rows if _benchmark(row) == benchmark]
        if not bench_rows:
            continue
        lines.append(r"\clearpage")
        lines.append(rf"\section{{{_tex_escape_text(benchmark)}}}")
        for idx, row in enumerate(bench_rows):
            if idx > 0:
                lines.append(r"\clearpage")
            number = sorted_rows.index(row) + 1
            lines.append(_render_tex_card(number, row, parents_by_id.get(str(row.get("problem_id")), [])))
    other_rows = [row for row in sorted_rows if _benchmark(row) not in BENCHMARK_ORDER]
    if other_rows:
        lines.append(r"\clearpage")
        lines.append(r"\section{other}")
        for idx, row in enumerate(other_rows):
            if idx > 0:
                lines.append(r"\clearpage")
            number = sorted_rows.index(row) + 1
            lines.append(_render_tex_card(number, row, parents_by_id.get(str(row.get("problem_id")), [])))
    lines.append(r"\end{document}")
    return "\n\n".join(lines) + "\n"


def _filter_rows(rows: List[Dict[str, Any]], benchmark: str) -> List[Dict[str, Any]]:
    if benchmark == "all":
        return rows
    return [row for row in rows if _benchmark(row) == benchmark]


def build_gallery(input_path: Path, output_md: Path, output_tex: Path, benchmark: str) -> None:
    rows = _filter_rows(_read_jsonl(input_path), benchmark)
    source_index = _load_source_index(rows)
    csv_index = _load_csv_index()
    parents_by_id = {
        str(row.get("problem_id")): _resolve_parents(row, source_index, csv_index)
        for row in rows
    }
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(rows, parents_by_id), encoding="utf-8")
    output_tex.write_text(_render_tex(rows, parents_by_id), encoding="utf-8")
    print(f"wrote {output_md.relative_to(REPO_ROOT)} ({len(rows)} cards)")
    print(f"wrote {output_tex.relative_to(REPO_ROOT)} ({len(rows)} cards)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument(
        "--benchmark",
        choices=["minif2f", "proofnet", "putnambench", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_gallery(
        input_path=args.input if args.input.is_absolute() else REPO_ROOT / args.input,
        output_md=args.output_md if args.output_md.is_absolute() else REPO_ROOT / args.output_md,
        output_tex=args.output_tex if args.output_tex.is_absolute() else REPO_ROOT / args.output_tex,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
