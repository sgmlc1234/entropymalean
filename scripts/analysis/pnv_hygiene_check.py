#!/usr/bin/env python3
"""EML ProofNet parent hygiene check against ProofNet-Verified.

ProofNet-Verified audited every ProofNet formalization (q2 faithfulness to
the informal statement, q3 provability, q4 error type) and corrected the
errors. Our EML ProofNet children were bred from the UNaudited lineage, so
any child whose seed ancestry touches a flagged formalization inherits that
risk. This script maps every released EML ProofNet row to its seed
exercises, joins the PNV audit verdicts, and reports flagged ancestry.

Outputs:
  data/evaluation/pnv_hygiene/report.json   (machine-readable)
  docs/eml_proofnet_hygiene_report.md       (human summary)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PNV_ROOT = REPO_ROOT / "references" / "ProofNet-Verified"
RELEASE = REPO_ROOT / "release" / "huggingface" / "EML-1" / "accepted.jsonl"
OUT_JSON = REPO_ROOT / "data" / "evaluation" / "pnv_hygiene" / "report.json"
OUT_MD = REPO_ROOT / "docs" / "eml_proofnet_hygiene_report.md"

_SEED_RE = re.compile(r"exercise_\d+(?:_\d+)*[a-z]{0,2}")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*|[^\sA-Za-z0-9]")

FLAGGED_FAITHFULNESS = {"stronger", "weaker", "incomparable", "nl_ambiguous", "nl_wrong"}


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text or "")))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _norm_stmt(text: str) -> str:
    text = re.sub(r"--.*?$", "", str(text or ""), flags=re.M)
    text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    return re.sub(r"\s+", "", text.split(":=")[0])


def _suffix(name: str) -> str:
    name = str(name)
    if "|" in name:
        name = name.split("|")[-1]
    index = name.find("exercise")
    return name[index:] if index >= 0 else name


def load_source_statements():
    """Normalized statements of both upstream lineages, keyed by seed suffix.

    Our local benchmark turned out to be a MIXED port: most rows follow
    ProofNet# but several keep the pre-correction original. The audit verdict
    that applies to a seed is the one for the lineage its statement actually
    matches, so both sources are loaded and matched per seed.
    """
    original: Dict[str, str] = {}
    sharp: Dict[str, str] = {}
    orig_path = (
        PNV_ROOT / "data" / "proofnet" / "proofnet_lean4-6deae98.jsonl"
    )
    if orig_path.is_file():
        for line in orig_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                original.setdefault(
                    _suffix(row["name"]), _norm_stmt(row.get("formal_statement"))
                )
    sharp_path = PNV_ROOT / "data" / "proofnet#" / "proofnet#.jsonl"
    if sharp_path.is_file():
        for line in sharp_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                sharp.setdefault(
                    _suffix(row["id"]), _norm_stmt(row.get("lean4_formalization"))
                )
    return original, sharp


def load_local_statements() -> Dict[str, List[str]]:
    """Normalized statements of our local ProofNet benchmark, by seed name."""
    local: Dict[str, List[str]] = {}
    data_dir = REPO_ROOT / "data" / "benchmarks" / "proofnet" / "data"
    try:
        import pandas as pd
    except ImportError:
        return local
    for split in ("valid", "test"):
        path = data_dir / f"{split}-00000-of-00001.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            local.setdefault(str(row["name"]), []).append(
                _norm_stmt(row.get("formal_statement"))
            )
    return local


def resolve_lineage(seed: str, local, original, sharp) -> str:
    """Which upstream lineage our local statement for ``seed`` equals."""
    mine = local.get(seed) or []
    if not mine:
        return "unknown"
    in_sharp = seed in sharp and sharp[seed] in mine
    in_orig = seed in original and original[seed] in mine
    if in_sharp and in_orig:
        return "identical"
    if in_sharp:
        return "proofnet_sharp"
    if in_orig:
        return "proofnet_original"
    return "unmatched"


def load_pnv():
    rows = [
        json.loads(line)
        for line in (PNV_ROOT / "data" / "proofnet-verified.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    by_stem = {}
    for row in rows:
        by_stem[f"proofnet-{row['index']}"] = row
    taxonomy = {"proofnet_sharp": {}, "proofnet_original": {}}
    for key, folder in (
        ("proofnet_sharp", "proofnet#"),
        ("proofnet_original", "proofnet"),
    ):
        tax_path = PNV_ROOT / "error_taxonomy" / folder / "results.jsonl"
        for line in tax_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                taxonomy[key][record["stem"]] = record
    # suffix (our naming) -> list of PNV rows
    by_suffix = {}
    for stem, row in by_stem.items():
        name = str(row["name"])
        # Textbook prefixes may be multi-word (Dummit_Foote_…); our seed
        # names start at "exercise_".
        marker = name.find("exercise")
        suffix = name[marker:] if marker >= 0 else name
        by_suffix.setdefault(suffix, []).append((stem, row))
    return by_stem, by_suffix, taxonomy


def seed_names_for_row(row: dict) -> set:
    haystacks = [str((row.get("provenance") or {}).get("source_problem_id") or "")]
    haystacks.append(str(row.get("problem_id") or ""))
    for parent in row.get("parents") or []:
        haystacks.append(str(parent.get("parent_id") or ""))
    return set(
        seed for haystack in haystacks for seed in _SEED_RE.findall(haystack)
    )


def resolve_seed(seed: str, by_suffix, our_statement: str):
    """Pick the PNV entry for a seed name, using statement overlap to break
    textbook ambiguity."""
    candidates = by_suffix.get(seed) or []
    if not candidates:
        return None, "unmatched", []
    if len(candidates) == 1:
        return candidates[0], "unique", [candidates[0][1]["name"]]
    scored = sorted(
        candidates,
        key=lambda item: _similarity(our_statement, item[1]["formal_stmt"]),
        reverse=True,
    )
    return scored[0], "ambiguous_resolved_by_statement", [
        candidate[1]["name"] for candidate in scored
    ]


def main() -> None:
    by_stem, by_suffix, taxonomy = load_pnv()
    original_src, sharp_src = load_source_statements()
    local_src = load_local_statements()
    released = [
        json.loads(line)
        for line in RELEASE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proofnet_rows = [row for row in released if row.get("benchmark") == "proofnet"]

    seed_verdicts = {}
    row_reports = []
    for row in proofnet_rows:
        seeds = sorted(seed_names_for_row(row))
        seed_entries = []
        row_flagged = False
        for seed in seeds:
            if seed not in seed_verdicts:
                match, how, candidates = resolve_seed(
                    seed, by_suffix, str(row.get("formal_statement") or "")
                )
                if match is None:
                    seed_verdicts[seed] = {
                        "seed": seed,
                        "pnv_name": None,
                        "match": how,
                        "verdict": "unmatched",
                        "candidates": candidates,
                    }
                else:
                    stem, pnv_row = match
                    lineage = resolve_lineage(
                        seed, local_src, original_src, sharp_src
                    )
                    # A seed whose local statement we could not match to either
                    # upstream is scored against BOTH taxonomies and takes the
                    # stricter verdict, so an unresolved lineage never silently
                    # clears a row.
                    if lineage in {"proofnet_sharp", "proofnet_original"}:
                        audit = taxonomy[lineage].get(stem) or {}
                    else:
                        candidates_audit = [
                            taxonomy[key].get(stem) or {}
                            for key in ("proofnet_original", "proofnet_sharp")
                        ]
                        audit = max(
                            candidates_audit,
                            key=lambda rec: (
                                rec.get("q3_provability") == "false",
                                rec.get("q2_faithfulness") in FLAGGED_FAITHFULNESS,
                            ),
                        )
                    faithfulness = str(audit.get("q2_faithfulness") or "unknown")
                    provable = str(audit.get("q3_provability") or "unknown")
                    flagged = (
                        faithfulness in FLAGGED_FAITHFULNESS or provable == "false"
                    )
                    seed_verdicts[seed] = {
                        "seed": seed,
                        "pnv_name": pnv_row["name"],
                        "pnv_stem": stem,
                        "match": how,
                        "candidates": candidates,
                        "lineage": lineage,
                        "faithfulness": faithfulness,
                        "provable": provable,
                        "error_type": audit.get("q4_error_type"),
                        "reasoning": str(audit.get("reasoning") or "")[:400],
                        "verdict": "flagged" if flagged else "clean",
                    }
            entry = seed_verdicts[seed]
            seed_entries.append(entry)
            if entry["verdict"] == "flagged":
                row_flagged = True
        row_reports.append(
            {
                "problem_id": row.get("problem_id"),
                "seeds": seeds,
                "flagged": row_flagged,
                "flagged_seeds": [
                    entry["seed"]
                    for entry in seed_entries
                    if entry["verdict"] == "flagged"
                ],
                "unmatched_seeds": [
                    entry["seed"]
                    for entry in seed_entries
                    if entry["verdict"] == "unmatched"
                ],
            }
        )

    flagged_rows = [row for row in row_reports if row["flagged"]]
    unmatched_rows = [
        row
        for row in row_reports
        if row["unmatched_seeds"] and not row["flagged"]
    ]
    verdict_counts = Counter(entry["verdict"] for entry in seed_verdicts.values())

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "released_proofnet_rows": len(proofnet_rows),
                "distinct_seeds": len(seed_verdicts),
                "seed_verdict_counts": dict(verdict_counts),
                "lineage_counts": dict(
                    Counter(
                        str(entry.get("lineage")) for entry in seed_verdicts.values()
                    )
                ),
                "flagged_row_count": len(flagged_rows),
                "seed_verdicts": sorted(
                    seed_verdicts.values(), key=lambda entry: entry["seed"]
                ),
                "rows": row_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EML ProofNet 부모 위생 점검 (ProofNet-Verified 감사 대조)",
        "",
        f"- 릴리스된 ProofNet 행: **{len(proofnet_rows)}**",
        f"- 고유 시드 조상: **{len(seed_verdicts)}** "
        f"(clean {verdict_counts.get('clean', 0)} / flagged {verdict_counts.get('flagged', 0)} "
        f"/ unmatched {verdict_counts.get('unmatched', 0)})",
        f"- **조상에 flagged 시드를 가진 릴리스 행: {len(flagged_rows)}**",
        "",
        "PNV 감사 기준: `q2_faithfulness ∈ {stronger, weaker, incomparable, "
        "nl_ambiguous, nl_wrong}` 또는 `q3_provability == false` 인 formalization을 "
        "flagged로 판정 (ProofNet# 계보 기준).",
        "",
        "## 시드별 판정",
        "",
        "| seed | PNV 매칭 | 로컬 계보 | faithfulness | provable | error_type | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in sorted(seed_verdicts.values(), key=lambda e: e["seed"]):
        lines.append(
            f"| {entry['seed']} | {entry.get('pnv_name') or '—'} "
            f"| {entry.get('lineage', '—')} "
            f"| {entry.get('faithfulness', '—')} | {entry.get('provable', '—')} "
            f"| {entry.get('error_type', '—')} | **{entry['verdict']}** |"
        )
    if flagged_rows:
        lines += ["", "## Flagged 행 (조상에 오류 formalization 포함)", ""]
        for row in flagged_rows:
            lines.append(
                f"- `{row['problem_id']}` ← seeds: {', '.join(row['flagged_seeds'])}"
            )
        lines += [
            "",
            "### Flagged 시드의 PNV 근거",
            "",
        ]
        for entry in sorted(seed_verdicts.values(), key=lambda e: e["seed"]):
            if entry["verdict"] == "flagged":
                lines.append(
                    f"- **{entry['seed']}** ({entry['pnv_name']}): "
                    f"{entry.get('reasoning', '')}"
                )
    if unmatched_rows:
        lines += ["", f"## Unmatched-only 행: {len(unmatched_rows)}건 (수동 확인 필요)"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"rows={len(proofnet_rows)} seeds={len(seed_verdicts)} "
        f"verdicts={dict(verdict_counts)} flagged_rows={len(flagged_rows)}"
    )
    print(f"report: {OUT_MD}")


if __name__ == "__main__":
    main()
