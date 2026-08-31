#!/usr/bin/env python3
"""Build per-benchmark campaign inputs from the pilot combo CSVs and
the curated treatment JSONL.

Usage:
    python scripts/generate/prepare_campaign_inputs.py --out-dir /tmp/eml_campaign

The script writes four main-result files under ``--out-dir``:

    {minif2f,proofnet}_control.csv   (concatenation of the pilot combos)
    {minif2f,proofnet}_treatment.jsonl (filtered slice of the accepted ledger)

Calling it is idempotent — re-running overwrites the outputs in place.
The launcher script ``scripts/archive/run_eml_campaign.sh`` invokes this first
before kicking off ``run_proof_evaluation.py`` per benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO = _repo_root()
BENCHMARKS = ("minif2f", "proofnet")
PILOT_COMBOS = ("", "2", "3", "4")
ACCEPTED_JSONL = REPO / "data" / "evaluation" / "treatment_inventory" / "final_curated" / "accepted.jsonl"


def concat_control(benchmark: str, out_csv: Path) -> int:
    """Concatenate the four pilot combos for ``benchmark`` into a single CSV.

    Preserves the column order of the first combo and uses ``csv.DictReader``
    + ``csv.DictWriter`` so multi-line cells (Lean ``verification_code`` etc.)
    survive the round-trip. Duplicates are not expected because each combo
    seeds a disjoint slot range.
    """
    inputs = [
        REPO / "data" / "raw" / f"{benchmark}_pilot{suffix}_5.csv"
        for suffix in PILOT_COMBOS
    ]
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        raise SystemExit(f"no pilot combos found for {benchmark}")
    with inputs[0].open() as f:
        fieldnames = next(csv.reader(f))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set = set()
    rows_written = 0
    with out_csv.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path in inputs:
            with path.open() as f:
                for row in csv.DictReader(f):
                    pid = row.get("id") or row.get("problem_id") or ""
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def _eval_problem_id(record: dict, duplicate_ids: set[str]) -> str:
    """Return the row id used by the evaluator.

    The accepted ledger can contain multiple certified theorem surfaces with
    the same lineage ``problem_id``. The proof evaluator resumes and dedups by
    problem id, so duplicate lineage ids need a stable row-level suffix in the
    per-campaign input.
    """
    problem_id = str(record.get("problem_id") or record.get("id") or "")
    if problem_id not in duplicate_ids:
        return problem_id
    hashes = record.get("hashes") if isinstance(record.get("hashes"), dict) else {}
    digest = (
        hashes.get("statement_sha256")
        or hashes.get("formal_statement_sha256")
        or ""
    )
    suffix = str(digest)[:10]
    if not suffix:
        import hashlib

        surface = "\n".join(
            str(record.get(key) or "")
            for key in ("statement", "formal_statement", "lean_code", "theorem_name")
        )
        suffix = hashlib.sha256(surface.encode("utf-8")).hexdigest()[:10]
    return f"{problem_id}__eval_{suffix}"


def split_treatment(benchmark: str, accepted_jsonl: Path, out_jsonl: Path) -> int:
    """Slice rows of the curated accepted ledger whose benchmark matches."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    id_counts: dict[str, int] = {}
    for line in accepted_jsonl.open():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("benchmark") == benchmark:
            records.append(record)
            problem_id = str(record.get("problem_id") or record.get("id") or "")
            id_counts[problem_id] = id_counts.get(problem_id, 0) + 1
    duplicate_ids = {pid for pid, count in id_counts.items() if count > 1}
    with out_jsonl.open("w") as out:
        for record in records:
            record = dict(record)
            record["eval_problem_id"] = _eval_problem_id(record, duplicate_ids)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/eml_campaign"),
        help="directory to write the per-benchmark control + treatment inputs",
    )
    p.add_argument(
        "--accepted-jsonl",
        type=Path,
        default=ACCEPTED_JSONL,
        help="accepted treatment ledger to slice",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for bench in BENCHMARKS:
        control = args.out_dir / f"{bench}_control.csv"
        treatment = args.out_dir / f"{bench}_treatment.jsonl"
        n_ctrl = concat_control(bench, control)
        n_trt = split_treatment(bench, args.accepted_jsonl, treatment)
        print(f"{bench:14} control={n_ctrl:3} -> {control}")
        print(f"{bench:14} treatment={n_trt:3} -> {treatment}")


if __name__ == "__main__":
    main()
