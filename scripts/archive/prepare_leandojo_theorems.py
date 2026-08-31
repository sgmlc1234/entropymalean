#!/usr/bin/env python3
"""Map our control / treatment problem IDs to LeanDojo Theorem records.

Once a benchmark repo has been traced by LeanDojo
(``trace(LeanGitRepo(url, commit))``), every theorem in it is reachable
via ``TracedRepo.get_traced_theorems()``. This script looks up our
problem IDs in that index and writes a small JSONL of records like::

    {"problem_id": "mathd_numbertheory_188",
     "arm":        "control",
     "file_path":  "MiniF2F/Test.lean",
     "full_name":  "mathd_numbertheory_188"}

which ``scripts/archive/run_leandojo_eval.py`` consumes directly. For benchmarks
that are not yet traced (or for our EML-1 treatment theorems which live
in a separate small project we trace ourselves), call this script once
per (benchmark, arm) pair.

Usage:
    python scripts/archive/prepare_leandojo_theorems.py \
        --repo-url https://github.com/yangky11/miniF2F-lean4 \
        --repo-commit 5746b7d6c47855ce1294bed87329618ff7f1bc31 \
        --benchmark miniF2F \
        --arm control \
        --control-csv /tmp/eml_campaign/minif2f_control.csv \
        --output /tmp/eml_campaign/minif2f_control_leandojo.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from lean_dojo import LeanGitRepo, get_traced_repo_path


def _problem_ids_from_csv(csv_path: Path) -> List[str]:
    ids: List[str] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            pid = (row.get("id") or row.get("problem_id") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def _problem_ids_from_jsonl(jsonl_path: Path) -> List[str]:
    ids: List[str] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pid = (record.get("problem_id") or record.get("id") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def _index_traced_theorems(repo: LeanGitRepo) -> Dict[str, Tuple[str, str]]:
    """Return ``{full_name: (file_path, full_name)}`` for every theorem
    in the traced repo. Cache hit only — does not trigger a re-trace.
    """
    path = get_traced_repo_path(repo)
    # Import here so the module loads quickly even when the trace cache
    # is absent (e.g. in tests).
    from lean_dojo import TracedRepo

    traced = TracedRepo.load_from_disk(path)
    out: Dict[str, Tuple[str, str]] = {}
    for thm in traced.get_traced_theorems():
        out[thm.theorem.full_name] = (
            str(thm.theorem.file_path),
            thm.theorem.full_name,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-url", required=True,
                   help="Git URL of the traced benchmark repo")
    p.add_argument("--repo-commit", required=True,
                   help="Commit hash of the traced revision")
    p.add_argument("--benchmark", required=True,
                   choices=["miniF2F", "proofnet", "putnambench"])
    p.add_argument("--arm", required=True, choices=["control", "treatment"])
    p.add_argument("--control-csv", type=Path, default=None,
                   help="CSV with control problem IDs (column 'id' or 'problem_id')")
    p.add_argument("--treatment-jsonl", type=Path, default=None,
                   help="JSONL with treatment problem IDs")
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL of mapped Theorem records")
    args = p.parse_args()

    if args.arm == "control":
        if not args.control_csv:
            raise SystemExit("--control-csv required when --arm control")
        problem_ids = _problem_ids_from_csv(args.control_csv)
    else:
        if not args.treatment_jsonl:
            raise SystemExit("--treatment-jsonl required when --arm treatment")
        problem_ids = _problem_ids_from_jsonl(args.treatment_jsonl)

    repo = LeanGitRepo(args.repo_url, args.repo_commit)
    index = _index_traced_theorems(repo)
    print(f"traced repo contains {len(index)} theorems")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    missing: List[str] = []
    with args.output.open("w") as out:
        for pid in problem_ids:
            hit = index.get(pid)
            if hit is None:
                missing.append(pid)
                continue
            file_path, full_name = hit
            record = {
                "problem_id": pid,
                "benchmark": args.benchmark,
                "arm": args.arm,
                "repo_url": args.repo_url,
                "repo_commit": args.repo_commit,
                "file_path": file_path,
                "full_name": full_name,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            found += 1

    print(f"matched {found}/{len(problem_ids)} → {args.output}")
    if missing:
        print(f"missing {len(missing)}:")
        for pid in missing:
            print(f"  {pid}")


if __name__ == "__main__":
    main()
