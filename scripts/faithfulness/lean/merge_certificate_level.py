#!/usr/bin/env python3
"""Fold the audit and comparator verdicts into one `certificate_level` per row.

Three checks run separately and at very different costs — the statement probe is
seconds, the proof replay is a minute, the comparator is many minutes — so they
land in three different files. A reader should not have to join them to answer
"how well is this row actually certified?", and a downstream filter should not
have to encode the ordering itself.

This writes the ladder onto the row, along with the individual flags it was
derived from, so a claim can always be traced back to the check that supports
it:

  none              nothing holds
  statement_checked the statement elaborates under our pin
  proof_checked     …and the proof compiles with a clean axiom closure
  kernel_replayed   …and an independent kernel replayed the exported term
  reproducible      …and that term exported byte-identically on a second platform

The levels are cumulative on purpose: `reproducible` is not a separate opinion
about the row, it is the strongest of the four, and demoting a row for a failed
comparator run has to leave the weaker claims standing.

Usage:
  python scripts/faithfulness/lean/merge_certificate_level.py \
    --rows data/benchmarks/proofnet_verified/raw/seeds_50_rows.jsonl \
    --axiom-audit data/benchmarks/proofnet_verified/raw/seeds_50_axiom_audit.json \
    --replay-cert data/benchmarks/proofnet_verified/raw/seeds_50_replay_cert.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Set

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


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.levels import (  # noqa: E402
    LEVEL_KERNEL,
    LEVEL_NONE,
    LEVEL_PROOF,
    LEVEL_REPRODUCIBLE,
    LEVEL_STATEMENT,
)


def load_axiom_pass(path: Path) -> Set[str]:
    """Names whose axiom closure stayed inside the allowlist."""
    if not path or not path.is_file():
        return set()
    report = json.loads(path.read_text(encoding="utf-8"))
    passing: Set[str] = set()
    for entry in report.get("rows_detail") or report.get("rows_report") or []:
        if entry.get("passed") or (entry.get("audit") or {}).get("passed"):
            passing.add(str(entry.get("problem_id")))
    if passing:
        return passing
    # Older report shape carries only aggregate verdicts. A run with no failures
    # and no unresolved probes passed for every row it covered; anything less
    # specific than that is not enough to grant a level, so we say so instead of
    # guessing per row.
    verdicts = report.get("row_verdicts") or {}
    if set(verdicts) == {"True"} and not report.get("disallowed_axiom_counts"):
        return {"__ALL__"}
    return set()


def load_replay_cert(path: Path) -> Dict[str, bool]:
    if not path or not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    # `replay_certified` is the pre-rename spelling; reports written before the
    # ladder gained its fourth level still use it, and re-running the comparator
    # to change a key would cost hours for nothing.
    return {
        str(r.get("name")): bool(
            r.get("kernel_replayed", r.get("replay_certified", False))
        )
        for r in report.get("results") or []
    }


def load_export_digests(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {str(r.get("name")): r for r in report.get("results") or []}


def level_of(
    row: Dict[str, Any], axiom_ok: bool, replayed: bool, reproduced: bool
) -> str:
    if not row.get("statement_checked"):
        return LEVEL_NONE
    if not (row.get("proof_replays") and axiom_ok):
        return LEVEL_STATEMENT
    if not replayed:
        return LEVEL_PROOF
    # Reproducing a weaker check on more platforms does not make it a stronger
    # check, so the top level requires the kernel replay underneath it.
    return LEVEL_REPRODUCIBLE if reproduced else LEVEL_KERNEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--axiom-audit", type=Path, default=None)
    parser.add_argument("--replay-cert", type=Path, default=None)
    parser.add_argument("--export-digests", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.rows

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    axiom_pass = load_axiom_pass(args.axiom_audit)
    replay = load_replay_cert(args.replay_cert)
    exports = load_export_digests(args.export_digests)
    blanket = "__ALL__" in axiom_pass

    for row in rows:
        name = str(row.get("name"))
        axiom_ok = blanket or name in axiom_pass
        # Absent from the comparator report means not attempted, not refuted.
        replayed = replay.get(name, False)
        export = exports.get(name) or {}
        reproduced = bool(export.get("reproducible"))
        row["axiom_audit_passed"] = axiom_ok
        row["kernel_replayed"] = replayed
        row["replay_cert_attempted"] = name in replay
        row["reproducible"] = reproduced
        row["export_digest"] = export.get("export_digest")
        row["platforms_verified"] = (
            ["macos-aarch64", "linux-aarch64"] if reproduced else []
        )
        row["certificate_level"] = level_of(row, axiom_ok, replayed, reproduced)

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(r["certificate_level"] for r in rows)
    print(f"rows={len(rows)} -> {output}")
    for level in (LEVEL_REPRODUCIBLE, LEVEL_KERNEL, LEVEL_PROOF, LEVEL_STATEMENT, LEVEL_NONE):
        if counts.get(level):
            print(f"  {level:18s} {counts[level]}")
    unattempted = sum(1 for r in rows if not r["replay_cert_attempted"])
    if unattempted:
        print(f"  (comparator not attempted on {unattempted} rows)")


if __name__ == "__main__":
    main()
