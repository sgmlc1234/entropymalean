#!/usr/bin/env python3
"""Freeze the episode-level evidence behind the paper's panel tables.

The raw episode logs live under `data/evaluation/exam/`, which `.gitignore`
excludes wholesale through `*.jsonl`. That is right for the working tree --- it
holds ablations, aborted runs and pre-replay backups --- and wrong for the
twenty-six cells the paper actually reports, which a reviewer has to be able to
read. This script copies exactly those, gzipped, into a tracked directory, and
records for each one the SHA-256 of the bytes it copied, the episode count, and
the Pass@3 it yields per benchmark.

The manifest is the audit trail: the numbers in it are recomputed from the
bundled bytes, so a reviewer can check the paper's Table against the manifest
and the manifest against the files without running a model.

    python3 scripts/evaluate/bundle_exam_evidence.py           # write bundle
    python3 scripts/evaluate/bundle_exam_evidence.py --verify  # check, write nothing
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/exam_cells.json"
OUT = ROOT / "data/evaluation/exam_evidence"

# The panel as the paper reports it. A model absent here was withdrawn and its
# cells are deliberately not bundled; see the `_withdrawn` block in the config.
REPORTED = [
    "bfs", "goedel", "pythagoras", "leanstral",
    "qwen3_14b", "nemotron_nano_9b", "muse", "nemotron", "qwen36", "gptoss",
    "luna", "muse_spark", "gemini_flash",
]
EXPECTED = {"controls": 300, "treatments": 1605}


def primary_file(cell_dir: str) -> Path:
    """The one file a cell is scored from.

    `finalize_panel_numbers.py` globs `episodes_*.jsonl` and drops
    `before-replay` backups; anything left is merged. If that ever leaves more
    than one file the scoring silently sums two runs into one cell, so this
    refuses rather than guesses.
    """
    files = [f for f in glob.glob(f"{ROOT / cell_dir}/episodes_*.jsonl")
             if "before-replay" not in f]
    if len(files) != 1:
        raise SystemExit(f"{cell_dir}: expected one scored file, found {len(files)}: {files}")
    return Path(files[0])


# An attempt counted as measured is one Lean actually judged. `no_code`,
# `error` and `over_budget` are model or budget outcomes, not lost episodes;
# the tactic-step player records `steps` instead of `attempt_log` and is
# therefore exempt from the check below.
JUDGED = {"rejected", "accepted", "solved", "sketch_only"}


def _compromised(row: dict) -> bool:
    """True when transport failed and Lean never judged a single attempt.

    Such an episode is scored as a failure the model never got to earn. An
    episode whose generator hiccuped on one attempt but reached the verifier on
    another is measured and is not counted here.
    """
    if not (row.get("generator_empty") or row.get("generator_error")):
        return False
    log = row.get("attempt_log")
    if log is None:
        return False
    return not any(a.get("status") in JUDGED for a in log)


def summarise(raw: bytes) -> dict:
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    by_seed = collections.defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    pass3, counts, lost = {}, collections.Counter(), 0
    for episodes in by_seed.values():
        bench = episodes[0].get("benchmark")
        counts[bench] += 1
        pass3.setdefault(bench, []).append(any(e.get("success") for e in episodes))
        if all(_compromised(e) for e in episodes):
            lost += 1
    return {
        "episodes": len(rows),
        "rows": sum(counts.values()),
        "compromised_episodes": sum(1 for r in rows if _compromised(r)),
        "rows_with_no_measured_episode": lost,
        "pass3": {b: round(100.0 * sum(v) / len(v), 1) for b, v in sorted(pass3.items())},
        "rows_per_benchmark": dict(sorted(counts.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="recompute against an existing bundle and write nothing")
    args = ap.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest, problems = {}, []

    for arm in ("controls", "treatments"):
        for model in REPORTED:
            cell = config[arm].get(model)
            if cell is None:
                problems.append(f"{model}/{arm}: no cell in {CONFIG.name}")
                continue
            src = primary_file(cell)
            raw = src.read_bytes()
            entry = summarise(raw)
            entry.update(source=str(src.relative_to(ROOT)),
                         sha256=hashlib.sha256(raw).hexdigest())
            if entry["episodes"] != EXPECTED[arm]:
                problems.append(f"{model}/{arm}: {entry['episodes']} episodes, expected {EXPECTED[arm]}")
            if entry["rows_with_no_measured_episode"]:
                problems.append(f"{model}/{arm}: "
                                f"{entry['rows_with_no_measured_episode']} rows with no measured episode")

            name = f"{model}_{arm[:-1]}.jsonl.gz"
            entry["bundled"] = name
            dest = OUT / name
            if args.verify:
                if not dest.exists():
                    problems.append(f"{name}: missing from bundle")
                elif hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest() != entry["sha256"]:
                    problems.append(f"{name}: bundled bytes differ from {src}")
            else:
                OUT.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(gzip.compress(raw, 6))
            manifest[f"{model}/{arm[:-1]}"] = entry
            print(f"  {model:18s}{arm[:-1]:10s}{entry['episodes']:5d} ep  "
                  f"{' '.join(f'{b.split('_')[0]}={p}' for b, p in entry['pass3'].items())}")

    if not args.verify:
        (OUT / "MANIFEST.json").write_text(
            json.dumps({"cells": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        total = sum((OUT / e["bundled"]).stat().st_size for e in manifest.values())
        print(f"\nwrote {len(manifest)} cells to {OUT.relative_to(ROOT)} ({total / 1e6:.1f} MB)")

    if problems:
        print("\nproblems:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)
    print("\nall reported cells present, complete, and fully measured.")


if __name__ == "__main__":
    main()
