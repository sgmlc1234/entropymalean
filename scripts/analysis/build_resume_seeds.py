"""Turn a finished group's deepest certified rows into gen0 seeds for a longer run.

The pipeline has no resume: a run starts from a seed CSV at generation 0 and
stops at `--max-generations`. Continuing a lineage past that means handing the
next run the rows the last one ended on, dressed as seeds.

The rows are already proved, so the receiving run takes `--skip-gen0` and the
proof travels in `verification_code`. The identifier travels unchanged, which
is what keeps the lineage honest: `child_id` reads the operator chain out of
the parent's id and appends to it, so a child of `Rudin_exercise_4_3__mh.me__…`
is `Rudin_exercise_4_3__mh.me.mh__…` and `lineage_depth` keeps counting from
the original benchmark seed rather than restarting at zero.

Seeds are taken deepest-first. A group rarely ends with a full pool at its
final generation -- two or three certified rows is typical -- so the remainder
is filled from the generation below, and so on. Filling upward like this keeps
the run as deep as the group actually got instead of averaging it back down.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

#: The column order the runner's own `*.gen0_seeds.csv` uses. A seed row is read
#: by `row_to_input`, which keeps everything except `id`/`statement`/`answer` as
#: metadata, so extra columns are harmless and missing ones are not.
COLUMNS = [
    "release_id", "id", "statement", "answer", "solution", "verification_code",
    "formal_statement", "lean_header", "formal_status", "operation", "difficulty",
    "difficulty_label", "generation", "source_run", "source_file", "source_slot",
    "parent_ids", "ancestor_ids", "statement_sha256", "answer_sha256",
]


def proof_of(row: dict) -> str:
    """The proof body, wherever this row happens to carry it.

    Certified rows written by the pipeline hold the whole compiled file in
    `lean_code` and leave `verification_code` empty; seeds read the latter. The
    header is stored separately and re-emitted by the receiving run, so passing
    the full file would duplicate the imports.
    """
    for key in ("verification_code", "proof", "lean_code"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        if key == "lean_code":
            header = str(row.get("lean_header") or "").strip()
            if header and value.startswith(header):
                value = value[len(header):].strip()
        return value
    return ""


def seeds_from(path: Path, count: int) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    certified = [r for r in rows if r.get("status") == "certified" and (r.get("formal_statement") or "").strip()]
    # Deepest first; within a generation, the order the run produced them.
    certified.sort(key=lambda r: -(r.get("generation") or 0))
    picked, seen = [], set()
    for row in certified:
        statement = " ".join(str(row.get("formal_statement") or "").split())
        if statement in seen:
            continue
        seen.add(statement)
        picked.append(row)
        if len(picked) >= count:
            break
    return picked


def to_seed(row: dict) -> Dict[str, str]:
    problem_id = str(row.get("problem_id") or "")
    return {
        "release_id": problem_id,
        "id": problem_id,
        "statement": str(row.get("statement") or ""),
        "answer": str(row.get("answer") or ""),
        "solution": "",
        "verification_code": proof_of(row),
        "formal_statement": str(row.get("formal_statement") or ""),
        "lean_header": str(row.get("lean_header") or ""),
        "formal_status": "certified",
        "operation": "seed",
        "difficulty": str(row.get("difficulty") or ""),
        "difficulty_label": str(row.get("difficulty_label") or ""),
        # Reported as 0 because it is generation 0 *of the receiving run*. The
        # real depth is in the identifier, which is where anything downstream
        # reads it from.
        "generation": "0",
        "source_run": str(row.get("run_name") or ""),
        "source_file": "",
        "source_slot": "",
        "parent_ids": json.dumps(row.get("parent_ids") or []),
        "ancestor_ids": json.dumps(row.get("ancestor_ids") or []),
        "statement_sha256": "",
        "answer_sha256": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/certified/run-a"))
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/certified/run-b/seeds"))
    parser.add_argument("--pool-size", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        path = args.source / f"{group}.jsonl"
        if not path.is_file():
            print(f"  {group}: no source run at {path}")
            continue
        picked = seeds_from(path, args.pool_size)
        if len(picked) < args.pool_size:
            print(f"  {group}: only {len(picked)} distinct certified rows, need {args.pool_size} — skipped")
            continue
        target = args.out_dir / f"{group}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in picked:
                writer.writerow(to_seed(row))
        depths = [r.get("generation") for r in picked]
        missing = sum(1 for r in picked if not proof_of(r))
        print(f"  {group}: {len(picked)} seeds from generations {depths}"
              + (f"  — {missing} WITHOUT A PROOF" if missing else ""))


if __name__ == "__main__":
    main()
