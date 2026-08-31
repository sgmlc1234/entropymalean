"""Materialise a group plan as seed CSVs, carrying each seed's row over intact.

`plan_seed_groups` decides which seeds share a mating pool; it does not touch
what a seed is. The rows here are the originals -- same statement, same proof,
same header -- rearranged into different files. Nothing is regenerated, so a
run against these seeds differs from the last one only in who could be crossed
with whom.

The receiving run still needs its gen0 proofs, and they are already in the row,
so this preserves `verification_code` and `formal_status` exactly as the
previous campaign's seed files carried them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


def load_rows(pattern: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    fields: List[str] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or fields
            for row in reader:
                if row.get("id"):
                    rows[row["id"]] = row
    return rows, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("data/certified/run-b/seed_plan.json"))
    parser.add_argument("--source", default="data/certified/run-a/seeds/proofnet_g*.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/certified/run-b/seeds"))
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))["groups"]
    rows, fields = load_rows(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    placed = set()
    for group, members in plan.items():
        missing = [m for m in members if m not in rows]
        if missing:
            print(f"  {group}: no source row for {', '.join(missing)} — skipped")
            continue
        target = args.out_dir / f"{group}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name in members:
                writer.writerow(rows[name])
                placed.add(name)
        proved = sum(1 for m in members if str(rows[m].get("verification_code") or "").strip())
        print(f"  {group}: {len(members)} seeds, {proved} carrying a proof")

    print(f"\n{len(placed)} of {len(rows)} source seeds placed")
    for name in sorted(set(rows) - placed):
        print(f"  unplaced: {name}")


if __name__ == "__main__":
    main()
