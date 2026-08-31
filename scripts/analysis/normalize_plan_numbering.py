#!/usr/bin/env python3
"""Strip the numbering a plan step wrote for itself, in every arm at once.

The plan generator asks for a JSON list and most replies come back as a list of
already-numbered sentences — `"1. First analyze..."`. The players number the
list again when they render it, so the model reads `1. 1. First analyze...`
while an unnumbered neighbour reads `1. The argument begins...`. Two rows in
the same arm therefore get the aid in two different shapes.

It is not an arm asymmetry — the control plans carry the prefix on 241 of 319
steps — which is exactly why the repair has to be applied to every plan file in
one pass. Normalising the treatment file alone would create the asymmetry that
does not currently exist, and it would land on the difference the experiment
measures.

Only a leading ordinal is removed. `2.5` inside a sentence, a step that opens
with a year, and a bare `(1)` citation are left alone: the pattern requires the
number to be followed by a period or parenthesis and then whitespace, at the
very start of the step.

Read-only unless `--write` is passed.

Usage:
  python3 scripts/analysis/normalize_plan_numbering.py \
    data/evaluation/exam/machine_plans_all.json \
    data/evaluation/exam/machine_plans_treatment114.json --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: A step's own ordinal: digits, a period or paren, then whitespace, anchored at
#: the start. `"1. First"` matches; `"2.5 times the radius"` does not, because
#: no whitespace follows the period.
ORDINAL = re.compile(r"^\s*\d{1,2}\s*[.)]\s+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="apply; otherwise report only")
    args = parser.parse_args()

    total_steps = total_stripped = 0
    for path in args.files:
        plans = json.loads(path.read_text(encoding="utf-8"))
        steps = stripped = 0
        for entry in plans.values():
            new = []
            for step in entry.get("plan") or []:
                steps += 1
                fixed = ORDINAL.sub("", str(step), count=1).strip()
                if fixed != str(step).strip():
                    stripped += 1
                # A step that was *only* an ordinal would empty out; keep the
                # original rather than hand the model a blank line.
                new.append(fixed or str(step))
            entry["plan"] = new
        total_steps += steps
        total_stripped += stripped
        print(f"{path.name:38s} {stripped:4d}/{steps:4d} steps carried their own number")
        if args.write:
            path.write_text(json.dumps(plans, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    verb = "stripped" if args.write else "would strip"
    print(f"\n{verb} {total_stripped}/{total_steps} steps across {len(args.files)} files"
          + ("" if args.write else "  (dry run — pass --write to apply)"))


if __name__ == "__main__":
    main()
