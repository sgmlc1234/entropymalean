#!/usr/bin/env python3
"""Sort `Unknown constant` failures into ones a palette can fix and ones it cannot.

Two episodes can fail with the same Lean message for opposite reasons. On
`Artin_exercise_2_11_3` the model reached for a lemma it could not name, and
supplying the name fixed it — 0/3 became 2/3. On `mathd_algebra_208` the
palette handed over `Real.sqrt_eq_iff_eq_sq` and the model wrote
`Real.sqrt_eq_iff_sq_eq`, transposing the tail; it failed 3/3 across every arm,
including the one that gave it the finished proof. Lean reports both as
`Unknown constant`.

The distinction decides what an environment can be expected to do. A palette
supplies knowledge, so it repairs the first kind and cannot touch the second —
and averaging them together makes the palette look unreliable when it is simply
being asked to solve a problem it is not addressed to.

The classifier is edit distance against what the palette offered:

  not_in_palette   the name is nothing like anything we supplied → knowledge gap
  misquoted        close to a supplied name → transcription, not knowledge
  no_palette       nothing was supplied, so neither claim is available

`misquoted` is deliberately narrow. A name within a couple of edits of one we
handed over is a copy that went wrong; anything further is more likely a
different lemma the model invented, and calling that a transcription error
would inflate the category this script exists to isolate.

Usage:
  python scripts/analysis/classify_lean_failures.py \
    --episodes 'data/evaluation/exam/*/episodes_*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_UNKNOWN = re.compile(r"[Uu]nknown (?:constant|identifier) `([^`]+)`")

#: Distance at which "the same name, mistyped" stops being the better reading.
#: Two edits covers a transposed pair or a dropped underscore; beyond that the
#: strings are usually different lemmas, not the same one written badly.
#:
#: An absolute threshold is wrong at both ends of the length range. `sr` against
#: `ZMod` is 4 edits on strings of length 2 and 4 — nothing alike, yet close to
#: the cutoff — while `Nat.card_eq_zero_iff.mp` was filed as a mistyping of
#: `Nat.eq_zero_or_pos`. So the absolute bound is kept only as a floor for short
#: names, and the real test is the edit distance as a *fraction* of the name.
MISQUOTE_MAX_EDITS = 2
#: A name is a mistyping when the edits are a small share of its length. 0.15
#: lets a 20-character lemma absorb three, and stops `sr`/`ZMod` from
#: qualifying on any number.
MISQUOTE_MAX_RATIO = 0.15

#: Segments a model appends or prepends when it is *inventing* a plausible name
#: rather than mistyping a real one: `exists_prime_orderOf_dvd_card_of_dvd` for
#: the offered `exists_prime_orderOf_dvd_card`. Mathlib's naming is regular
#: enough to extrapolate from, which is exactly why the model does it. This is
#: a third failure, distinct from both a typo and plain ignorance, and it was
#: being counted as ignorance — the category that a palette is supposed to fix.
_AFFIX = re.compile(r"^(.*?)((?:_(?:of|iff|left|right|self|mp|mpr|le|lt|eq|"
                    r"ne|zero|one|succ|pred|add|sub|mul|div|pow|neg|inv|"
                    r"cast|coe|dvd|card|mem|not|comm|assoc|symm|trans))+)$")


def edits(a: str, b: str) -> int:
    """Levenshtein distance, computed with a rolling row."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _mistyped(a: str, b: str) -> bool:
    """Whether `a` reads as `b` written badly, rather than as another name."""
    if not a or not b:
        return False
    distance = edits(a, b)
    if distance == 0:
        return True
    # Both bounds have to hold: the ratio alone would let two short strings
    # match on a single edit, and the absolute alone is what mistook `sr` for
    # `ZMod`.
    return distance <= MISQUOTE_MAX_EDITS and distance <= len(b) * MISQUOTE_MAX_RATIO


def _invented_from(name: str, offered: str) -> bool:
    """Whether `name` is `offered` with Mathlib-shaped decoration added.

    `exists_prime_orderOf_dvd_card_of_dvd` is not a typo of
    `exists_prime_orderOf_dvd_card` and it is not ignorance of it either — the
    model had the stem and extended it into a lemma that does not exist.
    """
    if name == offered or len(name) <= len(offered):
        return False
    if not name.startswith(offered):
        return False
    return bool(_AFFIX.match(name[len(offered):]))


def classify(name: str, palette: Sequence[str]) -> str:
    if not palette:
        return "no_palette"
    tail = name.split(".")[-1]
    for offered in palette:
        # Compare on the full name and on the final component: a model that
        # keeps the namespace but garbles the lemma is still miscopying.
        if _mistyped(name, offered) or _mistyped(tail, offered.split(".")[-1]):
            return "misquoted"
    for offered in palette:
        if _invented_from(name, offered) or _invented_from(tail, offered.split(".")[-1]):
            return "invented"
    return "not_in_palette"


def palette_for(row: Dict[str, Any], seeds: Dict[str, Any]) -> List[str]:
    seed = seeds.get(str(row.get("seed"))) or {}
    return sorted((seed.get("palette") or {}).get("theorems") or {})


def messages_of(row: Dict[str, Any]) -> List[str]:
    """Lean's complaints, wherever the player happened to record them."""
    out = []
    for attempt in row.get("attempt_log") or []:
        if attempt.get("message"):
            out.append(str(attempt["message"]))
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("message"):
            out.append(str(step["message"]))
    # The tactic player records its refusals separately: it has no attempts,
    # only candidates Lean declined one at a time.
    for rejection in row.get("rejections") or []:
        if rejection.get("message"):
            out.append(str(rejection["message"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="glob over episode JSONL")
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[
            "data/benchmarks/proofnet_verified/raw/seeds_50_rows.jsonl",
            "data/benchmarks/minif2f_v2/raw/exam_rows_v2.jsonl",
        ],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    seeds: Dict[str, Any] = {}
    for path in args.seeds:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                seeds[str(row.get("name"))] = row

    by_run: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for path in sorted(glob.glob(args.episodes)):
        run = Path(path).stem.replace("episodes_", "")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("outcome") == "solved":
                continue
            palette = palette_for(row, seeds)
            for message in messages_of(row):
                for name in _UNKNOWN.findall(message):
                    kind = classify(name, palette)
                    by_run[run][kind] += 1
                    if len(examples[kind]) < 8:
                        closest = min(
                            palette, key=lambda o: edits(name, o), default=""
                        )
                        examples[kind].append(
                            {"run": run, "seed": row.get("seed"),
                             "wrote": name, "offered": closest}
                        )

    report = {
        "by_run": {k: dict(v.most_common()) for k, v in sorted(by_run.items())},
        "totals": dict(sum(by_run.values(), Counter()).most_common()),
        "examples": examples,
    }
    out = args.output or Path("data/evaluation/exam/failure_kinds.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"},
                     ensure_ascii=False, indent=2))
    for kind, rows in examples.items():
        print(f"\n[{kind}]")
        for e in rows[:4]:
            print(f"  {e['run'][:24]:24s} {str(e['seed'])[:26]:26s} "
                  f"wrote {e['wrote']!r} vs offered {e['offered']!r}")


if __name__ == "__main__":
    main()
