"""Does topic fit predict whether a crossover works?

The first attempt at this used the textbook a seed came from as its topic, and
found nothing: crossovers within one book were rejected as `parallel` 90% of
the time and across books 90% of the time. That proxy is too coarse to carry
the question -- Dummit & Foote alone spans groups, rings and fields, so two
seeds can share a book and share no mathematics.

This reads the topic off the Lean instead. A statement's signature is the set
of names it mentions: the typeclasses it demands, the Mathlib constants it
applies, the operators it relates them with. Two statements fit when they talk
about the same objects, whatever book they were printed in.

The test is the one the grouping decision needs: over every crossover this
corpus produced, does the overlap between its two parents' signatures separate
the ones that certified from the ones the judge called `parallel`? A grouping
rule is only worth writing if it does.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Set

from src.orchestration.problem_ids import roots_of

#: Lean identifiers, of any case. Mathlib mixes `IsCompact` with `closure` and
#: `is_topology`, and a pattern keyed on capitals drops a third of the content.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z0-9_']+)*")

#: Syntax, not mathematics. These appear in nearly every statement and would
#: dominate any overlap measure built on raw token sets.
_STOP = {
    "theorem", "lemma", "by", "fun", "let", "in", "if", "then", "else", "with",
    "Type", "Prop", "Sort", "sorry", "this", "have", "show", "from", "at",
}

#: Relations and quantifiers carry topic too: a statement about ⊆ and ∈ is set
#: theory whatever it names, and one about ∣ and % is arithmetic.
_SYMBOLS = "⊆ ∈ ∪ ∩ ∀ ∃ ∣ ≤ < ∑ ∏ → ↔ ≠ ⁻¹ ∘ ∅ ℕ ℤ ℚ ℝ ℂ".split()


def signature(statement: str) -> Set[str]:
    """The objects a statement talks about, as a set of tokens."""
    text = str(statement or "")
    # Binder names are local and say nothing about the subject: `(f : I → I)`
    # contributes `Continuous` and `→`, not `f` and `I`. Single letters and
    # short primed names are dropped for that reason.
    names = {
        token for token in _IDENT.findall(text)
        if token not in _STOP and len(token.split(".")[0]) > 2
    }
    names |= {symbol for symbol in _SYMBOLS if symbol in text}
    return names


def overlap(a: Set[str], b: Set[str]) -> float:
    """Jaccard. Zero when either side is empty, which is the honest reading:
    a statement whose signature we could not extract has no measured fit."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_seeds(pattern: str) -> Dict[str, str]:
    seeds: Dict[str, str] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("id"):
                    seeds[row["id"]] = row.get("formal_statement") or ""
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/certified/run-a/seeds/proofnet_g*.csv")
    parser.add_argument("--runs", default="data/certified/run-a/proofnet_g*.jsonl")
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    sig = {name: signature(text) for name, text in seeds.items()}
    sizes = sorted(len(s) for s in sig.values())
    print(f"{len(seeds)} seeds · signature size min {sizes[0]} median {sizes[len(sizes)//2]} max {sizes[-1]}")
    thin = [k for k, v in sig.items() if len(v) < 3]
    if thin:
        print(f"  {len(thin)} seed(s) with a signature under 3 tokens: {', '.join(sorted(thin)[:4])}")

    rows: List[dict] = []
    for path in sorted(glob.glob(args.runs)):
        if ".pre_" in path:
            continue
        rows += [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    measured = []
    for row in rows:
        if not str(row.get("op_type") or "").startswith("crossover"):
            continue
        roots = [r for r in roots_of(row.get("problem_id") or "") if r in sig]
        if len(roots) < 2:
            continue
        pairs = [overlap(sig[a], sig[b])
                 for i, a in enumerate(roots) for b in roots[i + 1:]]
        measured.append({
            "fit": max(pairs) if pairs else 0.0,
            "certified": row.get("status") == "certified",
            "parallel": "parallel_crossover" in str(row.get("failure_signature") or ""),
            "roots": roots,
        })

    print(f"\n{len(measured)} crossovers with two identifiable seed parents")
    if not measured:
        return

    measured.sort(key=lambda r: r["fit"])
    size = max(1, len(measured) // args.bins)
    print(f"\n{'topic fit':>18s} {'n':>4s} {'certified':>10s} {'parallel':>9s}")
    for index in range(0, len(measured), size):
        chunk = measured[index:index + size]
        if len(chunk) < 3:
            continue
        lo, hi = chunk[0]["fit"], chunk[-1]["fit"]
        cert = sum(1 for c in chunk if c["certified"])
        par = sum(1 for c in chunk if c["parallel"])
        print(f"{lo:7.2f} – {hi:5.2f} {len(chunk):5d} "
              f"{cert:5d} {100*cert/len(chunk):4.0f}% {par:4d} {100*par/len(chunk):4.0f}%")

    zero = [c for c in measured if c["fit"] == 0]
    some = [c for c in measured if c["fit"] > 0]
    for label, group in (("no shared vocabulary", zero), ("some shared vocabulary", some)):
        if group:
            cert = sum(1 for c in group if c["certified"])
            print(f"\n{label:24s} n={len(group):3d}  certified {cert:3d} ({100*cert/len(group):.0f}%)")


if __name__ == "__main__":
    main()
