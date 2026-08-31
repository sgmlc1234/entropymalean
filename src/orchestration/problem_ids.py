"""Build and read the identifier a generated row carries.

The old form appended `__theorem_gen1__<hash>` per generation, and the `1` was
a literal rather than a counter: across 146 released rows the segment appeared
209 times and said `gen1` every one of them. A fourth-generation descendant read

    Dummit_Foote_exercise_3_1_3a__theorem_gen1__09c322d9__theorem_gen1__edc654cc
      __theorem_gen1__af6e130a__theorem_gen1__3331197a

which is 134 characters, repeats a constant four times, and does not say which
operator produced any of them. It broke the report's contents list and the card
headings, and it told a reader nothing they could not get from `parent_ids`.

The identifier was doing lineage's job while lineage was already recorded. It
now carries the three things that are actually about *this* row:

    {root seeds}__{operator chain}__{statement fingerprint}
    Dummit_Foote_exercise_3_1_3a__mh.me.ms.me__3331197a

The chain reads left to right in generation order, its length is the lineage
depth, and a crossover keeps both roots joined by `__x__` and starts a fresh
chain. The fingerprint stays a hash of the statement, not a slot number: two
slots that produce the same theorem must collide, because that collision is what
lets the duplicate be dropped.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

#: Two letters per operator, so a four-generation chain costs eleven characters
#: rather than ninety.
CODES = {
    "mutation_easy": "me",
    "mutation_hard": "mh",
    "mutation_silent": "ms",
    "crossover_easy": "xe",
    "crossover_hard": "xh",
}
NAMES = {code: name for name, code in CODES.items()}

CROSS = "__x__"
#: `__<chain>__<8 hex>` at the end. The chain is dot-joined two-letter codes.
SUFFIX = re.compile(r"__(?:[a-z]{2})(?:\.[a-z]{2})*__[0-9a-f]{8}$")
#: The form this replaces, still parsed so old rows remain readable.
LEGACY_SUFFIX = re.compile(r"(__theorem_gen\d+(?:__[0-9a-f]{8})?|__gen\d+_[a-z_]+|__fallback[a-z_]*)+$")


def code_for(op_type: str, operator_variant: str) -> str:
    """The two-letter code for a slot, falling back on `op_type` alone."""
    variant = str(operator_variant or "").strip()
    if variant in CODES:
        return CODES[variant]
    return "xx" if str(op_type or "").startswith("crossover") else "mm"


def roots_of(problem_id: str) -> List[str]:
    """The benchmark seeds this row descends from, in order."""
    stem = strip_suffix(str(problem_id or ""))
    return [part for part in stem.split(CROSS) if part]


def strip_suffix(problem_id: str) -> str:
    """The identifier with its chain and fingerprint removed.

    Both forms are stripped, and repeatedly, because a legacy id stacks one
    suffix per generation while the current form carries exactly one.
    """
    text = str(problem_id or "")
    while True:
        shorter = LEGACY_SUFFIX.sub("", SUFFIX.sub("", text))
        if shorter == text:
            return text
        text = shorter


def chain_of(problem_id: str) -> List[str]:
    """The operator chain, as variant names. Empty for a seed or a legacy id."""
    match = SUFFIX.search(str(problem_id or ""))
    if not match:
        return []
    codes = match.group(0).strip("_").split("__")[0].split(".")
    return [NAMES.get(code, code) for code in codes]


def child_id(
    parent_ids: Iterable[str],
    *,
    op_type: str,
    operator_variant: str,
    fingerprint: str,
) -> str:
    """The identifier for a child of these parents.

    A mutation extends its parent's chain. A crossover joins both parents' roots
    and starts a new chain, because two chains cannot be merged into one linear
    history and `parent_ids` already records the branch precisely.
    """
    parents = [str(p) for p in parent_ids if str(p or "").strip()]
    if not parents:
        raise ValueError("a generated row must name at least one parent")
    code = code_for(op_type, operator_variant)
    if len(parents) >= 2:
        roots: List[str] = []
        for parent in parents:
            for root in roots_of(parent):
                if root not in roots:
                    roots.append(root)
        return f"{CROSS.join(roots)}__{code}__{fingerprint}"
    parent = parents[0]
    chain = [CODES.get(name, "mm") for name in chain_of(parent)]
    chain.append(code)
    return f"{strip_suffix(parent)}__{'.'.join(chain)}__{fingerprint}"


def fingerprint_of(problem_id: str) -> Optional[str]:
    match = SUFFIX.search(str(problem_id or ""))
    return match.group(0).rsplit("__", 1)[-1] if match else None


def root_set(problem_id: str) -> Set[str]:
    return set(roots_of(problem_id))
