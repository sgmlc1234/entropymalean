"""Refuse a statement the corpus already contains.

Every other gate was replaced by the judge, because every other gate was asking
a question of degree — is this different enough, is this fused enough — that a
parser cannot answer and a reader can. Identity is not a question of degree. Two
rows either state the same theorem or they do not, and a hash settles it without
opinion, so this one stays mechanical.

It has to span the whole corpus. The dedup that existed before read back only
the file the current run was writing, which caught a statement repeated between
generations of one seed group and missed the same statement produced by a
different group, or by an earlier campaign. Those are the collisions that matter
most: a released benchmark with two identical problems under different names
inflates its own size and silently double-counts whatever a prover does with
them.

What counts as the same statement is the mathematical surface: the declaration
name goes, binder names go, comments and whitespace go, and the proof body is
excluded, because proving one theorem two ways does not make two problems.

Binder names have to go *everywhere*, not just where they are declared. The
first version blanked the declaration site and left every use of the name in the
goal, so

    (a b : G) (n : ℕ) : closure {a, b} = closure {b * a * b ^ n, ...}
    (a b : G) (r : ℕ) : closure {a, b} = closure {b * a * b ^ r, ...}

hashed differently and both were released — the same theorem twice, under two
names, from the same parent. It also blanked only the last name of a group,
leaving `(a b : G)` as `(a _ : G)`. Names are now alpha-normalised to positions,
which is what "binder names go" was always supposed to mean.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

_THEOREM_NAME = re.compile(r"\b(theorem|lemma)\s+\S+")
_IDENT = r"[A-Za-z_][A-Za-z0-9_'!?₀-₉]*"
# A binder group's names are everything left of its `:`; instance binders like
# `[Group G]` bind nothing and must not consume `G`.
_EXPLICIT = re.compile(r"[({⦃]\s*((?:" + _IDENT + r"\s+)*" + _IDENT + r")\s*:")
# Binders the goal itself introduces. Their names vary as freely as the
# signature's do, and a statement differing only in them is the same statement.
_QUANTIFIED = re.compile(
    r"(?:∀|∃!|∃|Σ|λ|fun|⨆|⨅|∑|∏)\s*\(?\s*((?:" + _IDENT + r"\s+)*" + _IDENT + r")\s*(?=[:,)]|=>|↦)"
)


def _bound_names(text: str) -> list:
    """Every identifier the statement binds, in order of first binding."""
    seen: Dict[str, None] = {}
    for pattern in (_EXPLICIT, _QUANTIFIED):
        for match in pattern.finditer(text):
            for name in match.group(1).split():
                if name not in ("Type", "Prop", "Sort", "fun"):
                    seen.setdefault(name, None)
    return list(seen)


def dedup_surface(formal: Any) -> str:
    """A theorem statement reduced to what makes it that theorem."""
    text = str(formal or "")
    text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    text = re.sub(r"--[^\n]*", "", text)
    text = text.split(":= by", 1)[0]
    text = _THEOREM_NAME.sub("theorem _", text)
    for position, name in enumerate(_bound_names(text)):
        # `#n#` cannot be a Lean identifier, so a substitution can neither collide
        # with a name already in the text nor be re-substituted by a later pass.
        # Not preceded by a dot, so `Nat.succ` survives a binder named `succ`.
        text = re.sub(r"(?<![A-Za-z0-9_.'])" + re.escape(name) + r"(?![A-Za-z0-9_'!?₀-₉])",
                      f"#{position}#", text)
    return " ".join(text.split())


def fingerprint(formal_statement: str) -> str:
    """Full-length digest of the surface. Used for equality, not for ids."""
    surface = dedup_surface(formal_statement)
    return hashlib.sha256(surface.encode("utf-8")).hexdigest() if surface else ""


class CorpusIndex:
    """Fingerprints of every statement already released or generated.

    Built once per run and added to as the run produces rows, so a statement
    repeated inside the run is caught by the same index that catches one
    repeated from three campaigns ago.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, str] = {}
        self.sources = 0

    def add(self, formal_statement: str, problem_id: str = "") -> bool:
        """Record a statement. False when it was already present."""
        key = fingerprint(formal_statement)
        if not key:
            return True
        if key in self._seen:
            return False
        self._seen[key] = str(problem_id or "")
        return True

    def holder(self, formal_statement: str) -> Optional[str]:
        """Which row already states this, or None."""
        key = fingerprint(formal_statement)
        return self._seen.get(key) if key else None

    def __len__(self) -> int:
        return len(self._seen)

    def load_paths(self, paths: Iterable[Path], *, statuses: Optional[Set[str]] = None) -> "CorpusIndex":
        """Index every row in these JSONL files.

        Rows that failed are indexed too. A statement that could not be proved
        is still a statement the pipeline has already tried, and producing it
        again wastes the slot on a known dead end — but see `certified_only`
        below for the release-facing use.
        """
        wanted = statuses
        for path in paths:
            if not Path(path).is_file():
                continue
            self.sources += 1
            for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if wanted and str(row.get("status")) not in wanted:
                    continue
                self.add(
                    str(row.get("formal_statement") or ""),
                    str(row.get("problem_id") or row.get("id") or ""),
                )
        return self


def build_index(
    roots: Iterable[Path],
    *,
    certified_only: bool = True,
    exclude: Optional[Iterable[Path]] = None,
) -> CorpusIndex:
    """Index the corpus under these directories.

    `certified_only` by default: a duplicate of a row that never certified is
    not a corpus collision, and blocking it would stop a slot from retrying a
    statement whose first attempt failed for an unrelated reason — a missing
    `open`, a timeout.
    """
    skip = {Path(p).resolve() for p in (exclude or [])}
    index = CorpusIndex()
    paths = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("*.jsonl")))
    paths = [p for p in paths if p.resolve() not in skip and "burned" not in str(p)]
    return index.load_paths(paths, statuses={"certified"} if certified_only else None)
