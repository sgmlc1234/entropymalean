"""Did a silent mutation keep every hypothesis its parent had?

A silent mutation restates the parent in different Lean and changes no
mathematics. Dropping a hypothesis changes the mathematics: the child is then a
more general theorem, which is a different and usually harder claim, and adding
one makes it a weaker claim about a smaller domain. Either way it is not the
same problem, and difficulty invariance — the only reason this operator earns
its place — is gone.

The judge cannot be relied on for this. Asked inside a rubric whose vocabulary
is about difference, it read a dropped bound as evidence *for* the child:
on `Nat.choose n k = choose (n-1) k + choose (n-1) (k-1)` with `0 < n`, `0 < k`
and `k ≤ n`, the child restated it in successor form and dropped all three. The
judge named the drop in its own reasoning and passed the row anyway.

This is not a question of degree, so it does not go to the judge. Two binder
sets either match or they do not, and comparing them is parsing.
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.certification.dedup import dedup_surface

#: `(name : type)` groups. Parentheses are counted rather than matched by
#: pattern because binder types nest: `(hn : ¬ ((3:ℤ) ∣ y))` is two levels deep
#: and a one-level pattern skips it silently.
#: Ordinary Lean names, plus the `#0#` placeholders the dedup surface leaves
#: behind. Without the second form this parser finds no binders at all in a
#: normalised statement and reports every hypothesis as preserved.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_'!?₀-₉]*|#\d+#")
_NAME = re.compile(r"^\s*(?:theorem|lemma)\s+\S+")


def _groups(statement: str) -> List[tuple]:
    """(names, type) for every explicit binder group, in order."""
    text = _NAME.sub("", str(statement or "").strip())
    text = text.split(":=", 1)[0]
    out: List[tuple] = []
    index = 0
    while index < len(text):
        if text[index] not in "({":
            index += 1
            continue
        opener, closer = text[index], ")" if text[index] == "(" else "}"
        depth = 0
        for end in range(index, len(text)):
            if text[end] in "({":
                depth += 1
            elif text[end] in ")}":
                depth -= 1
                if depth == 0:
                    inner = text[index + 1 : end]
                    head, sep, kind = inner.partition(":")
                    names = head.split()
                    if sep and names and all(_IDENT.fullmatch(n) for n in names):
                        out.append((names, " ".join(kind.split())))
                    index = end
                    break
        index += 1
    return out


#: `(x)` around a single token. Lean writes `a * (x) ^ 2` and a restatement
#: writes `a * x ^ 2`; comparing the type text verbatim called those different
#: hypotheses and reported a drop the judge correctly refused to believe.
#: Only single-atom groups are removed -- `(a + b) * c` keeps its parentheses,
#: because there they carry meaning.
_REDUNDANT_PARENS = re.compile(r"\((\s*(?:[A-Za-z_][A-Za-z0-9_'!?₀-₉]*|#\d+#|\d+)\s*)\)")


def _normalise_type(kind: str) -> str:
    text = " ".join(str(kind or "").split())
    while True:
        shorter = _REDUNDANT_PARENS.sub(r"\1", text)
        shorter = " ".join(shorter.split())
        if shorter == text:
            return text.replace(" ", "")
        text = shorter


def hypotheses(statement: str) -> Dict[str, str]:
    """Binder name to its type, for binders that state a proposition.

    A datum binder (`n : ℕ`) is not a hypothesis; the type of a hypothesis
    mentions a relation. Rather than guess at type theory, anything whose type
    contains a relational or logical token counts, which is the same test the
    redundancy probe uses and errs toward calling something a hypothesis.
    """
    relational = ("=", "≤", "<", "≥", ">", "≠", "∣", "∈", "⊆", "∧", "∨", "→", "¬",
                  "≡", "Prime", "Odd", "Even", "Nonempty", "Fintype.card", "0 <")
    out: Dict[str, str] = {}
    for names, kind in _groups(statement):
        if any(token in kind for token in relational):
            for name in names:
                out[name] = _normalise_type(kind)
    return out


def compare(parent_statement: str, child_statement: str) -> Dict[str, object]:
    """What the child did to its parent's hypotheses.

    Both sides are alpha-normalised first — renaming variables is the one thing
    a silent mutation is unambiguously allowed to do, and comparing raw type
    text calls `0 < n` and `0 < m` different hypotheses. The dedup surface
    already maps binders onto positions, so the same normal form is reused
    rather than invented twice.

    A hypothesis counts as preserved when some child hypothesis has the same
    normalised type.
    """
    parent = hypotheses(dedup_surface(parent_statement))
    child = hypotheses(dedup_surface(child_statement))
    child_types = list(child.values())
    dropped = [f"{name} : {kind}" for name, kind in parent.items() if kind not in child_types]
    parent_types = list(parent.values())
    added = [f"{name} : {kind}" for name, kind in child.items() if kind not in parent_types]
    return {
        "measured": True,
        "parent_hypotheses": len(parent),
        "child_hypotheses": len(child),
        "dropped": dropped,
        "added": added,
        "preserved": not dropped and not added,
        "why": (
            "dropped " + "; ".join(dropped) if dropped
            else "added " + "; ".join(added) if added
            else ""
        ),
    }
