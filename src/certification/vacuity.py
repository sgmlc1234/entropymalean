"""Ask Lean whether a generated theorem's hypotheses can hold at all.

A theorem whose hypotheses are contradictory is provable by anything: every
goal follows from `False`, so the proof teaches nothing and the row measures
nothing. This is not hypothetical for bred problems — crossover in particular
invites it, because piling one parent's constraints onto another's is exactly
how a satisfiable system becomes an unsatisfiable one. The released corpus
contains a crossover whose set

    {y | 0 < y ∧ Int.gcd 40 y = y + 3 ∧ Int.lcm 40 y = y * (y + 3)}

is empty — `gcd 40 y = y + 3` forces `y + 3 ∣ 3`, impossible for `y > 0` — so
its `∀ y ∈ S` conclusion held for no y at all.

The seed screener asks this question of benchmark seeds before they enter the
pool. The same question belongs on generated rows, where the risk is higher:
a seed's hypotheses were written by a person, a child's were assembled by an
operator that does not check satisfiability.

The test is one-sided on purpose. Failing to derive `False` does not make a
row meaningful; deriving it makes the row worthless, and that is the half worth
paying for.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, Optional

#: Tactics tried in turn: `omega` for linear-arithmetic contradictions,
#: `simp_all` first where hypotheses must rewrite each other before the
#: arithmetic is reachable.
#:
#: Every branch ends in `done` (or is `omega`, which fails when it cannot
#: close). Without it `first` accepts the first branch
#: that does not *error*, and `norm_num at *` does not error when it merely
#: fails to close the goal — so the probe stopped at branch one with `⊢ False`
#: still open and reported "not vacuous" for hypotheses that plainly are.
REFUTATION = (
    "  first\n"
    "  | (exfalso; omega)\n"
    "  | (exfalso; norm_num at *; done)\n"
    "  | (exfalso; simp_all; done)\n"
    "  | (exfalso; simp_all; norm_num at *; done)\n"
    "  | (exfalso; simp_all; omega)"
)


def binders_of(statement: str) -> str:
    """Everything before the goal's colon: the theorem's hypotheses."""
    body = re.sub(r":=\s*by\s*$", "", str(statement or "").strip()).rstrip()
    depth = 0
    for index, char in enumerate(body):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and depth == 0 and body[index : index + 2] != ":=":
            return body[:index].rstrip()
    return body


def vacuity_probe(header: str, statement: str) -> str:
    """`example <binders> : False` — provable exactly when the row is vacuous."""
    binders = binders_of(statement)
    binders = re.sub(r"^\s*(theorem|lemma)\s+\S+", "", binders).strip()
    head = str(header or "").strip() or "import Mathlib"
    return (
        f"{head}\nset_option autoImplicit false\n\n"
        f"example {binders} : False := by\n{REFUTATION}\n"
    )


async def is_vacuous(
    verifier: Callable[..., Awaitable[Any]],
    header: str,
    statement: str,
    *,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Run the probe. `vacuous` is True only when Lean *proved* `False`.

    Any other outcome — the probe failed to elaborate, the verifier timed out,
    the tactics did not close it — returns False with the reason recorded. A
    check that cannot see the evidence must not convict on it, and a timeout is
    not evidence of consistency either, so both are distinguishable downstream.
    """
    if not str(statement or "").strip():
        return {"vacuous": False, "measured": False, "why": "no statement"}
    code = vacuity_probe(header, statement)
    try:
        verdict = await verifier(code, timeout=timeout)
    except TypeError:
        verdict = await verifier(code)
    except Exception as error:  # pragma: no cover - transport failures
        return {"vacuous": False, "measured": False, "why": f"probe error: {error}"[:120]}
    proved = bool(getattr(verdict, "complete", False) or getattr(verdict, "ok", False))
    system_error = getattr(verdict, "system_error", None)
    if system_error:
        return {"vacuous": False, "measured": False, "why": str(system_error)[:120]}
    return {
        "vacuous": proved,
        "measured": True,
        "why": "hypotheses entail False" if proved else "",
    }

#: `∀ y, y ∈ S → …` and `∀ y ∈ S, …`. The set is captured so the probe can ask
#: whether anything is in it.
_BOUNDED_FORALL = re.compile(
    r"∀\s*([A-Za-z_][A-Za-z0-9_']*)\s*(?::\s*[^,]+?)?\s*,\s*\1\s*∈\s*([A-Za-z_][A-Za-z0-9_']*)\s*→"
    r"|∀\s*([A-Za-z_][A-Za-z0-9_']*)\s*∈\s*([A-Za-z_][A-Za-z0-9_']*)\s*,"
)


def quantified_sets(formal_statement: str) -> list:
    """Set names the conclusion quantifies over, in order of appearance."""
    text = str(formal_statement or "").split(":= by", 1)[0]
    names = []
    for match in _BOUNDED_FORALL.finditer(text):
        name = match.group(2) or match.group(4)
        if name and name not in names:
            names.append(name)
    return names


def inhabited_probe(header: str, statement: str, set_name: str) -> str:
    """`example … : ∃ y, y ∈ S` — provable exactly when the quantifier bites.

    A conclusion of the form `∀ y ∈ S, P y` is true for free when `S` is empty,
    and the hypotheses can be perfectly consistent while that happens: the
    released corpus contains a crossover whose `S` is empty because
    `Int.gcd 40 y = y + 3` forces `y + 3 ∣ 3`. The vacuity probe cannot see it,
    because nothing is contradictory — the emptiness is inside the set, not in
    the hypotheses.
    """
    binders = binders_of(statement)
    binders = re.sub(r"^\s*(theorem|lemma)\s+\S+", "", binders).strip()
    head = str(header or "").strip() or "import Mathlib"
    return (
        f"{head}\nset_option autoImplicit false\n\n"
        f"example {binders} : ∃ y, y ∈ {set_name} := by\n"
        f"  first\n"
        f"  | (simp_all; done)\n"
        f"  | (decide)\n"
        f"  | (norm_num [Set.mem_setOf_eq]; done)\n"
    )


async def empty_quantified_sets(
    verifier: Callable[..., Awaitable[Any]],
    header: str,
    statement: str,
    *,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Which quantified sets Lean could not show to be inhabited.

    Reported, not decided. Failing to *prove* inhabitation is not proving
    emptiness — the witness may simply be beyond these tactics — so this returns
    the unresolved names and lets the caller treat them as a flag for review
    rather than a rejection. Anything stronger would need the negation proved,
    which is a much harder goal than the one being screened for.
    """
    names = quantified_sets(statement)
    if not names:
        return {"measured": True, "quantified_sets": [], "uninhabited": []}
    unproven = []
    for name in names[:3]:
        code = inhabited_probe(header, statement, name)
        try:
            verdict = await verifier(code, timeout=timeout)
        except TypeError:
            verdict = await verifier(code)
        except Exception as error:  # pragma: no cover
            return {"measured": False, "why": f"probe error: {error}"[:120]}
        if getattr(verdict, "system_error", None):
            return {"measured": False, "why": str(verdict.system_error)[:120]}
        if not bool(getattr(verdict, "complete", False)):
            unproven.append(name)
    return {
        "measured": True,
        "quantified_sets": names,
        "uninhabited": unproven,
    }

