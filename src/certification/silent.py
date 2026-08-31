"""Prove that a silent mutation changed the encoding and nothing else.

A silent mutation re-expresses a parent theorem in different Lean without
touching the mathematics — `% 10 = 8` becomes `≡ 8 [MOD 10]`, `11 ∣ x` becomes
`x ≡ 0 [ZMOD 11]`, `Real.logb 3 27` becomes `Real.log 27 / Real.log 3`. The name
is the genetic one: a substitution that alters the sequence without altering
what it encodes.

Every other operator is gated on being *different enough* from its parent, and
that question has no mechanical answer, which is why it ended up with a judge.
This operator inverts the question into one Lean can settle: is the child
*exactly* equivalent? So the novelty gates are not waived here, they are traded
for a stricter one. A silent row that cannot prove its equivalence is not a
weak row to be flagged; it is an unsupported claim, and it is discarded.

Equivalence is proved in both directions on purpose. One direction alone admits
a child that is merely implied by the parent — a weakening, which is a different
theorem and an easier one. Only the biconditional pins the child to the same
mathematical content, and difficulty-invariance is the whole reason this
operator is worth having: it is the one rung on the mutation ladder where a drop
in solve rate cannot be explained by the problem having become harder.

The statement is closed into a Prop before anything is asked of it. A theorem's
binders are already valid `∀` binders, so

    theorem f (s : ℕ) (hs : s = 1 + 2) : s % 4 = 2

closes to `∀ (s : ℕ) (hs : s = 1 + 2), s % 4 = 2` by moving the text, with no
parsing of the binder group into types and hypotheses.

The automatic ladder is offered first and expected to miss. Writing these five
by hand, two needed real proofs — `Int.ModEq` had to be unfolded to divisibility
before `omega` could see it, and the squares-mod-3 case needed the residue split
supplied explicitly. The generator will hit the same wall, so the worker is
asked for the two proofs and the ladder only saves the easy cases.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, Optional

from src.certification.vacuity import binders_of

#: Tried before the worker's own proof is used. These close the cases where the
#: two encodings are definitionally equal or fall to one simp set; anything
#: needing a case split or an unfolding lemma is expected to fail here.
#: Tried in order; the first branch that closes the goal wins.
#:
#: Every rung was about `Nat` and `Int` -- `omega`, `Nat.ModEq`, `Int.ModEq` --
#: with nothing for the reals. A silent mutation restating
#: `2 - √2 ≥ 2 - x - 1/(2x)` as `√2 ≤ x + 1/(2x)`, which is the same inequality
#: rearranged, failed in *both* directions with "omega could not prove the goal:
#: No usable constraints found", and the row was discarded as unproven. `omega`
#: is an integer decision procedure and cannot see a real inequality at all.
#: `linarith` closes a rearrangement, `nlinarith` a product or square, and
#: `field_simp` clears the denominator that made this one nonlinear.
AUTO_LADDER = (
    "  first\n"
    "  | exact h\n"
    "  | (simpa [Nat.ModEq, Int.ModEq, Int.emod_emod_of_dvd, Real.logb,\n"
    "            Int.dvd_iff_emod_eq_zero, Nat.dvd_iff_mod_eq_zero,\n"
    "            Set.mem_Icc, Set.mem_Ico, Set.mem_Ioc, Set.mem_Ioo] using h)\n"
    "  | (simp_all [Nat.ModEq, Int.ModEq, Real.logb,\n"
    "               Set.mem_Icc, Set.mem_Ico, Set.mem_Ioc, Set.mem_Ioo]; done)\n"
    "  | (intros; simp_all [Nat.ModEq, Int.ModEq, Real.logb,\n"
    "                       Set.mem_Icc, Set.mem_Ico, Set.mem_Ioc, Set.mem_Ioo]; done)\n"
    "  | (norm_num [Nat.ModEq, Int.ModEq, Real.logb] at h ⊢; omega)\n"
    # `linarith [h]` cannot use a universally quantified `h`: it needs the
    # instance, not the rule. When the statement's binders live in the goal
    # rather than in the signature -- `∀ x > 0, …` -- the hypothesis has to be
    # applied to the variables `intro` just produced. Arities one and two cover
    # the shapes the corpus actually contains; anything deeper is what the
    # worker's own `equivalence_forward`/`equivalence_backward` are for.
    "  | (intro x hx; linarith [h x hx])\n"
    "  | (intro x hx; nlinarith [h x hx])\n"
    "  | (intro x hx; field_simp at hx ⊢; nlinarith [h x hx])\n"
    "  | (intro x; linarith [h x])\n"
    "  | (intro x; nlinarith [h x])\n"
    "  | (intros; linarith [h])\n"
    "  | (intros; nlinarith [h])\n"
    "  | (intros; field_simp at h ⊢; linarith [h])\n"
    "  | (intros; field_simp at h ⊢; nlinarith [h])\n"
    "  | (intros; ring_nf at h ⊢; linarith [h])\n"
    "  | (intros; omega)\n"
    "  | linarith [h]\n"
    "  | nlinarith [h]\n"
    "  | omega"
)

_NAME = re.compile(r"^\s*(?:theorem|lemma)\s+\S+")


def prop_of(statement: str) -> str:
    """A theorem statement as a closed Prop.

    `theorem f (n : ℕ) : P n` becomes `∀ (n : ℕ), P n`; a statement with no
    binders is already a Prop and is returned as its goal.
    """
    body = re.sub(r":=\s*(by)?\s*$", "", str(statement or "").strip()).rstrip()
    binders = _NAME.sub("", binders_of(body)).strip()
    goal = body[len(binders_of(body)):].lstrip()
    goal = goal[1:].strip() if goal.startswith(":") else goal.strip()
    if not goal:
        return ""
    return f"∀ {binders}, {goal}" if binders else goal


def equivalence_probe(
    header: str,
    parent_statement: str,
    child_statement: str,
    *,
    forward_proof: str = "",
    backward_proof: str = "",
) -> str:
    """Both implications as two theorems, so a half-proof cannot pass as a whole.

    `forward` is parent ⟹ child and `backward` is child ⟹ parent. They are
    separate declarations rather than one `↔` because Lean then reports which
    direction failed, and that is what the retry brief needs to say.
    """
    parent = prop_of(parent_statement)
    child = prop_of(child_statement)
    if not parent or not child:
        return ""
    head = str(header or "").strip() or "import Mathlib"
    forward = _indent(forward_proof) or AUTO_LADDER
    backward = _indent(backward_proof) or AUTO_LADDER
    # `maxRecDepth` is raised because the ladder's simp sets recurse on
    # congruence goals — `10^m - (-1)^m ≡ 0 [ZMOD 11]` blew the default stack
    # and reported it as a simp failure, which reads as "not equivalent" when it
    # only meant "not reachable from here".
    return (
        f"{head}\nset_option autoImplicit false\n"
        f"set_option maxHeartbeats 800000\nset_option maxRecDepth 8000\n\n"
        f"theorem silent_forward (h : {parent}) : {child} := by\n{forward}\n\n"
        f"theorem silent_backward (h : {child}) : {parent} := by\n{backward}\n"
    )


#: A worker asked for a tactic block often answers with the declaration syntax
#: it writes everywhere else. The probe already supplies `:= by`, so a proof
#: that repeats it lands as `:= by\n  := by ...` and Lean reports `unexpected
#: token 'by'` -- which reads downstream as "the equivalence does not hold"
#: rather than "the proof was pasted with its opener".
_OPENER = re.compile(r"^\s*(?::=)?\s*by\b[ \t]*\n?")


def _indent(proof: str) -> str:
    text = _OPENER.sub("", str(proof or "").strip("\n"), count=1).strip("\n")
    if not text.strip():
        return ""
    lines = text.splitlines()
    if all(line.startswith(("  ", "\t")) or not line.strip() for line in lines):
        return text
    return "\n".join(f"  {line}" if line.strip() else line for line in lines)


async def check_equivalence(
    verifier: Callable[..., Awaitable[Any]],
    header: str,
    parent_statement: str,
    child_statement: str,
    *,
    forward_proof: str = "",
    backward_proof: str = "",
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Whether Lean accepts both directions.

    `equivalent` is True only when it did. Everything else — a probe that would
    not elaborate, a timeout, a transport failure — is False with `measured`
    False, so a row is never certified silent on the strength of a check that
    did not run.
    """
    code = equivalence_probe(
        header,
        parent_statement,
        child_statement,
        forward_proof=forward_proof,
        backward_proof=backward_proof,
    )
    if not code:
        return {"equivalent": False, "measured": False, "why": "no statement to compare"}
    try:
        verdict = await verifier(code, timeout=timeout)
    except TypeError:
        verdict = await verifier(code)
    except Exception as error:  # pragma: no cover - transport failures
        return {"equivalent": False, "measured": False, "why": f"probe error: {error}"[:160]}
    if getattr(verdict, "system_error", None):
        return {
            "equivalent": False,
            "measured": False,
            "why": str(verdict.system_error)[:160],
        }
    proved = bool(getattr(verdict, "complete", False))
    summary = getattr(verdict, "summary", None)
    message = str(
        (summary() if callable(summary) else "")
        or getattr(verdict, "error", "")
        or getattr(verdict, "message", "")
        or ""
    )
    return {
        "equivalent": proved,
        "measured": True,
        "used_auto_ladder": not (forward_proof.strip() or backward_proof.strip()),
        "failed_direction": _failed_direction(message, code) if not proved else "",
        "why": "" if proved else message[:400],
    }


def _failed_direction(message: str, code: str = "") -> str:
    """Which implication Lean rejected, for the retry brief.

    Lean reports the line, not the declaration, so the two theorems are told
    apart by where the backward one starts. Matching on the declaration name
    instead returned "unknown" for every real failure.
    """
    text = str(message or "")
    lines = str(code or "").splitlines()
    split = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.startswith("theorem silent_backward")
        ),
        0,
    )
    numbers = [int(n) for n in re.findall(r"line (\d+)", text)]
    if not (split and numbers):
        return "unknown"
    forward = any(number < split for number in numbers)
    backward = any(number >= split for number in numbers)
    if forward and backward:
        return "both"
    return "forward" if forward else "backward"


#: What may change without changing the mathematics. Given to the worker as its
#: whole licence: anything outside this list is a different theorem.
ALLOWED_CHANGES = (
    "rename variables and binders",
    "swap a notation for an equivalent one "
    "(`a % n = r` ↔ `a ≡ r [MOD n]`, `n ∣ a` ↔ `a ≡ 0 [ZMOD n]`)",
    "name an inline expression as a hypothesis "
    "(`: f 3 = 2` becomes `(s : ℕ) (hs : s = f 3) : s = 2`)",
    "unfold or fold a definition (`Real.logb b x` ↔ `Real.log x / Real.log b`)",
    "restate a dichotomy as the exclusion of its complement",
    "reorder independent hypotheses",
    "re-tell the problem as a concrete situation -- ages, coins, distances, a "
    "shop's stock -- provided the setting adds and removes no condition. This is "
    "the strongest re-encoding available: it leaves the mathematics untouched "
    "and moves everything a memorising solver keys on",
)

#: What breaks difficulty-invariance, and so is out of bounds even when the
#: result is a true theorem. The third is the one that catches people: turning
#: `x % 10 = 8` into `∃ q, x = 10 * q + 8` is a true rewriting and a strictly
#: harder problem, because the solver now has to produce the witness.
FORBIDDEN_CHANGES = (
    "change what kind of thing is concluded (an equality must stay an equality)",
    "add or remove a quantifier that was not already there",
    "restate the goal so that a witness must be constructed",
    "strengthen or weaken any hypothesis",
    "add a hypothesis that was not derivable from the parent's",
)


def worker_rules() -> str:
    """The silent-mutation contract, as prompt text."""
    return (
        "Silent mutation rules:\n"
        "- Re-express the parent theorem in different Lean. The mathematics must not "
        "change at all; only the encoding does.\n"
        "- You must also return `equivalence_forward` and `equivalence_backward`: Lean "
        "tactic blocks proving parent ⟹ child and child ⟹ parent. Both are checked. A "
        "child whose equivalence does not compile is discarded, not flagged.\n"
        "- Allowed changes:\n"
        + "".join(f"  - {item}\n" for item in ALLOWED_CHANGES)
        + "- Forbidden, because they change the difficulty rather than the encoding:\n"
        + "".join(f"  - {item}\n" for item in FORBIDDEN_CHANGES)
        + "- Aim for an encoding a solver that had memorised the parent's surface form "
        "would not recognise, while a solver that understood the parent would.\n"
    )
