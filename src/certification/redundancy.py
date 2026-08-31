"""Ask Lean whether a crossover actually needed both of its parents.

Two certified crossovers judged `strong` turned out not to need one parent at
all, and both failed in the same way. `gcd_lcm_mod_three_pipeline` carried
`Int.gcd 40 y = y + 3` and `Int.lcm 40 y = y * (y + 3)` as hypotheses and
concluded `y^2 + (y+3)^2 ≡ 2 [ZMOD 3]`, which follows from `¬(3 ∣ y)` alone.
`normalized_six_step_difference_mod_three` concluded
`∃ q, 5^(n+6) - 5^n = 7q ∧ (¬(3 ∣ q) ↔ q^2 ≡ 1 [ZMOD 3])`, where the second
conjunct holds for every integer, so the child is equivalent to its first
parent.

The judge missed both, and for a reason worth naming: it read the shared
variable `q` as evidence that the parents interacted. A variable appearing in
two places is a syntactic fact. Whether one of those places is free is a
mathematical one, and Lean can answer it.

Two probes, both one-sided:

  `universal`   a parent whose statement is provable with no hypotheses is a
                fact Mathlib already supplies. Applying it to anything the other
                parent constructed costs nothing, so it cannot be the parent
                that made the child harder.

  `droppable`   a child still provable after a parent's hypotheses are removed
                did not need them.

One-sided means a probe that fails to prove has shown nothing: these ladders are
weak on purpose, so what they do prove is certainly free, and what they cannot
is left to the judge with the measurement attached. The alternative — a ladder
strong enough to settle every case — would reject rows for being provable by a
tactic rather than for being redundant.
"""

from __future__ import annotations

import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

#: Deliberately modest. It closes statements that are true by arithmetic,
#: residue arithmetic or a finite decision, which is the shape a free-standing
#: supplier takes; it does not do induction or case analysis on structure.
LADDER = (
    "  intros\n"
    "  first\n"
    "  | omega\n"
    "  | decide\n"
    "  | (norm_num [Int.ModEq, Nat.ModEq, Int.emod_emod_of_dvd]; done)\n"
    "  | (simp_all [Int.ModEq, Nat.ModEq]; done)\n"
    "  | (norm_num [Int.ModEq, Nat.ModEq] at *; omega)\n"
    "  | (constructor <;> omega)\n"
    "  | (positivity)\n"
    "  | (nlinarith [sq_nonneg 1]; done)"
)

_NAME = re.compile(r"^\s*(?:theorem|lemma)\s+\S+")
#: Lean identifiers are not ASCII. The corpus names its hypotheses `h₀`, `h₁`,
#: `h₂` with subscript digits, and a class of `[A-Za-z0-9_']` matches none of
#: them: the scanner found no hypotheses at all, dropped nothing, and then
#: "proved" each parent with its hypotheses still in place -- reporting 18 of 38
#: crossovers redundant, almost all of them wrongly. Subscripts, primes, Greek
#: and the exclamation/question marks Lean permits all belong here.
_IDENT = re.compile(
    r"[A-Za-z_\u00c0-\u024f\u0370-\u03ff]"
    r"[A-Za-z0-9_'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u209c\u2070-\u207f]*"
)


def _binders(text: str):
    """Every `(name : type)` group, matched by counting parentheses.

    A regex cannot do this: the types nest. `(hn : ¬ ((3:ℤ) ∣ y))` is two levels
    deep and a pattern that allows one level silently skips the binder, which
    then never gets tested. Counting is exact at any depth.
    """
    out = []
    index = 0
    while index < len(text):
        if text[index] != "(":
            index += 1
            continue
        depth = 0
        for end in range(index, len(text)):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    inner = text[index + 1 : end]
                    head, sep, kind = inner.partition(":")
                    name = head.strip()
                    if sep and _IDENT.fullmatch(name):
                        out.append((name, kind.strip(), index, end + 1))
                    index = end
                    break
        index += 1
    return out


def _split(statement: str) -> tuple[str, str]:
    """Binder text and goal text of a theorem statement."""
    body = re.sub(r":=\s*(?:by)?\s*$", "", str(statement or "").strip()).rstrip()
    body = _NAME.sub("", body).strip()
    depth = 0
    for index, char in enumerate(body):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and depth == 0 and body[index : index + 2] != ":=":
            return body[:index].rstrip(), body[index + 1 :].strip()
    return body, ""


def hypothesis_names(statement: str) -> List[str]:
    """Named binders whose type is a proposition-looking thing, in order.

    Type binders (`n : ℕ`, `f : ℝ → ℝ`) are skipped: dropping one makes the
    statement fail to elaborate rather than testing anything.
    """
    binders, _ = _split(statement)
    out: List[str] = []
    for name, kind, _start, _end in _binders(binders):
        if not _is_data(kind):
            out.append(name)
    return out


#: Types that introduce data rather than an assumption. Everything else is
#: treated as a hypothesis.
#:
#: The test used to run the other way -- a keyword list of things that look like
#: propositions -- and a list of that kind is always missing something. It missed
#: `Even (card G)`, `IsClosed A`, `Continuous (f i)`, `H.Normal`, and even
#: `2 * n ≡ 15 [MOD 47]`, whose `≡` was absent from the character class. Each
#: omission called a conditional parent unconditional and flagged a crossover
#: that was doing real work.
#:
#: Inverted, an unrecognised type counts as a hypothesis, so the error falls on
#: the side of finding fewer redundancies. For a gate that discards rows, that is
#: the only acceptable direction to be wrong in.
#: `(?![\w.])` after each name: `Nat` is a type, `Nat.digits 3 x = [2,2,2,1]` is
#: an equation. Matching the prefix alone classified that hypothesis as data, so
#: a parent that plainly assumed something was reported as assuming nothing.
#: `Nat.Prime p` and `Int.gcd a b = c` fail the same way.
_DATA_HEAD = re.compile(
    r"^\s*(?:"
    r"(?:ℕ|ℤ|ℝ|ℚ|ℂ|Nat|Int|Real|Rat|Complex|Bool|Char|String)(?![\w.])"
    r"|Type\b|Sort\b|Prop\b"
    r"|Set\b|Finset\b|Multiset\b|List\b|Fin\b|Vector\b|Matrix\b"
    r"|Polynomial\b|Subgroup\b|Submodule\b|Ideal\b|Filter\b"
    r"|[A-Za-z_][A-Za-z0-9_']*\s*→"
    r")"
)


def _is_data(kind: str) -> bool:
    """Whether a binder introduces a value or a type, not an assumption."""
    text = " ".join(str(kind or "").split())
    if not text:
        return True
    if _DATA_HEAD.match(text):
        return True
    # `α → β` where both sides are data, and bare single-letter type variables.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*(\s*→\s*[A-Za-z_ℕℤℝℚℂ][A-Za-z0-9_']*)+", text):
        return True
    # A bare identifier with no operator applied is a type variable -- `(x : G)`,
    # `(a : α)`. A proposition of that shape would have to be a `Prop`-valued
    # variable, which this corpus does not use.
    return bool(re.fullmatch(r"[A-Za-zΑ-Ωα-ω_][A-Za-z0-9Α-Ωα-ω_']*\s*\*?", text))


def is_unconditional(statement: str) -> bool:
    """True when the statement assumes nothing at all.

    Binders are not the only place a hypothesis lives. `∀ b c : ℝ, 1 ≤ b → b ≤ 5
    → (∃ x, 2*x^2 + b*x + c = 0 ↔ c ≤ b^2/8)` has no propositional binder and is
    still conditional: its assumptions are arrows in the goal. Reading binders
    alone called that parent unconditional and flagged a crossover that was
    doing real work -- the ratio from one parent had to be shown to satisfy
    `1 ≤ b ≤ 5` before the other parent applied.

    An arrow inside a nested `∃`/`↔` body is part of the claim rather than an
    assumption, so only arrows at the top level of the goal count.
    """
    if hypothesis_names(statement):
        return False
    _binders, goal = _split(statement)
    depth = 0
    for index, char in enumerate(goal):
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif depth == 0 and goal.startswith("→", index):
            return False
    return bool(goal.strip())


def without(statement: str, names: Sequence[str]) -> str:
    """The statement with those binders removed."""
    binders, goal = _split(statement)
    drop = set(names)
    keep = []
    cursor = 0
    for name, _kind, start, end in _binders(binders):
        if name in drop:
            keep.append(binders[cursor:start])
            cursor = end
    keep.append(binders[cursor:])
    binders = " ".join("".join(keep).split())
    return f"theorem _probe {binders} : {goal}" if binders else f"theorem _probe : {goal}"


def probe(header: str, statement: str, drop: Sequence[str] = ()) -> str:
    """Lean file that tries to prove the statement without `drop`, by ladder."""
    goal = without(statement, drop)
    if not goal.strip():
        return ""
    head = str(header or "").strip() or "import Mathlib"
    return (
        f"{head}\nset_option autoImplicit false\nset_option maxHeartbeats 600000\n\n"
        f"{goal} := by\n{LADDER}\n"
    )


async def _provable(
    verifier: Callable[..., Awaitable[Any]], code: str, timeout: float
) -> Optional[bool]:
    """True/False, or None when the probe could not be evaluated at all."""
    if not code:
        return None
    try:
        verdict = await verifier(code, timeout=timeout)
    except TypeError:
        verdict = await verifier(code)
    except Exception:  # pragma: no cover - transport failures
        return None
    if getattr(verdict, "system_error", None):
        return None
    return not getattr(verdict, "errors", None)


#: Asked of a model when the ladder cannot settle a probe. The ladder is a
#: filter for statements that are true by arithmetic; the cases that matter here
#: needed a residue split and an unfolding of `Int.ModEq` to divisibility, which
#: no fixed tactic list produces. Proving "this holds without that hypothesis"
#: is theorem proving, so it is given to a prover and checked by Lean.
_PROVE_SYSTEM = (
    "You are a Lean 4 (Mathlib) prover. You are given a theorem statement. "
    "Return ONLY a Lean tactic block that proves it, with no explanation, no "
    "code fences, and no restatement of the theorem. If you cannot prove it, "
    "return exactly: CANNOT."
)


def _prove_prompt(goal: str) -> str:
    return (
        "Prove this theorem. It may or may not be true; if you cannot close it, "
        "answer CANNOT rather than guessing.\n\n"
        f"{goal} := by\n\n"
        "Return the tactic block only, indented by two spaces."
    )


async def _llm_attempt(
    verifier: Callable[..., Awaitable[Any]],
    prover: Optional[Callable[..., Awaitable[Any]]],
    header: str,
    statement: str,
    drop: Sequence[str],
    timeout: float,
) -> Optional[bool]:
    """Ask a model to prove the reduced statement, then check it in Lean.

    The model's answer is never trusted; only Lean's verdict is. A model that
    hallucinates a proof produces a Lean error and the probe reports nothing,
    which is the same as not knowing.
    """
    if prover is None:
        return None
    goal = without(statement, drop)
    if not goal.strip():
        return None
    try:
        reply = await prover(_PROVE_SYSTEM, _prove_prompt(goal))
    except Exception:  # pragma: no cover - transport
        return None
    text = str(reply or "").strip()
    if not text or "CANNOT" in text.upper()[:40]:
        return False
    text = re.sub(r"^```[a-zA-Z0-9]*\n?|```$", "", text, flags=re.M).strip("\n")
    body = "\n".join(
        line if line.startswith((" ", "\t")) else "  " + line
        for line in text.splitlines()
        if line.strip()
    )
    head = str(header or "").strip() or "import Mathlib"
    code = (
        f"{head}\nset_option autoImplicit false\nset_option maxHeartbeats 600000\n\n"
        f"{goal} := by\n{body}\n"
    )
    return await _provable(verifier, code, timeout)


#: Where each fusion mechanism puts the contributing parent's work, and so what
#: has to be removed to ask whether that parent was needed. Dropping a hypothesis
#: answers the question for three of them and not for the other two: a witness
#: lives inside an existential and a goal shape lives in the conclusion, and
#: neither is a binder that can be deleted.
MECHANISM_PROBES = {
    "invariant_transplant": ("universal", "droppable"),
    "obstruction_as_lemma": ("universal", "droppable"),
    "parameter_coupling": ("universal", "droppable"),
    "witness_exchange": ("universal", "free_witness"),
    "goal_form_transplant": ("universal", "forced_witness"),
    "sequential_composition": ("universal", "droppable"),
    "": ("universal", "droppable"),
}

_EXISTS = re.compile(r"∃\s*!?\s*")


def strip_to_existence(statement: str) -> str:
    """The child's conclusion with its hypotheses removed, keeping the existential.

    `witness_exchange` claims parent A supplied the object parent B needed. If the
    bare existential closes without A's hypotheses, some other witness was
    reachable and the exchange bought nothing.
    """
    binders, goal = _split(statement)
    if "∃" not in goal:
        return ""
    keep = []
    cursor = 0
    for name, kind, start, end in _binders(binders):
        if re.search(r"[=≤<≥>∣≠∈∧∨↔→¬]|ModEq|Prime", kind):
            keep.append(binders[cursor:start])
            cursor = end
    keep.append(binders[cursor:])
    kept = " ".join("".join(keep).split())
    return f"theorem _probe {kept} : {goal}" if kept else f"theorem _probe : {goal}"


def weaken_unique(statement: str) -> str:
    """`∃!` demoted to `∃`.

    `goal_form_transplant` earns its keep when the shape forces work the parent's
    content did not. If the child still closes once uniqueness is dropped, the
    uniqueness half was the whole transplant, and the `exists_unique_wrapper`
    failure is that half being immediate from the hypotheses.
    """
    binders, goal = _split(statement)
    if "∃!" not in goal:
        return ""
    return f"theorem _probe {binders} : {goal.replace('∃!', '∃')}" if binders else (
        f"theorem _probe : {goal.replace('∃!', '∃')}"
    )


# `single_parent_probe` and `prop_of` lived here and are gone. They built
# `(hP : parent) (child binders) : child goal` and read Lean closing it as "the
# parent implies the child". Every certified child makes that statement true
# with `hP` unused, so it was provable for the whole corpus and reported prover
# strength. Both probes that remain in this module weaken the child instead --
# a dropped hypothesis, a stripped uniqueness step -- which is the only
# direction in which Lean's success carries information about redundancy.


#: Verdicts already reached for a parent statement, keyed on its text.
#:
#: The prover is not deterministic, and the same parent appears in many
#: crossovers: `nondivisible_square_mod_three` is behind five of them. Asking
#: separately gave `free` for one child and `not free` for another on identical
#: input, which is the one thing a gate cannot do -- two rows with the same
#: parent must get the same verdict. Caching also removes most of the calls,
#: since a scan sees far fewer distinct parents than rows.
_FREE_CACHE: Dict[str, bool] = {}


def _cache_key(statement: str) -> str:
    return " ".join(str(statement or "").split())


def clear_free_cache() -> None:
    _FREE_CACHE.clear()


async def _batch_provable(
    verifier: Callable[..., Awaitable[Any]],
    goals: Sequence[tuple],
    header: str,
    timeout: float,
) -> Dict[str, Optional[bool]]:
    """Try every goal in one file, then fall back to one file each.

    A single failing goal makes the whole batch report errors, so the batch is
    only trusted when it passes cleanly: that means every goal closed. Otherwise
    the goals are retried individually, which costs what the old path cost and
    only for the batches that had a failure in them.
    """
    head = str(header or "").strip() or "import Mathlib"
    blocks = []
    for index, (_name, goal) in enumerate(goals):
        if goal.strip():
            blocks.append(goal.replace("theorem _probe", f"theorem _probe{index}", 1) + " := by\n" + LADDER)
    if not blocks:
        return {name: None for name, _ in goals}
    combined = (
        f"{head}\nset_option autoImplicit false\nset_option maxHeartbeats 600000\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    if await _provable(verifier, combined, timeout):
        return {name: True for name, _ in goals}
    out: Dict[str, Optional[bool]] = {}
    for name, goal in goals:
        out[name] = await _provable(verifier, probe(header, goal), timeout)
    return out


async def check_parents(
    verifier: Callable[..., Awaitable[Any]],
    header: str,
    child_statement: str,
    parents: Sequence[Dict[str, str]],
    *,
    prover: Optional[Callable[..., Awaitable[Any]]] = None,
    mechanism: str = "",
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Which parents Lean can show were not needed.

    `universal_parents` names parents whose own statement the ladder proves with
    no hypotheses. `free_hypotheses` names child hypotheses the ladder can drop
    and still close the goal. Either one is evidence the crossover leans on one
    parent; neither is a verdict, and both are handed to the judge as such.
    """
    # No prover is needed here, and using one was a mistake worth recording.
    # A parent is a row this pipeline already certified, so its statement is
    # already proved. If that statement carries no hypotheses, what was proved is
    # an unconditional fact -- Mathlib-grade, available anywhere, and therefore
    # incapable of being the parent that made the child harder. The question is
    # answered by reading the binders, not by proving anything, and this probe is
    # eight of the nine redundancy findings so far.
    # Unconditional is necessary and not sufficient. `amc12_2000_p6` assumes
    # nothing and is still a competition problem Mathlib does not have: a child
    # that uses it has to do that work again, so calling it a free supplier was
    # wrong. What made the two known defects free was that their parent was
    # unconditional *and* fell out of a residue split -- available for the asking.
    #
    # So the syntactic test selects candidates and Lean decides. A parent the
    # ladder closes is genuinely free; one it cannot is a real fact, whatever its
    # binders look like.
    candidates = [
        parent
        for parent in parents
        if str(parent.get("statement") or "").strip()
        and is_unconditional(str(parent.get("statement")))
    ]
    universal: List[str] = []
    # Only positives are cached, and that asymmetry is the whole point. Finding a
    # proof settles the question; not finding one means the attempt failed, and
    # the attempt is a model that proved this same statement on one run and not
    # on the next. Caching a negative would freeze a coin-flip into a verdict,
    # which is exactly what happened when it did: two rows sharing a parent got
    # opposite answers, then the same wrong answer.
    known = [p for p in candidates if _FREE_CACHE.get(_cache_key(str(p.get("statement"))))]
    for parent in known:
        universal.append(str(parent.get("name") or ""))
    candidates = [p for p in candidates if p not in known]
    if candidates:
        verdicts = await _batch_provable(
            verifier,
            [
                (str(p.get("name") or f"p{i}"), without(str(p.get("statement")), []))
                for i, p in enumerate(candidates)
            ],
            header,
            timeout,
        )
        for i, p in enumerate(candidates):
            got = bool(verdicts.get(str(p.get("name") or f"p{i}")))
            if got:
                universal.append(str(p.get("name") or ""))
                _FREE_CACHE[_cache_key(str(p.get("statement")))] = True
        # The ladder is a filter for arithmetic, and "free" is wider than that.
        # `¬(3 ∣ n) ↔ n^2 ≡ 1 [ZMOD 3]` needs a residue split, which no fixed
        # tactic list produces, and it is the parent behind both known defects.
        # A prover is affordable here in a way it was not before: the syntactic
        # test has already cut the field to the handful of parents that assume
        # nothing, so this is about one call per crossover rather than eight.
        if prover is not None:
            for index, parent in enumerate(candidates):
                name = str(parent.get("name") or "")
                if name in universal:
                    continue
                # Retried, because the probe is one-sided: another attempt can
                # only add a finding, never remove one. A single try on the
                # parent behind both known defects succeeded roughly one run in
                # three.
                got = False
                for _ in range(int(os.getenv("REDUNDANCY_PROVER_TRIES", "3"))):
                    if await _llm_attempt(
                        verifier, prover, header, str(parent.get("statement")), (), timeout
                    ):
                        got = True
                        break
                if got:
                    _FREE_CACHE[_cache_key(str(parent.get("statement")))] = True
                    universal.append(name)

    probes = MECHANISM_PROBES.get(str(mechanism or ""), MECHANISM_PROBES[""])

    # The mechanism-independent probe is not run, and the reason is worth keeping.
    # "Is the child derivable from the other parent alone" is the right question
    # — it subsumes every mechanism — but handing the prover the child's own
    # binders answers a different one. A crossover often carries a parent's
    # hypothesis rather than its conclusion, so that parent's content is already
    # inside the goal being proved, and the probe reports "derivable without it"
    # for a row where it plainly was not. It flagged both known-good crossovers
    # on its first run.
    #
    # Making it sound needs the child's binders partitioned by which parent each
    # came from, which is the attribution problem the deterministic crossover
    # gate failed at before the judge replaced it. Until that is solved, the
    # probe would add false positives to a gate whose whole value is that it
    # never convicts without proof.
    derivable: List[str] = []

    # Witness and goal-shape probes. Neither is expressible as "drop a binder":
    # the contribution sits inside the conclusion, so the conclusion is what gets
    # weakened. Both stay one-sided -- Lean accepting the weakened form is
    # decisive, failing to prove it says nothing.
    shape: Dict[str, Any] = {}
    if "free_witness" in probes:
        goal = strip_to_existence(child_statement)
        if goal:
            code = probe(header, goal.replace("theorem _probe", "theorem _x", 1))
            got = await _provable(verifier, code, timeout)
            if not got:
                got = await _llm_attempt(verifier, prover, header, goal, (), timeout)
            shape["witness_reachable_without_parent"] = bool(got)
    if "forced_witness" in probes:
        goal = weaken_unique(child_statement)
        if goal:
            got = await _provable(verifier, probe(header, goal), timeout)
            if not got:
                got = await _llm_attempt(verifier, prover, header, goal, (), timeout)
            shape["existence_without_uniqueness"] = bool(got)

    free: List[str] = []
    measured = True
    if "droppable" not in probes:
        return {
            "measured": True,
            "mechanism": str(mechanism or ""),
            "universal_parents": universal,
            "free_hypotheses": [],
            "derivable_without": derivable,
            "shape": shape,
            "redundant": bool(universal or derivable)
            or bool(shape.get("witness_reachable_without_parent")),
        }
    # Batched: the probes are independent theorems, so they go in one file and
    # one Mathlib load answers all of them. Serially this was 3.4s of process
    # startup per hypothesis; together it is 3.4s for the set.
    names = hypothesis_names(child_statement)[:6]
    if names:
        results = await _batch_provable(
            verifier,
            [(name, without(child_statement, [name])) for name in names],
            header,
            timeout,
        )
        for name in names:
            if results.get(name) is None:
                measured = False
            elif results[name]:
                free.append(name)
    # The prover is a fallback for this probe only, and off unless asked: it is
    # the expensive half and it produced two of fourteen findings.
    if prover is not None and os.getenv("REDUNDANCY_USE_PROVER", "0") == "1":
        for name in [n for n in names if n not in free]:
            if await _llm_attempt(verifier, prover, header, child_statement, [name], timeout):
                free.append(name)

    return {
        "measured": measured,
        "mechanism": str(mechanism or ""),
        "universal_parents": universal,
        "free_hypotheses": free,
        "derivable_without": derivable,
        "shape": shape,
        "redundant": bool(universal or free or derivable),
    }


def brief(evidence: Dict[str, Any]) -> str:
    """The finding as one line for the judge's prompt, or empty."""
    if not evidence.get("measured"):
        return ""
    parts = []
    if evidence.get("universal_parents"):
        parts.append(
            "parents whose statement Lean proves with no hypotheses, so applying "
            "them to anything the other parent built is free: "
            + ", ".join(str(p) for p in evidence["universal_parents"])
        )
    if evidence.get("free_hypotheses"):
        parts.append(
            "hypotheses Lean can drop and still close the goal: "
            + ", ".join(evidence["free_hypotheses"])
        )
    if evidence.get("derivable_without"):
        parts.append(
            "parents Lean showed were not needed, because the child follows from "
            "the other one alone: " + ", ".join(evidence["derivable_without"])
        )
    shape = dict(evidence.get("shape") or {})
    if shape.get("witness_reachable_without_parent"):
        parts.append(
            "the existential closes without the supplying parent, so the witness "
            "it exchanged was not the only one reachable"
        )
    if shape.get("existence_without_uniqueness"):
        parts.append(
            "existence alone is provable here, so the transplanted `∃!` shape rests "
            "entirely on a uniqueness step -- check that step is not immediate"
        )
    return "; ".join(parts)


async def check_mutation(
    verifier: Callable[..., Awaitable[Any]],
    header: str,
    child_statement: str,
    parent_statement: str,
    *,
    prover: Optional[Callable[..., Awaitable[Any]]] = None,
    variant: str = "",
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Whether the child carries hypotheses its own proof never needed.

    This asked a different question until a full scan of the release exposed it
    as unanswerable. The probe was `(hP : parent) (child binders) : child goal`,
    read as "the parent alone implies the child". It does not say that. Every
    row reaching this point is a certified theorem, so `child binders ⊢ child
    goal` already closes, and adding `hP` cannot take that away: the statement
    is provable for every certified child regardless of what the parent says,
    and `hP` need never be used. A control run substituting `hP : True` -- a
    hypothesis with no mathematics in it at all -- closed four of six rows the
    probe had just flagged. What it measured was prover strength.

    The differential form does not rescue it either. If the parent version
    closes where `True` does not, the parent was a useful lemma -- which is what
    every mutation's parent is by construction. Building on the parent is the
    operator working, not failing.

    What survives is the probe crossover already relies on, and it survives for
    a structural reason: it *weakens* the child instead of strengthening the
    context. Deleting a hypothesis makes the statement strictly harder, so Lean
    closing it afterwards is a fact about the child that nothing else can
    explain -- the hypothesis was decoration. That stays one-sided in the usual
    direction: a drop that fails to prove has shown nothing.

    `mutation_silent` is exempt. It is equivalent to its parent on purpose, that
    equivalence is proved in both directions elsewhere, and flagging it here
    would report the operator working as designed.
    """
    if str(variant or "") == "mutation_silent":
        return {"measured": False, "why": "silent mutations are equivalent by design"}
    if not str(child_statement or "").strip():
        return {"measured": False, "why": "no statement to probe"}

    names = hypothesis_names(child_statement)[:6]
    if not names:
        return {"measured": True, "free_hypotheses": [], "redundant": False}

    free: List[str] = []
    measured = True
    results = await _batch_provable(
        verifier,
        [(name, without(child_statement, [name])) for name in names],
        header,
        timeout,
    )
    for name in names:
        if results.get(name) is None:
            measured = False
        elif results[name]:
            free.append(name)
    if prover is not None and os.getenv("REDUNDANCY_USE_PROVER", "0") == "1":
        for name in [n for n in names if n not in free]:
            if await _llm_attempt(verifier, prover, header, child_statement, [name], timeout):
                free.append(name)

    return {
        "measured": measured,
        "free_hypotheses": free,
        "redundant": bool(free),
        "why": (
            "hypotheses the child states but its goal does not need, so the mutation "
            "added assumptions rather than difficulty"
            if free
            else ""
        ),
    }
