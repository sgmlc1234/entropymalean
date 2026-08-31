"""Two model judges that decide whether a bred problem is worth keeping.

The deterministic gates in `novelty.py` measure real things, and several of them
turned out to measure the wrong thing. Their record, on the 245-row miniF2F
campaign and the samples read from it:

  decorative_mutation    the one that held up. Closing-lemma identity caught the
                         two rows a reader had independently called decoration.
  parallel_crossover     broken. 53% of its verdicts came from proofs where
                         attribution found 0 or 1 parent, so depth-1 was
                         unreachable by construction; of ten flagged rows read
                         by hand, five were genuine fusions. It also passes
                         `hbridge : A ∧ B` as a meeting point and `4 * 900 =
                         3600` as coupling.
  new_literal_bound      noisy. The bounds it flagged were 10, 16, 18 — the
                         problem's own constants, not `k ≤ 100` tractability
                         caps.
  exists_unique_wrapper  syntactic. Fires on legitimate uniqueness claims.
  parent_proof_embedded  plausible, unvalidated, 8 rows.

What separates a redecoration from a variation, or an adhesive from a bridge, is
what the statement *means* — whether `hbridge : A ∧ B` joins two arguments or
merely staples two answers. No amount of graph parsing reaches that, and five
successive heuristics failed at it. So the measurements stay, as evidence handed
to a reader, and the verdict moves to a model.

The judges are deliberately separate. A mutation has one parent and fails by
growing notation; a crossover has two and fails by keeping them apart. Sharing a
prompt would blur both failure catalogues into generalities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

JUDGE_SYSTEM = (
    "You review problems bred from benchmark seeds for a Lean 4 evaluation set. "
    "The set exists to test whether a prover reasons or recalls, so a child that "
    "a model could solve by remembering its parent is worthless to it, however "
    "different the child looks. Judge the mathematics, not the presentation. "
    "Return JSON only."
)

_MUTATION_RUBRIC = """A mutation takes ONE parent and should produce a problem whose proof is not the
parent's proof. These are the ways that succeeds, observed in this corpus:

  parameterise   a constant becomes a variable and the argument must survive it
                 (`x^2 - 6x + 13` has least value 4  ->  `(x-t)^2 + (t+1)` has
                 least value `t+1`)
  generalise     one instance becomes the law (`logb 3 27 = 3`  ->
                 `logb 3 (3^n) = n`)
  change shape   the conclusion becomes a different kind of claim (a value
                 becomes a uniqueness statement; a minimum becomes a
                 characterisation of the whole range; divisibility becomes a
                 congruence)
  weaken hypotheses  the premise is loosened and the conclusion still holds
                 (`c = I`  ->  `c^2 = -1`)
  strengthen     the modulus, exponent or bound moves so a new case split is
                 needed (`8 | n^2-1`  ->  `16 | a^4-b^4`)

And these are the ways it fails, all of which type-check and all of which were
produced by this pipeline:

  decoration     the statement grows, the proof does not. A coefficient is
                 enlarged (`x*y` -> `x*(q^2+r^2+1)`) and one extra step
                 discharges its nonzero-ness. The proof still ends by applying
                 the parent's final lemma.
  exists-unique wrapper   the parent's conclusion is wrapped as
                 `∃! x, P x ∧ (parent's equation) = x`, where the witness is
                 forced immediately. New tactics appear; no new mathematics
                 does. NOTE: a uniqueness claim that must actually be proved is
                 a legitimate change of shape — the wrapper is only the case
                 where the witness falls out of the hypotheses.
  conjunction    the parent's conclusion is kept and trivial side facts are
                 appended (`a ∈ [a-r, a+r] ∧ ...` given `0 ≤ r`).
  tractability bound   a ceiling appears that the parent did not have, so a
                 finite check closes what was an open claim (`hk : k ≤ 100`).
                 A bound inherited from the parent, or intrinsic to the problem,
                 is not this.
  weakening      the parent's hard hypotheses are kept and the conclusion is
                 made easier.
  recall         the parent's proof appears inside the child's, so remembering
                 the parent solves the child.

The PLAN names the tier, and the tier changes what counts as success. Judging an
`easy` slot by the `hard` standard rejects children that did exactly what they
were asked:

  mutation_hard    must demand reasoning the parent's proof does not supply. A
                   child that is merely different is not enough here; something
                   in the parent's argument has to break.
  mutation_easy    must not be solvable by recalling the parent, and that is the
                   whole bar. It is explicitly not asked to be harder -- its
                   instruction prefers a certifiable child over an aggressive
                   one. A child that needs a different lemma, or reaches the
                   same kind of claim by a different route, passes even if it is
                   shorter than its parent. Reject it for decoration, for a
                   forced witness, for an unused hypothesis -- not for being
                   easy.
  mutation_silent  is meant to be the same mathematics in a different Lean
                   surface. Do not judge it for failing to add anything; that is
                   the operator working, and every difficulty test belongs to
                   the other two tiers. You are asked two things instead.
                   First, is it the same mathematics? A child that drops a
                   conjunct, weakens a bound, or states something else true
                   about the same objects is not a silent mutation, it is a
                   different and usually easier problem. Second, is the surface
                   changed enough to matter -- would a solver that had memorised
                   the parent's exact form fail to recognise it, while a solver
                   that understood the parent would not? A rename, a reordered
                   conjunction, or an unfolding Lean does definitionally is too
                   little.
                   You may be shown an equivalence probe. Read it as a weak
                   signal only: it asks Lean to derive each statement from the
                   other, and because both are theorems the tactic block can
                   close either direction without using the hypothesis at all.
                   It has passed a child that dropped a conjunct and a child
                   that was unrelated. A failed probe is worth something; a
                   passed one establishes nothing you should not check yourself."""

_SILENT_RUBRIC = """A silent mutation takes ONE parent and should produce the SAME mathematics in
different Lean. Every other tier is judged on how much the child differs from its
parent. This one is judged on how little, and the two questions are opposite
enough that reading it against the mutation rubric gets it backwards: a dropped
hypothesis reads there as the child being more general, and here it means the
child is a different theorem.

Ask two things, in this order.

  1. Is it the same mathematics? The child must hold exactly where the parent
     holds and nowhere else. A child that drops a hypothesis is more general, a
     child that adds one is weaker, a child that concludes something else true
     about the same objects is neither -- and none of the three is silent, no
     matter how good a problem it is on its own. Say so and reject; the slot can
     be replanned as a mutation.

  1b. A concrete re-telling -- the theorem restated as a question about ages,
     coins, distances, a shop's stock -- is the strongest form this operator
     has, not a gimmick. Judge it on the same two questions and nothing else:
     the setting must add and remove no condition (a story that needs a
     quantity positive where the theorem does not say so is `not_equivalent`),
     and the Lean must have moved as well, since renaming binders after the
     story's characters is invisible to everything downstream.

  2. Is the surface changed enough to be worth releasing? The point of the
     operator is to hold difficulty fixed so that a drop in solver performance
     can only be the surface. That only works if the surface actually moved.
     A rename, a reordered conjunction, an unfolding Lean does definitionally,
     a double negation, or wrapping the goal in `¬¬` is not a re-encoding: it is
     the same text with noise on it.

What you must NOT hold against it:

  - that a solver who memorised the parent is helped. That is the measurement,
    not a defect. `recall` is not a silent failure mode and naming it here means
    the question was misread.
  - that the child's proof reuses the parent's argument. It should. A silent
    mutation that needed a new argument would have changed the mathematics.
  - that it adds nothing. It is not supposed to.

Failure names for this tier:

  not_equivalent   a hypothesis was dropped or added, or the claim's scope moved
  alias_only       the only change is naming a subexpression, and the statement
                   is otherwise the parent's; harmless once, but a later slot
                   that removes the alias lands back on the parent
  decoration       double negation, reordered independent conjuncts, a rename,
                   a definitional unfolding -- the surface did not really move
  wrong_tier       a good problem that is not a silent mutation; say what tier
                   it belongs to
  transcribed      the statement names a Lean construct instead of saying the
                   mathematics, or reads as the Lean translated word by word
  inaccurate       the prose does not describe the theorem the Lean states

You may be shown a `hypothesis_preservation` measurement. It compares the
parent's and child's binders after alpha-normalisation and is a parser's answer,
not a judgement: a named drop is decisive, an empty result is not evidence of
anything."""


_CROSSOVER_RUBRIC = """A crossover takes TWO parents and should produce a problem that needs both.
These are the ways that succeeds here:

  pipeline       one parent's conclusion becomes an input to the other's
                 derivation (`3 + 1/x = 7/x` forces `x = 2`, and that `x` is
                 then substituted into the other parent's inequality)
  incompatibility   the two conclusions are shown to be mutually exclusive
                 (one parent forces `x = 18`, the other `x = 53`; the child
                 proves no `x` satisfies both)
  specialise and invert   one parent is instantiated at a value the other
                 supplies, and their combination yields an iff
                 (`p | n ↔ n^2 ≡ 0 [p]` at p=3, plus `a^2 ≡ 0 ∨ 1 [3]`, gives
                 `¬(3|n) ↔ n^2 ≡ 1 [3]`)
  structural transfer   one parent's construction (an index set, a product
                 range, a recurrence) reshapes the other's statement

And these are the ways it fails:

  arithmetic on answers   the two parents' results are multiplied or added
                 (`4 * 900 = 3600`). The parents are present as numbers, not as
                 arguments.
  conjunction    the two statements are joined by `∧`, or a `have` of the form
                 `A ∧ B` is presented as the point where they meet.
  constant supplier   one parent exists only to fix a numeral used by the other,
                 or only to show a coefficient is nonzero / a set is nonempty.
  empty domain   the combined constraints admit nothing, so a `∀ y ∈ S` in the
                 conclusion holds for no `y` at all. The hypotheses can be
                 perfectly consistent while this happens — check whether the
                 set the conclusion quantifies over can contain anything.
  parallel       each parent is discharged on its own and the results are
                 combined once, at the last line, with nothing flowing between
                 them.
  repeated device   the same pair of parents combined the same way as a row this
                 run already kept. Two children of one pair that share their
                 opening lemma and differ only in the equation that couples them
                 are one problem written twice. Judge this against the siblings
                 listed below, not only against the parents: a pipeline that
                 finds one good fusion and then re-emits it produces a corpus
                 whose size overstates its content.
  universal supplier   one parent's statement is true with no hypotheses at all,
                 so it is a fact Mathlib already provides. Applying it to
                 something the other parent constructed is not interaction, and
                 a shared variable between two conjuncts is not either: ask
                 whether the conjunct would still hold for an arbitrary value in
                 that position. A certified child concluded
                 `∃ q, 5^(n+6) - 5^n = 7q ∧ (¬(3 ∣ q) ↔ q^2 ≡ 1 [ZMOD 3])`,
                 and the second conjunct holds for every integer, so the child
                 says exactly what its first parent said."""

_EVIDENCE_NOTE = """The measurements below come from a parser and are advisory. Their known
failure modes, so you can discount them correctly:

  coupling_depth 0 means no `have` was found carrying material from both
  parents. When `attributed_nodes` is 0 or 1 the parser could not tell the
  parents apart at all, and depth 0 then says nothing — read the proof yourself.
  Depth 1 is also not proof of fusion: an `A ∧ B` step satisfies it.

  literal bounds are reported without judging whether they are intrinsic.

  closing_lemma_match is the strongest single signal for decoration.

  redundancy is Lean's answer, not a parser's, and it is one-sided. A named
  `universal_parent` means Lean accepted a proof of that parent's statement with
  every hypothesis removed; a named `free_hypothesis` means Lean accepted a proof
  of this child without it. Either is decisive against the parent -- do not argue
  around it. An empty result means nothing was found, not that nothing is there.

Weigh them as a reader would weigh a colleague's notes."""



_PROSE_RUBRIC = """The STATEMENT is the released problem, not a caption for the Lean. It is what a
reader meets first and the only part of a row most people will read, so it is
judged, not skimmed. Three things make it wrong:

  transcribed    a Lean name survives in the prose -- `Int.gcd b x`,
                 `Nat.choose n k`, `Set.univ`, `IsExtrOn`, a hypothesis label.
                 The mathematics has to be said in mathematics: "the greatest
                 common divisor of $b$ and $x$".
  unreadable     every binder packed into one universally quantified English
                 sentence, with no notation. The source benchmarks write "What
                 is the remainder when 2003 is divided by 11?"; a released
                 statement should read like its parent, terse and often as a
                 question, with symbols in $...$.
  inaccurate     the prose claims something the Lean does not, or omits a
                 hypothesis the Lean has. This is the serious one: it makes the
                 row wrong rather than ugly.

You may be shown a `goal_roundtrip` measurement. Lean elaborated the statement,
a model that saw only that Lean wrote it back as prose, and a third compared the
two texts. `prose_matches_the_elaborated_goal: false` is strong evidence of the
third failure -- read the round-trip's own prose against the statement and say
which one is right. A missing measurement is not evidence either way.

A statement that is merely ugly is a `weak`, not a rejection, unless the row has
nothing else going for it. A statement that is inaccurate is a rejection whatever
the mathematics underneath is worth."""


_OUTPUT = """Return exactly this JSON object and nothing else:

{"verdict": "keep" | "reject",
 "quality": "strong" | "acceptable" | "weak",
 "failure": "" | one of the failure names above, in snake_case,
 "reason": "one or two sentences naming the specific mathematics that decides it",
 "retry_plan": "" | "one or two sentences naming the concrete change to make",
 "fix_scope": "" | "rewrite" | "replan"}

Judge `keep` when the child demands reasoning the parent's proof does not
supply. Judge `reject` when a solver who had memorised the parent would be
substantially helped. When the evidence genuinely does not settle it, answer
`keep` with quality `acceptable` and say what you could not determine — a
corpus loses more from discarding good problems than from admitting mediocre
ones, because the gate runs before anyone has seen them.

On `reject`, fill `retry_plan` and `fix_scope`.

`retry_plan` names the change in this row's own mathematics. There is a generic
corrective for every failure name, and it will be appended, so do not restate
it: write the sentence only you can write, having read this proof. If you found
that a parent's result was avoidable, say which claim should depend on it
instead; if a constant collapsed an expression, say which quantity should have
stayed symbolic.

`fix_scope` is decided against THE PLAN shown above, not against the child
alone. The question is whether the instruction was sound and went unmet, or
whether the instruction itself is what produced the failure.
  `rewrite` — the plan asked for something substantive and the child did not
  deliver it. The next attempt can succeed from the same plan and the same
  parents.
  `replan`  — the child did what the plan asked, and the plan is the defect.
  Choose this when the plan itself specifies a change you would reject on
  sight — swapping an index type, renaming a bound, restating a conclusion —
  or when it pairs parents that have no point of contact, or when your own
  retry_plan would need a parent this slot was not given.

Do not default to `rewrite`. A child that obeyed its instructions is not a
generation failure, and sending it back to be rewritten against the same
instruction produces the same row a second time."""


_PLAN_FIELDS = (
    "op_type",
    "operator_variant",
    "goal",
    "operator_goal",
    "variation_axis",
    "composition_pattern",
    "parent_contributions",
    "fusion_contract",
    "required_checkpoints",
    "quality_target",
)


_SILENT_OUTPUT = """Return exactly this JSON object and nothing else:

{"verdict": "keep" | "reject",
 "quality": "strong" | "acceptable" | "weak",
 "failure": "" | one of the failure names above, in snake_case,
 "reason": "one or two sentences naming the specific mathematics that decides it",
 "retry_plan": "" | "one or two sentences naming the concrete change to make",
 "fix_scope": "" | "rewrite" | "replan"}

Judge `keep` when the child states the parent's theorem in a surface a solver
who had memorised the parent's exact form would not recognise, while a solver
who understood it would. Judge `reject` when the mathematics moved, or when the
surface did not.

`strong` is a re-encoding that a reader has to think about to see is the same
claim. `acceptable` is a real but shallow re-encoding. `weak` means it should
not be released as a silent mutation.

Unlike the other tiers, do not fall back on `keep` when the evidence is thin.
A silent mutation whose equivalence you cannot convince yourself of is exactly
the row that must not be released, because the whole operator rests on the
claim that difficulty is unchanged."""


def _sibling_block(siblings: Optional[Sequence[Dict[str, Any]]]) -> str:
    """Children already kept this run from an overlapping pair of parents.

    The judge sees a child against its parents and nothing else, which makes one
    failure invisible to it by construction: two rows from the same pair, built
    the same way. It happened -- `coupled_recurrences_inconsistent` and
    `incompatible_terminal_coupling` share a parent pair, share fifteen lines of
    opening proof, differ in one coupling hypothesis, and were both judged
    `strong` because neither was ever shown the other.
    """
    if not siblings:
        return ""
    lines = []
    for row in siblings[:5]:
        lines.append(
            f"- [{row.get('quality') or 'kept'}] {_clip(row.get('statement'), 260)}"
        )
    return (
        "Children already KEPT this run from parents overlapping this one. If this "
        "child is another way of writing one of them -- same pair, same device, a "
        "different coupling equation -- it is a repeat, whatever its own merits:\n"
        + "\n".join(lines)
        + "\n"
    )


def _plan_block(plan: Optional[Dict[str, Any]]) -> str:
    """What the slot was instructed to produce.

    Without it the judge can see that a child is a restatement but not who is
    answerable for that. Measured over 41 generated rows across two groups, 15
    of 17 rejections were routed back to the generator, including one where the
    plan had said in as many words "replace the fixed `Fin n` indexing surface
    by a bounded finite index type" — the generator complied exactly, the judge
    named the result `recall`, and the retry went to the party that had done
    what it was told. A plan that specifies a trivial transformation cannot be
    fixed by writing it again.
    """
    if not plan:
        return (
            "PLAN: not available. Judge the child on its own terms and leave "
            "fix_scope as `rewrite`.\n"
        )
    kept = {
        key: plan[key]
        for key in _PLAN_FIELDS
        if plan.get(key) not in (None, "", [], {})
    }
    return (
        "PLAN the slot was given (the instruction the child was written "
        "against):\n" + json.dumps(kept, ensure_ascii=False, indent=1) + "\n"
    )


def _clip(text: Any, limit: int) -> str:
    body = str(text or "").strip()
    return body if len(body) <= limit else body[:limit] + "\n… (truncated)"


def mutation_prompt(
    parent_statement: str,
    parent_proof: str,
    child_statement: str,
    child_proof: str,
    evidence: Optional[Dict[str, Any]] = None,
    precedents: str = "",
    plan: Optional[Dict[str, Any]] = None,
    siblings: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    return f"""{_MUTATION_RUBRIC}

{_plan_block(plan)}
{_sibling_block(siblings)}
PARENT statement:
{_clip(parent_statement, 1200)}

PARENT proof:
{_clip(parent_proof, 1800)}

CHILD statement:
{_clip(child_statement, 1200)}

CHILD proof:
{_clip(child_proof, 2600)}

Measured evidence:
{json.dumps(evidence or {}, ensure_ascii=False, indent=1)}

{_EVIDENCE_NOTE}

{_PROSE_RUBRIC}

{precedents}
{_OUTPUT}"""


def silent_prompt(
    parent_statement: str,
    parent_proof: str,
    child_statement: str,
    child_proof: str,
    evidence: Optional[Dict[str, Any]] = None,
    precedents: str = "",
    plan: Optional[Dict[str, Any]] = None,
    siblings: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """The silent tier gets its own prompt, not a paragraph inside mutation's.

    Judged under the mutation rubric, two of four silent rows in one run were
    misread in the same direction: a child that dropped all three of its
    parent's bounds was passed with the drop named in the judge's own reasoning,
    and a child that merely removed an alias its parent had introduced -- and is
    provably equivalent to its grandparent -- was called materially different.
    Both readings are correct for a rubric about difference and wrong for this
    tier.
    """
    return f"""{_SILENT_RUBRIC}

{_plan_block(plan)}
{_sibling_block(siblings)}
PARENT statement:
{_clip(parent_statement, 1200)}

PARENT proof:
{_clip(parent_proof, 1800)}

CHILD statement:
{_clip(child_statement, 1200)}

CHILD proof:
{_clip(child_proof, 2600)}

Measured evidence:
{json.dumps(evidence or {}, ensure_ascii=False, indent=1)}

{_EVIDENCE_NOTE}

{_PROSE_RUBRIC}

{precedents}
{_SILENT_OUTPUT}"""


def crossover_prompt(
    parents: Sequence[Dict[str, str]],
    child_statement: str,
    child_proof: str,
    evidence: Optional[Dict[str, Any]] = None,
    precedents: str = "",
    plan: Optional[Dict[str, Any]] = None,
    siblings: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    blocks = []
    for index, parent in enumerate(parents[:2], 1):
        blocks.append(
            f"PARENT {index} ({parent.get('name', '?')}) statement:\n"
            f"{_clip(parent.get('statement'), 900)}\n\n"
            f"PARENT {index} proof:\n{_clip(parent.get('proof'), 1200)}"
        )
    return f"""{_CROSSOVER_RUBRIC}

{_plan_block(plan)}
{_sibling_block(siblings)}
{chr(10).join(blocks)}

CHILD statement:
{_clip(child_statement, 1400)}

CHILD proof:
{_clip(child_proof, 3000)}

Measured evidence:
{json.dumps(evidence or {}, ensure_ascii=False, indent=1)}

{_EVIDENCE_NOTE}

{_PROSE_RUBRIC}

{precedents}
{_OUTPUT}"""


_VALID_VERDICT = {"keep", "reject"}
_VALID_QUALITY = {"strong", "acceptable", "weak"}
_VALID_SCOPE = {"rewrite", "replan"}


def parse_verdict(raw_text: str) -> Dict[str, Any]:
    """Read the judge's JSON, or record that it did not produce any.

    An unparseable answer is not a rejection. The row keeps its place and the
    failure is recorded against the judge, because a gate that silently drops
    rows when its own output is malformed corrupts the corpus in the direction
    that is hardest to notice.
    """
    match = re.search(r"\{.*\}", str(raw_text or ""), re.S)
    if not match:
        return {"verdict": "keep", "quality": "acceptable",
                "failure": "", "reason": "", "judge_error": "no JSON in reply"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return {"verdict": "keep", "quality": "acceptable",
                "failure": "", "reason": "", "judge_error": f"bad JSON: {error}"}
    verdict = str(data.get("verdict") or "").strip().lower()
    quality = str(data.get("quality") or "").strip().lower()
    scope = str(data.get("fix_scope") or "").strip().lower()
    return {
        "verdict": verdict if verdict in _VALID_VERDICT else "keep",
        "quality": quality if quality in _VALID_QUALITY else "acceptable",
        "failure": str(data.get("failure") or "").strip(),
        "reason": str(data.get("reason") or "").strip()[:400],
        "retry_plan": str(data.get("retry_plan") or "").strip()[:400],
        # Absent or unrecognised means the generator tries again on the same
        # plan. Defaulting the other way would let one unparsed field discard a
        # parent pairing that was never shown to be unworkable.
        "fix_scope": scope if scope in _VALID_SCOPE else "rewrite",
        "judge_error": "" if verdict in _VALID_VERDICT else "verdict missing or unknown",
    }

#: What to do instead, keyed by the failure the judge named. The judge's own
#: `reason` says what went wrong on *this* row; these say what would be
#: different next time. A retry given only the reason tends to repair the
#: sentence rather than the mathematics — it renames the coefficient it was told
#: was decorative — so the corrective has to name the move, not just the fault.
_CORRECTIVE = {
    "decoration": (
        "Change what must be proved. Move the conclusion to a different kind of "
        "claim, strengthen a modulus or exponent so a new case split is forced, "
        "or turn a constant into a second variable the parent's argument cannot "
        "absorb. Do not enlarge a coefficient or add a factor whose nonzero-ness "
        "is immediate."
    ),
    "exists_unique_wrapper": (
        "If uniqueness is the point, make it something to prove: choose a "
        "setting where more than one candidate is plausible and the argument has "
        "to rule the others out. Do not wrap an equation whose witness the "
        "hypotheses already fix."
    ),
    "conjunction": (
        "Drop the appended facts and change the single claim instead. A "
        "conjunct that follows from the hypotheses in one step adds length, not "
        "difficulty."
    ),
    "tractability_bound": (
        "Remove the ceiling and prove the general case, or replace the bound "
        "with a hypothesis that carries mathematical content rather than "
        "shrinking the search. A cap that lets a finite check close the goal "
        "changes the problem into a different, easier one."
    ),
    "weakening": (
        "Keep the strength of the conclusion. If the parent's hypotheses are "
        "hard to satisfy, either derive more from them or relax them — do not "
        "keep them and ask for less."
    ),
    "recall": (
        "The parent's proof appears inside yours, so remembering the parent "
        "solves your problem. Restate the target so the parent's proof no longer "
        "applies as written."
    ),
    "arithmetic_on_answers": (
        "The parents appear as numbers, not as arguments. Make one parent's "
        "conclusion an input the other's derivation consumes: a bound, a "
        "modulus, an index, a case distinction."
    ),
    "constant_supplier": (
        "One parent only fixes a numeral or discharges a side condition. Give it "
        "a role the other parent's argument depends on structurally."
    ),
    "empty_domain": (
        "The combined constraints admit nothing, so the conclusion holds "
        "vacuously. Exhibit a concrete witness satisfying every hypothesis and "
        "state it in proof_plan; if none exists, relax a constraint."
    ),
    "parallel": (
        "Each parent is discharged separately and the results are joined at the "
        "end. Restructure so that a step of the proof needs both at once."
    ),
}


def judge_brief(verdict: Dict[str, Any]) -> str:
    """The paragraph handed to the generator on its one retry.

    Carries three things: what was judged wrong, the judge's own words about
    *this* row, and the move that would answer it. The middle part is why the
    judge exists — a deterministic gate can say `parallel_crossover`, but only a
    reader can say which two things failed to meet and where.
    """
    if verdict.get("verdict") != "reject":
        return ""
    failure = str(verdict.get("failure") or "").strip()
    reason = str(verdict.get("reason") or "").strip()
    plan = str(verdict.get("retry_plan") or "").strip()
    corrective = _CORRECTIVE.get(failure, "")
    parts = ["REJECTED by problem-quality review."]
    if failure:
        parts.append(f"Failure: {failure}.")
    if reason:
        parts.append(f"Finding: {reason}")
    # The row's own plan leads. The corrective below is selected by failure
    # name, so every `constant_supplier` row was getting the same sentence
    # regardless of which constant collapsed which expression; it stays as the
    # floor when the judge wrote no plan, not as the whole instruction.
    if plan:
        parts.append(f"Required change: {plan}")
        if corrective:
            parts.append(f"In general: {corrective}")
    elif corrective:
        parts.append(f"Required change: {corrective}")
    else:
        parts.append(
            "Required change: make the child demand reasoning its parent's proof "
            "does not supply, rather than restating the parent at greater length."
        )
    return " ".join(parts)

