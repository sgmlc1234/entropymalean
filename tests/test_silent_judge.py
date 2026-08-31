"""The silent tier gets its own prompt, and its own question.

Judged under the mutation rubric, two of four silent rows in one four-generation
run came back wrong in the same direction. A child that restated Pascal's rule in
successor form and dropped all three of its parent's bounds was passed, with the
drop named in the judge's own reasoning -- because under a rubric about
difference, a dropped hypothesis reads as the child being more general. A child
that removed an alias its own parent had introduced was called "materially
different" from a statement it is provably equivalent to.

Neither is the judge being careless. Both are correct readings of the wrong
question.
"""

from __future__ import annotations

import pytest

from src.certification.hypothesis_preservation import compare, hypotheses
from src.certification.judges import mutation_prompt, silent_prompt

PARENT = "theorem p (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₁ : k ≤ n) : Nat.choose n k = 1"
CHILD = "theorem c (p q : ℕ) : Nat.choose (p + 1) (q + 1) = 1"


def prompt(**kwargs) -> str:
    return silent_prompt(PARENT, "by simp", CHILD, "by simp", **kwargs)


def test_the_silent_prompt_asks_for_sameness_not_difference():
    text = prompt()
    assert "SAME mathematics" in text
    assert "judged on how little" in text


def test_recall_is_not_offered_as_a_silent_failure():
    """`recall` is the measurement, not a defect, and naming it means the
    question was misread. One run rejected a silent row with that label."""
    text = prompt()
    assert "`recall` is not a silent failure mode" in text
    for name in ("not_equivalent", "alias_only", "decoration", "wrong_tier"):
        assert name in text


def test_the_silent_prompt_does_not_inherit_the_mutation_decision_rule():
    """The shared rule says keep when the child demands reasoning the parent's
    proof does not supply, which a silent mutation never does by design."""
    silent = prompt()
    assert "demands reasoning the parent's proof does not\nsupply" not in silent
    assert "surface a solver" in silent


def test_a_thin_case_is_not_kept_by_default_here():
    """Every other tier falls back on `keep`; this one must not, because the
    operator's whole claim is that difficulty did not move."""
    assert "do not fall back on `keep`" in prompt()
    assert "corpus loses more from discarding good problems" in mutation_prompt(
        PARENT, "by simp", CHILD, "by simp"
    )


def test_the_plan_and_siblings_still_reach_the_silent_judge():
    text = prompt(
        plan={"operator_variant": "mutation_silent", "operator_goal": "restate in successor form"},
        siblings=[{"problem_id": "sib", "statement": "theorem s : True", "quality": "strong"}],
    )
    assert "successor form" in text
    # The sibling block carries each kept child's statement, not its id — the
    # judge is being asked to recognise a repeat, and an identifier does not
    # tell it anything about the mathematics.
    assert "theorem s : True" in text
    assert "already KEPT this run" in text


# --------------------------------------------------------------- preservation

PRESERVING = [
    ("rename only", "theorem p (n : ℕ) (h : 0 < n) : n % 3 = 0",
                    "theorem c (m : ℕ) (hm : 0 < m) : 3 ∣ m"),
    ("notation swap", "theorem p (a n r : ℕ) (h : a % n = r) : True",
                      "theorem c (x y z : ℕ) (hh : x % y = z) : True"),
    ("reordered hypotheses", "theorem p (n : ℕ) (h1 : 0 < n) (h2 : n < 9) : True",
                             "theorem c (n : ℕ) (h2 : 0 < n) (h1 : n < 9) : True"),
]

CHANGING = [
    ("bound dropped", "theorem p (n : ℕ) (h : 0 < n) : True", "theorem c (n : ℕ) : True"),
    ("hypothesis added", "theorem p (n : ℕ) : True", "theorem c (n : ℕ) (h : 0 < n) : True"),
    ("the released case", PARENT, CHILD),
]


@pytest.mark.parametrize("label,parent,child", PRESERVING, ids=[c[0] for c in PRESERVING])
def test_a_legitimate_re_encoding_preserves_its_hypotheses(label, parent, child):
    assert compare(parent, child)["preserved"] is True


@pytest.mark.parametrize("label,parent,child", CHANGING, ids=[c[0] for c in CHANGING])
def test_a_changed_hypothesis_set_is_reported(label, parent, child):
    result = compare(parent, child)
    assert result["preserved"] is False
    assert result["why"]


def test_a_datum_binder_is_not_counted_as_a_hypothesis():
    assert "n" not in hypotheses("theorem p (n : ℕ) : True")
    assert hypotheses("theorem p (n : ℕ) (h : 0 < n) : True")
