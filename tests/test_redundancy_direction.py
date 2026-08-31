"""A redundancy probe must delete something, never add something.

The mutation probe used to build `(hP : parent) (child binders) : child goal`
and read Lean closing it as "the parent implies the child". Every certified
child makes that statement true with `hP` unused, so it was provable for the
whole corpus and reported prover strength: over the release it called 66 of 165
rows redundant, and a control substituting `hP : True` reproduced four of the
first six findings.

The failure was not a bad threshold, it was a direction. Adding a hypothesis
can only make a proof easier, so no probe that hands the prover more context
can convict a row. What these tests pin is that property, not the wording: a
probe is admissible only if what it gives Lean is strictly harder than what the
row claims.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from src.certification import redundancy
from src.certification.redundancy import check_mutation, without

PARENT = "theorem p (n : ℕ) (h : 0 < n) : n ^ 2 + n ≤ n !"
CHILD = "theorem c (n : ℕ) (h₀ : 4 ≤ n) (hspare : 0 < n) : n ^ 2 + n ≤ n !"


class FakeVerifier:
    """Lean, standing in. Closes exactly the statements it is told to close.

    `_provable` reads `errors`, not `ok`; a stub without that attribute reports
    every probe as closing and the whole suite passes vacuously.
    """

    def __init__(self, closes=None):
        self.closes = closes or (lambda code: False)
        self.seen = []

    async def __call__(self, code, *args, **kwargs):
        self.seen.append(code)
        ok = bool(self.closes(code))
        return type("R", (), {"errors": None if ok else ["probe did not close"],
                              "system_error": None})()


def drops_only(name: str):
    """Closes the single-goal file in which `name` is the hypothesis removed.

    The batch puts every deletion in one file, so a marker taken from one probe
    also appears in the batch; keying on the goal count is what separates the
    individual retry from it.
    """
    def predicate(code: str) -> bool:
        return code.count("theorem _probe") == 1 and name not in code
    return predicate


def _body(function) -> str:
    """Source with the docstring removed -- what the function does, not what it
    says about what it used to do."""
    source = inspect.getsource(function)
    doc = inspect.getdoc(function) or ""
    for line in doc.splitlines():
        source = source.replace(line, "")
    return source


def test_the_probe_never_offers_the_parent_as_an_extra_hypothesis():
    """The exact shape that could not fail. A parent binder named in the code
    the probe builds means the question is answerable by the child's own proof."""
    source = _body(check_mutation)
    assert "hP" not in source
    assert "prop_of" not in source
    # The two helpers that existed only to build that shape are gone, so a
    # future edit cannot reach for them by name.
    assert not hasattr(redundancy, "single_parent_probe")
    assert not hasattr(redundancy, "prop_of")


def test_what_lean_is_asked_is_strictly_harder_than_the_row():
    """Deleting a hypothesis is the whole mechanism: the probe statement must be
    the child minus something, never the child plus something."""
    reduced = without(CHILD, ["hspare"])
    assert "hspare" not in reduced
    assert "h₀ : 4 ≤ n" in reduced
    assert "n ^ 2 + n ≤ n !" in reduced


def test_a_hypothesis_lean_can_delete_is_reported():
    verifier = FakeVerifier(drops_only("hspare"))
    evidence = asyncio.run(check_mutation(verifier, "import Mathlib", CHILD, PARENT))
    assert evidence["measured"] is True
    assert evidence["free_hypotheses"] == ["hspare"]
    assert evidence["redundant"] is True
    assert evidence["why"]


def test_a_row_whose_hypotheses_all_matter_is_not_convicted():
    evidence = asyncio.run(check_mutation(FakeVerifier(), "import Mathlib", CHILD, PARENT))
    assert evidence["measured"] is True
    assert evidence["free_hypotheses"] == []
    assert evidence["redundant"] is False
    assert not evidence["why"]


def test_the_parent_is_not_needed_to_ask_the_question():
    """It takes a parent argument for call-site symmetry with the crossover
    probe, but the question is about the child alone. Passing nothing must not
    change the verdict -- if it did, the parent would be back in the statement."""
    with_parent = asyncio.run(check_mutation(
        FakeVerifier(drops_only("hspare")), "import Mathlib", CHILD, PARENT))
    without_parent = asyncio.run(check_mutation(
        FakeVerifier(drops_only("hspare")), "import Mathlib", CHILD, ""))
    assert with_parent["free_hypotheses"] == without_parent["free_hypotheses"]


def test_a_silent_mutation_is_exempt_rather_than_passed():
    """Equivalent to its parent by design, so a finding would report the
    operator working. That must read as inapplicable, not as measured-and-clean."""
    evidence = asyncio.run(check_mutation(
        FakeVerifier(), "import Mathlib", CHILD, PARENT, variant="mutation_silent"))
    assert evidence["measured"] is False
    assert "design" in evidence["why"]


@pytest.mark.parametrize("statement", ["", "   "])
def test_nothing_to_probe_is_not_a_clean_bill(statement):
    evidence = asyncio.run(check_mutation(FakeVerifier(), "import Mathlib", statement, PARENT))
    assert evidence["measured"] is False
