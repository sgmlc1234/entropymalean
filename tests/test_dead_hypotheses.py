"""The dead-hypothesis check must let Lean decide, and must prune one at a time.

Two things went wrong before this existed. The name search called a hypothesis
dead because `omega` never wrote its name -- it was right 24 times out of 53.
And removing every candidate at once broke a proof whose hypotheses were each
individually removable, because dropping one made the next load-bearing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.certification.hypotheses import candidates, drop_binder, prune_dead, scanning_tactics

STATEMENT = "theorem t (a : ℝ) (ha_lo : -10 ≤ a) (ha_hi : a ≤ 10) (hf : a = a) : a = a"
PROOF = STATEMENT + " := by\n  exact hf"


def verifier_that(accepts):
    """A stand-in Lean whose verdict is decided by `accepts(code)`."""
    calls = []

    async def verify(code, timeout=None):
        calls.append(code)
        return SimpleNamespace(complete=bool(accepts(code)), ok=False, system_error=None)

    return verify, calls


def test_candidates_are_names_absent_from_the_proof_body():
    assert candidates(STATEMENT, PROOF) == ["ha_lo", "ha_hi"]


def test_a_named_hypothesis_is_not_a_candidate():
    assert "hf" not in candidates(STATEMENT, PROOF)


def test_drop_binder_counts_nested_parentheses():
    text = "theorem t (n : ℤ) (hn : ¬ ((3:ℤ) ∣ n)) : True"
    assert drop_binder(text, "hn") == "theorem t (n : ℤ)  : True"


def test_scanning_tactics_are_reported_from_the_body_only():
    assert scanning_tactics("theorem omega_thing : True := by\n  simp_all") == ["simp_all"]


@pytest.mark.asyncio
async def test_a_hypothesis_the_proof_still_needs_is_kept():
    # Lean rejects every trimmed version, so nothing may be removed.
    verify, _ = verifier_that(lambda code: False)
    evidence = await prune_dead(verify, STATEMENT, PROOF)
    assert evidence["removed"] == []
    assert evidence["used_silently"] == ["ha_lo", "ha_hi"]
    assert evidence["formal_statement"] == STATEMENT


@pytest.mark.asyncio
async def test_a_dead_hypothesis_is_removed_from_statement_and_proof():
    verify, _ = verifier_that(lambda code: True)
    evidence = await prune_dead(verify, STATEMENT, PROOF)
    assert evidence["removed"] == ["ha_lo", "ha_hi"]
    assert "ha_lo" not in evidence["formal_statement"]
    assert "ha_hi" not in evidence["lean_code"]
    assert "hf" in evidence["formal_statement"]


@pytest.mark.asyncio
async def test_removals_are_confirmed_one_at_a_time():
    """The second candidate is tested against the survivor, not the original.

    This is the interaction a release row actually hit: three hypotheses each
    removable on their own, and a broken proof when all three went at once.
    """
    # Dropping `ha_lo` is fine; dropping `ha_hi` afterwards is not, because this
    # Lean only accepts a proof that still has `ha_hi` once `ha_lo` has gone.
    def accepts(code):
        return "ha_lo" not in code and "ha_hi" in code
    verify, calls = verifier_that(accepts)
    evidence = await prune_dead(verify, STATEMENT, PROOF)
    assert evidence["removed"] == ["ha_lo"]
    assert evidence["used_silently"] == ["ha_hi"]
    assert "ha_hi" in evidence["formal_statement"]
    # The trimmed text handed to Lean the second time must already lack `ha_lo`.
    assert "ha_lo" not in calls[1]


@pytest.mark.asyncio
async def test_a_verifier_error_leaves_the_row_untouched():
    async def verify(code, timeout=None):
        return SimpleNamespace(complete=False, ok=False, system_error="lake died")
    evidence = await prune_dead(verify, STATEMENT, PROOF)
    assert evidence["removed"] == []
    assert evidence["formal_statement"] == STATEMENT
    assert "lake died" in evidence["why"]


@pytest.mark.asyncio
async def test_nothing_to_check_is_measured_not_skipped():
    verify, calls = verifier_that(lambda code: True)
    statement = "theorem t (n : ℕ) : n = n"
    evidence = await prune_dead(verify, statement, statement + " := by rfl")
    assert evidence["measured"] is True
    assert evidence["removed"] == []
    assert calls == []


@pytest.mark.asyncio
async def test_candidate_budget_is_reported_when_it_bites():
    verify, _ = verifier_that(lambda code: False)
    statement = "theorem t " + " ".join(f"(h{i} : True)" for i in range(9)) + " : True"
    evidence = await prune_dead(verify, statement, statement + " := by trivial", max_candidates=2)
    assert "exceeds max_candidates" in evidence["why"]
    assert len(evidence["used_silently"]) == 2


@pytest.mark.asyncio
async def test_removal_repeats_until_nothing_more_comes_out():
    """A hypothesis can become dead once another one goes.

    This is the case a single pass missed on a release row: `h3` was needed
    while `h4` was present and dead once it was gone, and only the judge
    noticed.
    """
    statement = "theorem t (h3 : True) (h4 : True) (hf : 1 = 1) : 1 = 1"
    proof = statement + " := by\n  exact hf"

    def accepts(code):
        # h4 may always go; h3 may go only after h4 has.
        if "h4" in code:
            return False
        return True

    verify, _ = verifier_that(accepts)
    evidence = await prune_dead(verify, statement, proof)
    assert sorted(evidence["removed"]) == ["h3", "h4"]
    assert evidence["used_silently"] == []
    assert "h3" not in evidence["formal_statement"]


@pytest.mark.asyncio
async def test_the_result_does_not_depend_on_candidate_order():
    statement = "theorem t (hb : True) (ha : True) (hf : 1 = 1) : 1 = 1"
    proof = statement + " := by\n  exact hf"
    verify, _ = verifier_that(lambda code: "hb" not in code)
    evidence = await prune_dead(verify, statement, proof)
    # `ha` is only removable once `hb` has gone, and `hb` is listed second.
    assert sorted(evidence["removed"]) == ["ha", "hb"]
