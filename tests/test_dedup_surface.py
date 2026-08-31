"""The dedup surface must be an alpha-normal form, not a partial erasure.

The first version blanked binder declaration sites and left every *use* of the
name in the goal, so two rows stating the same closure identity -- one over `n`,
one over `r` -- hashed differently and both were released. It also blanked only
the last name of a group, leaving `(a b : G)` as `(a _ : G)`. These cases are the
ones that were wrong, plus the ones that must stay right.
"""

from __future__ import annotations

import pytest

from src.certification.dedup import CorpusIndex, fingerprint

SAME = [
    ("renamed binder", "theorem a (n : ℕ) : n + 0 = n", "theorem b (r : ℕ) : r + 0 = r"),
    ("multi-name group", "theorem a (x y : ℤ) : x + y = y + x", "theorem b (p q : ℤ) : p + q = q + p"),
    ("goal quantifier", "theorem a : ∀ n : ℕ, n ≥ 0", "theorem b : ∀ k : ℕ, k ≥ 0"),
    ("existential", "theorem a : ∃ n : ℕ, n = n", "theorem b : ∃ w : ℕ, w = w"),
    ("implicit binder", "theorem a {G : Type*} [Group G] (a : G) : a = a",
                        "theorem b {H : Type*} [Group H] (z : H) : z = z"),
    # A binder named `digits` must not rewrite the `digits` in `Nat.digits`.
    ("qualified name", "theorem a (digits : ℕ) : Nat.digits 10 digits = []",
                       "theorem b (z : ℕ) : Nat.digits 10 z = []"),
    # Placeholders must not collide with identifiers already present.
    ("placeholder clash", "theorem a (v0 v1 : ℕ) : v0 = v1", "theorem b (v1 v0 : ℕ) : v1 = v0"),
    ("proof body", "theorem a (n : ℕ) : n = n := by rfl", "theorem a (n : ℕ) : n = n := by simp"),
    ("comments", "theorem a (n : ℕ) : n = n", "-- note\ntheorem a (n : ℕ) : n = n /- x -/"),
    ("the released collision",
     "theorem closure_eq_closure_shifted {G : Type*} [Group G] (a b : G) (n : ℕ) : "
     "Subgroup.closure ({a, b} : Set G) = Subgroup.closure ({b * a * b ^ n, b * a * b ^ (n + 1)} : Set G)",
     "theorem parameterized_two_generator_closure {G : Type*} [Group G] (a b : G) (r : ℕ) : "
     "Subgroup.closure ({a, b} : Set G) = Subgroup.closure ({b * a * b ^ r, b * a * b ^ (r + 1)} : Set G)"),
]

DIFFERENT = [
    ("constant differs", "theorem a (n : ℕ) : n + 0 = n", "theorem b (n : ℕ) : n + 1 = n"),
    ("binder order", "theorem a (n : ℕ) (m : ℤ) : True", "theorem b (m : ℤ) (n : ℕ) : True"),
    ("extra hypothesis", "theorem a (n : ℕ) : n ≥ 0", "theorem b (n : ℕ) (h : n > 0) : n ≥ 0"),
    ("type differs", "theorem a (n : ℕ) : n = n", "theorem b (n : ℤ) : n = n"),
    ("relation differs", "theorem a (n : ℕ) : n ≤ n", "theorem b (n : ℕ) : n < n"),
]


@pytest.mark.parametrize("label,left,right", SAME, ids=[c[0] for c in SAME])
def test_same_theorem_hashes_the_same(label, left, right):
    assert fingerprint(left) == fingerprint(right)


@pytest.mark.parametrize("label,left,right", DIFFERENT, ids=[c[0] for c in DIFFERENT])
def test_different_theorems_stay_apart(label, left, right):
    assert fingerprint(left) != fingerprint(right)


def test_index_rejects_the_second_copy():
    index = CorpusIndex()
    assert index.add("theorem a (n : ℕ) : n + 0 = n", "first") is True
    assert index.add("theorem b (r : ℕ) : r + 0 = r", "second") is False
    assert index.holder("theorem c (q : ℕ) : q + 0 = q") == "first"


def test_empty_statement_is_not_a_duplicate():
    index = CorpusIndex()
    assert index.add("", "a") is True
    assert index.add("", "b") is True
