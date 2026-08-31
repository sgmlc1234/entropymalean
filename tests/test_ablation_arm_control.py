"""A single-operator arm has to hold, or the two arms are not comparable.

`OP_TYPE_LOCK` stops replanning from changing an operator and makes
`crossover_count` exact, and the plan validator rejects a plan that misses it.
Reserve slots are built after that check, from the previous generation's failure
profile rather than from the plan, and that path chose crossover on its own: the
mutation-only arm produced one crossover per generation, and two of them reached
the corpus. The failures of one operator were being counted as the output of the
other, which is the one thing the arm exists to prevent.
"""

from __future__ import annotations

import pytest

from src.orchestration.pool_generation import (
    CertificationInput,
    CertificationResult,
    _build_reserve_work_items,
)


def pool(n: int = 3):
    return [
        CertificationInput(
            id=f"seed{i}", statement=f"Seed {i}.", answer="",
            metadata={
                "formal_statement": f"theorem seed{i} (n : ℕ) : n + {i} = {i} + n := by",
                "lean_code": f"theorem seed{i} (n : ℕ) : n + {i} = {i} + n := by omega",
                "problem_style": "theorem_proof",
            },
        )
        for i in range(n)
    ]


def failed_results():
    return [
        CertificationResult(
            problem_id=f"seed0__theorem_gen1__{i}",
            status="certified",
            lean_level=3,
            op_type="mutation",
            operator_variant="mutation_easy",
            parent_ids=["seed0"],
            target_style="theorem_proof",
            certification_route="theorem_prover",
            formal_statement=f"theorem child{i} (n : ℕ) : n = n",
            lean_code=f"theorem child{i} (n : ℕ) : n = n := by rfl",
            quality_verdict="weak",
        )
        for i in range(2)
    ]


def build(monkeypatch, *, locked: str, crossover_count: int):
    monkeypatch.setenv("OP_TYPE_LOCK", locked)
    return _build_reserve_work_items(
        pool(), failed_results(),
        target_accepted=5, reserve_budget=2, crossover_count=crossover_count,
    )


def test_mutation_arm_reserve_slots_are_never_crossover(monkeypatch):
    items = build(monkeypatch, locked="1", crossover_count=0)
    assert items, "the reserve path must still produce slots"
    assert all(item["op_type"] == "mutation" for item in items), \
        [item["op_type"] for item in items]


def test_crossover_arm_reserve_slots_may_be_crossover(monkeypatch):
    items = build(monkeypatch, locked="1", crossover_count=4)
    assert any(item["op_type"] == "crossover" for item in items), \
        "the crossover arm's reserve slot should follow its arm"


def test_unlocked_reserve_keeps_its_own_judgement(monkeypatch):
    items = build(monkeypatch, locked="0", crossover_count=0)
    assert items
    # The mixed pipeline is allowed to reach for a crossover here; only a locked
    # arm is constrained.
    assert all(item["op_type"] in {"mutation", "crossover"} for item in items)


@pytest.mark.parametrize("crossover_count", [0, 1, 4])
def test_reserve_slots_always_name_a_parent(monkeypatch, crossover_count):
    for item in build(monkeypatch, locked="1", crossover_count=crossover_count):
        assert item["parent_ids"], "a reserve slot with no parent cannot be executed"
        if item["op_type"] == "crossover":
            assert len(item["parent_ids"]) == 2
