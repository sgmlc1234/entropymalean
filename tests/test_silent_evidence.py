"""A silent mutation is gated differently; it is not audited differently.

The silent branch of `verify_slot_quality` replaces the novelty heuristics with
the two-way equivalence, which is right -- the other variants are asked to be
different enough from their parent and this one is asked to be exactly
equivalent. What it must not replace is the record. Written as a fresh dict, it
dropped thirty-five evidence keys including the judge's verdict and reasoning,
dedup, vacuity, inhabitation, the dead-hypothesis result and the goal
round-trip. The checks had all run and all gated the row; only their record
vanished, so the first silent row ever produced reached the corpus reading as
though nothing had been checked.
"""

from __future__ import annotations

import pytest

from src.orchestration.pool_generation import CertificationInput, CertificationResult
from src.orchestration.quality import verify_slot_quality

PARENT = CertificationInput(
    id="parent", statement="Three divides n.", answer="",
    metadata={"formal_statement": "theorem parent (n : ℕ) : 3 ∣ n ↔ n % 3 = 0 := by"},
)


def silent_result(*, equivalent: bool = True, judge_quality: str = "strong") -> CertificationResult:
    return CertificationResult(
        problem_id="parent__theorem_gen1__abc",
        status="certified",
        lean_level=3,
        op_type="mutation",
        operator_variant="mutation_silent",
        parent_ids=["parent"],
        target_style="theorem_proof",
        certification_route="theorem_prover",
        statement="A restatement of the parent.",
        formal_statement="theorem child (n : ℕ) : n ≡ 0 [MOD 3] ↔ 3 ∣ n",
        lean_code="theorem child (n : ℕ) : n ≡ 0 [MOD 3] ↔ 3 ∣ n := by omega",
        parent_contributions={"parent": "same statement, different Lean surface"},
        quality_evidence={
            "silent": {"checked": True, "measured": True, "equivalent": equivalent,
                       "used_auto_ladder": True, "failed_direction": "", "why": ""},
            "judge": {"ran": True, "verdict": "keep" if judge_quality != "weak" else "reject",
                      "quality": judge_quality,
                      "reason": "The child restates the parent's divisibility claim."},
            "dedup": {"checked": True, "duplicate": False, "corpus_size": 2037},
            "vacuity": {"measured": True, "vacuous": False},
            "inhabited": {"measured": True, "uninhabited": []},
            "dead_hypotheses": {"measured": True, "removed": []},
            "alignment_evidence": {"equivalent": True, "elaborated_goal": "..."},
            "redundancy": {"measured": True, "free_hypotheses": [], "redundant": False},
        },
    )


CARRIED = ["judge", "dedup", "vacuity", "inhabited", "dead_hypotheses",
           "alignment_evidence", "redundancy"]


@pytest.mark.parametrize("key", CARRIED)
def test_every_check_survives_the_silent_branch(key):
    quality = verify_slot_quality(
        silent_result(), {"op_type": "mutation", "operator_variant": "mutation_silent"}, [PARENT]
    )
    assert key in quality.quality_evidence, f"{key} was dropped by the silent branch"


def test_the_judges_reasoning_is_not_discarded():
    quality = verify_slot_quality(
        silent_result(), {"op_type": "mutation", "operator_variant": "mutation_silent"}, [PARENT]
    )
    judge = quality.quality_evidence.get("judge") or {}
    assert judge.get("quality") == "strong"
    assert "restates the parent" in (judge.get("reason") or "")


def test_the_verdict_comes_from_the_judge_not_the_probe():
    """The probe cannot decide it, so it does not.

    `silent_backward (h : child) : parent` asks Lean to prove the parent, which
    is a theorem, so the tactic block closes it with or without `h`. Measured
    directly, the probe called a child that dropped a conjunct equivalent, and
    an unrelated true statement equivalent. A failing probe must therefore not
    condemn a row the judge kept, and a passing probe must not save one it
    rejected.
    """
    kept = verify_slot_quality(
        silent_result(equivalent=False, judge_quality="strong"),
        {"op_type": "mutation", "operator_variant": "mutation_silent"}, [PARENT])
    assert kept.quality_verdict == "strong"
    assert kept.quality_flags == []

    rejected = verify_slot_quality(
        silent_result(equivalent=True, judge_quality="weak"),
        {"op_type": "mutation", "operator_variant": "mutation_silent"}, [PARENT])
    assert rejected.quality_verdict == "weak"
    assert rejected.quality_evidence["verdict_source"] == "judge"


def test_an_unavailable_judge_fails_open():
    result = silent_result(equivalent=True, judge_quality="")
    result.quality_evidence["judge"] = {"ran": False}
    quality = verify_slot_quality(
        result, {"op_type": "mutation", "operator_variant": "mutation_silent"}, [PARENT])
    assert quality.quality_verdict == "acceptable"


def test_the_silent_findings_win_where_they_overlap():
    quality = verify_slot_quality(
        silent_result(), {"op_type": "mutation", "operator_variant": "mutation_silent"}, [])
    assert quality.quality_evidence["signature_group"] == "silent"
    assert quality.quality_evidence["reasoning_signature"].startswith("silent:")
    assert quality.quality_evidence["silent"]["equivalent"] is True
