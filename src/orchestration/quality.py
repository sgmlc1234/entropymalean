"""Deterministic quality checks for pool-generation slots."""

from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.certification import CertificationInput, CertificationResult
from src.no_go_policy import (
    ACCEPTED_PROXY_SEVERE_FLAGS,
    MISFORMALIZATION_MISREPRESENTATION_FLAGS,
    MISFORMALIZATION_SEMANTIC_FLAGS,
)
import langsmith as ls

from src.utils.lean_templates import detect_family


DERIVED_PARAM_KEYS = {
    "gcd_divisor_sum": {"gcd"},
    "divisor_sum_mod": {"modulus"},
}
NON_WEAK_CROSSOVER_FLAGS = {
    "sequential_composition",
    "same_role_crossover",
    "pipeline_composite",
    "lemma_bundle_master",
}
INFORMAL_STATEMENT_INTERNAL_TERM_PATTERNS = (
    (r"\bcheckpoint\b", "checkpoint"),
    (r"\bparent\b", "parent"),
    (r"\bcertified\b", "certified"),
    (r"\bgenerated\b", "generated"),
    (r"\bmutation\b", "mutation"),
    (r"\bcrossover\b", "crossover"),
    (r"\bpipeline\b", "pipeline"),
    (r"\boperator\b", "operator"),
    (r"\bproof obligation\b", "proof obligation"),
    (r"\bLean\b", "Lean"),
    (r"\bformal\b", "formal"),
)


class QualityResult(BaseModel):
    quality_verdict: str = "acceptable"
    quality_flags: List[str] = Field(default_factory=list)
    interestingness_score: float = 0.5
    feedback_for_next_generation: str = ""
    semantic_parent_contribution: Dict[str, str] = Field(default_factory=dict)
    interestingness_features: Dict[str, Any] = Field(default_factory=dict)
    quality_evidence: Dict[str, Any] = Field(default_factory=dict)


def informal_statement_internal_term_hits(statement: Optional[str]) -> List[str]:
    """Return workflow/internal terms that should not appear in public statements."""
    text = str(statement or "")
    hits = [
        label
        for pattern, label in INFORMAL_STATEMENT_INTERNAL_TERM_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    return sorted(set(hits))


def _ints(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def derive_misformalization_taxonomy(
    result: CertificationResult,
    quality_flags: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map verifier-derived signals to the compact Formal Conjectures taxonomy."""
    evidence = dict(evidence or {})
    signals = set(str(flag) for flag in (quality_flags or []) if str(flag).strip())
    for key in ("missing_checkpoints", "novelty_flags", "solution_verify_flags"):
        value = evidence.get(key)
        if isinstance(value, list):
            signals.update(str(flag) for flag in value if str(flag).strip())
    failure_class = str(evidence.get("failure_class") or result.failure_signature or "")
    if failure_class:
        signals.add(failure_class)
    status = str(result.status or "")
    if status and status not in {"certified", "survivor"}:
        signals.add(status)
    proof_summary = str(result.proof_verify_summary or result.error or "").lower()

    source_reporting = {
        "axiom_backed_seed_or_child",
        "parent_proof_surface_missing",
        "proof_surface_unavailable",
        "invalid_parent_proof_surface",
        "missing_parent_proof_body",
    }
    syntactic = {
        "llm_json_parse_error",
        "invalid_formal_shape",
        "lean_syntax_error",
        "proof_contains_sorry",
        "theorem_generation_failed",
        "statement_typecheck_failed",
    }
    semantic = set(MISFORMALIZATION_SEMANTIC_FLAGS) | {
        "trivial_negation_chain",
        "trivial_add_zero_padding",
        "syntactic_wrapper_only",
        "typeclass_narrowing_only",
        "projection_only_theorem",
        "divisibility_weaken_only_theorem",
        "fin_one_vacuity_theorem",
        "fin_one_concrete_arithmetic_theorem",
        "unit_product_closure_only",
        "solution_modulus_mismatch",
        "solution_answer_mismatch",
        "solution_formula_mismatch",
        "tautological_checkpoint_theorem",
        "auxiliary_conjunct_only_theorem",
        "parameter_shift_only_theorem",
        "piecewise_branch_only_theorem",
        "concrete_native_decide_projection",
        "informal_statement_internal_terms",
    }
    misrepresentation = set(MISFORMALIZATION_MISREPRESENTATION_FLAGS) | {
        "statement_lean_alignment_failed",
        "alignment_failed",
        "missing_parent_contribution",
    }
    implicit = {
        "theorem_too_broad",
        "missing_formal_statement",
        "missing_proof_plan",
        "missing_quality_checkpoints",
        "unsupported_quality_claim",
        "unsupported",
        "numeric_template_unsupported",
    }
    mathematical = {
        "proof_failed",
        "lean_type_check_failed",
        "lean_check_failed",
        "wrong_answer",
        "projection_params_mismatch",
        "projection_check_failed",
    }

    if signals & source_reporting or "axiom " in proof_summary:
        level, category = "source", "reporting"
        rationale = "Source/proof artifact was missing, invalid, or axiom-backed."
    elif signals & syntactic or "syntax" in proof_summary or "parse" in proof_summary:
        level, category = "translation", "syntactic"
        rationale = "Lean or JSON surface was syntactically malformed."
    elif signals & misrepresentation:
        level, category = "translation", "misrepresentation"
        rationale = "Formal artifact did not faithfully represent the intended statement or parent contribution."
    elif signals & semantic:
        level, category = "translation", "semantic"
        rationale = "Formal artifact type-checks or nearly type-checks but changes only superficial or vacuous semantics."
    elif signals & implicit:
        level, category = "underspecified", "implicit_conventions"
        rationale = "The generated theorem relied on missing conventions, hidden assumptions, or an underspecified route."
    elif signals & mathematical:
        level, category = "source", "mathematical"
        rationale = "Lean verification rejected the mathematical claim or projected answer."
    else:
        level, category = "none", "none"
        rationale = "No misformalization signal detected."

    return {
        "level": level,
        "category": category,
        "signals": sorted(signals),
        "rationale": rationale,
    }


def _with_misformalization(
    result: CertificationResult,
    evidence: Dict[str, Any],
    quality_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out = dict(evidence or {})
    out["misformalization"] = derive_misformalization_taxonomy(result, quality_flags, out)
    out["accepted_proxy"] = derive_accepted_proxy(result, quality_flags, out)
    out["entropy_direction"] = derive_entropy_direction(result, quality_flags, out)
    out["curation_decision"] = derive_curation_decision(result, quality_flags, out)
    return out


def derive_curation_decision(
    result: CertificationResult,
    quality_flags: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify certified artifacts for paper-facing QA without changing worker schema.

    ``accepted_proxy`` remains a frontier-style filter. This layer separates
    paper-grade candidates from useful scaffold/contrast results so planner
    memory does not learn scaffold rows as successes.
    """
    evidence = dict(evidence or {})
    flags = {str(flag) for flag in (quality_flags or []) if str(flag).strip()}
    accepted_proxy = dict(evidence.get("accepted_proxy") or {})
    entropy = evidence.get("entropy_direction")
    entropy_direction = (
        str(entropy.get("direction") or "")
        if isinstance(entropy, dict)
        else str(entropy or "")
    )

    if result.status != "certified" or result.op_type in {
        "survivor",
        "fallback_survivor",
        "seed_proof_completion",
    }:
        return {
            "curation_class": "reject",
            "paper_grade": False,
            "scaffold_ok": False,
            "reason": "not_a_generated_certified_candidate",
            "flags": sorted(flags | {result.status or "not_certified"}),
        }

    reject_flags = {
        "same_formal_statement_as_parent",
        "same_statement_rephrase",
        "formal_surface_not_changed",
        "syntactic_wrapper_only",
        "projection_only_theorem",
        "trivial_negation_chain",
        "trivial_add_zero_padding",
        "typeclass_narrowing_only",
        "side_by_side_conjunction",
        "mutation_like_crossover",
        "weak_inspiration_only_crossover",
        "parent_checkpoint_not_consumed",
        "unused_checkpoint",
        "same_lineage_crossover",
        "informal_statement_internal_terms",
        "statement_lean_alignment_failed",
        "alignment_failed",
        "exact_duplicate_memory",
        "near_duplicate",
    }
    scaffold_flags = {
        "direct_parent_corollary_only",
        "ap_index_only_theorem",
        "ap_shifted_local_corollary_only",
        "ap_bound_padding_only",
        "ap_interval_bound_padding_only",
        "linear_equation_shift_corollary_only",
        "affine_index_drift_only",
        "cardinality_only_window",
        "aggregate_helper_only",
        "proof_infrastructure_only",
        "fixed_finite_aggregate_computation",
        "native_decide_fixed_domain_computation",
        "cardinality_arithmetic_pipeline_only",
        "mod_inverse_same_conclusion_paraphrase",
        "mod_inverse_arithmetic_corollary_only",
        "finite_mod_inverse_window_restatement",
        "finite_residue_bookkeeping_only",
        "residue_finset_cardinality_restatement",
        "solved_parameter_quotient_corollary_only",
        "theorem_local_corollary_dominated",
        "cyclic_transport_same_target_role",
        "topology_direct_consequence_only",
        "same_target_role_already_accepted",
        "witness_packaging_only",
        "coefficient_engineering_only",
    }
    hard_reject = sorted(flags & reject_flags)
    scaffold = sorted(flags & scaffold_flags)
    if hard_reject:
        return {
            "curation_class": "reject",
            "paper_grade": False,
            "scaffold_ok": False,
            "reason": "hard_reject_flags",
            "flags": hard_reject,
        }
    if scaffold:
        return {
            "curation_class": "scaffold",
            "paper_grade": False,
            "scaffold_ok": True,
            "reason": "certified_scaffold_not_paper_grade",
            "flags": scaffold,
        }
    if accepted_proxy.get("pass") and entropy_direction != "decrease":
        return {
            "curation_class": "paper",
            "paper_grade": True,
            "scaffold_ok": False,
            "reason": "accepted_proxy_pass",
            "flags": sorted(flags),
        }
    if entropy_direction == "increase":
        return {
            "curation_class": "scaffold",
            "paper_grade": False,
            "scaffold_ok": True,
            "reason": "certified_entropy_increase_but_not_paper_grade",
            "flags": sorted(flags),
        }
    return {
        "curation_class": "reject",
        "paper_grade": False,
        "scaffold_ok": False,
        "reason": "not_entropy_increasing_or_proxy_failed",
        "flags": sorted(flags),
    }


def derive_entropy_direction(
    result: CertificationResult,
    quality_flags: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify whether a certified child increases or decreases task entropy.

    This is deliberately coarser than accepted-grade QA. A direct corollary or
    checkpoint may fail the frontier-style accepted proxy while still adding a new
    formal surface that is useful for later generations. Pure wrappers,
    restatements, vacuity, and failed proofs are entropy decreases.
    """
    evidence = dict(evidence or {})
    flags = {str(flag) for flag in (quality_flags or []) if str(flag).strip()}
    accepted_proxy = dict(evidence.get("accepted_proxy") or {})
    feature_delta = dict(evidence.get("feature_delta") or {})
    status = str(result.status or "")
    if status not in {"certified", "survivor"}:
        return {
            "direction": "decrease",
            "signals": sorted(flags | {status or "not_certified"}),
            "rationale": "The artifact did not produce a reusable Lean-certified surface.",
        }

    entropy_decrease_flags = {
        "same_formal_statement_as_parent",
        "same_statement_rephrase",
        "syntactic_wrapper_only",
        "projection_only_theorem",
        "trivial_negation_chain",
        "trivial_add_zero_padding",
        "typeclass_narrowing_only",
        "side_by_side_conjunction",
        "mutation_like_crossover",
        "weak_inspiration_only_crossover",
        "parent_checkpoint_not_consumed",
        "unused_checkpoint",
        "fin_one_vacuity_theorem",
        "fin_one_concrete_arithmetic_theorem",
        "tautological_checkpoint_theorem",
        "statement_lean_alignment_failed",
        "alignment_failed",
    }
    blocking = sorted(flags & entropy_decrease_flags)
    if blocking:
        return {
            "direction": "decrease",
            "signals": blocking,
            "rationale": "The child mainly restates, wraps, or fails to consume the parent theorem surface.",
        }

    formal_changed = feature_delta.get("formal_surface_changed")
    if formal_changed is False:
        return {
            "direction": "decrease",
            "signals": sorted(flags | {"formal_surface_not_changed"}),
            "rationale": "The formal surface did not materially change from the parent.",
        }

    if accepted_proxy.get("pass"):
        return {
            "direction": "increase",
            "signals": sorted(flags),
            "rationale": "The child passes the frontier-style accepted proxy.",
        }

    if status == "certified":
        return {
            "direction": "increase",
            "signals": sorted(flags),
            "rationale": "The child is Lean-certified and changes the task surface, but remains outside the frontier-style accepted proxy while still useful for generation support.",
        }

    return {
        "direction": "decrease",
        "signals": sorted(flags),
        "rationale": "No certified entropy-increasing signal was detected.",
    }


def derive_accepted_proxy(
    result: CertificationResult,
    quality_flags: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Approximate manual-accepted candidacy from verifier-derived signals only."""
    evidence = dict(evidence or {})
    flags = {str(flag) for flag in (quality_flags or []) if str(flag).strip()}
    proxy_flags: List[str] = []
    if result.status != "certified":
        proxy_flags.append("not_certified")
    if result.op_type in {"survivor", "fallback_survivor", "seed_proof_completion"}:
        proxy_flags.append("not_generated")
    severe = sorted(flags & ACCEPTED_PROXY_SEVERE_FLAGS)
    proxy_flags.extend(severe)
    weak_like_flags = sorted(flags - NON_WEAK_CROSSOVER_FLAGS - ACCEPTED_PROXY_SEVERE_FLAGS)
    if weak_like_flags:
        proxy_flags.extend(weak_like_flags)

    is_theorem_route = (
        result.target_style == "theorem_proof"
        or result.certification_route == "theorem_prover"
        or str(evidence.get("signature_group") or "") == "theorem_proof"
    )
    feature_delta = dict(evidence.get("feature_delta") or {})
    if is_theorem_route and result.op_type not in {"survivor", "fallback_survivor"}:
        if feature_delta.get("formal_surface_changed") is False:
            proxy_flags.append("formal_surface_not_changed")
        if not result.formal_statement and not result.lean_code:
            proxy_flags.append("missing_theorem_surface")

    if result.op_type == "crossover":
        parent_contribution = dict(evidence.get("parent_contribution") or {})
        checkpoint_consumption = dict(evidence.get("parent_checkpoint_consumption") or {})
        consumed = [
            item
            for item in checkpoint_consumption.values()
            if isinstance(item, dict) and item.get("consumed_in_lean_surface")
        ]
        if len(parent_contribution) < 2 and len(consumed) < 2:
            proxy_flags.append("crossover_parent_usage_not_observable")

    proxy_flags = sorted(set(proxy_flags))
    return {
        "pass": not proxy_flags,
        "accepted_grade_pass": not proxy_flags,
        "flags": proxy_flags,
        "reason": "accepted_proxy_pass" if not proxy_flags else ",".join(proxy_flags),
    }


def _family_params(problem: CertificationInput) -> Dict[str, Any]:
    statement = problem.statement
    family = detect_family(statement)
    if family in {"gcd", "gcd_divisor_sum"}:
        nums = _ints(statement)
        return {"a": nums[0], "b": nums[1]} if len(nums) >= 2 else {}
    if family == "units_digit":
        match = re.search(r"units digit of\s+(\d+)\s*\^\s*\{?(\d+)\}?", statement, re.I)
        return {"base": int(match.group(1)), "exp": int(match.group(2))} if match else {}
    if family == "divisor_sum":
        nums = _ints(statement)
        return {"n": nums[-1]} if nums else {}
    if family == "divisor_sum_mod":
        nums = _ints(statement)
        return {"n": nums[0], "a": nums[1]} if len(nums) >= 2 else {}
    if family == "stars_and_bars":
        total_match = re.search(r"=\s*(\d+)", statement)
        var_count = len(set(re.findall(r"x_(\d+)", statement)))
        return {"vars": var_count, "sum": int(total_match.group(1))} if total_match else {}
    if family == "arithmetic_series":
        nums = _ints(statement)
        if len(nums) >= 5:
            return {"n_terms": nums[0], "first": nums[1], "diff": nums[2] - nums[1]}
        return {}
    if family == "modular_congruence":
        match = re.search(r"Find\s+(\d+)\s+mod\s+(\d+)", statement, re.I)
        return {"a": int(match.group(1)), "m": int(match.group(2))} if match else {}
    return {}


def _difficulty_metric(family: Optional[str], params: Dict[str, Any]) -> Optional[int]:
    if family in {"gcd", "gcd_divisor_sum"} and {"a", "b"} <= set(params):
        a = int(params["a"])
        b = int(params["b"])
        return max(a, b) + 50 * _prime_factor_count(math.gcd(a, b))
    if family == "units_digit" and {"base", "exp"} <= set(params):
        return int(params["exp"])
    if family == "divisor_sum" and "n" in params:
        n = int(params["n"])
        return n + 100 * _prime_factor_count(n)
    if family == "divisor_sum_mod" and {"n", "a"} <= set(params):
        return int(params["n"]) + int(params["a"]) // 1000
    if family == "stars_and_bars" and {"vars", "sum"} <= set(params):
        return int(params["vars"]) * 100 + int(params["sum"])
    if family == "arithmetic_series" and {"n_terms", "diff"} <= set(params):
        return int(params["n_terms"]) * max(1, int(params["diff"]))
    if family == "modular_congruence" and {"a", "m"} <= set(params):
        return int(params["m"]) * 100 + len(str(int(params["a"])))
    return None


def _prime_factor_count(n: int) -> int:
    count = 0
    d = 2
    value = abs(int(n))
    while d * d <= value:
        if value % d == 0:
            count += 1
            while value % d == 0:
                value //= d
        d += 1 if d == 2 else 2
    if value > 1:
        count += 1
    return count


def _divisor_count(n: int) -> int:
    total = 1
    d = 2
    value = abs(int(n))
    while d * d <= value:
        exponent = 0
        while value % d == 0:
            exponent += 1
            value //= d
        if exponent:
            total *= exponent + 1
        d += 1 if d == 2 else 2
    if value > 1:
        total *= 2
    return total


def _sum_of_divisors(n: int) -> int:
    return sum(d for d in range(1, int(n) + 1) if int(n) % d == 0)


def _is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _interestingness_features(family: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if family in {"gcd", "gcd_divisor_sum"} and {"a", "b"} <= set(params):
            g = math.gcd(int(params["a"]), int(params["b"]))
            features = {
                "gcd": g,
                "gcd_prime_factor_count": _prime_factor_count(g),
                "gcd_nontrivial": g > 1,
            }
            if family == "gcd_divisor_sum":
                features["reasoning_chain_depth"] = 2
                features["divisor_count_of_gcd"] = _divisor_count(g)
            return features
        if family == "divisor_sum" and "n" in params:
            n = int(params["n"])
            return {
                "prime_factor_count": _prime_factor_count(n),
                "divisor_count": _divisor_count(n),
            }
        if family == "divisor_sum_mod" and {"n", "a"} <= set(params):
            n = int(params["n"])
            modulus = sum(d for d in range(1, n + 1) if n % d == 0)
            return {
                "reasoning_chain_depth": 2,
                "prime_factor_count": _prime_factor_count(n),
                "divisor_count": _divisor_count(n),
                "derived_modulus": modulus,
                "remainder": int(params["a"]) % modulus,
            }
        if family == "modular_congruence" and {"a", "m"} <= set(params):
            return {
                "dividend": int(params["a"]),
                "modulus": int(params["m"]),
                "modulus_is_prime": _is_prime(int(params["m"])),
                "dividend_digits": len(str(int(params["a"]))),
                "remainder": int(params["a"]) % int(params["m"]),
            }
        if family == "units_digit" and {"base", "exp"} <= set(params):
            return {"cycle_reasoning": True, "exponent": int(params["exp"])}
        if family == "stars_and_bars" and {"vars", "sum"} <= set(params):
            return {"variables": int(params["vars"]), "total": int(params["sum"])}
    except (TypeError, ValueError, ZeroDivisionError):
        return {}
    return {}


def _evidence_text(result: CertificationResult) -> str:
    return " ".join(
        str(value or "")
        for value in [
            result.statement,
            result.axis_applied,
            result.generation_notes,
            result.quality_target,
            result.generated_params,
            result.reasoning_pattern,
            result.solution_skeleton,
            result.solution,
            result.projected_params,
            result.projection_check,
        ]
    ).lower()


def _expected_answer_for_family(family: Optional[str], params: Dict[str, Any]) -> Optional[int]:
    try:
        if family == "gcd" and {"a", "b"} <= set(params):
            return math.gcd(int(params["a"]), int(params["b"]))
        if family == "gcd_divisor_sum" and {"a", "b"} <= set(params):
            return _sum_of_divisors(math.gcd(int(params["a"]), int(params["b"])))
        if family == "units_digit" and {"base", "exp"} <= set(params):
            return pow(int(params["base"]), int(params["exp"]), 10)
        if family == "divisor_sum" and "n" in params:
            return _sum_of_divisors(int(params["n"]))
        if family == "divisor_sum_mod" and {"n", "a"} <= set(params):
            return int(params["a"]) % _sum_of_divisors(int(params["n"]))
        if family == "stars_and_bars" and {"vars", "sum"} <= set(params):
            return math.comb(int(params["sum"]) + int(params["vars"]) - 1, int(params["vars"]) - 1)
        if family == "arithmetic_series" and {"n_terms", "first", "diff"} <= set(params):
            n_terms = int(params["n_terms"])
            first = int(params["first"])
            diff = int(params["diff"])
            return sum(first + diff * i for i in range(n_terms))
        if family == "modular_congruence" and {"a", "m"} <= set(params):
            return int(params["a"]) % int(params["m"])
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _answer_claim_from_solution(solution: Optional[str]) -> Optional[int]:
    text = str(solution or "")
    if not text.strip():
        return None
    patterns = [
        r"(?:answer|therefore|so|hence|remainder)\s*(?:is|=|:)\s*(-?\d+)",
        r"(?:boxed|\\boxed)\{?(-?\d+)\}?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return int(matches[-1])
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


def _named_claim_from_solution(solution: Optional[str], names: List[str]) -> Optional[int]:
    text = str(solution or "")
    for name in names:
        pattern = rf"(?:{re.escape(name)})\s*(?:\([^)]*\))?\s*(?:is|=|:)\s*(-?\d+)"
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return int(matches[-1])
    return None


def _claim_after_patterns(solution: Optional[str], patterns: List[str]) -> Optional[int]:
    text = str(solution or "")
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return int(matches[-1])
    return None


def _solution_verification(
    result: CertificationResult,
    family: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Check that saved solution/skeleton agrees with the canonical params answer."""
    flags: List[str] = []
    expected = _expected_answer_for_family(family, params)
    canonical_answer = _coerce_int(result.answer)
    skeleton = result.solution_skeleton if isinstance(result.solution_skeleton, dict) else {}
    skeleton_answer = _coerce_int(skeleton.get("expected_answer"))
    solution_answer = _answer_claim_from_solution(result.solution)

    if expected is not None and canonical_answer is not None and canonical_answer != expected:
        flags.append("canonical_answer_mismatch")
    if expected is not None and skeleton_answer is not None and skeleton_answer != expected:
        flags.append("solution_skeleton_answer_mismatch")
    if expected is not None and solution_answer is not None and solution_answer != expected:
        flags.append("solution_answer_mismatch")

    derived: Dict[str, Any] = {}
    if family == "divisor_sum_mod" and "n" in params:
        n = int(params["n"])
        modulus = _sum_of_divisors(int(params["n"]))
        derived["expected_modulus"] = modulus
        modulus_claim = _claim_after_patterns(
            result.solution,
            [
                r"(?<!mod )\bm\s*(?:is|=|:)\s*(-?\d+)",
                r"\bmodulus\s*(?:is|=|:)\s*(-?\d+)",
                rf"\bsigma\s*\(\s*{n}\s*\)\s*(?:is|=|:)\s*(-?\d+)",
                rf"\bsum of (?:all )?(?:positive )?divisors of\s+{n}\s*(?:is|=|:)\s*(-?\d+)",
                rf"\bdivisor sum of\s+{n}\s*(?:is|=|:)\s*(-?\d+)",
            ],
        )
        if modulus_claim is not None and modulus_claim != modulus:
            flags.append("solution_modulus_mismatch")
    if family == "gcd_divisor_sum" and {"a", "b"} <= set(params):
        a = int(params["a"])
        b = int(params["b"])
        gcd_value = math.gcd(int(params["a"]), int(params["b"]))
        derived["expected_gcd"] = gcd_value
        gcd_claim = _claim_after_patterns(
            result.solution,
            [
                rf"\bgcd\s*\(\s*{a}\s*,\s*{b}\s*\)\s*(?:is|=|:)\s*(-?\d+)",
                rf"\bgcd\s*\(\s*{b}\s*,\s*{a}\s*\)\s*(?:is|=|:)\s*(-?\d+)",
                r"\bn\s*(?:is|=|:)\s*(-?\d+)",
            ],
        )
        if gcd_claim is not None and gcd_claim != gcd_value:
            flags.append("solution_gcd_mismatch")

    return {
        "passed": not flags,
        "flags": sorted(set(flags)),
        "expected_answer": expected,
        "canonical_answer": canonical_answer,
        "solution_answer_claim": solution_answer,
        "skeleton_expected_answer": skeleton_answer,
        **derived,
    }


def _reasoning_signature(
    family: Optional[str],
    reasoning_pattern: Optional[str],
    params: Dict[str, Any],
) -> str:
    pattern = str(reasoning_pattern or "unknown").strip() or "unknown"
    if family in {"gcd", "gcd_divisor_sum"}:
        g = 0
        if {"a", "b"} <= set(params):
            try:
                g = math.gcd(int(params["a"]), int(params["b"]))
            except (TypeError, ValueError):
                g = 0
        return f"{family}:{pattern}:gpf{_prime_factor_count(g)}"
    if family in {"divisor_sum", "divisor_sum_mod"}:
        n = int(params.get("n") or 0)
        return f"{family}:{pattern}:pf{_prime_factor_count(n)}:tau{_divisor_count(n) if n else 0}"
    if family == "units_digit":
        try:
            cycle = len({pow(int(params.get("base") or 0), i, 10) for i in range(1, 5)})
        except (TypeError, ValueError):
            cycle = 0
        return f"{family}:{pattern}:cycle{cycle}"
    if family == "modular_congruence":
        return f"{family}:{pattern}:m_digits{len(str(params.get('m') or ''))}"
    return f"{family or 'unknown'}:{pattern}"


def _signature_group(family: Optional[str], reasoning_signature: str) -> str:
    text = f"{family or ''}:{reasoning_signature}".lower()
    if "gcd_divisor_sum" in text or ("gcd" in text and "sigma" in text):
        return "gcd_sigma"
    if "divisor_sum_mod" in text or "sigma_then_mod" in text:
        return "sigma_mod"
    if "modular_congruence" in text:
        return "modular"
    if "stars_and_bars" in text:
        return "counting"
    if "arithmetic_series" in text:
        return "arithmetic"
    if "units_digit" in text:
        return "units_digit"
    if "divisor_sum" in text:
        return "sigma"
    if "gcd" in text:
        return "gcd"
    return str(family or "unknown")


def _fusion_contract(work_item: Dict[str, Any]) -> Dict[str, Any]:
    value = work_item.get("fusion_contract") or {}
    return value if isinstance(value, dict) else {}


def _fusion_parent(fusion_contract: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = fusion_contract.get(key) or {}
    return value if isinstance(value, dict) else {}


def _checkpoint_visible(checkpoint: str, result: CertificationResult, features: Dict[str, Any]) -> bool:
    checkpoint = checkpoint.strip().lower()
    if not checkpoint:
        return True
    evidence = _evidence_text(result)
    params = dict(result.generated_params or {})
    family = str(result.family or result.target_family or "")
    if checkpoint == "family_certified":
        return result.status == "certified" and result.lean_level >= 2
    if checkpoint == "rich_factorization":
        count = int(features.get("gcd_prime_factor_count") or features.get("prime_factor_count") or 0)
        divisor_count = int(features.get("divisor_count") or features.get("divisor_count_of_gcd") or 0)
        if not count and family in {"gcd", "gcd_divisor_sum"} and {"a", "b"} <= set(params):
            count = _prime_factor_count(math.gcd(int(params["a"]), int(params["b"])))
        if not count and "n" in params:
            count = _prime_factor_count(int(params["n"]))
        return count >= 3 or divisor_count >= 16
    if checkpoint == "rich_gcd":
        return bool(features.get("gcd_nontrivial")) and int(
            features.get("gcd_prime_factor_count") or 0
        ) >= 2
    if checkpoint == "nontrivial_modular_reduction":
        modulus = int(features.get("modulus") or params.get("m") or 0)
        dividend = int(features.get("dividend") or params.get("a") or 0)
        return family == "modular_congruence" and modulus > 10 and dividend > modulus
    if checkpoint == "nontrivial_mod_remainder":
        modulus = int(features.get("modulus") or params.get("m") or 0)
        remainder = int(features.get("remainder") or 0)
        return modulus > 2 and remainder not in {0, 1, modulus - 1}
    if checkpoint == "binomial_formula":
        return family == "stars_and_bars" and bool(result.solution_skeleton)
    if checkpoint == "arithmetic_sum_formula":
        return family == "arithmetic_series" and bool(result.solution_skeleton or params)
    if checkpoint == "numeric_answer_verified":
        if "solution_verification_passed" in features:
            return bool(features.get("solution_verification_passed"))
        verification = result.quality_evidence.get("solution_verification") if result.quality_evidence else None
        if isinstance(verification, dict):
            return bool(verification.get("passed"))
        return bool(result.answer and result.solution_skeleton)
    if "certified" in checkpoint or "lean-certifiable" in checkpoint:
        return result.status == "certified" and (
            "family" in checkpoint or family.replace("_", " ") in checkpoint or family in checkpoint
        )
    if "answer exceeds" in checkpoint:
        match = re.search(r"answer exceeds\s+(\d+)", checkpoint)
        try:
            return bool(match and int(str(result.answer).strip()) > int(match.group(1)))
        except (TypeError, ValueError):
            return False
    if "dividend" in checkpoint and "at least" in checkpoint:
        match = re.search(r"at least\s+(\d+)", checkpoint)
        return bool(match and int(features.get("dividend") or params.get("a") or 0) >= int(match.group(1)))
    if "vars=" in checkpoint:
        match = re.search(r"vars\s*=\s*(\d+)", checkpoint)
        return bool(match and int(params.get("vars") or 0) == int(match.group(1)))
    if "modulus is prime" in checkpoint:
        modulus = int(features.get("modulus") or params.get("m") or 0)
        if not _is_prime(modulus):
            return False
        range_match = re.search(r"\[(\d+)\s*,\s*(\d+)\]", checkpoint)
        return not range_match or int(range_match.group(1)) <= modulus <= int(range_match.group(2))
    if "prime factor" in checkpoint and "at least" in checkpoint:
        match = re.search(r"at least\s+(\d+)", checkpoint)
        if not match:
            return False
        required = int(match.group(1))
        count = int(features.get("gcd_prime_factor_count") or features.get("prime_factor_count") or 0)
        if not count and family in {"gcd", "gcd_divisor_sum"} and {"a", "b"} <= set(params):
            count = _prime_factor_count(math.gcd(int(params["a"]), int(params["b"])))
        if not count and "n" in params:
            count = _prime_factor_count(int(params["n"]))
        return count >= required
    if "divisor count" in checkpoint and "at least" in checkpoint:
        match = re.search(r"at least\s+(\d+)", checkpoint)
        count = int(features.get("divisor_count") or features.get("divisor_count_of_gcd") or 0)
        return bool(match and count >= int(match.group(1)))
    if "sigma" in checkpoint and ("formula" in checkpoint or "multiplicative" in checkpoint or "computed" in checkpoint):
        return any(token in evidence for token in ["sigma", "divisor", "positive divisors"]) and any(
            token in evidence for token in ["factor", "prime", "product", "multiply", "*"]
        )
    if "binomial coefficient" in checkpoint and "computed" in checkpoint:
        return family == "stars_and_bars" and bool(result.solution_skeleton)
    if "sum formula" in checkpoint and family == "arithmetic_series":
        return bool(result.solution_skeleton or params)
    if "answer verified numerically" in checkpoint:
        if "solution_verification_passed" in features:
            return bool(features.get("solution_verification_passed"))
        verification = result.quality_evidence.get("solution_verification") if result.quality_evidence else None
        if isinstance(verification, dict):
            return bool(verification.get("passed"))
        return bool(result.answer and result.solution_skeleton)
    if checkpoint in {"reasoning_pattern", "nonempty_reasoning_pattern"}:
        return bool(result.reasoning_pattern)
    if checkpoint in {"solution_skeleton", "nonempty_solution_skeleton"}:
        return bool(result.solution_skeleton)
    if checkpoint in {"projected_params", "params_projected"}:
        return bool(result.projected_params or result.generated_params)
    if checkpoint in {"two_step_reasoning", "derived_object_chain"}:
        return int(features.get("reasoning_chain_depth") or 0) >= 2 or any(
            token in evidence for token in ["then", "derived", "sigma", "modulus", "gcd"]
        )
    if checkpoint in {"semantic_parent_contribution", "parent_contribution"}:
        return "parent_contributions" in evidence or "parent" in evidence
    return all(token in evidence for token in re.findall(r"[a-zA-Z0-9_]+", checkpoint))


def _default_checkpoints(result: CertificationResult, op_type: str, family: Optional[str]) -> List[str]:
    if op_type == "survivor":
        return []
    checkpoints = ["reasoning_pattern", "solution_skeleton", "projected_params"]
    if family in {"gcd_divisor_sum", "divisor_sum_mod"}:
        checkpoints.append("two_step_reasoning")
    if op_type == "crossover":
        checkpoints.append("semantic_parent_contribution")
    return checkpoints


def _contribution_visible(contribution: str, evidence: str) -> bool:
    numbers = {str(number) for number in _ints(contribution)}
    if numbers and any(number in evidence for number in numbers):
        return True
    words = {
        word
        for word in re.findall(r"[a-zA-Z_]{4,}", contribution.lower())
        if word not in {"parent", "problem", "family", "template", "parameter", "numeric"}
    }
    return bool(words and any(word in evidence for word in words))


def _param_values(params: Dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for value in params.values():
        try:
            values.add(int(value))
        except (TypeError, ValueError):
            continue
    return values


def _answer_int(problem: CertificationInput) -> Optional[int]:
    try:
        return int(str(problem.answer).strip())
    except (TypeError, ValueError):
        return None


def _composite_semantic_contribution(
    *,
    target_family: Optional[str],
    parent: CertificationInput,
    parent_family: Optional[str],
    generated_params: Dict[str, Any],
    evidence: str,
) -> Optional[str]:
    if target_family == "gcd_divisor_sum":
        if parent_family == "gcd":
            parent_params = _family_params(parent)
            if (
                {"a", "b"} <= set(parent_params)
                and int(generated_params.get("a", -1)) == int(parent_params["a"])
                and int(generated_params.get("b", -1)) == int(parent_params["b"])
            ):
                return "gcd_inputs_projected_to_composite_params"
            if "gcd" in evidence and "divisor" in evidence:
                return "gcd_structure_used_in_reasoning_skeleton"
        if parent_family == "divisor_sum" and any(
            token in evidence for token in ["divisor", "sigma", "sum divisors", "positive divisors"]
        ):
            return "divisor_sum_operation_used_in_reasoning_skeleton"
    if target_family == "divisor_sum_mod":
        if parent_family == "divisor_sum" and any(
            token in evidence for token in ["modulus", "divisor", "sigma", "sum divisors"]
        ):
            return "divisor_sum_operation_defines_modulus"
        if parent_family == "modular_congruence" and any(
            token in evidence for token in ["mod", "remainder", "reduction", "dividend"]
        ):
            return "modular_reduction_structure_used"
    return None


def _theorem_surface_without_name(text: Any) -> str:
    surface = str(text or "")
    surface = surface.split(":= by", 1)[0]
    surface = re.sub(r"\b(theorem|lemma)\s+[A-Za-z0-9_'.]+", r"\1 _", surface)
    return re.sub(r"\s+", " ", surface).strip().lower()


def _theorem_surface_without_name_or_numbers(text: Any) -> str:
    surface = _theorem_surface_without_name(text)
    surface = re.sub(r"\b\d+\b", "#", surface)
    return re.sub(r"\s+", " ", surface).strip()


def _theorem_conclusion_surface(text: Any) -> str:
    surface = _theorem_surface_without_name(text)
    if ":" not in surface:
        return surface
    return surface.rsplit(":", 1)[-1].strip()


def _theorem_final_goal_surface(text: Any) -> str:
    """Approximate the theorem's final goal after hypothesis arrows.

    This is intentionally shallow. It exists to distinguish a real pipeline
    dependency from a parent checkpoint that is merely introduced as an
    implication premise and then passed through unused.
    """
    conclusion = _theorem_conclusion_surface(text)
    for arrow in ("→", "->"):
        if arrow in conclusion:
            conclusion = conclusion.rsplit(arrow, 1)[-1].strip()
    return conclusion


def _is_parameter_shift_only_theorem(
    child_surface: str,
    parent_surfaces: List[str],
    result: CertificationResult,
) -> bool:
    if not child_surface or result.op_type != "mutation" or result.status != "certified":
        return False
    child_key = _theorem_surface_without_name_or_numbers(child_surface)
    if not child_key:
        return False
    if not any(child_key == _theorem_surface_without_name_or_numbers(surface) for surface in parent_surfaces):
        return False
    proof_text = f"{result.lean_code or ''} {result.proof_plan or ''}".lower()
    proof_is_mechanical = any(
        token in proof_text for token in ("native_decide", "decide", "norm_num", "linarith")
    )
    return proof_is_mechanical


def _is_auxiliary_conjunct_only_theorem(
    child_surface: str,
    parent_surfaces: List[str],
    result: CertificationResult,
) -> bool:
    if not child_surface or result.op_type != "mutation" or result.status != "certified":
        return False
    child_conclusion = _theorem_conclusion_surface(child_surface)
    if "∧" not in child_conclusion and " and " not in child_conclusion:
        return False
    parent_conclusions = [
        _theorem_conclusion_surface(surface)
        for surface in parent_surfaces
        if _theorem_conclusion_surface(surface)
    ]
    for parent_conclusion in parent_conclusions:
        if len(parent_conclusion) >= 8 and parent_conclusion in child_conclusion:
            return True
    return False


def _is_fin_one_vacuity_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}".lower()
    return "fin 1" in surface and any(
        marker in surface for marker in ("vacuous", "subsingleton.elim", "no distinct", "cannot be distinct")
    )


def _is_concrete_native_decide_projection(result: CertificationResult) -> bool:
    surface = f"{result.formal_statement or ''} {result.lean_code or ''}"
    if "native_decide" not in surface:
        return False
    return bool(
        re.search(r"\bNat\.divisors\s+\d+\b", surface)
        or re.search(r"\bdivisors\s+\d+\b", surface)
        or re.search(r"\b\d+\s*[+*^%]\s*\d+\b", surface)
    )


def _is_tautological_checkpoint_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''}".lower()
    if "n = n" in surface or "reflexive side condition" in surface:
        return True
    if re.search(r"h\w*\s*:\s*[^:]+=\s*24", surface) and "24 ∣" in str(result.formal_statement or result.lean_code):
        proof = str(result.lean_code or "").lower()
        return "rw [h" in proof or "simpa [h" in proof or "simp [h" in proof
    return False


def _is_piecewise_branch_only_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.proof_plan or ''}".lower()
    return (
        "solution" in surface
        and "= 0" in surface
        and any(marker in surface for marker in ("zero-branch", "zero branch", "n = 3", "n = 4", "if n ="))
    )


def _is_trivial_negation_chain_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}"
    compact = re.sub(r"\s+", "", surface).lower()
    text = surface.lower()
    return (
        "-(-" in compact
        or "neg_neg" in text
        or "negation wrapper" in text
        or "double negation" in text
        or "triple negation" in text
    )


def _is_trivial_add_zero_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}"
    compact = re.sub(r"\s+", "", surface).lower()
    text = surface.lower()
    return "+0" in compact or "0+" in compact or "add_zero" in text or "zero padding" in text


def _is_typeclass_narrowing_only_theorem(
    child_surface: str,
    parent_surfaces: List[str],
    result: CertificationResult,
) -> bool:
    if not child_surface or result.op_type != "mutation" or result.status != "certified":
        return False
    narrowing_pairs = [
        ("[commring", "[ring"),
        ("[linearorderedring", "[ring"),
        ("[field", "[divisionring"),
        ("[field", "[ring"),
        ("[commgroup", "[group"),
    ]
    child_conclusion = _theorem_conclusion_surface(child_surface)
    if not child_conclusion:
        return False
    for parent_surface in parent_surfaces:
        parent_conclusion = _theorem_conclusion_surface(parent_surface)
        if not parent_conclusion or parent_conclusion != child_conclusion:
            continue
        if any(stronger in child_surface and weaker in parent_surface for stronger, weaker in narrowing_pairs):
            return True
    return False


def _is_projection_only_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}".lower()
    if any(
        marker in surface
        for marker in (
            "project the",
            "projected from",
            "second component",
            "first component",
            "exact poddprime.1",
            "exact poddprime.2",
            ".1",
            ".2",
        )
    ) and any(marker in surface for marker in ("∧", " and ", "odd p ∧", "p.prime", "nat.prime")):
        return True
    formal = str(result.formal_statement or result.lean_code or "")
    return bool(
        re.search(r"\([^:()]+:\s*[^()]+∧[^()]+\)\s*:\s*[^:=]+", formal)
        and re.search(r"\bexact\s+[A-Za-z_][A-Za-z0-9_']*\.[12]\b", str(result.lean_code or ""))
    )


def _is_divisibility_weaken_only_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}".lower()
    if "divis" not in surface and "∣" not in surface:
        return False
    if not any(marker in surface for marker in ("24", "(24", ":ℤ)∣", ": ℤ) ∣")):
        return False
    weaker_targets = ("(3 : ℤ) ∣", "(4 : ℤ) ∣", "(8 : ℤ) ∣", "3 divides", "4 divides", "8 divides")
    if not any(target.lower() in surface for target in weaker_targets):
        return False
    proof_text = f"{result.proof_plan or ''} {result.lean_code or ''}".lower()
    return any(
        marker in proof_text
        for marker in (
            "unpack 24 divisibility",
            "24 divides",
            "exhibit 8*k",
            "exhibit 3*k",
            "exhibit 6*k",
            "quotient witnessing",
            "24*k",
        )
    )


def _is_fin_one_concrete_arithmetic_theorem(result: CertificationResult) -> bool:
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}".lower()
    if "fin 1" not in surface or "matrix.det" not in surface:
        return False
    return any(
        marker in surface
        for marker in (
            "digits 1 through",
            "digits 6 through",
            "one-by-one",
            "normalize each fin 1 determinant",
            "constant integer entries",
        )
    )


def _is_unit_product_closure_only_theorem(result: CertificationResult) -> bool:
    if result.op_type != "mutation" or result.status != "certified":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    ).lower()
    if "isunit" not in surface or "*" not in surface:
        return False
    if any(marker in surface for marker in ("orderof", "exists", "∃", "irreducible", "tendsto")):
        return False
    return any(
        marker in surface
        for marker in (
            ".mul",
            "isunit.mul",
            "product closure",
            "products of units",
            "unit product",
        )
    )


def _is_syntactic_wrapper_only_theorem(result: CertificationResult) -> bool:
    text = f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}".lower()
    wrapper_markers = (
        "syntactic wrapper",
        "wrapper only",
        "by simpa",
        "simpa only",
        "immediate by simpa",
    )
    if not any(marker in text for marker in wrapper_markers):
        return False
    return any(marker in text for marker in ("same theorem", "same conclusion", "unchanged conclusion", "without new obligation"))


def _theorem_gen_depth(problem_id: Any) -> int:
    return str(problem_id or "").count("__theorem_gen")


def _is_ap_index_only_theorem(result: CertificationResult) -> bool:
    """Reject AP children that only hide/recompute one index or one term.

    A closed-form or parameter-characterization theorem is a real role shift.
    A new ``m = affine(p,q)+c`` or ``a m = constant`` row is usually just the
    original AMC computation with a different wrapper.
    """
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    ap_markers = (
        "arithmetic-sequence",
        "arithmetic sequence",
        "arithmetic-progression",
        "arithmetic progression",
        "a(n+2)-a(n+1)",
        "a(n + 2) - a(n + 1)",
        "3 * p - q",
        "3*p-q",
    )
    if not any(marker in lower or marker in compact for marker in ap_markers):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    statement_plan = f"{result.statement or ''} {result.proof_plan or ''}".lower()
    if (
        any(marker in statement_plan for marker in ("closed form", "all indices", "every term"))
        or "∀" in final_goal
        or "forall" in final_goal
    ):
        return False
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    hidden_index = any(
        marker in lower or marker in compact
        for marker in (
            "hidden real parameter",
            "hidden parameter",
            "hmidx",
            "(m:ℝ)=",
            "(m : ℝ) =",
        )
    )
    single_index_goal = (
        bool(re.search(r"\ba\s+m\s*=", final_goal))
        or bool(re.search(r"\bm\s*=\s*\d+\b", final_goal))
        or "am=" in goal_compact
        or "m=2010" in goal_compact
        or "a m = 8041" in final_goal
    )
    return hidden_index and single_index_goal


def _is_ap_shifted_local_corollary_only(result: CertificationResult) -> bool:
    """Reject AP children that only prove a nearby shifted local corollary.

    The accepted target for the AMC AP family is a closed form or parameter
    characterization. Rows about ``a (m + k)``, a local gap, or a midpoint
    relation around the same hidden ``m`` are Lean-valid but manual-QA weak.
    """
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    ap_markers = (
        "arithmetic-sequence",
        "arithmetic sequence",
        "arithmetic-progression",
        "arithmetic progression",
        "a(n+2)-a(n+1)",
        "a(n + 2) - a(n + 1)",
        "3 * p - q",
        "3*p-q",
    )
    if not any(marker in lower or marker in compact for marker in ap_markers):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    statement_text = str(result.statement or "").lower()
    if (
        any(marker in statement_text for marker in ("closed form", "all indices", "every term"))
        or "∀" in final_goal
        or "forall" in final_goal.lower()
    ):
        return False
    has_hidden_shift = any(
        marker in lower or marker in compact
        for marker in (
            "hidden natural parameter",
            "hidden parameter",
            "hm:m+",
            "hm : m +",
            "hmidx",
            "m+5=2014",
            "m + 5 = 2014",
        )
    )
    shifted_terms = len(re.findall(r"a\s*\(\s*m\s*\+\s*\d+\s*\)", final_goal))
    shifted_terms += len(re.findall(r"a\(m\+\d+\)", goal_compact))
    local_relation = any(
        marker in goal_compact
        for marker in (
            "a(m+",
            "-a(m+",
            "+a(m+",
            "2*a(m+",
        )
    ) or any(
        marker in lower
        for marker in (
            "local gap",
            "centered midpoint",
            "shifted term",
            "centered shifted",
        )
    )
    return has_hidden_shift and local_relation and shifted_terms >= 1


def _is_mod_inverse_same_conclusion_paraphrase(result: CertificationResult) -> bool:
    """Reject repeated modulo-398 inverse rows that keep the same ``=57`` goal."""
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    if "398" not in compact or "*7" not in compact:
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    if not any(marker in goal_compact for marker in ("n=57", "k=57")):
        return False
    modulo_surface = "%398=1" in compact or "modulo 398" in lower or "remainder 1 modulo 398" in lower
    explicit_quotient = any(
        marker in compact
        for marker in (
            "=398*q+1",
            "=1+398*q",
            "=398* q+1",
            "=1+398* q",
        )
    ) or "quotient" in lower
    if modulo_surface:
        return True
    # Keep the first explicit quotient conversion as a useful representative;
    # descendants with the same final goal are paraphrases.
    return explicit_quotient and _theorem_gen_depth(result.problem_id) >= 2


def _is_solved_parameter_quotient_corollary_only(result: CertificationResult) -> bool:
    """Reject descendants that only expose arithmetic corollaries of ``n = 57``.

    These rows are Lean-complete, but manual QA treats them as solved-parameter
    bookkeeping: once the modulo-398 inverse forces ``n = 57``, conclusions like
    ``n / 19 = 3`` or ``2 ^ (n / 19) = 8`` add no new theorem role.
    """
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    if "398" not in compact or "*7" not in compact:
        return False
    if not any(marker in lower or marker in compact for marker in ("n = 57", "n=57", "force n", "bounded inverse")):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    quotient_goal = any(
        marker in goal_compact
        for marker in (
            "n/19=3",
            "n=19*q",
            "n=19*q∧",
            "2^(n/19)=8",
            "2^(n/19)%8=0",
            "2^q%8=0",
        )
    ) or any(
        marker in lower
        for marker in (
            "quotient n / 19",
            "quotient q",
            "exposed by n = 19q",
            "2 raised to the quotient",
        )
    )
    return quotient_goal


def _is_mod_inverse_arithmetic_corollary_only(result: CertificationResult) -> bool:
    """Reject direct remainder/divisibility corollaries after solving ``n = 57``."""
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    if "398" not in compact or "*7" not in compact:
        return False
    if not any(marker in lower or marker in compact for marker in ("n = 57", "n=57", "force n", "bounded inverse")):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    return any(marker in goal_compact for marker in ("n%19=0", "19∣n", "n=57"))


def _is_residue_finset_cardinality_restatement(result: CertificationResult) -> bool:
    """Reject fixed residue-set rows whose final goal is only the old card/power fact."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    residue_domain = (
        "residue modulo 8" in lower
        or "gcd(m,8)" in compact
        or "nat.gcdm8=1" in compact
        or "m%8=1∨m%8=3∨m%8=5∨m%8=7" in compact
        or "{1,3,5,7}" in compact
    )
    if not residue_domain:
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    if "3^n%8=1" in goal_compact or "s.card=4" in goal_compact or "n=4" in goal_compact:
        return True
    return (
        "cardinality" in lower
        and "3^n" in compact
        and not any(marker in compact for marker in ("s.sum", "s.prod", "*n", "a*n"))
    )


def _is_fixed_finite_aggregate_computation(result: CertificationResult) -> bool:
    """Reject fixed finite-set aggregate expressions closed by computation only."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    residue_domain = (
        "finset.icc17" in compact
        or "{1,3,5,7}" in compact
        or "m%8=1∨m%8=3∨m%8=5∨m%8=7" in compact
        or "residue modulo 8" in lower
    )
    aggregate_goal = any(marker in compact for marker in ("s.sum", "s.prod", "s.card"))
    fixed_computation = "native_decide" in lower or "rewrite to the explicit finite set" in lower
    if residue_domain and aggregate_goal and fixed_computation:
        return True
    return False


def _is_cardinality_arithmetic_pipeline_only(result: CertificationResult) -> bool:
    """Reject fixed cardinality facts combined only by arithmetic/modular wrapping."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    divisor_card = (
        "endpoint-erased divisor" in lower
        or "nat.divisors(30^4)" in compact
        or "nat.divisors(30 ^ 4)" in lower
        or "((nat.divisors(30^4)).erase1).erase(30^4)" in compact
    )
    residue_card = (
        "reduced residues from 1 through 7" in lower
        or "finset.icc17" in compact
        or "nat.gcdx8=1" in compact
        or "nat.gcd x 8 = 1" in lower
    )
    if not (divisor_card and residue_card):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_lower = final_goal.lower()
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    arithmetic_wrapper = (
        "3^" in goal_compact
        or "%8" in goal_compact
        or "modulo 8" in lower
        or "raised to" in lower
        or ".card+" in goal_compact
        or "+(" in goal_compact
    )
    richer_role = any(
        marker in goal_lower or marker in lower
        for marker in (
            "exists",
            "unique",
            "least",
            "greatest",
            "iff",
            "if and only if",
            "classification",
            "characterization",
        )
    ) or "∃" in final_goal or "∀" in final_goal or "↔" in final_goal
    return arithmetic_wrapper and not richer_role


def _is_linear_equation_shift_corollary_only(result: CertificationResult) -> bool:
    """Reject one-step linear corollaries after solving the same ``y = 9`` parent."""
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    algebra_parent = (
        "y+6" in compact
        and ("2*12" in compact or "12-(y+6)=y-12" in compact)
        and (
            "y=9" in compact
            or "y = 9" in lower
            or "shifted conclusion" in lower
            or "resulting linear equation" in lower
        )
    )
    if not algebra_parent:
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    if goal_compact in {"y=9", "(y:ℝ)=9"}:
        return False
    shifted_linear_goal = bool(
        re.search(r"\by\s*[+\-]\s*\d+\s*=\s*\d+\b", final_goal)
        or re.search(r"\bt\s*≤\s*y\s*[+\-]\s*\d+", surface)
        or re.search(r"\bt\s*≤\s*\d+\b", final_goal)
    )
    if not shifted_linear_goal:
        return False
    return any(
        marker in lower
        for marker in (
            "nearby consequence",
            "shifted conclusion",
            "same-sort real arithmetic",
            "linear arithmetic conclusion",
            "derive the checkpoint y = 9",
            "derive the internal checkpoint y = 9",
        )
    ) or "linarith" in lower


def _is_ap_bound_padding_only(result: CertificationResult) -> bool:
    """Reject AP children that only wrap the solved 21st term in a bound.

    A theorem such as ``a + 20*d ≤ B`` from ``a + 20*d = 135`` and ``135 ≤ B``
    is a valid corollary, but it is not an accepted-grade harder problem.
    """
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    ap_markers = (
        "arithmetic sequence",
        "arithmetic-sequence",
        "a+6*d=30",
        "a + 6 * d = 30",
        "a+10*d=60",
        "a + 10 * d = 60",
    )
    if not any(marker in lower or marker in compact for marker in ap_markers):
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    if "≤" not in final_goal and "<=" not in final_goal and "\\le" not in final_goal:
        return False
    if "a+20*d" not in goal_compact and "a+20*d+t" not in goal_compact:
        return False
    padding_hypothesis = any(
        marker in lower or marker in compact
        for marker in (
            "135 ≤ b",
            "135<=b",
            "135 + t ≤ b",
            "135+t<=b",
            "135 + t ≤ c",
            "135+t<=c",
            "135 + s ≤ m",
            "135+s<=m",
            "c ≤ b",
            "c<=b",
            "at least 135 plus",
            "upper bound",
            "shifted inequality",
            "two-step bound",
            "combine that checkpoint with the added real bound",
        )
    )
    richer_role = (
        any(
            marker in lower or marker in final_goal
            for marker in (
                "unique",
                "least",
                "greatest",
                "if and only if",
                "∀",
                "∃",
                "closed form",
                "parameter characterization",
            )
        )
        or bool(re.search(r"\biff\b", lower))
        or "↔" in final_goal
    )
    return padding_hypothesis and not richer_role


def _is_ap_interval_bound_padding_only(result: CertificationResult) -> bool:
    """Reject AP children that only put the solved 21st term in a padded interval."""
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    ap_parent = (
        "arithmetic sequence" in lower
        or "a+6*d=30" in compact
        or "a+10*d=60" in compact
    )
    if not ap_parent or "a+20*d" not in compact:
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    interval_goal = (
        "set.ioc" in goal_compact
        or "set.icc" in goal_compact
        or "∈set." in goal_compact
        or "lies in the interval" in lower
        or "closed interval" in lower
    )
    if not interval_goal:
        return False
    padding_hypothesis = any(
        marker in lower or marker in compact
        for marker in (
            "135 ≤ m",
            "135≤m",
            "l ≤ 135",
            "l≤135",
            "135 ≤ u",
            "135≤u",
            "135 ≤ b",
            "135≤b",
            "135 ≤ upper",
            "135 + s ≤ m",
            "135+s<=m",
            "upper bound",
            "real bound",
        )
    )
    richer_role = any(
        marker in lower or marker in final_goal
        for marker in (
            "unique",
            "least",
            "greatest",
            "if and only if",
            "∀",
            "∃",
            "characterization",
        )
    ) or "↔" in final_goal
    return padding_hypothesis and not richer_role


def _is_finite_mod_inverse_window_restatement(result: CertificationResult) -> bool:
    """Reject modulo-inverse children restated as finite filtered membership."""
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    inverse_parent = (
        "398" in compact
        and ("n*7%398=1" in compact or "multiplicative inverse of 7 modulo 398" in lower)
    )
    if not inverse_parent:
        return False
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    finite_window = (
        "finset.range398" in goal_compact
        or "finset.icc5060" in goal_compact
        or "finite filtered window" in lower
        or "filtered window" in lower
    )
    repeats_inverse = "k*7%398=1" in goal_compact or "k * 7 % 398 = 1" in final_goal
    repeats_known_remainder = (
        "k%19=0" in goal_compact
        or "k % 19 = 0" in final_goal
        or "remainder 0 modulo 19" in lower
        or "at most 57" in lower
    )
    richer_role = any(
        marker in lower or marker in final_goal
        for marker in (
            "sum",
            "product",
            "exists",
            "unique",
            "least",
            "greatest",
            "classification",
            "↔",
        )
    )
    return finite_window and repeats_inverse and repeats_known_remainder and not richer_role


def _is_parent_theorem_assumption_smuggling(result: CertificationResult) -> bool:
    """Reject crossovers that take a parent theorem as a broad hypothesis."""
    if result.status != "certified" or result.op_type != "crossover":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    parent_theorem_hypothesis = (
        "supplied" in lower
        or "parent divisor-sum" in lower
        or "h₃:∀c" in compact
        or "h3:∀c" in compact
        or "hprime_source:∀c" in compact
        or "h_prime_source:∀c" in compact
    )
    parent_applied_directly = any(
        marker in compact
        for marker in (
            "exacth₃",
            "exacth3",
            "exacthprime_source",
            "exacth_prime_source",
            "h₃a",
            "h3a",
            "hprime_sourcea",
        )
    )
    theorem_shape = (
        "∀c:ℕ" in compact
        and "nat.divisors500" in compact
        and "nat.prime" in compact
    )
    return parent_theorem_hypothesis and parent_applied_directly and theorem_shape


def _is_native_decide_fixed_domain_computation(result: CertificationResult) -> bool:
    if result.status != "certified":
        return False
    surface = f"{result.formal_statement or ''} {result.lean_code or ''} {result.proof_plan or ''}"
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    if "native_decide" not in lower:
        return False
    fixed_domain = any(
        marker in compact
        for marker in (
            "finset.icc17",
            "{1,3,5,7}",
            "m%8=1∨m%8=3∨m%8=5∨m%8=7",
            "nat.divisors500",
            "finset.range98",
        )
    )
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    final_lower = final_goal.lower()
    symbolic_role = any(
        marker in lower
        for marker in (
            "for every",
            "exists",
            "∃",
            "unique",
            "least",
            "greatest",
            "iff",
            "if and only if",
        )
    ) or "∀" in final_goal or "exists" in final_lower or "∃" in final_goal
    return fixed_domain and not symbolic_role


def _is_artificial_bridge_to_existing_pipeline(result: CertificationResult) -> bool:
    """Reject crossovers that bolt an artificial AP bridge onto an existing pipeline.

    Example: use an AP bound plus ``(n : ℝ) = a + 20*d + t`` only to recover
    ``n < 398``, then run an already accepted inverse+coprime-count modular
    exponent theorem. The AP parent is technically consumed, but only through a
    contrived bridge hypothesis rather than a natural theorem role.
    """
    if result.status != "certified" or result.op_type != "crossover":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    has_ap_bridge = (
        ("a+20*d" in compact or "a+20*d+t" in compact)
        and (
            "hn_shift" in lower
            or "(n:ℝ)=a+20*d+t" in compact
            or "(n : ℝ) = a + 20 * d + t" in lower
            or "bridge" in lower
        )
        and ("b<398" in compact or "hB : B < 398".lower() in lower or "obtain n < 398" in lower)
    )
    has_existing_mod_pipeline = (
        ("n*7%398=1" in compact or "multiplicative inverse of 7 modulo 398" in lower)
        and ("finset.icc17" in compact or "gcd(x,8)=1" in compact or "coprime to 8" in lower)
        and ("3^(n+c)%8=3" in compact or "3 ^ (n + c) % 8 = 3" in lower)
    )
    return has_ap_bridge and has_existing_mod_pipeline


def _is_numeric_bound_fitting_crossover(result: CertificationResult) -> bool:
    """Reject crossovers whose fusion is only a fitted numeric inequality.

    Example: AP parent gives ``a+20*d=135`` and inverse parent gives
    ``n/19=3``; the child states ``a+20*d ≤ 132 + n/19``. Both parents appear,
    but the coupling is a hand-fitted constant equality rather than a new
    mathematical object or reusable theorem role.
    """
    if result.status != "certified" or result.op_type != "crossover":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    has_ap_value = "a+20*d" in compact and ("135" in compact or "21st term" in lower)
    has_inverse_quotient = (
        (
            "n*7%398=1" in compact
            or "inverse of 7 modulo 398" in lower
        )
        and ("n/19" in compact or "quotient" in lower)
    )
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    surface_bound = "a+20*d≤132+" in compact or "a+20*d<=132+" in compact
    fitted_bound = (
        (
            (("≤" in final_goal or "<=" in final_goal) and "132+" in goal_compact)
            or surface_bound
        )
        and ("n/19" in goal_compact or "n/19" in compact or "quotient" in lower)
    )
    return has_ap_value and has_inverse_quotient and fitted_bound


def _is_same_target_role_already_accepted(result: CertificationResult) -> bool:
    """Catch repeated unit/orderOf targets that manual QA already rejected.

    This is intentionally local and conservative: it does not inspect the
    accepted ledger. It only catches the repeatedly observed surface where the
    child restates that a negated unit object, or its inverse, has the same
    order as the original unit object.
    """
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()

    has_order_unit_surface = (
        "orderof" in compact
        and (
            "unit object" in lower
            or "unit witness" in lower
            or "units" in lower
            or "isunit" in compact
            or "ˣ" in surface
        )
        and (
            "negated unit" in lower
            or "negating u" in lower
            or "-u" in surface
            or "inverse" in lower
            or "⁻¹" in surface
        )
    )
    if not has_order_unit_surface:
        return False

    # Keep genuinely new orderOf theorems that ask for a universal divisor
    # transfer or an arithmetic divisibility target rather than another same
    # order/inverse surface.
    if ("∀" in final_goal or "forall" in lower) and ("∣" in final_goal or "divides" in lower):
        return False
    if "16∣" in goal_compact or "16 ∣" in final_goal or "odd" in lower:
        return False

    same_order_target = any(
        marker in goal_compact
        for marker in (
            "orderofv⁻¹=orderofv",
            "orderofv=orderofv⁻¹",
            "orderof(-",
            "orderof(-u",
            "orderof(unit",
        )
    ) or any(
        marker in lower
        for marker in (
            "has the same order as",
            "same order as its inverse",
            "inverse has the same order",
            "underlying value -u",
        )
    )
    return bool(same_order_target)


def _is_witness_packaging_only(result: CertificationResult) -> bool:
    """Reject explicit-witness packaging when it is the final theorem role."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    gaussian_surface = any(
        marker in lower or marker in compact
        for marker in (
            "gaussian integer",
            "gaussian integers",
            "gaussianint",
            "ℤ[i]",
            "(1+i)^2",
            "(1+i)",
            "1+i",
        )
    )
    if not gaussian_surface:
        return False
    witness_surface = any(
        marker in lower
        for marker in (
            "explicit witness",
            "witness equality",
            "witness w",
            "there exists an explicit witness",
            "packaged as a local implication",
        )
    ) or ("∃" in surface and "w" in surface and ("∣" in surface or "divides" in lower))
    divisibility_surface = "divides" in lower or "∣" in surface or "dvd" in compact
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_lower = final_goal.lower()
    goal_is_witness_or_same_divisibility = (
        "∃" in final_goal
        or "∃" in surface
        or "exists" in goal_lower
        or "there exists" in lower
        or ("∣" in final_goal and ("2" in final_goal or "1+i" in goal_lower))
    )
    return bool(witness_surface and divisibility_surface and goal_is_witness_or_same_divisibility)


def _is_coefficient_engineering_only(result: CertificationResult) -> bool:
    """Reject irrationality variants that only change the rational factor."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    final_goal = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    goal_compact = re.sub(r"\s+", "", final_goal).lower()
    if "irrational" not in lower:
        return False
    # Do not reject broader translate-style generalizations merely because
    # they are about irrationality.
    if ("∀q:ℚ" in goal_compact or "∀(q:ℚ)" in goal_compact) and "x+q" in goal_compact:
        return False
    coefficient_markers = (
        "nonzero rational coefficient",
        "nonzero rational factor",
        "rational expression",
        "rational factor",
        "q^2 + 1",
        "q ^ 2 + 1",
        "square_add_one",
        "ratcast",
        "rat.cast",
        "mul_ratcast",
        "3/5 - 1/10",
        "-3/2",
        "(ab - c)^2 + 1",
    )
    has_coefficient_marker = any(marker in lower or marker in compact for marker in coefficient_markers)
    goal_is_factor_mul = (
        ("irrational" in goal_compact or "irrational" in compact)
        and (
            "x*" in goal_compact
            or "*x" in goal_compact
            or "x*(" in goal_compact
            or "x*" in compact
            or "*x" in compact
            or "multiplied by" in lower
        )
    )
    return bool(has_coefficient_marker and goal_is_factor_mul)


def _is_theorem_local_corollary_dominated(result: CertificationResult) -> bool:
    """Reject local corollaries already dominated by stronger accepted roles.

    This is deliberately narrow: it catches the recurring miniF2F units-digit
    surface ``20 ∣ k -> (k^2 + 2^k) % 10 = 6``, which is weaker than existing
    mod-20, shifted-exponent, and non-primality accepted variants.
    """
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = f"{result.statement or ''} {result.formal_statement or ''} {result.proof_plan or ''}"
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    has_units_digit_goal = "units digit" in lower or "%10=6" in compact or "%10 = 6" in lower
    has_twenty_multiple = (
        "20∣k" in compact
        or "20 ∣ k" in surface
        or "divisible by 20" in lower
        or ("positive multiple" in lower and "20" in compact)
    )
    has_square_power = "k^2+2^k" in compact or "k ^ 2 + 2 ^ k" in lower
    richer_goal = (
        any(marker in compact for marker in ("%20=16", "¬nat.prime", "nat.prime", "k+20*t", "↔"))
        or "if and only if" in lower
        or "not prime" in lower
    )
    return bool(has_units_digit_goal and has_twenty_multiple and has_square_power and not richer_goal)


def _is_definitional_extensionality_only(result: CertificationResult) -> bool:
    """Reject equality of two finite objects defined by the same predicate."""
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''} {result.solution or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    finite_set_equality = (
        ("s t : finset" in lower or "s,t:finset" in compact or "(s t : finset" in lower)
        and ("t=s" in compact or "s=t" in compact)
    )
    same_predicate_hypotheses = (
        ("∀n" in compact or "forall n" in lower or "every natural number" in lower)
        and ("↔" in surface or "->" in surface or "→" in surface)
        and any(marker in compact for marker in ("n∈s", "nins"))
        and any(marker in compact for marker in ("n∈t", "nint"))
    )
    extensionality_proof = any(
        marker in lower
        for marker in (
            "extensionality",
            "prove equality of finite sets",
            "extensionality theorem",
        )
    ) or "ext " in lower
    return bool(finite_set_equality and same_predicate_hypotheses and extensionality_proof)


def _is_pid_definition_restatement(result: CertificationResult) -> bool:
    """Reject PID children that only restate the constructor data.

    These are Lean-valid and sometimes useful scaffold, but manual QA rejected
    them as paper-grade rows because the final theorem is just
    ``∀ I, I.IsPrincipal -> IsPrincipalIdealRing`` under another name.
    """
    if result.status != "certified" or result.op_type not in {"mutation", "crossover"}:
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''} {result.solution or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    pid_target = "isprincipalidealring" in compact or "principal ideal ring" in lower
    explicit_generator = (
        "∀i:ideal" in compact
        or "∀i:ideal" in compact.replace(" ", "")
        or "idealgenerator" in compact
        or "idealspangenerator" in compact
        or "every ideal" in lower and "principal" in lower
    )
    constructor_like = (
        "exact⟨" in compact
        or "exact<" in compact
        or "is assigned a principal generator" in lower
        or "span {g(i)}" in lower
        or "span({idealgeneratori}" in compact
    )
    return bool(pid_target and explicit_generator and constructor_like)


def _is_standard_library_theorem_restatement(result: CertificationResult) -> bool:
    """Reject direct restatements of a named library theorem.

    The known ProofNet surface is the algebraically-closed root-set cardinality
    iff separability theorem. Keep crossovers that feed this theorem with a real
    upstream checkpoint; reject only mutation rows whose whole role is exposing
    the existing theorem with an extra displayed degree parameter.
    """
    if result.status != "certified" or result.op_type != "mutation":
        return False
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''} {result.solution or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    rootset_sep = (
        "rootset" in compact
        and "natdegree" in compact
        and "separable" in compact
        and ("if and only if" in lower or "↔" in surface)
    )
    library_proof = "card_rootset_eq_natdegree_iff_of_splits" in compact
    has_real_upstream_checkpoint = any(
        marker in compact
        for marker in (
            "hrep:",
            "hn:odd",
            "isunit",
            "ideal.span",
            "p.derivative",
            "p^r",
        )
    )
    return bool(rootset_sep and library_proof and not has_real_upstream_checkpoint)


def _theorem_accepted_grade_flags(result: CertificationResult) -> List[str]:
    """Manual-QA inspired accepted-grade filters for certified theorem children.

    These flags do not broaden worker output. They only reclassify Lean-complete
    but paper-unusable artifacts as retry/selection failures.
    """
    if result.status != "certified" or result.op_type in {"survivor", "fallback_survivor"}:
        return []
    surface = (
        f"{result.statement or ''} {result.formal_statement or ''} "
        f"{result.lean_code or ''} {result.proof_plan or ''} {result.solution or ''}"
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    flags: List[str] = []

    prime_domain_surface = (
        "prime divisors" in lower
        or "nat.prime" in lower
        or "filter(funx" in compact
    )
    divisor_sum_500_surface = (
        "positive divisors of 500" in lower
        or "nat.divisors500" in compact
        or "sigma_1(500)" in lower
        or "sigma 1) 500" in lower
    )
    ap_surface = (
        "arithmetic progression" in lower
        or "finset.range98" in compact
        or "u(n+1)" in lower
        or "u(n + 1)" in lower
    )

    if _is_ap_index_only_theorem(result):
        flags.append("ap_index_only_theorem")
    if _is_ap_shifted_local_corollary_only(result):
        flags.append("ap_shifted_local_corollary_only")
    if _is_ap_bound_padding_only(result):
        flags.append("ap_bound_padding_only")
    if _is_ap_interval_bound_padding_only(result):
        flags.append("ap_interval_bound_padding_only")
    if _is_mod_inverse_same_conclusion_paraphrase(result):
        flags.append("mod_inverse_same_conclusion_paraphrase")
    if _is_solved_parameter_quotient_corollary_only(result):
        flags.append("solved_parameter_quotient_corollary_only")
    if _is_mod_inverse_arithmetic_corollary_only(result):
        flags.append("mod_inverse_arithmetic_corollary_only")
    if _is_residue_finset_cardinality_restatement(result):
        flags.append("residue_finset_cardinality_restatement")
    if _is_finite_mod_inverse_window_restatement(result):
        flags.append("finite_mod_inverse_window_restatement")
    if _is_fixed_finite_aggregate_computation(result):
        flags.append("fixed_finite_aggregate_computation")
    if _is_cardinality_arithmetic_pipeline_only(result):
        flags.append("cardinality_arithmetic_pipeline_only")
    if _is_linear_equation_shift_corollary_only(result):
        flags.append("linear_equation_shift_corollary_only")
    if _is_native_decide_fixed_domain_computation(result):
        flags.append("native_decide_fixed_domain_computation")
    if _is_artificial_bridge_to_existing_pipeline(result):
        flags.append("artificial_bridge_to_existing_pipeline")
    if _is_numeric_bound_fitting_crossover(result):
        flags.append("numeric_bound_fitting_crossover")
    if _is_parent_theorem_assumption_smuggling(result):
        flags.append("parent_theorem_assumption_smuggling")
    if _is_same_target_role_already_accepted(result):
        flags.append("same_target_role_already_accepted")
    if _is_witness_packaging_only(result):
        flags.append("witness_packaging_only")
    if _is_coefficient_engineering_only(result):
        flags.append("coefficient_engineering_only")
    if _is_theorem_local_corollary_dominated(result):
        flags.append("theorem_local_corollary_dominated")
    if _is_definitional_extensionality_only(result):
        flags.append("definitional_extensionality_only")
    if _is_pid_definition_restatement(result):
        flags.append("pid_definition_restatement")
    if _is_standard_library_theorem_restatement(result):
        flags.append("standard_library_theorem_restatement")
    if (
        result.op_type == "crossover"
        and "orderof" in compact
        and (
            "iforderof" in compact
            or "indicator" in lower
            or "selector" in lower
            or "selected by orderof" in lower
        )
        and ("irrational" in lower or "rational factor" in lower)
    ):
        flags.append("order_equality_selector_only")

    if result.op_type == "mutation":
        helper_markers = (
            "prime_divisor_finset",
            "finset of prime divisors",
            "has cardinality",
            ".card =",
            "product of the prime divisors",
            ".prod id",
            "sum of the prime divisors",
        )
        if prime_domain_surface and divisor_sum_500_surface and any(marker in lower for marker in helper_markers):
            flags.append("proof_infrastructure_only")
        if ap_surface and any(
            marker in lower
            for marker in (
                "odd-indexed",
                "even-indexed",
                "odd indexed",
                "even indexed",
                "sum of u(1) through u(98)",
                "first 98 terms",
            )
        ) and not prime_domain_surface:
            flags.append("direct_parent_corollary_only")
        if len(result.problem_id.split("__theorem_gen")) >= 5 and not any(
            marker in lower
            for marker in (
                "pipeline",
                "master theorem",
                "prime divisors",
                "cardinality",
                "finite sum of its elements",
            )
        ):
            flags.append("lineage_complexity_without_new_role")

    if result.op_type == "crossover" and ap_surface and prime_domain_surface:
        has_domain_sum = (
            "h_prime_sum" in surface
            or "finite sum of its elements" in lower
            or "rational finite sum" in lower
            or "sum(s)" in compact
        )
        has_domain_card = (
            "h_card" in surface
            or "cardinality" in lower
            or ".card=4" in compact
            or ".card = 4" in lower
        )
        shifted_or_scaled_index = any(
            marker in compact
            for marker in (
                "u(p+1)",
                "u(p+2)",
                "u(p+3)",
                "u(2*p)",
                "u(2*p",
                "u((2*p",
                "u(((finset.filter",
                ".card)*p+1",
            )
        )
        plain_domain_sum = (
            "sum over the distinct prime divisors" in lower
            or "sum over distinct prime divisors" in lower
            or "∑p∈finset.filter" in compact
        )
        if shifted_or_scaled_index and not (has_domain_card and has_domain_sum):
            flags.append("affine_index_drift_only")
        if has_domain_card and not has_domain_sum and any(
            marker in compact for marker in ("finset.range", "u(2*k.succ)", "u(k.succ)")
        ):
            flags.append("cardinality_only_window")
        if plain_domain_sum and not shifted_or_scaled_index:
            return sorted(set(flags))
        if has_domain_card and has_domain_sum:
            return sorted(set(flags))
    elif result.op_type == "crossover" and len(result.problem_id.split("__theorem_gen")) >= 6:
        if not any(marker in lower for marker in ("master theorem", "pipeline", "finite sum of its elements")):
            flags.append("lineage_complexity_without_new_role")

    return sorted(set(flags))


def _is_side_by_side_conjunction_theorem(
    child_surface: str,
    parent_surfaces: List[str],
    result: CertificationResult,
) -> bool:
    if result.op_type != "crossover" or result.status != "certified":
        return False
    child_conclusion = _theorem_conclusion_surface(child_surface)
    if "∧" not in child_conclusion and " and " not in child_conclusion:
        return False
    parent_conclusions = [
        _theorem_conclusion_surface(surface)
        for surface in parent_surfaces
        if _theorem_conclusion_surface(surface)
    ]
    contained = [
        parent_conclusion
        for parent_conclusion in parent_conclusions
        if len(parent_conclusion) >= 8 and parent_conclusion in child_conclusion
    ]
    text = f"{result.statement or ''} {result.proof_plan or ''} {result.lean_code or ''}".lower()
    return len(contained) >= 2 or any(
        marker in text
        for marker in (
            "side-by-side",
            "side by side",
            "independent conjunction",
            "prove both parent theorems",
        )
    )


def _is_pipeline_composite_theorem(result: CertificationResult, work_item: Dict[str, Any]) -> bool:
    if result.op_type != "crossover" or result.status != "certified":
        return False
    text = (
        f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} "
        f"{result.proof_plan or ''} {getattr(result, 'unified_obligation', '') or ''} "
        f"{getattr(result, 'why_not_conjunction', '') or ''}"
    ).lower()
    fusion_contract = _fusion_contract(work_item)
    mechanism = str(fusion_contract.get("fusion_mechanism") or "").lower()
    if mechanism == "sequential_composition":
        return True
    return any(
        marker in text
        for marker in (
            "pipeline",
            "feeds into",
            "used as input",
            "becomes the hypothesis",
            "derived object",
            "intermediate lemma",
            "checkpoint from parent",
        )
    )


def _is_lemma_bundle_master_theorem(result: CertificationResult, work_item: Dict[str, Any]) -> bool:
    if result.op_type != "crossover" or result.status != "certified":
        return False
    text = (
        f"{result.statement or ''} {result.formal_statement or ''} {result.lean_code or ''} "
        f"{result.proof_plan or ''} {result.solution or ''} "
        f"{work_item.get('operator_card') or {}} {work_item.get('operator_goal') or ''} "
        f"{work_item.get('goal') or ''} {work_item.get('fusion_goal') or ''}"
    ).lower()
    if not any(
        marker in text
        for marker in (
            "lemma_bundle_master",
            "bundle→master",
            "bundle-to-master",
            "master theorem",
            "multiple parent checkpoints",
            "different proof obligation",
        )
    ):
        return False
    if "∧" in _theorem_conclusion_surface(result.formal_statement or result.lean_code):
        return False
    parent_usage_count = len(
        dict(result.semantic_parent_contribution or result.parent_contributions or {})
    )
    obligation_count = len(result.proof_obligations or [])
    subgoal_markers = sum(
        1
        for marker in (
            "subgoal",
            "checkpoint",
            "intermediate lemma",
            "local lemma",
            "derive",
            "consume",
        )
        if marker in text
    )
    return parent_usage_count >= 2 and (obligation_count >= 2 or subgoal_markers >= 2)


THEOREM_USAGE_STOPWORDS = {
    "theorem",
    "lemma",
    "import",
    "mathlib",
    "aesop",
    "open",
    "set_option",
    "by",
    "exact",
    "simp",
    "simpa",
    "rw",
    "intro",
    "intros",
    "have",
    "show",
    "from",
    "fun",
    "forall",
    "true",
    "false",
    "type",
    "sort",
    "prop",
    "proof",
    "parent",
    "child",
}


def _lean_tokens(text: Any) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*", str(text or ""))
        if len(token) >= 3
    }
    return {token for token in tokens if token not in THEOREM_USAGE_STOPWORDS}


def _parent_checkpoint_consumption(
    result: CertificationResult,
    parents: List[CertificationInput],
) -> Dict[str, Dict[str, Any]]:
    """Best-effort evidence that parent checkpoints reached child proof surfaces.

    Worker ``parent_usage`` text is useful but too easy to satisfy with prose.
    This check looks for parent-specific Lean/formal atoms in the generated
    formal statement or proof body. It is intentionally conservative and only
    used as a quality signal; Lean certification remains the correctness gate.
    """
    lean_surface = f"{result.formal_statement or ''}\n{result.lean_code or ''}"
    proof_surface = f"{result.proof_plan or ''}\n{result.statement or ''}"
    final_goal_surface = _theorem_final_goal_surface(result.formal_statement or result.lean_code)
    lean_tokens = _lean_tokens(lean_surface)
    proof_tokens = _lean_tokens(proof_surface)
    final_goal_tokens = _lean_tokens(final_goal_surface)
    evidence: Dict[str, Dict[str, Any]] = {}
    for parent in parents:
        metadata = parent.metadata or {}
        parent_surface = "\n".join(
            str(part or "")
            for part in (
                parent.id,
                parent.statement,
                metadata.get("formal_statement"),
                metadata.get("lean_code"),
                metadata.get("solution"),
            )
        )
        parent_tokens = _lean_tokens(parent_surface)
        # The declaration name often changes, so prefer mathematical atoms over ids.
        parent_tokens.discard(str(parent.id or "").lower())
        overlap_lean = sorted(parent_tokens & lean_tokens)
        overlap_proof = sorted(parent_tokens & proof_tokens)
        overlap_final_goal = sorted(parent_tokens & final_goal_tokens)
        evidence[parent.id] = {
            "lean_overlap": overlap_lean[:12],
            "proof_plan_overlap": overlap_proof[:12],
            "final_goal_overlap": overlap_final_goal[:12],
            "lean_overlap_count": len(overlap_lean),
            "proof_plan_overlap_count": len(overlap_proof),
            "final_goal_overlap_count": len(overlap_final_goal),
            "consumed_in_lean_surface": len(overlap_lean) >= 2,
            "mentioned_in_plan": len(overlap_proof) >= 2,
            "consumed_in_final_goal": len(overlap_final_goal) >= 2,
        }
    return evidence


def _has_unused_parent_checkpoint(
    consumption: Dict[str, Dict[str, Any]],
    result: CertificationResult,
) -> bool:
    if result.op_type != "crossover" or result.status != "certified":
        return False
    text = f"{result.lean_code or ''}\n{result.proof_plan or ''}\n{result.statement or ''}".lower()
    pass_through_marker = any(
        marker in text
        for marker in (
            "have _ :=",
            "have _ :",
            "have hpoly_checkpoint",
            "checkpoint_consumed",
            "passed through",
            "pass through",
            "formal hypothesis",
        )
    )
    if not pass_through_marker:
        return False
    for item in consumption.values():
        if item.get("consumed_in_lean_surface") and not item.get("consumed_in_final_goal"):
            return True
    return False


def _root_lineage(problem_id: str) -> str:
    root = str(problem_id or "")
    for marker in ("__x__", "__theorem_gen", "__fallback"):
        if marker in root:
            root = root.split(marker, 1)[0]
    return root


def _parent_formal_surface(parent: CertificationInput) -> str:
    metadata = parent.metadata or {}
    return _theorem_surface_without_name(
        metadata.get("formal_statement") or metadata.get("lean_code") or ""
    )


#: Composition patterns whose whole point is that the parents interact. The
#: generator's catalog reserves `shared_parameter_binding` for loose coupling
#: and these three for genuine dependency, so asking for one of them and
#: getting a proof whose parents never meet is a broken contract regardless of
#: what difficulty label the slot carried.
COUPLED_PATTERNS = frozenset({
    "serial_pipeline", "coupled_system_extension", "cross_family_bridge",
})


def _parent_lean(parent: Any) -> str:
    """A parent's *proof*, wherever the input model happens to keep it.

    `CertificationInput` carries only id/statement/answer and a metadata bag,
    so the Lean is in the bag; a survivor row reaching here as a plain dict is
    read the same way. Returning "" is what marks the check unmeasurable, so
    this must not raise on a shape it does not recognise.

    It must be the proof and nothing else. This fell back to `formal_statement`,
    which is not a weaker proof but a different object, and the fallback fired
    on every seed parent because a seed keeps its proof under
    `verification_code` and leaves `lean_code` empty. The coupling gate then
    received two statements, found no conclusion in either (`:= by` never
    appears), matched no cited lemma, attributed no node to any parent, and
    recorded `coupling_depth: 0` -- which reads as "the parents never met" and
    was in fact "the parents were never read". Twenty-one of twenty-one
    certified crossovers in one campaign carry that reading. Returning "" here
    instead makes the caller's existing guard fire and the check report itself
    unmeasurable, which is the true answer.
    """
    if parent is None:
        return ""
    metadata = getattr(parent, "metadata", None)
    if metadata is None and isinstance(parent, dict):
        metadata = parent.get("metadata") or parent
    metadata = metadata or {}
    for key in ("lean_code", "verification_code", "proof"):
        value = str(metadata.get(key) or "")
        # A proof body, not a signature. Seeds store the tactic block alone, so
        # `:= by` is not required -- what is required is that something follows
        # the statement.
        if value.strip():
            return value
    return ""


def _structural_novelty(
    result: CertificationResult,
    parents: List[CertificationInput],
    work_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Judge the child's *proof* against its parents', not its prose.

    Returns the evidence dict that goes on the row, with ``flag`` set when the
    child should be rejected. Anything unreadable — no Lean, no parent proof,
    an unparseable script — returns no flag: a check that cannot see the
    evidence must not convict on it.
    """
    with ls.trace(
        name="tool.structural_novelty",
        run_type="tool",
        inputs={
            "problem_id": str(getattr(result, "problem_id", "") or ""),
            "op_type": str(getattr(result, "op_type", "") or ""),
            "parent_ids": [str(getattr(p, "id", "") or "") for p in parents],
            "requested_pattern": str(work_item.get("composition_pattern") or ""),
            "difficulty_label": str(work_item.get("difficulty_label") or ""),
        },
        tags=["pool-generation", "novelty-gate"],
    ) as run:
        evidence = _structural_novelty_inner(result, parents, work_item)
        # A rejection that is invisible in the trace cannot be audited later,
        # and the unmeasured cases matter as much as the flagged ones: two runs
        # of this gate silently measured nothing at all before this was caught.
        run.end(outputs={
            "measured": bool(evidence.get("measured")),
            "why": evidence.get("why", ""),
            "flag": evidence.get("flag", ""),
            "coupling_depth": evidence.get("coupling_depth"),
            "added_tactics": evidence.get("added_tactics"),
        })
        return evidence


def _structural_novelty_inner(
    result: CertificationResult,
    parents: List[CertificationInput],
    work_item: Dict[str, Any],
) -> Dict[str, Any]:
    from src.certification.novelty import judge_crossover, judge_mutation
    from src.exam_env.palette import TACTIC_DOCS

    child_lean = str(getattr(result, "lean_code", "") or "")
    if not child_lean:
        return {"measured": False, "why": "no lean_code"}

    op_type = str(getattr(result, "op_type", "") or "")
    if op_type == "crossover":
        pack = [
            {
                "key": str(getattr(parent, "id", "") or f"parent{index}"),
                "lean_code": _parent_lean(parent),
            }
            for index, parent in enumerate(parents)
        ]
        if len([p for p in pack if p["lean_code"]]) < 2:
            return {"measured": False, "why": "fewer than two parent proofs"}
        # The contract is what the planner *asked for*, not what the generator
        # says it did. The generator emits `composition_pattern_used`, but a
        # self-report cannot be trusted to grade itself, and the coupling depth
        # measured from the proof answers the same question without asking.
        requested = str(
            work_item.get("composition_pattern")
            or (work_item.get("fusion_contract") or {}).get("composition_pattern")
            or ""
        )
        difficulty = str(work_item.get("difficulty_label") or "")
        if requested in COUPLED_PATTERNS:
            difficulty = "hard"
        verdict = judge_crossover(child_lean, pack, difficulty=difficulty)
        verdict.detail["requested_pattern"] = requested
    else:
        parent = next((p for p in parents if _parent_lean(p)), None)
        if parent is None:
            return {"measured": False, "why": "no parent proof"}
        metadata = getattr(parent, "metadata", None) or {}
        parent_tactics = list(
            metadata.get("gt_tactics")
            or work_item.get("parent_tactics")
            or []
        )
        if not parent_tactics:
            from src.certification.novelty import tactic_skeleton

            parent_tactics = tactic_skeleton(_parent_lean(parent), list(TACTIC_DOCS))
        verdict = judge_mutation(child_lean, parent_tactics, list(TACTIC_DOCS))

    evidence = {"measured": True, **verdict.detail}
    if not verdict.ok:
        evidence["flag"] = (
            "decorative_mutation" if verdict.kind == "decorative" else "parallel_crossover"
        )
        evidence["reason"] = verdict.reason
        evidence["retry_brief"] = verdict.brief()
    return evidence


def verify_slot_quality(
    result: CertificationResult,
    work_item: Dict[str, Any],
    parents: List[CertificationInput],
) -> QualityResult:
    """Score slot quality without changing the certification status."""
    is_theorem_route = result.target_style == "theorem_proof" or result.certification_route == "theorem_prover"
    if result.status not in {"certified", "survivor"} and not is_theorem_route:
        return QualityResult(
            quality_verdict="weak",
            quality_flags=["certification_not_successful"],
            interestingness_score=0.0,
            feedback_for_next_generation="Fix Lean certification before judging problem quality.",
            quality_evidence=_with_misformalization(result, {
                "checkpoint_coverage": 0.0,
                "missing_checkpoints": ["lean_certification"],
                "reasoning_signature": "uncertified",
                "signature_group": "uncertified",
                "parent_contribution": {},
                "feature_delta": {},
                "novelty_flags": ["certification_not_successful"],
            }, ["certification_not_successful"]),
        )
    if result.op_type == "survivor":
        return QualityResult(
            quality_verdict="acceptable",
            interestingness_score=0.5,
            feedback_for_next_generation="Survivor carried forward unchanged.",
            quality_evidence=_with_misformalization(result, {
                "checkpoint_coverage": 1.0,
                "missing_checkpoints": [],
                "reasoning_signature": f"survivor:{result.problem_id}",
                "signature_group": "survivor",
                "parent_contribution": {},
                "feature_delta": {},
                "novelty_flags": [],
            }, []),
        )
    # A silent mutation is exempt from the novelty gates. Those ask whether the
    # child is different enough from its parent — `parameter_shift_only_theorem`,
    # `standard_library_theorem_restatement`, `definitional_extensionality_only`
    # and a dozen more — and a silent mutation is a restatement by construction,
    # so every one of them would fire on a row that is doing exactly what it was
    # asked.
    #
    # What replaced them was the equivalence probe, promoted to a verdict on the
    # belief that Lean settles "is this the same mathematics". It does not.
    # `silent_backward (h : child) : parent` asks Lean to prove the parent, and
    # the parent is a theorem, so the tactic block closes it whether or not `h`
    # is used: measured directly, the probe called a child that had dropped a
    # conjunct equivalent, and an unrelated true statement equivalent too. A
    # check that cannot fail on the thing it is meant to catch is a heuristic
    # wearing a proof's clothes, and it was the last one still deciding a row.
    #
    # So the verdict is the judge's, as it is for every other variant, and the
    # probe travels with the row as evidence. It still tells a reader something
    # — that the two directions are at least tactically reachable — but it no
    # longer claims to have established equivalence.
    variant = str(
        getattr(result, "operator_variant", "") or work_item.get("operator_variant") or ""
    )
    if variant == "mutation_silent":
        evidence = dict(result.quality_evidence or {})
        silent = dict(evidence.get("silent") or {})
        judge = dict(evidence.get("judge") or {})
        quality = str(judge.get("quality") or "")
        return QualityResult(
            # `judge_unavailable` fails open, the same way it does elsewhere: a
            # judge that timed out is not evidence against the row.
            quality_verdict=quality if quality in {"strong", "acceptable", "weak"} else "acceptable",
            quality_flags=[],
            interestingness_score={"strong": 0.8, "acceptable": 0.4}.get(quality, 0.2),
            feedback_for_next_generation=(
                str(judge.get("reason") or "")
                or "Silent re-encoding; the judge decides whether it is the same "
                "mathematics in a surface a memorising solver would miss."
            ),
            # Built on top of the evidence the row already carries, not in place
            # of it. Written as a fresh dict, this branch dropped thirty-five
            # keys — the judge's verdict and reasoning, dedup, vacuity,
            # inhabitation, the dead-hypothesis result, the goal round-trip, the
            # redundancy probe. Every one of those checks still ran and still
            # gated the row; only the record of them disappeared, so a silent
            # row reached the release reading as though nothing had been checked.
            quality_evidence=_with_misformalization(result, {
                **evidence,
                "checkpoint_coverage": 1.0,
                "missing_checkpoints": [],
                "reasoning_signature": f"silent:{result.problem_id}",
                "signature_group": "silent",
                "parent_contribution": dict(result.parent_contributions or {}),
                "feature_delta": {},
                "novelty_flags": [],
                "silent": silent,
                "verdict_source": "judge",
            }, []),
        )
    if is_theorem_route:
        parent_contribution = dict(result.semantic_parent_contribution or result.parent_contributions or {})
        flags: List[str] = []
        base_evidence = dict(result.quality_evidence or {})
        statement_term_hits = informal_statement_internal_term_hits(result.statement)
        op_type = result.op_type or work_item.get("op_type")
        parent_contribution_source = "explicit" if parent_contribution else "not_available"
        child_surface = _theorem_surface_without_name(result.formal_statement or result.lean_code)
        parent_surfaces = [_parent_formal_surface(parent) for parent in parents]
        operator_goal = str(
            work_item.get("operator_goal") or work_item.get("goal") or result.quality_target or ""
        ).lower()
        theorem_crossover_kind = "not_crossover"
        parent_checkpoint_consumption: Dict[str, Dict[str, Any]] = {}
        if result.status not in {"certified", "survivor"}:
            flags.append("certification_not_successful")
        if statement_term_hits:
            flags.append("informal_statement_internal_terms")
        if result.status == "alignment_failed":
            flags.append("statement_lean_alignment_failed")
        if not result.lean_code:
            flags.append("missing_lean_code")
        if not result.formal_statement:
            flags.append("missing_formal_statement")
        if not result.proof_plan:
            flags.append("missing_proof_plan")
        if parents and not parent_contribution:
            if result.status == "certified" and op_type != "crossover":
                parent_contribution = {
                    parents[0].id: (
                        "lean_code: certified theorem mutation preserves the parent theorem "
                        "route; explicit field-grounded evidence should be filled by the worker."
                    )
                }
                parent_contribution_source = "inferred_theorem_mutation"
            else:
                flags.append("missing_parent_contribution")
        if result.status == "certified" and op_type not in {"survivor", "fallback_survivor"}:
            if child_surface and any(child_surface == surface for surface in parent_surfaces if surface):
                flags.append("same_formal_statement_as_parent")
            if _is_parameter_shift_only_theorem(child_surface, parent_surfaces, result):
                flags.append("parameter_shift_only_theorem")
            if _is_auxiliary_conjunct_only_theorem(child_surface, parent_surfaces, result):
                flags.append("auxiliary_conjunct_only_theorem")
            if "same_statement_repair" in operator_goal:
                flags.append("repair_not_harder")
            if "native_decide" in str(result.lean_code or "") and op_type == "crossover":
                flags.append("computational_crossover_only")
            if _is_fin_one_vacuity_theorem(result):
                flags.append("fin_one_vacuity_theorem")
            if _is_concrete_native_decide_projection(result):
                flags.append("concrete_native_decide_projection")
            if _is_tautological_checkpoint_theorem(result):
                flags.append("tautological_checkpoint_theorem")
            if _is_piecewise_branch_only_theorem(result):
                flags.append("piecewise_branch_only_theorem")
            if _is_trivial_negation_chain_theorem(result):
                flags.append("trivial_negation_chain")
            if _is_trivial_add_zero_theorem(result):
                flags.append("trivial_add_zero_padding")
            if _is_typeclass_narrowing_only_theorem(child_surface, parent_surfaces, result):
                flags.append("typeclass_narrowing_only")
            if _is_projection_only_theorem(result):
                flags.append("projection_only_theorem")
            if _is_divisibility_weaken_only_theorem(result):
                flags.append("divisibility_weaken_only_theorem")
            if _is_fin_one_concrete_arithmetic_theorem(result):
                flags.append("fin_one_concrete_arithmetic_theorem")
            if _is_unit_product_closure_only_theorem(result):
                flags.append("unit_product_closure_only")
            if _is_syntactic_wrapper_only_theorem(result):
                flags.append("syntactic_wrapper_only")
            flags.extend(_theorem_accepted_grade_flags(result))
        if result.status == "certified" and op_type == "crossover" and len(parents) >= 2:
            parent_checkpoint_consumption = _parent_checkpoint_consumption(result, parents)
            consumed_parent_count = sum(
                1
                for item in parent_checkpoint_consumption.values()
                if item.get("consumed_in_lean_surface")
            )
            if consumed_parent_count < min(2, len(parents)):
                flags.append("parent_checkpoint_not_consumed")
            if _has_unused_parent_checkpoint(parent_checkpoint_consumption, result):
                flags.append("unused_checkpoint")
            roots = {_root_lineage(parent.id) for parent in parents}
            if len(roots) < len(parents):
                flags.append("same_lineage_crossover")
            if _is_side_by_side_conjunction_theorem(child_surface, parent_surfaces, result):
                flags.append("side_by_side_conjunction")
                theorem_crossover_kind = "side_by_side_conjunction"
            elif _is_lemma_bundle_master_theorem(result, work_item):
                theorem_crossover_kind = "lemma_bundle_master"
            elif _is_pipeline_composite_theorem(result, work_item):
                theorem_crossover_kind = "pipeline_composite"
            elif parent_contribution and len(parent_contribution) >= 2:
                theorem_crossover_kind = "true_fusion"
            else:
                theorem_crossover_kind = "mutation_like"
                flags.append("missing_parent_contribution")
        elif op_type == "crossover":
            theorem_crossover_kind = "mutation_like"

        # Structural novelty belongs here as much as on the numeric route, and
        # more so: a theorem-style child is exactly the shape that can restate
        # its parent in new notation and still compile.
        theorem_structure = _structural_novelty(result, parents, work_item)
        if theorem_structure.get("flag"):
            flags.append(theorem_structure["flag"])

        verdict = "acceptable" if not flags else "weak"
        if theorem_crossover_kind == "pipeline_composite":
            non_weak_flags = {"pipeline_composite"}
            weak_flags = [flag for flag in flags if flag not in non_weak_flags]
            verdict = "acceptable" if not weak_flags else "weak"
        # The judge decides. The flags above are measurements and they stay —
        # they are what the judge is shown, and they feed the retry brief — but
        # they stopped deciding anything on 2026-08-08, when every flag raised on
        # a certified row was checked against the judge's reading of the same
        # row: parallel_crossover 9 times, projection_only_theorem 5, seven other
        # rules once or twice each, and in all 23 cases the judge said keep, 20 of
        # them `strong`. The flags were not catching what the judge missed; they
        # were removing rows the judge had endorsed, and because a weak verdict
        # becomes a `weak_quality` selection risk, those rows could not become
        # parents. Twenty-five rows the judge called strong were barred from
        # parenthood that way, which is why lineages stopped compounding at
        # depth 2.
        #
        # `parallel_crossover` is the clearest case: all nine of its verdicts had
        # `attributed_nodes: None`, the state its own docstring says makes a
        # depth of 0 meaningless.
        judge = dict(base_evidence.get("judge") or (result.quality_evidence or {}).get("judge") or {})
        # A judge that could not run leaves no opinion, and "no opinion" must not
        # become "weak": that is the heuristic gate returning through the back
        # door, one transient timeout at a time. The first smoke run after the
        # authority moved had exactly one — 180s on a certified crossover — and
        # the row came back weak on `parallel_crossover` and
        # `projection_only_theorem`, the two flags whose removal was the point.
        # An unjudged but certified row is carried as acceptable and marked, so
        # the gap is visible rather than silently restrictive.
        if not judge.get("ran") and result.status == "certified" and verdict == "weak":
            base_evidence["advisory_flags"] = list(flags)
            base_evidence["verdict_source"] = "judge_unavailable"
            base_evidence["judge_unavailable_why"] = str(judge.get("why") or "")[:200]
            verdict = "acceptable"
        if judge.get("ran") and judge.get("verdict"):
            heuristic_verdict = verdict
            verdict = {
                "strong": "strong",
                "acceptable": "acceptable",
                "weak": "acceptable",
            }.get(str(judge.get("quality") or ""), "acceptable") if judge.get("verdict") == "keep" else "weak"
            base_evidence["advisory_flags"] = list(flags)
            base_evidence["heuristic_verdict"] = heuristic_verdict
            base_evidence["verdict_source"] = "judge"
        # One flag decides again, and only where it can be shown to have measured
        # something. A crossover whose proof graph was read, whose nodes were
        # traced to both parents, and in which no `have` carries material from
        # both, is parallel by the contract the generator was given -- not an
        # opinion the judge can outvote.
        #
        # This flag was demoted in August 2026 for good reason: all nine of its
        # verdicts then had an empty attribution, where a depth of 0 means "not
        # measured" rather than "did not couple", and it was removing rows the
        # judge had called strong. Both halves of that have changed. The
        # attribution failure is fixed at its source -- `_parent_lean` was
        # handing the gate each parent's statement in place of its proof -- and
        # `coupling_depth` now returns unmeasurable instead of 0 when nothing
        # traces. Replayed over 22 certified crossovers with proofs actually
        # supplied, the gate convicts 14 and passes 8, and every one of the three
        # that survived two independent re-judge passes is in the passing set.
        # `depth is None` still passes: absence of evidence convicts nothing.
        if (theorem_structure.get("flag") == "parallel_crossover"
                and theorem_structure.get("measurable") is True
                and theorem_structure.get("coupling_depth") == 0):
            base_evidence["verdict_source"] = "coupling_gate"
            base_evidence["judge_verdict_overridden"] = verdict
            verdict = "weak"
        return QualityResult(
            quality_verdict=verdict,
            quality_flags=flags,
            interestingness_score=0.7 if verdict == "acceptable" else 0.2,
            feedback_for_next_generation=(
                "Preserve theorem style and complete Lean proof."
                if verdict == "acceptable"
                else f"Repair theorem artifact fields: {', '.join(flags)}."
            ),
            semantic_parent_contribution=parent_contribution,
            interestingness_features={
                "proof_obligation_count": len(result.proof_obligations or []),
                "proof_complete": result.status == "certified",
                "formal_surface_changed": (
                    bool(child_surface)
                    and not any(child_surface == surface for surface in parent_surfaces if surface)
                ),
            },
            quality_evidence=_with_misformalization(result, {
                "structural_novelty": theorem_structure,
                **base_evidence,
                "checkpoint_coverage": 1.0 if verdict == "acceptable" else 0.5,
                "missing_checkpoints": flags,
                "reasoning_signature": f"theorem_proof:{result.problem_id}",
                "signature_group": "theorem_proof",
                "parent_contribution": parent_contribution,
                "parent_contribution_source": parent_contribution_source,
                "parent_contribution_required": (
                    "explicit_for_crossover; inferred_allowed_for_certified_mutation"
                ),
                "feature_delta": {
                    "formal_surface_changed": (
                        bool(child_surface)
                        and not any(child_surface == surface for surface in parent_surfaces if surface)
                    ),
                    "parameter_shift_only": "parameter_shift_only_theorem" in flags,
                    "auxiliary_conjunct_only": "auxiliary_conjunct_only_theorem" in flags,
                    "operator_goal": operator_goal,
                },
                "novelty_flags": flags,
                "proof_verify_summary": result.proof_verify_summary,
                "crossover_kind": theorem_crossover_kind,
                "parent_checkpoint_consumption": parent_checkpoint_consumption,
                "informal_statement_internal_terms": statement_term_hits,
            }, flags),
        )

    flags: List[str] = []
    op_type = result.op_type or work_item.get("op_type")
    generated_params = dict(result.generated_params or {})
    target_family = result.family or result.target_family
    parent_params = [_family_params(parent) for parent in parents]
    parent_families = [detect_family(parent.statement) for parent in parents]
    features = _interestingness_features(target_family, generated_params)
    solution_verification = _solution_verification(result, target_family, generated_params)
    features["solution_verification_passed"] = bool(solution_verification.get("passed"))
    semantic_contribution: Dict[str, str] = {}
    feature_delta: Dict[str, Any] = {}
    semantic_roles: Dict[str, str] = {}
    fusion_contract = _fusion_contract(work_item)
    fusion_mechanism = str(fusion_contract.get("fusion_mechanism") or "")
    why_not_concatenation = str(fusion_contract.get("why_not_concatenation") or "")
    crossover_kind = "not_crossover"
    statement_term_hits = informal_statement_internal_term_hits(result.statement)
    if statement_term_hits:
        flags.append("informal_statement_internal_terms")

    if result.projected_params:
        projected = dict(result.projected_params or {})
        derived_keys = DERIVED_PARAM_KEYS.get(str(target_family or ""), set())
        canonical_subset_matches = all(
            key in derived_keys or str(projected.get(key)) == str(value)
            for key, value in generated_params.items()
        )
        if not canonical_subset_matches:
            flags.append("projection_params_mismatch")
    if result.projection_check and not bool(result.projection_check.get("passed", True)):
        flags.append("projection_check_failed")

    for flag in solution_verification.get("flags", []):
        flags.append(str(flag))

    if target_family in parent_families:
        parent_index = parent_families.index(target_family)
        parent_metric = _difficulty_metric(target_family, parent_params[parent_index])
        child_metric = _difficulty_metric(target_family, generated_params)
        if parent_metric is not None and child_metric is not None:
            feature_delta["difficulty_metric_delta"] = child_metric - parent_metric
            if child_metric <= parent_metric:
                flags.append("claimed_harder_but_metric_not_increased")
            if generated_params == parent_params[parent_index]:
                flags.append("no_required_param_delta")

    if target_family == "modular_congruence" and "m" in generated_params:
        try:
            answer = int(result.answer or 0)
            modulus = int(generated_params["m"])
            if answer in {0, 1, modulus - 1}:
                flags.append("trivial_mod_remainder")
        except (TypeError, ValueError):
            pass

    if op_type == "crossover":
        contributions = dict(work_item.get("parent_contributions") or {})
        if len(parents) < 2:
            flags.append("missing_parent_contribution")
            crossover_kind = "mutation_like"
        for parent_key in ("parent_A", "parent_B"):
            fusion_parent = _fusion_parent(fusion_contract, parent_key)
            parent_id = str(fusion_parent.get("id") or "")
            role = str(fusion_parent.get("semantic_role") or "")
            contribution = str(fusion_parent.get("contribution") or "")
            if parent_id:
                if role:
                    semantic_roles[parent_id] = role
                if contribution and not contributions.get(parent_id):
                    contributions[parent_id] = contribution
        evidence = _evidence_text(result)
        missing = []
        for parent, parent_family in zip(parents, parent_families):
            contribution = str(contributions.get(parent.id, ""))
            semantic = _composite_semantic_contribution(
                target_family=target_family,
                parent=parent,
                parent_family=parent_family,
                generated_params=generated_params,
                evidence=evidence,
            )
            if semantic:
                semantic_contribution[parent.id] = semantic
            if not contribution.strip():
                missing.append(parent.id)
            elif semantic:
                continue
            elif parent_family != target_family and not _contribution_visible(
                contribution, evidence
            ):
                missing.append(parent.id)
        if missing:
            flags.append("missing_parent_contribution")
            crossover_kind = "mutation_like"
        if (
            work_item.get("composition_pattern") == "parameter_transfer"
            and len(set(parent_families)) > 1
            and target_family not in {"gcd_divisor_sum", "divisor_sum_mod"}
        ):
            flags.append("weak_crossover_parameter_only")
        generated_values = _param_values(generated_params)
        for parent, parent_family in zip(parents, parent_families):
            if parent_family == target_family:
                continue
            if parent.id in semantic_contribution:
                continue
            answer_value = _answer_int(parent)
            contribution = str(contributions.get(parent.id, "")).lower()
            claims_parent_output = any(
                word in contribution
                for word in ["answer", "gcd", "divisor", "remainder", "modulus", "sum"]
            )
            if answer_value is not None and claims_parent_output:
                if answer_value not in generated_values:
                    flags.append("indirect_parent_contribution")
            elif not (_param_values(parent_params[parents.index(parent)]) & generated_values):
                flags.append("indirect_parent_contribution")
        if any(
            word in evidence
            for word in ["inspired", "inspiration", "capped", "cap", "double the original"]
        ):
            flags.append("weak_inspiration_only_crossover")
        distinct_roles = len(set(role for role in semantic_roles.values() if role)) >= 2
        same_family_or_signature = len(set(parent_families)) == 1
        if same_family_or_signature and semantic_roles and not distinct_roles:
            flags.append("same_role_crossover")
        if crossover_kind != "mutation_like":
            if fusion_mechanism == "sequential_composition" or not why_not_concatenation.strip():
                crossover_kind = "pipeline_composite"
                flags.append("sequential_composition")
            elif distinct_roles and len(semantic_contribution) >= 2:
                crossover_kind = "true_fusion"
            elif len(semantic_contribution) >= 2:
                crossover_kind = "pipeline_composite"
            else:
                crossover_kind = "mutation_like"
                flags.append("missing_parent_contribution")

    required_checkpoints = list(work_item.get("required_checkpoints") or [])
    for checkpoint in _default_checkpoints(result, str(op_type or ""), target_family):
        if checkpoint not in required_checkpoints:
            required_checkpoints.append(checkpoint)
    missing_checkpoints = []
    for checkpoint in required_checkpoints:
        normalized_checkpoint = checkpoint.strip().lower()
        if normalized_checkpoint in {"semantic_parent_contribution", "parent_contribution"}:
            if semantic_contribution:
                continue
        if not _checkpoint_visible(checkpoint, result, features):
            missing_checkpoints.append(checkpoint)
    if required_checkpoints:
        checkpoint_coverage = round(
            (len(required_checkpoints) - len(missing_checkpoints)) / len(required_checkpoints), 3
        )
    else:
        checkpoint_coverage = 1.0
    if missing_checkpoints:
        flags.append("missing_quality_checkpoints")

    unsupported_claims = [
        word
        for word in ["proof", "identity", "theorem", "lemma"]
        if word in str(result.generation_notes or "").lower()
    ]
    if unsupported_claims and result.lean_level < 3:
        flags.append("unsupported_quality_claim")

    reasoning_signature = _reasoning_signature(
        target_family,
        result.reasoning_pattern,
        generated_params,
    )
    signature_group = _signature_group(target_family, reasoning_signature)
    if reasoning_signature in set(work_item.get("avoid_signatures") or []):
        flags.append("repeated_reasoning_signature")
    novelty_flags = [
        flag
        for flag in sorted(set(flags))
        if flag
        in {
            "no_required_param_delta",
            "claimed_harder_but_metric_not_increased",
            "weak_crossover_parameter_only",
            "weak_inspiration_only_crossover",
            "missing_parent_contribution",
            "missing_quality_checkpoints",
            "repeated_reasoning_signature",
        }
    ]
    weak_flags = set(flags)
    if op_type == "crossover":
        weak_flags = {
            flag for flag in weak_flags if flag not in NON_WEAK_CROSSOVER_FLAGS
        }
        if crossover_kind == "mutation_like":
            weak_flags.add("mutation_like_crossover")

    # Structural novelty, read off the Lean proof rather than the prose. The
    # checks above compare statements and solution skeletons, which a
    # redecoration satisfies: it really does mention both parents and really is
    # a different string. What it does not do is change how the theorem is
    # proved, and that is what these two read.
    structure = _structural_novelty(result, parents, work_item)
    if structure.get("flag"):
        flags.append(structure["flag"])
        weak_flags.add(structure["flag"])

    quality_evidence = _with_misformalization(result, {
        "structural_novelty": structure,
        "checkpoint_coverage": checkpoint_coverage,
        "missing_checkpoints": missing_checkpoints,
        "reasoning_signature": reasoning_signature,
        "signature_group": signature_group,
        "parent_contribution": semantic_contribution,
        "semantic_roles": semantic_roles,
        "fusion_contract": fusion_contract,
        "crossover_kind": crossover_kind,
        "feature_delta": feature_delta,
        "novelty_flags": novelty_flags,
        "features": features,
        "solution_verification": solution_verification,
        "informal_statement_internal_terms": statement_term_hits,
    }, sorted(set(flags)))
    score = max(0.0, round(checkpoint_coverage - 0.15 * len(weak_flags), 2))
    if not weak_flags:
        strong_crossover = (
            op_type == "crossover"
            and crossover_kind == "true_fusion"
            and "sequential_composition" not in set(flags)
        )
        verdict = "strong" if strong_crossover else "acceptable"
        feedback = "Preserve this pattern; it satisfied the slot contract and quality checks."
    else:
        verdict = "weak"
        feedback = f"Avoid repeating: {', '.join(sorted(weak_flags))}."
    return QualityResult(
        quality_verdict=verdict,
        quality_flags=sorted(set(flags)),
        interestingness_score=score,
        feedback_for_next_generation=feedback,
        semantic_parent_contribution=semantic_contribution,
        interestingness_features=features,
        quality_evidence=quality_evidence,
    )
