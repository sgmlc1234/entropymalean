"""Central registry for accepted-grade no-go signals.

The detectors live in ``quality.py`` because they need access to generated
surfaces. This module owns the policy metadata that should stay consistent
across quality verdicts, retry feedback, planner memory, and yield audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple


@dataclass(frozen=True)
class NoGoRule:
    flag: str
    category: str
    retry_instruction: str
    memory_lesson: str | None = None
    misformalization_category: str = "semantic"


NO_GO_RULES: Tuple[NoGoRule, ...] = (
    NoGoRule(
        "projection_only_theorem",
        "surface_wrapper",
        "Do not return a projection-only theorem. Change the conclusion or add a necessary proof obligation.",
        "Avoid projection-only theorem children; require a changed theorem surface.",
    ),
    NoGoRule(
        "same_formal_statement_as_parent",
        "surface_wrapper",
        "Change formal_statement and lean_code to prove a small new theorem, not the exact parent surface; valid patches are hypothesis_specialization, conclusion_projection, or immediate_corollary.",
        "Avoid same_statement_repair unless the goal is Gen0 proof completion.",
        "misrepresentation",
    ),
    NoGoRule(
        "syntactic_wrapper_only",
        "surface_wrapper",
        "Do not submit a wrapper/simpa-only theorem. Change statement, formal_statement, or proof_obligations so the child has a new mathematical obligation.",
        "Avoid syntactic wrapper repairs that keep the same theorem content.",
    ),
    NoGoRule(
        "parameter_shift_only_theorem",
        "direct_corollary_helper",
        "Do not only change numerals in the parent theorem. Add a nonnumeric proof obligation or change the theorem shape.",
        "Do not plan theorem mutation as a plain numeral shift with mechanical proof.",
    ),
    NoGoRule(
        "auxiliary_conjunct_only_theorem",
        "direct_corollary_helper",
        "Do not keep the parent conclusion as one conjunct and append a side fact. Make the new obligation replace or drive the old goal form.",
        "Avoid appending a side conjunct to the parent conclusion.",
    ),
    NoGoRule(
        "trivial_negation_chain",
        "surface_wrapper",
        "Do not repeat negation wrappers such as -(-u) or triple negation. Add a semantic lemma, stronger hypothesis interaction, or nontrivial conclusion projection.",
        "Avoid theorem mutations that only add repeated negation wrappers.",
    ),
    NoGoRule(
        "trivial_add_zero_padding",
        "surface_wrapper",
        "Do not use + 0 or add_zero as the only change. Replace padding with a real theorem-level obligation.",
        "Avoid theorem mutations that only add + 0 or add_zero padding.",
    ),
    NoGoRule(
        "typeclass_narrowing_only",
        "surface_wrapper",
        "Do not only strengthen a typeclass while keeping the same conclusion. Use the stronger structure in the conclusion/proof obligation.",
        "Avoid making a theorem harder only by typeclass narrowing with the same conclusion.",
    ),
    NoGoRule(
        "divisibility_weaken_only_theorem",
        "direct_corollary_helper",
        "Do not weaken a divisibility theorem by unpacking a stronger parent conclusion. Add a new proof role or different target.",
        "Avoid divisibility weakening as an accepted-grade theorem.",
    ),
    NoGoRule(
        "side_by_side_conjunction",
        "parent_crossover",
        "Do not prove parent A and parent B side by side. Convert crossover into a pipeline where one parent feeds the other's target.",
        "Do not plan side-by-side conjunction crossover.",
        "misrepresentation",
    ),
    NoGoRule(
        "unused_checkpoint",
        "parent_crossover",
        "Do not pass a parent checkpoint through as an unused hypothesis; it must affect the final goal or a necessary proof step.",
        "Require parent checkpoints to affect the final goal or proof step.",
        "misrepresentation",
    ),
    NoGoRule(
        "parent_checkpoint_not_consumed",
        "parent_crossover",
        "Put each parent's checkpoint into formal_statement or lean_code so parent-derived Lean atoms are observable.",
        "Do not plan crossover from prose-only parent usage.",
        "misrepresentation",
    ),
    NoGoRule(
        "same_lineage_crossover",
        "parent_crossover",
        "Downgrade to mutation or choose a parent with a different root lineage; do not crossover a parent with its descendant.",
        "Do not crossover a seed with its descendant.",
        "misrepresentation",
    ),
    NoGoRule(
        "mutation_like_crossover",
        "parent_crossover",
        "Do not present a one-parent child as crossover. Either consume the second parent or downgrade to mutation.",
        "Avoid mutation-like crossover where one parent disappears.",
        "misrepresentation",
    ),
    NoGoRule(
        "weak_inspiration_only_crossover",
        "parent_crossover",
        "Do not use a parent as inspiration only. Make its contribution visible in statement, formal_statement, proof_plan, or lean_code.",
        "Avoid inspiration-only crossover.",
        "misrepresentation",
    ),
    NoGoRule(
        "indirect_parent_contribution",
        "parent_crossover",
        "Move parent contribution from prose into params, target_computation, formal_statement, or proof_obligations.",
        "Avoid indirect/prose-only parent contribution.",
        "misrepresentation",
    ),
    NoGoRule(
        "concrete_native_decide_projection",
        "fixed_computation",
        "Do not turn a theorem parent into a one-number native_decide computation. Keep a symbolic hypothesis/conclusion or reusable checkpoint.",
        "Avoid one-number native_decide theorem children from theorem parents.",
    ),
    NoGoRule(
        "tautological_checkpoint_theorem",
        "direct_corollary_helper",
        "Do not prove a conclusion from a hypothesis that already states the exact required value. Replace it with a derived local lemma.",
        "Avoid checkpoint hypotheses that make the conclusion immediate by restatement.",
    ),
    NoGoRule(
        "piecewise_branch_only_theorem",
        "fixed_computation",
        "Do not only select an easy branch of a piecewise solution. Add a theorem-level reason for why the branch applies.",
        "Avoid only selecting an easy branch of a piecewise solution function.",
    ),
    NoGoRule(
        "fin_one_vacuity_theorem",
        "fixed_computation",
        "Do not collapse a theorem into a Fin 1 vacuity lemma. Use a non-vacuous index set or meaningful checkpoint.",
        "Avoid Fin 1/vacuity theorem children.",
    ),
    NoGoRule(
        "fin_one_concrete_arithmetic_theorem",
        "fixed_computation",
        "Do not replace a theorem with a one-by-one concrete arithmetic Fin 1 computation.",
        "Avoid concrete Fin 1 arithmetic theorem children.",
    ),
    NoGoRule(
        "unit_product_closure_only",
        "direct_corollary_helper",
        "Do not use unit/product closure as the final theorem unless it creates a new mathematical role.",
        "Avoid unit/product closure-only theorem children.",
    ),
    NoGoRule(
        "computational_crossover_only",
        "fixed_computation",
        "Do not rely only on native_decide/computation. Add a theorem-style proof obligation that uses both parent roles.",
        "Do not treat sequential arithmetic pipelines as strong fusion.",
    ),
    NoGoRule(
        "proof_infrastructure_only",
        "direct_corollary_helper",
        "Do not make an exact helper fact such as finset/card/prod the final result. Patch statement/formal_statement/proof_plan so that helper appears as a hypothesis or intermediate lemma feeding a final theorem target.",
        "Do not treat helper facts such as exact finset/card/prod as accepted problems.",
    ),
    NoGoRule(
        "direct_parent_corollary_only",
        "direct_corollary_helper",
        "Do not return a direct projection/subset/index corollary of the parent. Add a new proof obligation or latent parameter target.",
        "Avoid direct parent corollaries unless a new proof obligation or latent parameter target is added.",
    ),
    NoGoRule(
        "linear_equation_shift_corollary_only",
        "direct_corollary_helper",
        "Do not solve the same linear equation and change only the final arithmetic corollary. Use a characterization or second checkpoint.",
        "Avoid solving the same linear equation and only changing the final corollary.",
    ),
    NoGoRule(
        "affine_index_drift_only",
        "domain_specific",
        "Do not only change the index map such as u(p+k), u(2p), or a window length. Patch fusion_goal/proof_plan/formal_statement so a second aggregate or checkpoint is consumed.",
        "Avoid repeating the same domain pipeline by only changing an affine index.",
    ),
    NoGoRule(
        "cardinality_only_window",
        "fixed_computation",
        "Cardinality alone is not accepted-grade. Add another consumed domain fact such as sum/product/membership.",
        "Do not use only a cardinality-controlled range window.",
    ),
    NoGoRule(
        "aggregate_helper_only",
        "direct_corollary_helper",
        "Do not stop at a standalone aggregate fact. Use the aggregate inside the final conclusion or proof_plan.",
        "Avoid aggregate helper-only rows.",
    ),
    NoGoRule(
        "lineage_complexity_without_new_role",
        "direct_corollary_helper",
        "Simplify the surface and name one new mathematical role; do not add lineage complexity without changing the final obligation.",
        "Avoid long lineage/id-exploded expressions unless a new mathematical role is clearly introduced.",
    ),
    NoGoRule(
        "order_equality_selector_only",
        "direct_corollary_helper",
        "Do not use orderOf equality only as an if/indicator selector inside an unrelated rational factor. Make the order theorem change the final group-theoretic target.",
        "Avoid orderOf equality selector artifacts; require the order checkpoint to drive a group-theoretic conclusion.",
    ),
    NoGoRule(
        "same_target_role_already_accepted",
        "direct_corollary_helper",
        "Do not produce another theorem with the same final target role already represented in accepted memory. Change the object, quantification, or final theorem role.",
        "Avoid same-target-role variants after an accepted theorem already covers that role; require a new final obligation.",
    ),
    NoGoRule(
        "witness_packaging_only",
        "direct_corollary_helper",
        "Do not make explicit witness packaging the final result. Use the witness inside a new theorem target or stronger characterization.",
        "Avoid explicit-witness packaging as the final generated theorem.",
    ),
    NoGoRule(
        "coefficient_engineering_only",
        "direct_corollary_helper",
        "Do not only engineer a nonzero rational coefficient around an existing irrationality theorem. Use a new algebraic object, quantifier, or theorem role.",
        "Avoid irrationality variants that only change a nonzero rational coefficient.",
    ),
    NoGoRule(
        "theorem_local_corollary_dominated",
        "direct_corollary_helper",
        "Do not submit a local corollary dominated by an already accepted stronger theorem. Change the theorem role or use it only as scaffold.",
        "Avoid local corollaries dominated by an accepted stronger theorem; use them only as scaffold evidence.",
    ),
    NoGoRule(
        "definitional_extensionality_only",
        "direct_corollary_helper",
        "Do not make extensional equality of two objects with the same defining predicate the final theorem. Use it only as scaffold or make the predicate feed a new target.",
        "Avoid definitional extensionality-only theorems such as S = T from identical membership predicates.",
    ),
    NoGoRule(
        "pid_definition_restatement",
        "direct_corollary_helper",
        "Do not make the PID constructor data itself the final theorem. Use principal generators inside a new algebraic target or keep the row as scaffold.",
        "Avoid PID definition/constructor restatements as paper-grade successes.",
    ),
    NoGoRule(
        "standard_library_theorem_restatement",
        "direct_corollary_helper",
        "Do not restate a named Mathlib theorem as the child theorem. Make the library theorem consume a new upstream checkpoint or change the final target role.",
        "Avoid direct standard-library theorem restatements such as rootSet-cardinality iff separability.",
    ),
    NoGoRule(
        "cyclic_transport_same_target_role",
        "domain_specific",
        "Do not keep producing cyclic rotation/order/commutation variants with the same final target role. Change the target role or object, not only arity/orientation.",
        "Avoid same-role cyclic transport variants after a representative theorem already exists.",
    ),
    NoGoRule(
        "topology_direct_consequence_only",
        "direct_corollary_helper",
        "Do not submit a direct topology/library consequence as a paper-grade theorem. Use it as scaffold unless it changes the target role.",
        "Avoid direct topology/library consequences as paper-grade successes.",
    ),
    NoGoRule(
        "finite_residue_bookkeeping_only",
        "domain_specific",
        "Do not submit residue, interval, or uniqueness bookkeeping around a fixed modulus as the final theorem. Make the residue feed a different theorem target.",
        "Avoid fixed-modulus residue bookkeeping as a final generated theorem.",
    ),
    NoGoRule(
        "ap_index_only_theorem",
        "domain_specific",
        "Do not generate another single-index AP evaluation. Patch to a closed-form, uniqueness, parameter-characterization, or extremal theorem.",
        "Avoid AP single-index or hidden-parameter evaluations.",
    ),
    NoGoRule(
        "ap_shifted_local_corollary_only",
        "domain_specific",
        "Do not generate local shifted AP corollaries. Patch to a closed-form, uniqueness, or parameter-characterization theorem.",
        "Avoid AP shifted local corollaries.",
    ),
    NoGoRule(
        "ap_bound_padding_only",
        "domain_specific",
        "Do not wrap the solved AP value in an arbitrary bound. Use the AP checkpoint as input to a real final theorem.",
        "Avoid AP bound-padding corollaries.",
    ),
    NoGoRule(
        "ap_interval_bound_padding_only",
        "domain_specific",
        "Do not wrap the solved AP value in arbitrary interval membership. Use the AP checkpoint as input to a real final theorem.",
        "Avoid AP interval-membership padding corollaries.",
    ),
    NoGoRule(
        "mod_inverse_same_conclusion_paraphrase",
        "domain_specific",
        "Do not keep the same modulo-inverse conclusion n=57 while only rewriting the hypothesis. Use the inverse as an input to another theorem or change the final goal type.",
        "Avoid modulo inverse paraphrases that keep the same final goal.",
    ),
    NoGoRule(
        "solved_parameter_quotient_corollary_only",
        "direct_corollary_helper",
        "Do not expose only arithmetic consequences of a solved parameter. Make it feed a separate theorem target.",
        "Avoid quotient/power corollaries that only expose arithmetic after a solved parameter.",
    ),
    NoGoRule(
        "mod_inverse_arithmetic_corollary_only",
        "direct_corollary_helper",
        "Do not expose only a remainder/divisibility arithmetic consequence of the solved modulo inverse. Make the inverse feed a separate theorem target.",
        "Avoid modulo-inverse arithmetic corollaries such as n % 19 = 0 as final problems.",
    ),
    NoGoRule(
        "residue_finset_cardinality_restatement",
        "domain_specific",
        "Do not restate fixed residue-set cardinality or modular power facts. Make them feed another theorem target.",
        "Avoid restating fixed residue finset/cardinality facts.",
    ),
    NoGoRule(
        "finite_mod_inverse_window_restatement",
        "domain_specific",
        "Do not restate the solved modulo-398 inverse as membership in a finite filtered window. Make the inverse feed another theorem target.",
        "Avoid finite-window restatements of the modulo inverse parent.",
    ),
    NoGoRule(
        "fixed_finite_aggregate_computation",
        "fixed_computation",
        "Do not submit a fixed finite-set sum/product/card expression closed by native_decide. Introduce a symbolic condition or pipeline input.",
        "Avoid fixed finite-set aggregate expressions closed by native_decide.",
    ),
    NoGoRule(
        "cardinality_arithmetic_pipeline_only",
        "fixed_computation",
        "Do not combine fixed cardinalities only as arithmetic. Make a checkpoint drive a symbolic condition or classification role.",
        "Avoid cardinality arithmetic wrappers as final generated problems.",
    ),
    NoGoRule(
        "native_decide_fixed_domain_computation",
        "fixed_computation",
        "Avoid fixed-domain native_decide computations as final generated problems. Move computation into an intermediate checkpoint.",
        "Avoid fixed-domain native_decide computations.",
    ),
    NoGoRule(
        "artificial_bridge_to_existing_pipeline",
        "domain_specific",
        "Do not add an artificial bridge hypothesis just to feed an already accepted pipeline. The new parent must change the final theorem role.",
        "Avoid bridge hypotheses that only route a new parent into an already accepted pipeline.",
    ),
    NoGoRule(
        "numeric_bound_fitting_crossover",
        "domain_specific",
        "Do not fuse parents by fitting constants into an inequality. Use a natural theorem target where both checkpoints define a reusable object or condition.",
        "Avoid constant-fitted numeric inequalities as crossover.",
    ),
    NoGoRule(
        "parent_theorem_assumption_smuggling",
        "parent_crossover",
        "Do not take a parent theorem as a broad hypothesis and immediately apply it. Inline the checkpoint proof or make it an intermediate lemma consumed by a new final target.",
        "Avoid smuggling a parent theorem as an assumption that already proves the generated conclusion.",
        "misrepresentation",
    ),
    NoGoRule(
        "near_duplicate",
        "novelty_memory",
        "Do not recreate an accepted or run-local problem. Add a new object, target role, proof obligation, or consumed checkpoint.",
        "Avoid near-duplicates of accepted or earlier run-local problems; require an explicit distinguishing delta.",
    ),
    NoGoRule(
        "exact_duplicate_memory",
        "novelty_memory",
        "Do not submit a statement/formal surface or numeric family+params already present in accepted/run-local memory.",
        "Avoid exact statement, formal surface, and family+params duplicates from novelty memory.",
    ),
)

NO_GO_RULES_BY_FLAG = {rule.flag: rule for rule in NO_GO_RULES}
ACCEPTED_PROXY_SEVERE_FLAGS = frozenset(NO_GO_RULES_BY_FLAG)
QUALITY_RETRYABLE_NO_GO_FLAGS = ACCEPTED_PROXY_SEVERE_FLAGS
RETRY_PATCH_INSTRUCTIONS = {
    flag: rule.retry_instruction for flag, rule in NO_GO_RULES_BY_FLAG.items()
}
PLANNER_MEMORY_LESSONS = {
    flag: (rule.memory_lesson or rule.retry_instruction)
    for flag, rule in NO_GO_RULES_BY_FLAG.items()
}
MISFORMALIZATION_SEMANTIC_FLAGS = frozenset(
    rule.flag for rule in NO_GO_RULES if rule.misformalization_category == "semantic"
)
MISFORMALIZATION_MISREPRESENTATION_FLAGS = frozenset(
    rule.flag for rule in NO_GO_RULES if rule.misformalization_category == "misrepresentation"
)


def reject_cluster(flags: Iterable[str]) -> str:
    categories = [
        NO_GO_RULES_BY_FLAG[flag].category
        for flag in flags
        if flag in NO_GO_RULES_BY_FLAG
    ]
    if "direct_corollary_helper" in categories:
        return "direct_corollary"
    if "fixed_computation" in categories:
        return "fixed_finite_computation"
    if "domain_specific" in categories:
        if any(flag in {"ap_index_only_theorem", "ap_shifted_local_corollary_only"} for flag in flags):
            return "ap_index_only"
        if "mod_inverse_same_conclusion_paraphrase" in set(flags):
            return "mod_inverse_paraphrase"
        if "residue_finset_cardinality_restatement" in set(flags):
            return "residue_cardinality_restatement"
        if any(flag in {"affine_index_drift_only", "cardinality_only_window"} for flag in flags):
            return "affine_drift"
        return "fixed_finite_computation"
    if "parent_crossover" in categories:
        return "unused_parent"
    if "surface_wrapper" in categories:
        return "projection_or_wrapper"
    flag_set = {str(flag) for flag in flags if str(flag)}
    if flag_set & {"same_lineage_crossover", "lineage_complexity_without_new_role", "repeated_reasoning_signature"}:
        return "lineage_repetition"
    if flag_set & {"not_certified", "certification_not_successful"}:
        return "not_certified"
    return "other"


def no_go_policy_summary() -> dict[str, object]:
    counts: dict[str, int] = {}
    for rule in NO_GO_RULES:
        counts[rule.category] = counts.get(rule.category, 0) + 1
    return {
        "total": len(NO_GO_RULES),
        "by_category": dict(sorted(counts.items())),
        "flags": sorted(NO_GO_RULES_BY_FLAG),
    }


def _rule_priority(
    rule: NoGoRule,
    *,
    op_type: str,
    target_style: str,
    target_family: str,
    operator_variant: str,
    recent_failure_flags: set[str],
) -> int:
    score = 0
    if rule.flag in recent_failure_flags:
        score += 100
    if target_style == "theorem_proof":
        if rule.category in {"surface_wrapper", "direct_corollary_helper"}:
            score += 30
        if rule.category == "fixed_computation":
            score += 18
    else:
        if rule.flag in {
            "fixed_finite_aggregate_computation",
            "native_decide_fixed_domain_computation",
            "concrete_native_decide_projection",
        }:
            score += 18
    if op_type == "crossover":
        if rule.category == "parent_crossover":
            score += 35
        if rule.flag in {
            "artificial_bridge_to_existing_pipeline",
            "numeric_bound_fitting_crossover",
            "cardinality_arithmetic_pipeline_only",
            "parent_theorem_assumption_smuggling",
        }:
            score += 25
    if op_type == "mutation":
        if rule.flag in {
            "direct_parent_corollary_only",
            "parameter_shift_only_theorem",
            "ap_index_only_theorem",
            "ap_shifted_local_corollary_only",
        }:
            score += 25
    if target_family in {"arithmetic_series", "theorem_proof"}:
        if rule.flag.startswith("ap_") or rule.flag == "affine_index_drift_only":
            score += 14
    if target_family in {"modular_congruence", "theorem_proof"}:
        if rule.flag in {
            "mod_inverse_same_conclusion_paraphrase",
            "residue_finset_cardinality_restatement",
            "solved_parameter_quotient_corollary_only",
            "mod_inverse_arithmetic_corollary_only",
            "finite_mod_inverse_window_restatement",
        }:
            score += 12
    if "hard" in operator_variant:
        score += 3
    return score


def build_no_go_policy_pack(
    *,
    op_type: str,
    target_style: str,
    target_family: str = "",
    operator_variant: str = "",
    recent_failure_flags: Iterable[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Select the no-go rules most relevant to one LLM execution surface."""
    recent = {str(flag) for flag in (recent_failure_flags or []) if str(flag).strip()}
    scored = [
        (
            _rule_priority(
                rule,
                op_type=op_type,
                target_style=target_style,
                target_family=target_family,
                operator_variant=operator_variant,
                recent_failure_flags=recent,
            ),
            rule.flag,
            rule,
        )
        for rule in NO_GO_RULES
    ]
    selected = [rule for score, _flag, rule in sorted(scored, reverse=True) if score > 0][:limit]
    if len(selected) < min(limit, 4):
        fallback_categories = (
            {"parent_crossover"} if op_type == "crossover" else {"surface_wrapper", "direct_corollary_helper"}
        )
        for rule in NO_GO_RULES:
            if rule in selected or rule.category not in fallback_categories:
                continue
            selected.append(rule)
            if len(selected) >= min(limit, 4):
                break
    categories: dict[str, int] = {}
    for rule in selected:
        categories[rule.category] = categories.get(rule.category, 0) + 1
    accepted_grade_target = (
        "pipeline_composite_or_lemma_bundle_master"
        if op_type == "crossover"
        else "semantic_mutation_or_bounded_generalization_with_new_final_obligation"
    )
    if target_style == "numeric_answer":
        accepted_grade_target = "canonical_family_params_with_nontrivial_reasoning"
    return {
        "source": "no_go_policy_registry",
        "total_rules": len(NO_GO_RULES),
        "op_type": op_type,
        "target_style": target_style,
        "target_family": target_family,
        "operator_variant": operator_variant,
        "accepted_grade_target": accepted_grade_target,
        "entropy_direction_target": "increase",
        "hard_no_go_flags": [rule.flag for rule in selected],
        "categories": categories,
        "no_go_docstrings": [
            f"{rule.flag}: {rule.retry_instruction}" for rule in selected
        ],
    }


def format_no_go_policy_pack(pack: dict[str, Any], *, title: str = "NoGoPolicyPack") -> str:
    docstrings = [str(item) for item in pack.get("no_go_docstrings") or [] if str(item).strip()]
    if not docstrings:
        return f"{title}: not_available"
    lines = [
        f"{title}:",
        "These are verifier-backed frontier-style no-go patterns for this execution surface.",
        f"- source: {pack.get('source', 'no_go_policy_registry')}",
        f"- total_registry_rules: {pack.get('total_rules', len(NO_GO_RULES))}",
        f"- accepted_grade_target: {pack.get('accepted_grade_target', 'not_available')}",
        f"- entropy_direction_target: {pack.get('entropy_direction_target', 'increase')}",
        "- If a plan or child would match a listed flag as the final theorem, avoid that pattern or return cannot_execute/giveup.",
        "- A helper/corollary may still be useful only when it is explicitly converted into a new entropy-increasing obligation or bounded generalization.",
        "- Use exact flag names in avoid/avoid_signatures when applicable.",
    ]
    lines.extend(f"- {line}" for line in docstrings)
    return "\n".join(lines)

#: Lessons the planner reads, in three layers.
#:
#: The registry grew one rule per observed failure — 58 of them, of which 46
#: never fired on 1,431 rows. Each was a real observation, and each stopped
#: generalising the moment it became a pattern match: a rule for "arithmetic
#: progression interval bound padding" catches that and nothing adjacent.
#: Distilling rather than listing is the measured choice — agents given raw
#: episode traces transferred worse than baseline (-9.5% / -7.5% on ALFWorld and
#: BabyAI) while the same experience distilled into principles transferred
#: positively (+6.5% / +9.0%).
#:
#: The layering follows the failure profile rather than tidiness. Over 2,800
#: generated rows, `parallel_crossover` (189) appears only under crossover and
#: `decorative_mutation` (36) only under mutation, but `projection_only_theorem`
#: splits 124/95 across both. Twenty-four flag types are shared and only
#: thirteen and eleven are operator-specific, so splitting everything would
#: halve the evidence behind the lessons both operators need — crossover has
#: just 878 rows to learn from as it is.
#:
#: Each layer carries what to do as well as what not to: a planner told only
#: what to avoid picks the nearest safe thing, which is how a pool fills with
#: corollaries.

SHARED_LESSONS: dict[str, str] = {
    "direct_corollary_helper": (
        "Do not plan a child that is a corollary of its parent. Shifting a "
        "numeral, weakening a divisibility, appending a side conjunct, or "
        "projecting out one component all leave the parent's proof sufficient. "
        "Plan a change that makes the parent's argument stop closing the goal."
    ),
    "surface_wrapper": (
        "Do not plan a child that changes only the surface. Renaming, "
        "re-parenthesising, adding `+ 0`, stacking negations, or narrowing a "
        "typeclass all type-check and none changes what must be proved. The "
        "test is whether a reader who knew the parent would have to think again."
    ),
    "fixed_computation": (
        "Do not plan a child whose proof is a finite check. Capping a variable "
        "so `decide` can enumerate, collapsing to a single index, or reducing to "
        "one arithmetic evaluation converts a theorem into a computation. If a "
        "bound is needed to make the statement true it belongs to the "
        "mathematics; if it is needed to make the proof close, it does not."
    ),
    "domain_specific": (
        "Do not re-mine a route the pool has already taken. Once a "
        "representative child exists for a transport, an index shift, or a "
        "residue bookkeeping pattern, further variants add rows without adding "
        "problems. Move to a different mechanism, not a different index."
    ),
    "novelty_memory": (
        "Do not plan a near-duplicate of an accepted row. Name the specific "
        "distinguishing obligation before the slot is dispatched; if it cannot "
        "be named, the slot is a duplicate however different the numbers look."
    ),
}

MUTATION_LESSONS: dict[str, str] = {
    "what_works": (
        "Plan one of these, and say which: turn a constant into a parameter the "
        "argument must survive; turn one instance into the law behind it; change "
        "the kind of claim (a value into a uniqueness statement, a minimum into "
        "a characterisation of the range, a divisibility into a congruence); "
        "loosen a hypothesis and keep the conclusion; move a modulus or exponent "
        "so a new case split is forced."
    ),
    "decorative_mutation": (
        "A child whose proof still ends by applying the parent's final lemma is "
        "decoration, however much notation it added. Enlarging a coefficient and "
        "discharging its nonzero-ness is the common form; wrapping the parent's "
        "equation in `∃!` whose witness the hypotheses already fix is the other."
    ),
    "lineage_complexity_without_new_role": (
        "Depth in the lineage is not difficulty. A child several generations "
        "down that plays the same role as its ancestor has added length only; "
        "plan a new role, not another layer."
    ),
}

CROSSOVER_LESSONS: dict[str, str] = {
    "what_works": (
        "Plan the route before the pair: one parent's conclusion becomes an "
        "input the other's derivation consumes (a bound, a modulus, an index, a "
        "case split); or the two conclusions are shown mutually exclusive; or "
        "one parent is instantiated at a value the other supplies and the "
        "combination yields an iff. If no such route exists for this pair, plan "
        "a mutation instead — an honest downgrade beats a fused-looking child "
        "whose halves never meet."
    ),
    "parallel_crossover": (
        "Discharging each parent separately and joining the results at the last "
        "line is the easy pattern, whatever the slot asked for. Multiplying two "
        "answers, conjoining two statements, or using one parent only to fix a "
        "constant or show a set is nonempty all leave the parents untouched by "
        "each other."
    ),
    "parent_checkpoint_not_consumed": (
        "A parent mentioned in prose but absent from the formal statement has "
        "not contributed. Both parents must appear as obligations the proof has "
        "to meet."
    ),
}


def _layer_for(op_type: str) -> dict[str, str]:
    op = str(op_type or "").strip().lower()
    if op == "crossover":
        return CROSSOVER_LESSONS
    if op == "mutation":
        return MUTATION_LESSONS
    return {}


def categories_of(flags: Iterable[str]) -> list[str]:
    """The categories a set of observed flags belongs to."""
    out: list[str] = []
    for flag in flags or []:
        rule = NO_GO_RULES_BY_FLAG.get(str(flag))
        if rule and rule.category not in out:
            out.append(rule.category)
    return out


def lessons_for_slot(
    op_type: str,
    *,
    observed_flags: Iterable[str] | None = None,
    shared_limit: int = 2,
) -> list[str]:
    """What this slot should read: its operator's layer, plus shared lessons.

    The operator layer always leads with `what_works`, because a planner given
    only prohibitions picks the nearest safe thing — which is how a pool fills
    with corollaries. Its remaining entries are included when this pool has
    actually produced that failure, so the planner reads about mistakes being
    made rather than mistakes once imagined.

    Shared lessons are ranked the same way and capped, because a pool of advice
    that grows without bound dilutes: the same few entries surface for every
    query and the rest never do. Two is a working guess, not a measured optimum.
    """
    observed = list(observed_flags or [])
    layer = _layer_for(op_type)
    out: list[str] = []
    if layer:
        out.append(layer["what_works"])
        for name, text in layer.items():
            if name != "what_works" and name in observed:
                out.append(text)

    seen_categories = [c for c in categories_of(observed) if c in SHARED_LESSONS]
    ranked = list(dict.fromkeys(seen_categories)) + [
        c for c in SHARED_LESSONS if c not in set(seen_categories)
    ]
    out.extend(SHARED_LESSONS[c] for c in ranked[:shared_limit])
    return out

def format_lessons(lessons: Iterable[str], *, title: str) -> str:
    """Render selected lessons as a prompt block."""
    items = [str(l).strip() for l in lessons if str(l).strip()]
    if not items:
        return ""
    body = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{body}\n"


def planner_lessons(observed_flags: Iterable[str] | None = None) -> list[str]:
    """Both operator layers plus shared, for an agent that plans both kinds.

    The planner decides which slots become mutations and which become
    crossovers, so withholding either layer would have it choose an operator
    without knowing how that operator fails. Workers get only their own layer;
    the planner is the one place both belong.
    """
    observed = list(observed_flags or [])
    out = [MUTATION_LESSONS["what_works"], CROSSOVER_LESSONS["what_works"]]
    for layer in (MUTATION_LESSONS, CROSSOVER_LESSONS):
        for name, text in layer.items():
            if name != "what_works" and name in observed:
                out.append(text)
    seen = [c for c in categories_of(observed) if c in SHARED_LESSONS]
    ranked = list(dict.fromkeys(seen)) + [c for c in SHARED_LESSONS if c not in set(seen)]
    out.extend(SHARED_LESSONS[c] for c in ranked[:2])
    return out

