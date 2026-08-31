#!/usr/bin/env python3
"""Audit generation yield against accepted-proxy and curated accepted rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.no_go_policy import (
    no_go_policy_summary,
    reject_cluster as policy_reject_cluster,
)


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _accepted_keys(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    keys: set[str] = set()
    for row in _rows(path):
        if row.get("_dedupe_key"):
            keys.add(str(row["_dedupe_key"]))
        if row.get("statement"):
            keys.add(f"statement:{_sha(row.get('statement'))}")
        if row.get("formal_statement"):
            keys.add(f"formal:{_sha(row.get('formal_statement'))}")
        hashes = row.get("hashes") if isinstance(row.get("hashes"), dict) else {}
        if hashes.get("statement_sha256"):
            keys.add(f"statement:{hashes['statement_sha256']}")
        if hashes.get("formal_statement_sha256"):
            keys.add(f"formal:{hashes['formal_statement_sha256']}")
    return {key for key in keys if key}


def _is_generated(row: Dict[str, Any]) -> bool:
    return row.get("op_type") not in {"survivor", "fallback_survivor", "seed_proof_completion"}


def _accepted_proxy(row: Dict[str, Any]) -> Dict[str, Any]:
    evidence = row.get("quality_evidence") if isinstance(row.get("quality_evidence"), dict) else {}
    proxy = evidence.get("accepted_proxy") if isinstance(evidence, dict) else {}
    return proxy if isinstance(proxy, dict) else {"pass": False, "flags": ["missing_accepted_proxy"]}


def _audit_accepted_grade_flags(row: Dict[str, Any]) -> List[str]:
    if row.get("status") != "certified" or row.get("op_type") in {"survivor", "fallback_survivor", "seed_proof_completion"}:
        return []
    surface = " ".join(
        str(row.get(key) or "")
        for key in ("statement", "formal_statement", "lean_code", "proof_plan", "solution", "problem_id")
    )
    lower = surface.lower()
    compact = re.sub(r"\s+", "", surface).lower()
    flags: List[str] = []
    prime_domain = "prime divisors" in lower or "nat.prime" in lower or "filter(funx" in compact
    divisor_sum_500 = "positive divisors of 500" in lower or "nat.divisors500" in compact or "sigma_1(500)" in lower
    ap_surface = (
        "arithmetic progression" in lower
        or "arithmetic sequence" in lower
        or "finset.range98" in compact
        or "u(n+1)" in lower
    )

    if row.get("op_type") == "mutation":
        if (
            ("20∣k" in compact or "20 ∣ k" in surface or "divisible by 20" in lower)
            and ("k^2+2^k" in compact or "k ^ 2 + 2 ^ k" in lower)
            and ("units digit" in lower or "%10=6" in compact)
            and not any(marker in compact for marker in ("%20=16", "¬nat.prime", "nat.prime", "k+20*t", "↔"))
        ):
            flags.append("theorem_local_corollary_dominated")
        if (
            ("s t : finset" in lower or "s,t:finset" in compact or "(s t : finset" in lower)
            and ("t=s" in compact or "s=t" in compact)
            and ("n∈s" in compact or "nins" in compact)
            and ("n∈t" in compact or "nint" in compact)
            and any(marker in lower for marker in ("extensionality", "prove equality of finite sets", "extensionality theorem"))
        ):
            flags.append("definitional_extensionality_only")
        if (
            ("isprincipalidealring" in compact or "principal ideal ring" in lower)
            and ("idealgenerator" in compact or "idealspangenerator" in compact or ("every ideal" in lower and "principal" in lower))
            and ("exact⟨" in compact or "is assigned a principal generator" in lower or "span {g(i)}" in lower)
        ):
            flags.append("pid_definition_restatement")
        if (
            "rootset" in compact
            and "natdegree" in compact
            and "separable" in compact
            and ("card_rootset_eq_natdegree_iff_of_splits" in compact)
            and not any(marker in compact for marker in ("hrep:", "hn:odd", "isunit", "ideal.span", "p.derivative", "p^r"))
        ):
            flags.append("standard_library_theorem_restatement")
        if prime_domain and divisor_sum_500 and any(
            marker in lower
            for marker in (
                "prime_divisor_finset",
                "finset of prime divisors",
                "has cardinality",
                ".card =",
                "product of the prime divisors",
                ".prod id",
            )
        ):
            flags.append("proof_infrastructure_only")
        if ap_surface and not prime_domain and any(
            marker in lower for marker in ("odd-indexed", "even-indexed", "odd indexed", "even indexed", "first 98 terms")
        ):
            flags.append("direct_parent_corollary_only")
        if (
            ap_surface
            and ("a+20*d" in compact or "a + 20 * d" in lower)
            and ("≤" in surface or "<=" in surface or "\\le" in surface)
            and any(
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
                    "two-step bound",
                    "shifted inequality",
                )
            )
        ):
            flags.append("ap_bound_padding_only")
        if (
            ap_surface
            and ("a+20*d" in compact or "a + 20 * d" in lower)
            and any(marker in compact for marker in ("set.ioc", "set.icc", "∈set."))
            and any(marker in lower or marker in compact for marker in ("135 ≤ m", "135≤m", "l≤135", "135≤u", "real bound"))
        ):
            flags.append("ap_interval_bound_padding_only")
        if (
            "398" in compact
            and "*7" in compact
            and any(marker in lower or marker in compact for marker in ("n = 57", "n=57", "bounded inverse", "force n"))
            and (
                any(marker in compact for marker in ("n/19=3", "n=19*q", "2^(n/19)=8", "2^(n/19)%8=0", "2^q%8=0"))
                or any(marker in lower for marker in ("quotient n / 19", "quotient q", "2 raised to the quotient"))
            )
        ):
            flags.append("solved_parameter_quotient_corollary_only")
        if (
            "398" in compact
            and "*7" in compact
            and any(marker in lower or marker in compact for marker in ("n = 57", "n=57", "bounded inverse", "force n"))
            and any(marker in compact for marker in ("n%19=0", "19∣n", "n=57"))
        ):
            flags.append("mod_inverse_arithmetic_corollary_only")
        if (
            "398" in compact
            and ("n*7%398=1" in compact or "multiplicative inverse of 7 modulo 398" in lower)
            and any(marker in compact for marker in ("finset.range398", "finset.icc5060"))
            and ("k*7%398=1" in compact or "remainder 0 modulo 19" in lower or "at most 57" in lower)
        ):
            flags.append("finite_mod_inverse_window_restatement")
    if row.get("op_type") == "crossover" and ap_surface and prime_domain:
        has_domain_sum = any(
            marker in lower for marker in ("h_prime_sum", "finite sum of its elements", "rational finite sum")
        )
        has_domain_card = "h_card" in lower or "cardinality" in lower or ".card=4" in compact
        shifted = any(marker in compact for marker in ("u(p+1)", "u(p+2)", "u(p+3)", "u(2*p)", ".card)*p+1"))
        if shifted and not (has_domain_card and has_domain_sum):
            flags.append("affine_index_drift_only")
        if has_domain_card and not has_domain_sum and "finset.range" in compact:
            flags.append("cardinality_only_window")
    if row.get("op_type") == "crossover":
        has_ap_bridge = (
            ("a+20*d" in compact or "a+20*d+t" in compact)
            and (
                "hn_shift" in lower
                or "(n:ℝ)=a+20*d+t" in compact
                or "(n : ℝ) = a + 20 * d + t" in lower
                or "bridge" in lower
            )
            and ("b<398" in compact or "hb : b < 398" in lower or "obtain n < 398" in lower)
        )
        has_existing_mod_pipeline = (
            ("n*7%398=1" in compact or "multiplicative inverse of 7 modulo 398" in lower)
            and ("finset.icc17" in compact or "gcd(x,8)=1" in compact or "coprime to 8" in lower)
            and ("3^(n+c)%8=3" in compact or "3 ^ (n + c) % 8 = 3" in lower)
        )
        if has_ap_bridge and has_existing_mod_pipeline:
            flags.append("artificial_bridge_to_existing_pipeline")

        has_ap_value = "a+20*d" in compact and ("135" in compact or "21st term" in lower)
        has_inverse_quotient = (
            ("n*7%398=1" in compact or "multiplicative inverse of 7 modulo 398" in lower)
            and ("n/19" in compact or "quotient" in lower)
        )
        fitted_bound = (
            ("≤" in surface or "<=" in surface)
            and "132+" in compact
            and ("n/19" in compact or "quotient" in lower)
        )
        if has_ap_value and has_inverse_quotient and fitted_bound:
            flags.append("numeric_bound_fitting_crossover")
        if (
            any(marker in compact for marker in ("h₃:∀c", "h3:∀c", "hprime_source:∀c", "h_prime_source:∀c"))
            and "nat.divisors500" in compact
            and "nat.prime" in compact
            and any(marker in compact for marker in ("exacth₃", "exacth3", "exacthprime_source", "exacth_prime_source"))
        ):
            flags.append("parent_theorem_assumption_smuggling")
    if len(str(row.get("problem_id") or "").split("__theorem_gen")) >= 6 and not any(
        marker in lower for marker in ("master theorem", "pipeline", "finite sum of its elements")
    ):
        flags.append("lineage_complexity_without_new_role")
    return sorted(set(flags))


def _accepted_grade_proxy(row: Dict[str, Any]) -> Dict[str, Any]:
    proxy = dict(_accepted_proxy(row))
    flags = sorted(set(list(proxy.get("flags") or []) + _audit_accepted_grade_flags(row)))
    proxy["flags"] = flags
    proxy["accepted_grade_pass"] = bool(proxy.get("pass")) and not flags
    return proxy


def _reject_cluster(flags: Iterable[str]) -> str:
    flag_set = {str(flag) for flag in flags if str(flag)}
    if flag_set & {"proof_infrastructure_only", "aggregate_helper_only"}:
        return "helper_only"
    return policy_reject_cluster(flag_set)


def _ledger_hit(row: Dict[str, Any], accepted_keys: set[str]) -> bool:
    candidates = {
        str(row.get("_dedupe_key") or ""),
        f"statement:{_sha(row.get('statement'))}" if row.get("statement") else "",
        f"formal:{_sha(row.get('formal_statement'))}" if row.get("formal_statement") else "",
    }
    return bool(accepted_keys & {item for item in candidates if item})


def _unique_surface_count(rows: Iterable[Dict[str, Any]]) -> int:
    keys: set[str] = set()
    for row in rows:
        if row.get("formal_statement"):
            keys.add(f"formal:{_sha(row.get('formal_statement'))}")
        elif row.get("statement"):
            keys.add(f"statement:{_sha(row.get('statement'))}")
        elif row.get("problem_id"):
            keys.add(f"problem:{row.get('problem_id')}")
    return len(keys)


def _counter_list(rows: Iterable[Dict[str, Any]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            counter.update(str(item) for item in value if str(item))
        elif value:
            counter[str(value)] += 1
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generation yield funnel.")
    parser.add_argument("--input", required=True, type=Path, help="Generation JSONL to audit.")
    parser.add_argument(
        "--accepted",
        type=Path,
        default=Path("data/evaluation/treatment_inventory/final_curated/accepted.jsonl"),
        help="Curated accepted ledger. Default final_curated/accepted.jsonl.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    args = parser.parse_args()

    rows = _rows(args.input)
    accepted_keys = _accepted_keys(args.accepted)
    by_generation: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        generation = row.get("generation")
        if isinstance(generation, int):
            by_generation[generation].append(row)

    generations = []
    total_flags: Counter[str] = Counter()
    total_reasons: Counter[str] = Counter()
    total_clusters: Counter[str] = Counter()
    for generation in sorted(by_generation):
        gen_rows = by_generation[generation]
        generated = [row for row in gen_rows if _is_generated(row)]
        certified = [row for row in generated if row.get("status") == "certified"]
        non_weak = [row for row in certified if row.get("quality_verdict") != "weak"]
        eligible = [row for row in certified if row.get("parent_eligible")]
        proxy_pass = [row for row in generated if _accepted_proxy(row).get("pass")]
        accepted_grade = [
            row for row in generated if _accepted_grade_proxy(row).get("accepted_grade_pass")
        ]
        unique_accepted_grade = _unique_surface_count(accepted_grade)
        ledger = [row for row in generated if _ledger_hit(row, accepted_keys)]
        proxy_flags = Counter(
            flag
            for row in generated
            for flag in list(_accepted_proxy(row).get("flags") or [])
        )
        reject_clusters = Counter(
            _reject_cluster(_accepted_grade_proxy(row).get("flags") or [])
            for row in generated
            if not _accepted_grade_proxy(row).get("accepted_grade_pass")
        )
        selection_reasons = Counter(
            str(row.get("selection_reason") or "unset")
            for row in generated
            if not row.get("parent_eligible")
        )
        total_flags.update(proxy_flags)
        total_reasons.update(selection_reasons)
        total_clusters.update(reject_clusters)
        generations.append(
            {
                "generation": generation,
                "generated": len(generated),
                "certified": len(certified),
                "non_weak": len(non_weak),
                "parent_eligible": len(eligible),
                "accepted_proxy": len(proxy_pass),
                "accepted_grade_proxy": len(accepted_grade),
                "unique_accepted_grade_proxy": unique_accepted_grade,
                "ledger_accepted": len(ledger),
                "reserve_generated": sum(1 for row in generated if row.get("source_kind") == "reserve_generated"),
                "proxy_flags": dict(proxy_flags.most_common()),
                "reject_clusters": dict(reject_clusters.most_common()),
                "selection_reasons": dict(selection_reasons.most_common()),
            }
        )

    report = {
        "input": str(args.input),
        "accepted": str(args.accepted),
        "rows": len(rows),
        "no_go_policy": no_go_policy_summary(),
        "generations": generations,
        "totals": {
            "proxy_flags": dict(total_flags.most_common()),
            "reject_clusters": dict(total_clusters.most_common()),
            "selection_reasons": dict(total_reasons.most_common()),
            "quality_flags": dict(_counter_list((row for row in rows if _is_generated(row)), "quality_flags").most_common()),
        },
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"# Yield audit: {args.input}")
    print()
    print("| gen | generated | certified | non_weak | parent_eligible | accepted_proxy | accepted_grade_proxy | unique_accepted_grade | ledger_accepted | reserve |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in generations:
        print(
            f"| {item['generation']} | {item['generated']} | {item['certified']} | "
            f"{item['non_weak']} | {item['parent_eligible']} | {item['accepted_proxy']} | "
            f"{item['accepted_grade_proxy']} | {item['unique_accepted_grade_proxy']} | "
            f"{item['ledger_accepted']} | {item['reserve_generated']} |"
        )
    print()
    print("## Accepted-proxy failure flags")
    for flag, count in total_flags.most_common(20):
        print(f"- `{flag}`: {count}")
    print()
    print("## Accepted-grade reject clusters")
    for cluster, count in total_clusters.most_common(20):
        print(f"- `{cluster}`: {count}")
    print()
    policy = no_go_policy_summary()
    print("## No-go policy")
    print(f"- total rules: {policy['total']}")
    for category, count in dict(policy["by_category"]).items():
        print(f"- `{category}`: {count}")
    print()
    print("## Selection reasons")
    for reason, count in total_reasons.most_common(20):
        print(f"- `{reason}`: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
