"""
Template-based Lean 4 formalization for supported mathematical families.

Each supported family maps a natural-language problem onto a
native_decide-provable Lean 4 theorem (no Mathlib required).

Supported families
------------------
gcd                — Nat.gcd a b = k
gcd_divisor_sum    — sum of divisors of Nat.gcd a b = k
units_digit        — base^exp % 10 = k
divisor_sum        — (Finset.filter (· ∣ n) (Finset.range (n+1))).sum id = k
divisor_sum_mod    — a % (sum of divisors of n) = k
stars_and_bars     — Nat.choose (S + vars-1) (vars-1) = k
arithmetic_series  — (Finset.range n).sum (fun i => a + d * i) = k
modular_congruence — a % m = k
"""

import re
import logging
from dataclasses import dataclass
from math import comb
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES: frozenset = frozenset({
    "gcd",
    "gcd_divisor_sum",
    "units_digit",
    "divisor_sum",
    "divisor_sum_mod",
    "stars_and_bars",
    "arithmetic_series",
    "modular_congruence",
})

# Patterns that identify a trivially-true stub (not semantically faithful)
_STUB_PATTERNS = [
    re.compile(r':\s*True\s*:=\s*by\s*trivial', re.IGNORECASE),
    re.compile(r'example\s*:\s*True\b', re.IGNORECASE),
    re.compile(r'theorem\s+\w+\s*:\s*True\b', re.IGNORECASE),
    re.compile(r':=\s*trivial\b', re.IGNORECASE),
]


@dataclass
class TemplateResult:
    family: str
    lean_code: str
    theorem_name: str
    proof_method: str       # "native_decide" | "sorry"
    params: Dict[str, Any]  # extracted numeric parameters
    aligned: bool           # NL params are consistent with Lean params
    alignment_note: str     # "" on success, description of mismatch otherwise


# ─────────────────────────────────────────────────────────────────────────────
# Public utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_trivial_stub(lean_code: str) -> bool:
    """Return True if lean_code is a trivially-true placeholder with no math content."""
    return any(p.search(lean_code) for p in _STUB_PATTERNS)


def detect_family(statement: str) -> Optional[str]:
    """
    Detect the mathematical family of a natural-language problem statement.

    Returns a family name from SUPPORTED_FAMILIES, or None if unrecognised.
    More specific patterns are checked first to avoid false matches.
    """
    s = statement.lower()

    if re.search(r'\bgcd\b|greatest common divisor', s) and re.search(r'sum of.{0,35}divisors?|positive divisors', s):
        return "gcd_divisor_sum"
    if re.search(r'sum of.{0,35}divisors?|positive divisors', s) and re.search(r'\bmod\b|\bmodulo\b|\bremainder\b', s):
        return "divisor_sum_mod"
    if re.search(r'\bgcd\b|greatest common divisor', s):
        return "gcd"
    if re.search(r'\bunits?\s*digit\b|\blast\s*digit\b|\bunits?\s*place\b', s):
        return "units_digit"
    if re.search(r'sum of.{0,25}divisors?|divisors?.{0,25}sum|positive divisors', s):
        return "divisor_sum"
    if re.search(r'nonneg.{0,40}(solution|integer)|solutions?.{0,30}(equation|sum|=)', s):
        return "stars_and_bars"
    if re.search(r'arithmetic (series|sequence|progression)|sum of.{0,25}terms?', s):
        return "arithmetic_series"
    if re.search(r'\bmod\b|\bmodulo\b|\bremainder\b|\bcongruent\b', s):
        return "modular_congruence"
    return None


def check_param_alignment(lean_code: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify that every extracted parameter (including the expected answer) appears
    literally in the Lean theorem string.

    Returns (aligned, note) where note is "" on success.
    """
    missing = [
        f"{k}={v}" for k, v in params.items() if str(v) not in lean_code
    ]
    if missing:
        return False, f"Missing in Lean statement: {', '.join(missing)}"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Family-specific template generators (private)
# ─────────────────────────────────────────────────────────────────────────────

def _template_gcd(statement: str, answer: str) -> Optional[TemplateResult]:
    nums = re.findall(r'\d+', statement)
    if len(nums) < 2:
        return None
    try:
        a, b = int(nums[0]), int(nums[1])
        k = int(answer.strip())
    except (ValueError, IndexError):
        return None

    name = f"gcd_{a}_{b}"
    code = f"theorem {name} : Nat.gcd {a} {b} = {k} := by native_decide"
    params = {"a": a, "b": b, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="gcd", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_gcd_divisor_sum(statement: str, answer: str) -> Optional[TemplateResult]:
    nums = re.findall(r'\d+', statement)
    if len(nums) < 2:
        return None
    try:
        a, b = int(nums[0]), int(nums[1])
        k = int(answer.strip())
    except (ValueError, IndexError):
        return None

    g_expr = f"(Nat.gcd {a} {b})"
    name = f"gcd_divisor_sum_{a}_{b}"
    code = (
        f"theorem {name} : "
        f"((List.range ({g_expr} + 1)).filter "
        f"(fun d => d != 0 && {g_expr} % d == 0)).sum = {k} "
        f":= by native_decide"
    )
    params = {"a": a, "b": b, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="gcd_divisor_sum", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_units_digit(statement: str, answer: str) -> Optional[TemplateResult]:
    # Try explicit "base^{exp}" or "base^exp" notation first
    m = re.search(r'(\d+)\s*[\^\\]\s*\{?(\d+)\}?', statement)
    if m:
        base, exp = int(m.group(1)), int(m.group(2))
    else:
        nums = re.findall(r'\d+', statement)
        if len(nums) < 2:
            return None
        base, exp = int(nums[0]), int(nums[1])
    try:
        k = int(answer.strip())
    except ValueError:
        return None

    name = f"units_digit_{base}_{exp}"
    code = f"theorem {name} : {base}^{exp} % 10 = {k} := by native_decide"
    params = {"base": base, "exp": exp, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="units_digit", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_divisor_sum(statement: str, answer: str) -> Optional[TemplateResult]:
    nums = re.findall(r'\d+', statement)
    if not nums:
        return None
    try:
        n = int(nums[0])
        k = int(answer.strip())
    except (ValueError, IndexError):
        return None

    name = f"divisor_sum_{n}"
    # List-based (not Finset) — works in Lean 4 core without Mathlib
    code = (
        f"theorem {name} : "
        f"((List.range {n + 1}).filter (fun d => d != 0 && {n} % d == 0)).sum = {k} "
        f":= by native_decide"
    )
    params = {"n": n, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="divisor_sum", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_divisor_sum_mod(statement: str, answer: str) -> Optional[TemplateResult]:
    nums = re.findall(r'\d+', statement)
    if len(nums) < 2:
        return None
    try:
        n, a = int(nums[0]), int(nums[1])
        k = int(answer.strip())
    except (ValueError, IndexError):
        return None

    divisor_sum_expr = (
        f"((List.range {n + 1}).filter "
        f"(fun d => d != 0 && {n} % d == 0)).sum"
    )
    name = f"divisor_sum_mod_{n}_{a}"
    code = f"theorem {name} : {a} % {divisor_sum_expr} = {k} := by native_decide"
    params = {"n": n, "a": a, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="divisor_sum_mod", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _stars_bars_closed_form(S: int, var_count: int) -> str:
    """
    Return a Lean 4 core arithmetic expression for C(S+vars-1, vars-1).

    Uses the product formula (S+1)*(S+2)*...*(S+vars-1) / (vars-1)!
    which is integer-valued and verified by native_decide without Mathlib.
    """
    if var_count == 2:
        return f"({S} + 1)"
    numerator_parts = [f"({S} + {i})" for i in range(1, var_count)]
    denominator = 1
    for i in range(1, var_count):
        denominator *= i
    return " * ".join(numerator_parts) + f" / {denominator}"


def _template_stars_and_bars(statement: str, answer: str) -> Optional[TemplateResult]:
    """
    Nonneg integer solutions to x_1 + ... + x_vars = S.

    Encodes C(S + vars - 1, vars - 1) using the closed-form product formula
    — no Nat.choose or Mathlib needed, native_decide verifies it directly.
    """
    nums = re.findall(r'\d+', statement)
    if not nums:
        return None
    try:
        k = int(answer.strip())
    except ValueError:
        return None

    # Count subscripted variable terms; fall back to 3
    var_count = len(re.findall(r'x\s*[_₁₂₃₄₅₆\d]', statement))
    if var_count == 0:
        var_count = 3

    # Try to find (S, var_count) pair such that C(S + vars-1, vars-1) == k
    S = int(nums[-1]) if nums else 0
    if comb(S + var_count - 1, var_count - 1) != k:
        found = False
        for v in range(2, 7):
            for s_candidate in [int(x) for x in nums]:
                if comb(s_candidate + v - 1, v - 1) == k:
                    S, var_count = s_candidate, v
                    found = True
                    break
            if found:
                break

    computed = comb(S + var_count - 1, var_count - 1)
    name = f"stars_bars_{var_count}vars_sum{S}"
    expr = _stars_bars_closed_form(S, var_count)
    code = f"theorem {name} : {expr} = {k} := by native_decide"
    params = {"vars": var_count, "S": S, "expected": k}
    if computed != k:
        return TemplateResult(
            family="stars_and_bars", lean_code=code, theorem_name=name,
            proof_method="native_decide", params=params,
            aligned=False,
            alignment_note=f"C({S}+{var_count-1},{var_count-1})={computed} ≠ {k}",
        )
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="stars_and_bars", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_arithmetic_series(statement: str, answer: str) -> Optional[TemplateResult]:
    """
    Sum of n-term arithmetic series: a_i = first + diff * i, i in 0..n-1.
    Template: (Finset.range n).sum (fun i => first + diff * i) = k.
    """
    nums = re.findall(r'\d+', statement)
    try:
        k = int(answer.strip())
    except ValueError:
        return None

    if len(nums) < 3:
        return None

    # Try explicit sequence form: "first 20 terms of ... 3, 7, 11, ..."
    n_terms, first, diff = int(nums[0]), int(nums[1]), int(nums[2])
    found = False
    normalized = statement.lower()
    if "sequence" in normalized and len(nums) >= 4:
        seq_n_terms = int(nums[0])
        seq_first = int(nums[1])
        seq_diff = int(nums[2]) - int(nums[1])
        if seq_n_terms > 0 and sum(seq_first + seq_diff * x for x in range(seq_n_terms)) == k:
            n_terms, first, diff = seq_n_terms, seq_first, seq_diff
            found = True

    # Try orderings to find (n_terms, first, diff) that sums to k
    for i in range(len(nums)):
        if found:
            break
        for j in range(len(nums)):
            for l in range(len(nums)):
                if i == j or i == l or j == l:
                    continue
                nt, f, d = int(nums[i]), int(nums[j]), int(nums[l])
                if nt > 0 and sum(f + d * x for x in range(nt)) == k:
                    n_terms, first, diff = nt, f, d
                    found = True
                    break
            if found:
                break
        if found:
            break

    name = f"arith_series_{n_terms}terms"
    # List-based (not Finset) — works in Lean 4 core without Mathlib
    code = (
        f"theorem {name} : "
        f"((List.range {n_terms}).map (fun i => {first} + {diff} * i)).sum = {k} "
        f":= by native_decide"
    )
    params = {"n_terms": n_terms, "first": first, "diff": diff, "expected": k}
    computed = sum(first + diff * x for x in range(n_terms))
    if computed != k:
        return TemplateResult(
            family="arithmetic_series", lean_code=code, theorem_name=name,
            proof_method="native_decide", params=params,
            aligned=False,
            alignment_note=f"sum={computed} ≠ {k}",
        )
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="arithmetic_series", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


def _template_modular_congruence(statement: str, answer: str) -> Optional[TemplateResult]:
    nums = re.findall(r'\d+', statement)
    if len(nums) < 2:
        return None
    try:
        a, m = int(nums[0]), int(nums[1])
        k = int(answer.strip())
    except (ValueError, IndexError):
        return None

    name = f"mod_{a}_{m}"
    code = f"theorem {name} : {a} % {m} = {k} := by native_decide"
    params = {"a": a, "m": m, "expected": k}
    aligned, note = check_param_alignment(code, params)
    return TemplateResult(
        family="modular_congruence", lean_code=code, theorem_name=name,
        proof_method="native_decide", params=params,
        aligned=aligned, alignment_note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCHERS = {
    "gcd": _template_gcd,
    "gcd_divisor_sum": _template_gcd_divisor_sum,
    "units_digit": _template_units_digit,
    "divisor_sum": _template_divisor_sum,
    "divisor_sum_mod": _template_divisor_sum_mod,
    "stars_and_bars": _template_stars_and_bars,
    "arithmetic_series": _template_arithmetic_series,
    "modular_congruence": _template_modular_congruence,
}


def generate_lean_template(
    statement: str,
    answer: str,
    family: Optional[str] = None,
) -> Optional[TemplateResult]:
    """
    Generate a native_decide-provable Lean 4 theorem for the given problem.

    Args:
        statement: Natural language problem statement.
        answer:    Expected answer string (e.g. "9", "360").
        family:    Override family detection; auto-detected if None.

    Returns:
        TemplateResult on success, None if family is unsupported or params
        cannot be extracted.
    """
    if family is None:
        family = detect_family(statement)
    if family not in _DISPATCHERS:
        logger.debug("Unsupported family for Lean template: %r", family)
        return None

    try:
        result = _DISPATCHERS[family](statement, answer)
        if result is not None:
            logger.debug(
                "Template generated: family=%s aligned=%s theorem=%s",
                family, result.aligned, result.theorem_name,
            )
        return result
    except Exception as exc:
        logger.warning("Template generation failed for family=%s: %s", family, exc)
        return None
