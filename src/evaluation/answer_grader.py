"""Boxed-answer extraction and symbolic regrading.

Implements the EntropyMaG-1 grading contract:
- last \\boxed{...} in the model's response is the canonical answer
- numeric normalization (strip commas, leading +, trailing zeros after decimal)
- symbolic fallback via sympy if available
- exact string fallback after canonicalization
"""

from __future__ import annotations

import re
from typing import Optional

_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the contents of the *last* \\boxed{...} in `text`.

    Uses brace-depth counting so nested braces inside the answer are kept.
    Returns None if no \\boxed{...} occurs.
    """
    if not text:
        return None
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_answer(ans: str) -> str:
    """Canonicalize an answer string for comparison.

    - strip whitespace and surrounding $...$
    - drop a leading `+`
    - drop thousands separators (commas between digits)
    - collapse `1.0` to `1`, `1.50` to `1.5`
    """
    if ans is None:
        return ""
    s = ans.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("+"):
        s = s[1:].strip()
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def _try_sympy_equal(pred: str, gold: str) -> Optional[bool]:
    """Return True/False if sympy can decide equality, else None."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except Exception:
        return None
    parsers = [sympy.sympify, parse_latex]
    last: Optional[bool] = None
    for parse in parsers:
        try:
            p = parse(pred)
            g = parse(gold)
        except Exception:
            continue
        try:
            diff = sympy.simplify(p - g)
            last = diff == 0
            if last:
                return True
        except Exception:
            continue
    return last


def grade_answer(prediction: Optional[str], gold: str) -> bool:
    """Return True if `prediction` matches `gold` after canonicalization."""
    if prediction is None:
        return False
    p = normalize_answer(prediction)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if p.lower() == g.lower():
        return True
    decision = _try_sympy_equal(p, g)
    if decision is not None:
        return decision
    return False
