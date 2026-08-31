"""Find hypotheses the proof does not need, by removing them and recompiling.

A released row read

    theorem shifted_square_unique_minimizer
      (a : ℝ) (ha_lower : -10 ≤ a) (ha_upper : a ≤ 10)
      (f : ℝ → ℝ) (hf : f = fun x => (x - a)^2) :
      ∀ y : ℝ, IsMinOn f Set.univ y ↔ y = a

and both bounds on `a` are irrelevant to the conclusion. A dead hypothesis makes
a problem look harder than it is and hands a prover a clue that leads nowhere,
so it is a defect in a benchmark even though the proof is perfectly valid. The
judge cannot catch it: it can read the hypothesis but not test it.

The obvious check -- look for the hypothesis's name in the proof body -- is wrong
often enough to be useless. `omega`, `simp_all`, `interval_cases`, `linarith`,
`norm_num`, `decide` and `aesop` scan the whole local context and consume
hypotheses without naming any. Measured over one release, that name search was
right 24 times out of 53.

So the name search is used only to pick candidates, and Lean decides each one:
drop the binder, recompile, and if the proof still closes the hypothesis was
dead. Removals are confirmed one at a time against the previous survivor rather
than all at once, because dropping one hypothesis can make the next
load-bearing -- on one row, three individually removable hypotheses left the
proof broken when all three went together.
"""

from __future__ import annotations

import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

#: Named hypotheses only. An instance binder or a plain datum is not the kind of
#: thing this is about, and `h`-prefixed names are the convention the generator
#: follows throughout.
_BINDER = re.compile(r"\((h[A-Za-z0-9_'₀-₉]*)\s*[:\s]")

#: Tactics that read the local context whole. Their presence is why the name
#: search cannot be trusted; it is recorded so a reader can see why a candidate
#: was worth testing.
CONTEXT_SCANNING = (
    "omega", "simp_all", "interval_cases", "decide", "norm_num",
    "linarith", "nlinarith", "aesop", "tauto", "positivity", "field_simp",
)


def candidates(formal_statement: str, lean_code: str) -> List[str]:
    """Named hypotheses whose name never appears in the proof body."""
    body = str(lean_code or "").split(":= by", 1)[-1]
    return [
        name
        for name in _BINDER.findall(str(formal_statement or ""))
        if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_'₀-₉])", body)
    ]


def drop_binder(text: str, name: str) -> Optional[str]:
    """`text` with the binder group introducing `name` removed.

    Parentheses are counted rather than matched by pattern, because binder types
    nest: `(hn : ¬ ((3:ℤ) ∣ y))` is two levels deep and a one-level pattern
    silently skips it.
    """
    opening = str(text or "").find("(" + name)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[:opening] + text[index + 1:]
    return None


def scanning_tactics(lean_code: str) -> List[str]:
    body = str(lean_code or "").split(":= by", 1)[-1]
    return sorted({tactic for tactic in CONTEXT_SCANNING if tactic in body})


async def prune_dead(
    verifier: Callable[..., Awaitable[Any]],
    formal_statement: str,
    lean_code: str,
    *,
    timeout: float = 300.0,
    max_candidates: int = 6,
) -> Dict[str, Any]:
    """Remove every hypothesis Lean confirms the proof does not need.

    Returns the pruned statement and proof alongside the evidence. When nothing
    is removed the originals come back unchanged, so a caller can assign the
    result unconditionally.
    """
    original_statement = str(formal_statement or "")
    original_code = str(lean_code or "")
    evidence: Dict[str, Any] = {
        "measured": False,
        "removed": [],
        "used_silently": [],
        "tactics": scanning_tactics(original_code),
        "formal_statement": original_statement,
        "lean_code": original_code,
        "why": "",
    }
    if not original_statement.strip() or not original_code.strip():
        evidence["why"] = "no statement or proof"
        return evidence

    names = candidates(original_statement, original_code)
    if not names:
        evidence["measured"] = True
        return evidence
    if len(names) > max_candidates:
        # One Lean call per candidate. A row with a dozen unnamed hypotheses is
        # unusual enough that spending a dozen calls on it in-flight is not
        # worth it; the offline scan has no such bound.
        evidence["why"] = f"{len(names)} candidates exceeds max_candidates={max_candidates}"
        names = names[:max_candidates]

    statement, code = original_statement, original_code
    remaining = list(names)
    stop = False
    # Removal is repeated to a fixpoint, not run once per candidate. Dependence
    # goes both ways: dropping one hypothesis can make another load-bearing, and
    # it can equally make another dead. On a release row, `h₃` survived the
    # single pass because `h₄` was still present, and only the judge noticed it
    # had become unused once `h₄` went. Each round re-tests everything still
    # standing, so the result no longer depends on the order of the candidates.
    while remaining and not stop:
        survivors = []
        removed_this_round = False
        for name in remaining:
            trimmed_code = drop_binder(code, name)
            trimmed_statement = drop_binder(statement, name)
            if trimmed_code is None or trimmed_statement is None:
                continue
            try:
                verdict = await verifier(trimmed_code, timeout=timeout)
            except TypeError:
                verdict = await verifier(trimmed_code)
            except Exception as error:  # pragma: no cover - transport failures
                evidence["why"] = f"probe error: {error}"[:120]
                stop = True
                survivors.append(name)
                break
            if getattr(verdict, "system_error", None):
                evidence["why"] = str(verdict.system_error)[:120]
                stop = True
                survivors.append(name)
                break
            # Only a proof that still closes is evidence the hypothesis was
            # dead. A failure to compile for any other reason leaves it in place.
            if bool(getattr(verdict, "complete", False) or getattr(verdict, "ok", False)):
                statement, code = trimmed_statement, trimmed_code
                evidence["removed"].append(name)
                removed_this_round = True
            else:
                survivors.append(name)
        remaining = survivors if removed_this_round else []
        if not removed_this_round:
            evidence["used_silently"] = survivors

    evidence["measured"] = True
    evidence["formal_statement"] = statement
    evidence["lean_code"] = code
    return evidence


def enabled() -> bool:
    """On by default: a dead hypothesis is a defect and the fix is mechanical."""
    return os.getenv("HYPOTHESIS_PRUNE", "1") == "1"
