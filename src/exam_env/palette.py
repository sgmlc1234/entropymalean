"""Palette construction for the Lean exam environment.

The palette mirrors a Lean game's tactic/theorem panels: the solver first
sees only NAMES; an ``inspect`` action reveals the card (signature + prose).
Theorem palettes are extracted from a row's certified ground-truth proof and
validated against the pinned Mathlib in one ``#check`` probe, so every card
is real — the environment never invents documentation.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Game-style cards for common Mathlib tactics: what it does + how to use it.
TACTIC_DOCS: Dict[str, str] = {
    "intro": (
        "Introduces the next ∀-bound variable or the antecedent of an "
        "implication into the local context. `intro n hn` introduces two at "
        "once. Use when the goal starts with `∀` or `→`."
    ),
    "exact": (
        "Closes the goal with a term whose type is exactly the goal. "
        "`exact Nat.add_zero n` applies a known theorem to the right "
        "arguments. Nothing happens to other goals."
    ),
    "apply": (
        "Applies a theorem whose CONCLUSION matches the goal, turning its "
        "hypotheses into new goals. Use when you know the final step but "
        "still owe its premises."
    ),
    "refine": (
        "Like `exact` but with `?_` placeholders for the parts you want to "
        "prove later: `refine ⟨?_, ?_⟩` splits a conjunction into two goals."
    ),
    "rw": (
        "Rewrites the goal with an equation, left to right. `rw [h]` uses "
        "hypothesis h : a = b to replace a by b. `rw [← h]` goes right to "
        "left; `rw [h] at h2` rewrites inside another hypothesis. Fails if "
        "the left-hand side does not literally occur."
    ),
    "simp": (
        "Simplifies the goal with the simp lemma set. `simp [f]` also "
        "unfolds f; `simp at h` simplifies a hypothesis. Powerful but "
        "opaque — prefer targeted lemmas when the goal is delicate."
    ),
    "norm_num": (
        "Decides numeric facts: arithmetic on literals, inequalities, "
        "divisibility. Closes goals like `(2 : ℝ) + 2 = 4` or `0 < 5`."
    ),
    "ring_nf": (
        "Normalizes both sides of a commutative-(semi)ring equation. Use "
        "for polynomial identities like `(a+b)^2 = a^2 + 2*a*b + b^2`."
    ),
    "linarith": (
        "Closes linear arithmetic goals that follow from linear hypotheses "
        "in the context. Feed it extra facts: `linarith [sq_nonneg a]`."
    ),
    "nlinarith": (
        "linarith with some nonlinear preprocessing (products, squares). "
        "Try when linarith fails on a goal with products of hypotheses."
    ),
    "omega": (
        "Decision procedure for linear integer/natural arithmetic — "
        "equalities, inequalities, divisibility by literals, `%` and `/`."
    ),
    "decide": (
        "Evaluates a decidable proposition to close the goal. Works for "
        "concrete finite checks; fails on quantifiers over infinite types."
    ),
    "constructor": (
        "Splits a goal built from a structure/inductive with one obvious "
        "constructor: `∧` becomes two goals, `↔` becomes two implications, "
        "`∃` asks for a witness then the property."
    ),
    "use": (
        "Provides a witness for an ∃-goal: `use 37` turns `∃ n, P n` into "
        "`P 37`. Chain witnesses with `use a, b`."
    ),
    "obtain": (
        "Destructures a hypothesis: `obtain ⟨n, hn⟩ := h` unpacks "
        "h : ∃ n, P n. Also splits ∧ and case-splits ∨ "
        "(`obtain h1 | h2 := h`)."
    ),
    "rcases": (
        "Recursive destructuring with one pattern: "
        "`rcases h with ⟨a, ha | hb⟩` unpacks nested ∃/∧/∨ in one step."
    ),
    "cases": (
        "Case analysis on an inductive value or hypothesis: "
        "`cases n with | zero => … | succ k => …`."
    ),
    "induction": (
        "Induction on a natural number or inductive value: "
        "`induction n with | zero => … | succ k ih => …` gives you the "
        "induction hypothesis `ih` in the successor case."
    ),
    "have": (
        "States and proves an intermediate fact: `have h2 : 0 < n := by "
        "omega` adds h2 to the context. The workhorse for structuring a "
        "proof the way you would on paper."
    ),
    "calc": (
        "Chains equalities/inequalities step by step:\n"
        "`calc a = b := by … \n  _ ≤ c := by …`. Each step names its own "
        "justification."
    ),
    "specialize": (
        "Instantiates a universally quantified hypothesis: "
        "`specialize h 5` turns h : ∀ n, P n into h : P 5."
    ),
    "change": (
        "Replaces the goal (or a hypothesis with `change … at h`) by "
        "something definitionally equal to it — useful to unfold a "
        "definition into its ε-N form before working on it."
    ),
    "rfl": (
        "Closes a goal that is true by definitional unfolding/reflexivity, "
        "like `a = a` or `2 + 2 = 4` for naturals."
    ),
    "push_neg": (
        "Pushes a negation inward through quantifiers and connectives: "
        "`¬ ∀ x, P x` becomes `∃ x, ¬ P x`. Also works `at h`."
    ),
    "by_contra": (
        "Starts a proof by contradiction: `by_contra h` assumes the "
        "negation of the goal as h and asks you to derive False."
    ),
    "field_simp": (
        "Clears denominators in field expressions, given the needed "
        "nonzero-ness facts are in context."
    ),
    "positivity": (
        "Closes goals of the form `0 < e` or `0 ≤ e` by structural "
        "analysis of the expression e."
    ),
    "gcongr": (
        "Generalized congruence for inequalities: reduces `f a ≤ f b` "
        "to `a ≤ b` for monotone contexts."
    ),
    "aesop": (
        "General-purpose finishing search combining intro/simp/safe rules. "
        "Good last resort for routine logic goals."
    ),
    "skip": "Does nothing. Useful only as a placeholder.",
}

# Baseline tactics every exam offers even if the GT proof did not use them.
CORE_TACTICS = ("intro", "exact", "apply", "have", "constructor", "rfl")

_LEAN_KEYWORDS = {
    "by", "at", "with", "fun", "then", "else", "if", "do", "let", "in",
    "match", "this", "show", "from", "to", "using", "theorem", "lemma",
    "sorry", "admit", "import", "set_option", "open", "end", "namespace",
    "section", "variable", "true", "false", "True", "False", "Prop",
    "Type", "Sort", "case", "next", "all_goals", "any_goals", "first",
    "try", "repeat", "exact?", "apply?",
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
# `#check @foo` prints `foo : sig` when all args are explicit but keeps the
# `@` prefix (`@foo : sig`) when the lemma has implicit binders.
_CHECK_LINE_RE = re.compile(r"^@?(?P<name>[A-Za-z_][A-Za-z0-9_.']*)\s+:\s+(?P<sig>.+)$")
_DIAG_LINE_RE = re.compile(r"^[^:\n]+:\d+:\d+:\s*(?:error|warning|info)")


def candidate_theorem_names(proof_body: str) -> List[str]:
    """Identifier tokens in a proof body that look like library lemmas."""
    seen: List[str] = []
    for token in _IDENT_RE.findall(proof_body or ""):
        base = token.split(".")[0]
        if token in _LEAN_KEYWORDS or base in _LEAN_KEYWORDS:
            continue
        if token in TACTIC_DOCS:
            continue
        # Library lemmas are dotted (Nat.add_zero), snake_case (mul_comm),
        # or capitalized (IsOpen); bare short locals (n, hn, ih) are not.
        if "." in token or "_" in token or (token[0].isupper() and len(token) > 2):
            if token not in seen:
                seen.append(token)
    return seen


def build_check_probe(lean_header: str, names: List[str]) -> str:
    header = (lean_header or "import Mathlib").rstrip()
    lines = [header, "", *(f"#check @{name}" for name in names)]
    return "\n".join(lines) + "\n"


def parse_check_probe_output(raw_output: str, names: List[str]) -> Dict[str, str]:
    """Map validated names to their signatures from the probe output.

    ``#check @Foo.bar`` prints ``Foo.bar : <signature>`` on plain lines
    (possibly wrapped); unknown names produce error diagnostics and are
    simply absent from the result.
    """
    wanted = set(names)
    signatures: Dict[str, str] = {}
    current: Optional[str] = None
    for line in (raw_output or "").splitlines():
        if _DIAG_LINE_RE.match(line):
            current = None
            continue
        match = _CHECK_LINE_RE.match(line.strip())
        if match and match.group("name") in wanted:
            current = match.group("name")
            signatures[current] = match.group("sig").strip()
        elif current is not None and line.startswith(" "):
            signatures[current] += " " + line.strip()
        else:
            current = None
    return signatures


def _default_check_runner(repo_root: Path) -> Callable[[str], Awaitable[str]]:
    async def _run(code: str) -> str:
        def _blocking() -> str:
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", suffix=".lean", delete=False
            ) as handle:
                handle.write(code)
                path = handle.name
            proc = subprocess.run(
                ["lake", "env", "lean", path],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return proc.stdout + "\n" + proc.stderr

        return await asyncio.to_thread(_blocking)

    return _run


def tactics_in_proof(proof_body: str) -> List[str]:
    found: List[str] = []
    for line in (proof_body or "").splitlines():
        stripped = line.strip().lstrip("·<;> ").strip()
        head = stripped.split(" ", 1)[0] if stripped else ""
        if head in TACTIC_DOCS and head not in found:
            found.append(head)
    return found


async def build_palette(
    *,
    lean_code: str,
    formal_statement: str,
    lean_header: str,
    check_runner: Optional[Callable[[str], Awaitable[str]]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Palette for one exam row, derived from its certified proof.

    Returns ``{"tactics": {name: doc}, "theorems": {name: signature}}``.
    Only names that resolve against the pinned Mathlib survive.
    """
    proof_body = lean_code or ""
    statement_match = re.search(r":=", proof_body)
    if statement_match:
        proof_body = proof_body[statement_match.end():]
    # Exclude names bound by the statement itself (theorem name, binders).
    statement_names = set(_IDENT_RE.findall(formal_statement or ""))
    candidates = [
        name
        for name in candidate_theorem_names(proof_body)
        if name not in statement_names
    ]
    signatures: Dict[str, str] = {}
    if candidates:
        runner = check_runner or _default_check_runner(repo_root or Path.cwd())
        raw = await runner(build_check_probe(lean_header, candidates))
        signatures = parse_check_probe_output(raw, candidates)
    tactic_names = list(dict.fromkeys([*tactics_in_proof(proof_body), *CORE_TACTICS]))
    return {
        "tactics": {name: TACTIC_DOCS[name] for name in tactic_names},
        "theorems": signatures,
    }
