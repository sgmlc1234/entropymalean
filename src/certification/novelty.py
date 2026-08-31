"""Decide whether a child is a new problem or a redecoration of its parent.

The Lean gate answers "does this hold?" and the anti-stub guard answers "is this
a placeholder?". Neither answers "is this worth having?", and a generator asked
for harder problems will happily satisfy both by leaving the mathematics alone
and enlarging the notation. Two failures showed up in the released corpus:

  decorative mutation
      `x * y` irrational became `x * (q^2 + r^2 + 1)` irrational, and then
      `(x + t) * ((a + b*c)^2 + (b - c)^2 + 1)`. The coefficient grows, the
      nonzero-ness stays obvious, and the proof does not change at all.

  parallel crossover
      A child proved one parent's obligation, proved the other's, and joined
      them at the last line. Both parents are present and neither informs the
      other; the prompt's own taxonomy calls this the *easy* pattern, and it
      was being produced for slots asked to be hard.

Both are invisible to a type-checker because both are true theorems. What
separates them from real variation is structural, so the tests here are
structural too:

  * a mutation must change the *proof skeleton*, not just the statement text;
  * a crossover must let one parent's result enter the other's derivation,
    which shows up as a meeting point below the root of the proof's dependency
    graph.

Neither test costs a Lean call. Both produce a human-readable reason, because a
rejection that cannot be explained cannot be repaired on the retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Tactics that close or normalise a goal without deciding anything about its
#: structure. Decoration is what makes these appear: enlarging a coefficient
#: forces one extra step to show it is nonzero, and that step is always one of
#: these. Measured against the released corpus, an equality test on tactic sets
#: caught nothing at all (0 of 20 mutations) precisely because the decoration
#: *adds* a tactic rather than reusing the parent's exactly.
CLOSING_TACTICS = frozenset({
    "positivity", "norm_num", "norm_cast", "push_cast", "linarith", "nlinarith",
    "simp", "simpa", "simp_all", "ring", "ring_nf", "field_simp", "decide",
    "trivial", "rfl", "omega", "aesop", "assumption", "exact", "linear_combination",
})

#: Tactics that introduce a case, a witness, or an induction — the moves a
#: genuinely different problem needs and a redecorated one does not.
STRUCTURAL_TACTICS = frozenset({
    "rcases", "obtain", "cases", "rintro", "induction", "by_contra", "by_cases",
    "interval_cases", "constructor", "refine", "use", "calc", "subst", "revert",
    "specialize", "contrapose", "wlog", "conv", "match",
})

#: Slot difficulties that are allowed to produce a parallel (depth-0) crossover.
#: The generator's own catalog maps `shared_parameter_binding` to *easy* and
#: describes it as loose coupling, so forbidding depth 0 outright would forbid a
#: pattern the pipeline is meant to produce. What is not acceptable is a slot
#: asked for a coupled system returning the easy pattern anyway.
LOOSE_COUPLING_OK = {"easy", "", None}

_COMMENT_LINE = re.compile(r"--.*?$", re.M)
_COMMENT_BLOCK = re.compile(r"/-.*?-/", re.S)
#: `have h : T := by …` — the unit the dependency graph is built from.
_HAVE = re.compile(r"^(\s*)have\s+([A-Za-z_][A-Za-z0-9_']*)\s*:\s*(.+?):=", re.M | re.S)
#: Names introduced by destructuring, which behave as graph nodes without types.
_BINDS = re.compile(
    r"^\s*(?:obtain|rcases|set|let)\b[^\n]*?([A-Za-z_][A-Za-z0-9_']*)\s*(?::|←|:=|with)",
    re.M,
)
_IDENT = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\.[A-Za-z_][A-Za-z0-9_']*)+)\b")


def strip_comments(code: str) -> str:
    return _COMMENT_BLOCK.sub("", _COMMENT_LINE.sub("", str(code or "")))


def proof_body(lean_code: str) -> str:
    """The tactic block, with the header and statement discarded."""
    code = strip_comments(lean_code)
    return code.split(":= by", 1)[1] if ":= by" in code else code


def tactic_skeleton(lean_code: str, vocabulary: Sequence[str]) -> List[str]:
    """The ordered set of distinct tactics a proof uses.

    Deliberately a *set* rather than a sequence: reordering two independent
    tactics is not a new proof, so counting order would call it one. Matching
    the enrichment script's extractor keeps the seed's recorded `gt_tactics`
    comparable with a freshly generated child's.
    """
    known = set(vocabulary)
    out: List[str] = []
    for line in (l.strip() for l in proof_body(lean_code).splitlines()):
        if not line:
            continue
        head = line.lstrip("·<;>| ").split(" ", 1)[0].strip()
        if head in known and head not in out:
            out.append(head)
    return out


@dataclass
class Verdict:
    """Why a child was accepted or rejected, in terms a retry can act on."""

    ok: bool
    kind: str = ""                       # "" | "decorative" | "parallel_crossover"
    reason: str = ""                     # one sentence, shown to a person
    detail: Dict[str, Any] = field(default_factory=dict)

    def brief(self) -> str:
        """The paragraph handed back to the generator on its one retry.

        A bare code ("decorative_mutation") tells the model it failed without
        telling it what to change, and the second attempt then differs from the
        first by luck. So the brief states the finding, the evidence, and the
        specific moves that will not work a second time.
        """
        if self.ok:
            return ""
        if self.kind == "decorative":
            shared = ", ".join(self.detail.get("shared_tactics") or []) or "the same tactics"
            return (
                f"REJECTED: this child is a redecoration of its parent, not a new "
                f"problem. Its proof uses {shared} — exactly the parent's skeleton — "
                f"so the added notation carries no mathematical content.\n"
                f"Do NOT retry by: enlarging or renaming coefficients; adding a factor "
                f"whose nonzero-ness is immediate; wrapping the target in a definition "
                f"that unfolds in one step; adding a quantifier over a variable the "
                f"conclusion ignores.\n"
                f"Instead change what must be *proved*: alter the conclusion's shape "
                f"(a divisibility into a congruence, an existence into a bound), "
                f"strengthen the modulus or the exponent so a new case split is "
                f"required, or generalise a constant into a second variable so the "
                f"parent's argument no longer closes the goal."
            )
        if self.kind == "parallel_crossover":
            a = self.detail.get("parent_a_contribution") or "parent A's obligation"
            b = self.detail.get("parent_b_contribution") or "parent B's obligation"
            return (
                f"REJECTED: the two parents never meet. The proof establishes {a} and "
                f"{b} independently and combines them only at the final step, which is "
                f"the *easy* pattern (shared_parameter_binding) for a slot asked to be "
                f"{self.detail.get('difficulty') or 'harder'}.\n"
                f"Do NOT retry by: joining the two conclusions with a conjunction; "
                f"using one parent's result solely to show a coefficient is nonzero or "
                f"a set is nonempty; selecting between constants with a condition that "
                f"is already implied by the hypotheses.\n"
                f"Instead make one parent's result an *input* to the other's "
                f"derivation: let its conclusion supply a bound, a modulus, an index, "
                f"or a case distinction that the second parent's argument then consumes."
            )
        return f"REJECTED: {self.reason}"


def judge_mutation(
    child_lean: str,
    parent_tactics: Sequence[str],
    vocabulary: Sequence[str],
    *,
    child_tactics: Optional[Sequence[str]] = None,
    parent_lean: str = "",
) -> Verdict:
    """A mutation must move the proof, not only the prose.

    The test is one-sided. A different skeleton is not proof that the child is
    interesting, but an identical skeleton is good evidence that it is not: the
    same tactics in the same roles means Lean closed the child exactly the way
    it closed the parent.
    """
    child = set(child_tactics if child_tactics is not None else tactic_skeleton(child_lean, vocabulary))
    parent = set(parent_tactics or [])
    if not child or not parent:
        # Nothing to compare. Silence is not evidence, so this passes rather
        # than rejecting a row for a missing field.
        return Verdict(True, detail={"comparable": False})

    added = child - parent
    dropped = parent - child
    detail = {
        "comparable": True,
        "added_tactics": sorted(added),
        "dropped_tactics": sorted(dropped),
        "parent_tactics": sorted(parent),
    }

    # The spine test, when both proofs have readable structure. Decoration
    # attaches a leaf — one `have` establishing that the new coefficient is
    # nonzero, that the new index is in range — and leaves the chain that
    # actually reaches the goal untouched. A real variation changes the chain.
    if parent_lean:
        parent_close = closing_lemma(parent_lean)
        child_close = closing_lemma(child_lean)
        detail["parent_closing_lemma"] = parent_close
        detail["child_closing_lemma"] = child_close
        if parent_close and parent_close == child_close:
            return Verdict(
                False,
                kind="decorative",
                reason=(
                    f"the child's proof ends with the parent's final move "
                    f"(`{parent_close}`); everything it adds only discharges the "
                    f"larger expression"
                ),
                detail={**detail, "shared_tactics": sorted(parent & child)},
            )
        spine = _spine_change(parent_lean, child_lean)
        detail.update(spine)
        if spine.get("comparable_spines") and not spine.get("spine_changed"):
            return Verdict(
                False,
                kind="decorative",
                reason=(
                    "the child's proof reaches its goal along the parent's chain; "
                    "what it adds hangs off to the side"
                ),
                detail={**detail, "shared_tactics": sorted(parent & child)},
            )

    # Fallback when the graphs cannot be compared: the parent's whole tactic
    # skeleton survives and everything new is a closing step.
    keeps_skeleton = not dropped
    only_closers = bool(added) and added <= CLOSING_TACTICS
    if keeps_skeleton and (not added or only_closers):
        return Verdict(
            False,
            kind="decorative",
            reason=(
                "the child keeps the parent's entire proof skeleton and adds only "
                "closing steps"
            ),
            detail={**detail, "shared_tactics": sorted(parent)},
        )
    if added & STRUCTURAL_TACTICS:
        detail["new_structure"] = sorted(added & STRUCTURAL_TACTICS)
    return Verdict(True, detail=detail)


_CLOSING_LEMMA = re.compile(r"\bexact\s+[A-Za-z_][A-Za-z0-9_']*\.([A-Za-z_][A-Za-z0-9_']*)")


def closing_lemma(lean_code: str) -> str:
    """The named lemma the proof's final move applies, if it has one.

    This is the sharpest available signal for decoration. A redecorated child
    grows its statement, so it needs extra steps to discharge the bigger
    expression — but those steps all serve the same role the parent's single
    side condition served, and the proof still ends by applying the same lemma.
    Three descendants of `Irrational (x * y)` in the released corpus close with
    `mul_ratCast`, exactly as their parent does, however much they added above.

    Returns "" when the closing move is a bare tactic (`linarith`, `omega`),
    because those say nothing: two unrelated theorems can both end that way.
    """
    _, closing = proof_graph(lean_code)
    hits = _CLOSING_LEMMA.findall(closing)
    return hits[-1] if hits else ""


def _spine(lean_code: str) -> List[str]:
    """The chain of `have` types that actually feeds the closing term.

    A proof is a DAG, but only part of it carries the argument: nodes the final
    term depends on, directly or transitively. The rest are side conditions.
    Comparing spines rather than whole graphs is what separates "proved a
    different way" from "same proof, one more obligation discharged".
    """
    nodes, closing = proof_graph(lean_code)
    if not nodes:
        return []
    by_name = {node["name"]: node for node in nodes}
    reached: List[str] = []
    seen: Set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name not in by_name:
            return
        seen.add(name)
        for source in by_name[name].get("uses") or []:
            walk(source)
        reached.append(_normalise(by_name[name]["type"]))

    for name in by_name:
        if re.search(r"\b" + re.escape(name) + r"\b", closing):
            walk(name)
    return reached


def _spine_change(parent_lean: str, child_lean: str) -> Dict[str, Any]:
    """Whether the child's load-bearing chain differs from the parent's."""
    parent_spine = _spine(parent_lean)
    child_spine = _spine(child_lean)
    if not parent_spine or not child_spine:
        return {"comparable_spines": False}
    kept = [step for step in child_spine if step in set(parent_spine)]
    return {
        "comparable_spines": True,
        "parent_spine_size": len(parent_spine),
        "child_spine_size": len(child_spine),
        # The child's chain is new if it drops or replaces any of the parent's
        # steps, or introduces one of its own into the load-bearing path.
        "spine_changed": (
            len(kept) < len(parent_spine) or len(child_spine) > len(kept)
        ),
    }


def proof_graph(lean_code: str) -> Tuple[List[Dict[str, Any]], str]:
    """The proof's `have` blocks and the edges between them.

    A node is one `have`; an edge runs from a node to any later node whose
    tactic text mentions it by name. What is left after the last `have` is the
    closing term, which is the graph's root — every chain ends there.
    """
    body = proof_body(lean_code)
    hits = [
        (m.group(2), " ".join(m.group(3).split()), m.start(), m.end())
        for m in _HAVE.finditer(body)
    ]
    nodes: List[Dict[str, Any]] = []
    for index, (name, kind, start, end) in enumerate(hits):
        stop = hits[index + 1][2] if index + 1 < len(hits) else len(body)
        nodes.append({"name": name, "type": kind, "text": body[end:stop]})
    for node in nodes:
        node["uses"] = [
            other["name"]
            for other in nodes
            if other["name"] != node["name"]
            and re.search(r"\b" + re.escape(other["name"]) + r"\b", node["text"])
        ]
    closing = body[hits[-1][2]:] if hits else body
    return nodes, closing


def _normalise(text: str) -> str:
    """Collapse whitespace and drop binders so two spellings of one claim match."""
    out = re.sub(r"\s+", "", str(text or ""))
    out = re.sub(r"[{(\[]\s*[a-zA-Z_][a-zA-Z0-9_']*\s*:\s*[^})\]]*[})\]]", "", out)
    return out


def _conclusion_of(lean_code: str) -> str:
    """The part of a theorem statement after the last top-level colon."""
    code = strip_comments(lean_code)
    head = code.split(":= by", 1)[0]
    match = re.search(r"theorem\s+\S+(.*)$", head, re.S)
    body = match.group(1) if match else head
    return _normalise(body.rsplit(":", 1)[-1] if ":" in body else body)


def _attribute(node: Dict[str, Any], parents: Sequence[Dict[str, str]]) -> Optional[str]:
    """Which parent a node belongs to, or None when the evidence is ambiguous.

    Conclusion match is trusted first: a `have` whose type is the parent's
    theorem *is* that parent's obligation, discharged. Lemma provenance is the
    weaker fallback and is only used when exactly one parent cites the name,
    because two parents drawing on the same Mathlib lemma says nothing.
    """
    node_type = _normalise(node.get("type"))
    for parent in parents:
        if node_type and node_type == _conclusion_of(parent.get("lean_code", "")):
            return parent["key"]
    cites = set(_IDENT.findall(node.get("text") or ""))
    owners = {
        parent["key"]
        for parent in parents
        if cites & set(_IDENT.findall(strip_comments(parent.get("lean_code", ""))))
    }
    return next(iter(owners)) if len(owners) == 1 else None


def coupling_depth(
    child_lean: str, parents: Sequence[Dict[str, str]]
) -> Tuple[Optional[int], Dict[str, Any]]:
    """How far below the closing term the two parents' lineages first meet.

    0 means they meet only at the root: each parent was discharged on its own
    and the results were combined once, at the end. 1 or more means some `have`
    consumed material from both, which is what coupling looks like.

    Returns ``(None, …)`` when the proof has no internal structure to read. The
    caller treats that as 0 — a crossover closed in one line did not couple
    anything — but the distinction is kept in the detail so the two can be
    counted separately.
    """
    nodes, _ = proof_graph(child_lean)
    if len(nodes) < 2:
        return None, {"measurable": False, "nodes": len(nodes)}

    owner = {node["name"]: _attribute(node, parents) for node in nodes}
    by_name = {node["name"]: node for node in nodes}

    def lineage(name: str, seen: Optional[Set[str]] = None) -> Set[str]:
        """Every parent whose material reaches `name`, directly or through inputs."""
        seen = seen or set()
        if name in seen:
            return set()
        seen.add(name)
        marks = {owner.get(name)} - {None}
        for source in by_name[name].get("uses") or []:
            marks |= lineage(source, seen)
        return marks

    attribution = {k: v for k, v in owner.items() if v}
    # Nothing traced to either parent. A depth computed over an empty attribution
    # is not a low score, it is no score: every node's lineage is empty, so no
    # node can carry two parents and the answer is 0 whatever the proof does.
    # Reported as unmeasurable so a caller cannot read "the parents never met"
    # off a run in which the parents were never identified. This state was known
    # -- the flag was demoted from deciding in August 2026 precisely because all
    # nine of its verdicts had it -- but it was recorded only in the detail, and
    # the verdict beside it still said 0.
    if not attribution:
        return None, {
            "measurable": False,
            "nodes": len(nodes),
            "attribution": {},
            "why": "no proof node could be traced to either parent",
        }

    depth: Optional[int] = 0
    meeting: Optional[Dict[str, Any]] = None
    for node in nodes:
        if len(lineage(node["name"])) >= 2:
            # Depth is measured from the root, and the root is the closing term;
            # any `have` that already carries both parents sits below it.
            depth = 1
            meeting = {"at": node["name"], "type": node["type"]}
            break
    detail = {
        "measurable": True,
        "nodes": len(nodes),
        "attribution": attribution,
        "meeting_point": meeting,
    }
    return depth, detail


def judge_crossover(
    child_lean: str,
    parents: Sequence[Dict[str, str]],
    *,
    difficulty: str = "",
) -> Verdict:
    """A crossover has to let the parents interact, unless the slot asked easy."""
    depth, detail = coupling_depth(child_lean, parents)
    detail["difficulty"] = difficulty
    # Unmeasurable is not a failing score. A one-line proof and a proof whose
    # nodes could not be traced to a parent both arrive here as `None`, and
    # convicting on either is convicting for the absence of evidence -- which is
    # what a one-sided probe must never do.
    if depth is None:
        detail["coupling_depth"] = None
        return Verdict(True, detail=detail)
    detail["coupling_depth"] = depth
    if depth >= 1 or difficulty in LOOSE_COUPLING_OK:
        return Verdict(True, detail=detail)
    marks = detail.get("attribution") or {}
    contributions = {}
    for name, key in marks.items():
        contributions.setdefault(key, name)
    keys = sorted(contributions)
    detail["parent_a_contribution"] = contributions.get(keys[0]) if keys else None
    detail["parent_b_contribution"] = contributions.get(keys[1]) if len(keys) > 1 else None
    return Verdict(
        False,
        kind="parallel_crossover",
        reason=(
            "the parents are discharged independently and combined only at the "
            "closing step"
        ),
        detail=detail,
    )

#: A hypothesis that bounds a variable by a literal. Generators reach for these
#: when a statement has grown past what they can prove in general: capping `k`
#: at 100 turns an open claim into a finite check that `decide` can close. The
#: mathematics is then about the first hundred cases rather than about the
#: theorem, and the child looks harder than the parent while being easier.
_LITERAL_BOUND = re.compile(
    r"[({]\s*[^:(){}]*:\s*[^(){}]*?\b[a-zA-Z_][a-zA-Z0-9_']*\s*(?:≤|<|≥|>)\s*(\d{2,})\s*[)}]"
)


def literal_bounds(formal_statement: str) -> List[int]:
    """Numeric ceilings imposed on variables in the hypotheses."""
    head = strip_comments(formal_statement).split(":= by", 1)[0]
    return sorted({int(m) for m in _LITERAL_BOUND.findall(head)})


def introduced_bounds(parent_statement: str, child_statement: str) -> List[int]:
    """Ceilings the child added that its parent did not have.

    A bound the parent already carried is part of the problem; one that appears
    only in the child is a change of what is being claimed, and the direction of
    that change is always toward the tractable.
    """
    parent = set(literal_bounds(parent_statement))
    return [b for b in literal_bounds(child_statement) if b not in parent]

