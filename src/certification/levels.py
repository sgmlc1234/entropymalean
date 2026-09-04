"""Certificate levels and alignment evidence: named, not lettered.

Naming policy (decided 2026-07-28). A numbered ladder was rejected because an
invented "L1/L2/L3" ladder carries no meaning outside our paper ("people are
familiar with Lean type checking, you can just say that"). So:

  * every certificate level is named after the standard artifact it requires,
    never a letter-number code;
  * the semantic axis reuses the faithfulness vocabulary of An et al.,
    "Ground False" (AI4Math @ ICML 2026), so our rows compose with the
    emerging community taxonomy instead of running parallel to it;
  * the legacy integer ``lean_level`` is frozen for backward compatibility.

Certificate levels (syntactic axis, monotone):

  ``statement_checked``
      The statement alone — proof body replaced by ``sorry``, elaborated with
      ``set_option autoImplicit false`` against the pinned Mathlib — is
      accepted. Guarantee: the statement is well-formed and every identifier
      resolves; nothing is silently auto-bound.

  ``proof_checked``
      Additionally, a complete ``sorry``-free proof is accepted AND the
      declaration's axiom closure (``#print axioms``) lies within
      ``PERMITTED_AXIOMS``. The axiom audit is part of the level, not an
      afterthought: an elaborator-accepted proof may still lean on a smuggled
      ``axiom``, on ``sorryAx``, or on ``Lean.ofReduceBool`` (``native_decide``).
      The same no-``sorry``/no-new-axiom constraint is what An et al. impose
      on their certifying prover.

  ``kernel_replayed``
      Additionally, ``leanprover/comparator`` accepts the proof on a hardened
      runner: the proved statement is *exactly* the trusted statement, the
      axioms are re-audited, and the term is replayed through the kernel via
      ``lean4export`` in a sandbox. This buys independence from Lean's
      elaborator — not from our environment, which is still the only one the
      proof has been seen in.

  ``reproducible``
      Additionally, that verdict was reached again on a second platform from a
      fully recorded environment: every package revision in ``lake-manifest``
      (not just Mathlib's), the toolchain, and the exported term, which a third
      party can replay with any kernel without running our elaborator at all.
      This is the level that answers "does it hold for anyone who runs it",
      and it is the one a released benchmark most often skips — we measured a
      third of ProofNet-Verified failing to compile under our pin despite
      shipping as verified.

Semantic faithfulness is deliberately NOT a certificate level — type checking
cannot certify that a statement means what the prose says. It is recorded on
an orthogonal axis as ``faithfulness`` (verdict) plus ``alignment_method``
(how the verdict was obtained).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- certificate levels (syntactic axis) -----------------------------------

LEVEL_NONE = "none"
LEVEL_STATEMENT = "statement_checked"
LEVEL_PROOF = "proof_checked"
LEVEL_KERNEL = "kernel_replayed"
LEVEL_REPRODUCIBLE = "reproducible"

LEVEL_ORDER = {
    LEVEL_NONE: 0,
    LEVEL_STATEMENT: 1,
    LEVEL_PROOF: 2,
    LEVEL_KERNEL: 3,
    LEVEL_REPRODUCIBLE: 4,
}

# The ladder measures two different things and the names now say which.
#
# Levels 1-3 increase *checker independence*: the elaborator accepted it, then
# its axiom closure was inspected, then a kernel that Lean's elaborator did not
# write replayed the exported term. All three still run in one environment —
# ours — so passing them says nothing about what anyone else would see.
#
# Level 4 is the other axis: the same verdict, reached again from a fully
# recorded environment on a second platform. It exists because we measured what
# its absence costs — a third of ProofNet-Verified, released as "verified", does
# not compile under our pin. Their checks were not wrong; they were local, and
# nothing in the artifact said what "local" meant.
#
# `reproducible` deliberately does not claim the proof holds under *any*
# Mathlib. No Lean artifact has that property, and Mathlib's own CI does not
# claim it either. It claims something checkable instead: take the pins we
# publish, and you get the result we publish.

#: Axioms Mathlib itself relies on; anything else fails the audit.
PERMITTED_AXIOMS = ("propext", "Quot.sound", "Classical.choice")

CERTIFICATE_SCHEMA_VERSION = "eml-certificate-v2"

# --- faithfulness (semantic axis) ------------------------------------------
# Verdict vocabulary from An et al., "Ground False" (AI4Math @ ICML 2026).

FAITHFUL = "faithful"
STRONGER = "stronger"
WEAKER = "weaker"
INCOMPARABLE = "incomparable"
NL_AMBIGUOUS = "nl_ambiguous"
FAITHFULNESS_UNAUDITED = "unaudited"

FAITHFULNESS_VERDICTS = (
    FAITHFUL,
    STRONGER,
    WEAKER,
    INCOMPARABLE,
    NL_AMBIGUOUS,
    FAITHFULNESS_UNAUDITED,
)

#: How a faithfulness verdict was established, weakest evidence first.
ALIGNMENT_METHODS = (
    "none",
    "llm_judge",          # single judge over the surface statement
    "goal_roundtrip",     # elaborated goal -> informalize -> independent judge
    "human_review",
    "formal_refutation",  # machine-checked proof of the negation
)


def derive_certificate_level(
    *,
    statement_checked: bool,
    proof_checked: bool,
    kernel_replayed: bool = False,
    reproducible: bool = False,
) -> str:
    """Highest level supported by the recorded gate outcomes (monotone).

    Monotone in the literal sense: a row cannot reach a level without every
    weaker one holding. `reproducible` therefore requires the kernel replay, not
    merely a second successful run — reproducing a weaker check more widely does
    not make it a stronger check.
    """
    if reproducible and kernel_replayed:
        return LEVEL_REPRODUCIBLE
    if kernel_replayed:
        return LEVEL_KERNEL
    if proof_checked:
        return LEVEL_PROOF
    if statement_checked:
        return LEVEL_STATEMENT
    return LEVEL_NONE


def level_at_least(level: str, minimum: str) -> bool:
    return LEVEL_ORDER.get(str(level), 0) >= LEVEL_ORDER.get(str(minimum), 0)


def axiom_audit(axiom_closure: Optional[List[str]]) -> Dict[str, Any]:
    """Classify a declaration's axiom closure against the allowlist.

    ``axiom_closure`` is the list parsed from ``#print axioms <decl>``;
    ``None`` means the audit did not run (recorded as such, never as a pass).
    """
    if axiom_closure is None:
        return {"ran": False, "passed": None, "axioms": None, "disallowed": None}
    disallowed = [
        name for name in axiom_closure if name not in PERMITTED_AXIOMS
    ]
    return {
        "ran": True,
        "passed": not disallowed,
        "axioms": list(axiom_closure),
        "disallowed": disallowed,
    }


@lru_cache(maxsize=8)
def runtime_pins(repo_root: str) -> Dict[str, Optional[Any]]:
    """The full environment a certificate is relative to, not a summary of it.

    Recording Mathlib's revision alone under-describes the pin by nine
    packages: a proof breaks on a `batteries` or `aesop` bump exactly as
    readily, and a consumer who matched the one revision we printed would still
    not have our environment. The digest is over every package revision plus
    the toolchain, so a mismatch is detectable without comparing ten fields by
    hand.
    """
    root = Path(repo_root)
    toolchain: Optional[str] = None
    toolchain_path = root / "lean-toolchain"
    if toolchain_path.is_file():
        toolchain = toolchain_path.read_text().strip() or None

    revisions: Dict[str, str] = {}
    manifest_path = root / "lake-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            for package in manifest.get("packages") or []:
                name, rev = package.get("name"), package.get("rev")
                if name and rev:
                    revisions[str(name)] = str(rev)
        except (OSError, json.JSONDecodeError):
            revisions = {}

    payload = json.dumps(
        {"toolchain": toolchain, "packages": revisions},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "lean_toolchain": toolchain,
        "mathlib_revision": revisions.get("mathlib"),
        "package_revisions": revisions,
        "manifest_digest": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }

def build_certificate_record(
    *,
    statement_checked: bool,
    proof_checked: bool,
    kernel_replayed: bool = False,
    reproducible: bool = False,
    axiom_closure: Optional[List[str]] = None,
    auto_implicit_false: bool = True,
    proof_method: Optional[str] = None,
    faithfulness: str = FAITHFULNESS_UNAUDITED,
    alignment_method: str = "none",
    verifier: Optional[str] = None,
    repo_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-readable certificate carried on every generated row.

    ``proof_checked`` is only honoured when the axiom audit ran and passed —
    an elaborator-accepted proof with an unaudited or violating axiom closure
    stops at ``statement_checked``.
    """
    audit = axiom_audit(axiom_closure)
    proof_level_ok = bool(proof_checked) and audit["passed"] is True
    level = derive_certificate_level(
        statement_checked=statement_checked or proof_level_ok,
        proof_checked=proof_level_ok,
        kernel_replayed=kernel_replayed and proof_level_ok,
        reproducible=reproducible and proof_level_ok,
    )
    pins = runtime_pins(str(repo_root or Path.cwd()))
    record: Dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "level": level,
        "statement_checked": bool(statement_checked or proof_level_ok),
        "proof_accepted": bool(proof_checked),
        "axiom_audit": audit,
        "kernel_replayed": bool(kernel_replayed and proof_level_ok),
        "reproducible": bool(reproducible and kernel_replayed and proof_level_ok),
        "statement_check_auto_implicit_false": bool(auto_implicit_false),
        "proof_method": proof_method,
        "faithfulness": faithfulness,
        "alignment_method": alignment_method,
        "verifier": verifier,
        "lean_toolchain": pins["lean_toolchain"],
        "mathlib_revision": pins["mathlib_revision"],
        # Every package, not just Mathlib: a proof can break on a batteries or
        # aesop bump exactly as easily, and recording one revision out of ten
        # makes the pin look tighter than it is.
        "package_revisions": pins["package_revisions"],
        "manifest_digest": pins["manifest_digest"],
        "platforms_verified": [],
    }
    if extra:
        record.update(extra)
    return record


def upgrade_to_kernel_replayed(
    certificate: Dict[str, Any],
    *,
    comparator_revision: str,
    permitted_axioms: Optional[List[str]] = None,
    runner: str,
) -> Dict[str, Any]:
    """Upgrade a ``proof_checked`` certificate after a comparator run."""
    upgraded = dict(certificate)
    if upgraded.get("level") != LEVEL_PROOF:
        raise ValueError(
            "kernel replay requires a proof_checked certificate, got "
            f"{upgraded.get('level')!r}"
        )
    upgraded["kernel_replayed"] = True
    upgraded["level"] = LEVEL_KERNEL
    upgraded["comparator_revision"] = comparator_revision
    upgraded["permitted_axioms"] = list(permitted_axioms or PERMITTED_AXIOMS)
    upgraded["replay_runner"] = runner
    return upgraded


def upgrade_to_reproducible(
    certificate: Dict[str, Any],
    *,
    platforms: List[str],
    export_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Upgrade a ``kernel_replayed`` certificate after a second platform agrees.

    Two platforms is the minimum that can distinguish "this proof holds" from
    "this machine says so", so one is refused rather than accepted with a
    caveat — a claim about reproducibility that never left one host is the
    claim we found broken in the benchmark we inherited.
    """
    upgraded = dict(certificate)
    if upgraded.get("level") != LEVEL_KERNEL:
        raise ValueError(
            "reproducible requires a kernel_replayed certificate, got "
            f"{upgraded.get('level')!r}"
        )
    distinct = sorted(set(platforms))
    if len(distinct) < 2:
        raise ValueError(
            f"reproducible requires >= 2 distinct platforms, got {distinct}"
        )
    upgraded["reproducible"] = True
    upgraded["level"] = LEVEL_REPRODUCIBLE
    upgraded["platforms_verified"] = distinct
    if export_digest:
        upgraded["export_digest"] = export_digest
    return upgraded


def legacy_lean_level_for(level: str) -> int:
    """Deprecated ``lean_level`` mapping for dashboards that still read it."""
    return {
        LEVEL_NONE: 0,
        LEVEL_STATEMENT: 1,
        LEVEL_PROOF: 3,
        LEVEL_KERNEL: 3,
        LEVEL_REPRODUCIBLE: 3,
    }.get(str(level), 0)
