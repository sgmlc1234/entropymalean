"""Select the rows that go into the release, and carry their evidence with them.

Selection: two independent passes of the same judge over the same corpus in the
same order agreed on keep-or-reject about four times in five. That puts a floor
under what one verdict is worth, so one verdict does not admit a row -- `strong`
from both passes does. It is deliberately conservative and discards rows that
are probably fine; a released benchmark is judged by its worst entry.

Export: the earlier version wrote the verdict and dropped the reasoning, which
is the one thing a reader needs to audit the verdict. Every row now carries all
three opinions it received -- the judge that saw it during generation and the
two re-judge passes -- including the ones that disagree, plus the result of each
mechanical check and who decided it: Lean, a hash, or a model.

Rejected rows are exported too, to `--rejected-output`. A corpus that shows only
its survivors cannot be checked. The discard pile, with the reason attached to
each row, is the evidence that the gates did anything at all.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.certification.dedup import fingerprint
from src.orchestration.problem_ids import chain_of

SOURCES = {
    "run-a": "data/certified/run-a/*.jsonl",
    # Ten generations over ten ProofNet groups, re-planned seeds. Its last four
    # groups are the first rows produced with the coupling gate deciding.
    "run-b": "data/certified/run-b/proofnet_g[0-9][0-9].jsonl",
    # Five miniF2F groups, ten generations, seeds picked on measured per-root
    # yield alone. The first campaign run with the coupling gate and its replan
    # in place from generation one.
    "run-c": "data/certified/run-c/minif2f_h[0-9][0-9].jsonl",
    # Ten more miniF2F groups, all 50 seeds used and no pair repeated from the
    # h-campaign. Ran with the coupling gate and its replan from generation one.
    "run-d": "data/certified/run-d/minif2f_k[0-9][0-9].jsonl",
    # Ten ProofNet groups, ten generations, seeds chosen on measured
    # per-root admission rate. p02 carries both of ProofNet's two
    # number-theory seeds, swapped in so the topic had more than one root
    # to reach the release through.
    "run-e": "data/certified/run-e/proofnet_p[0-9][0-9].jsonl",
    "ablation/crossover": "data/certified/ablation/crossover/*.jsonl",
    "ablation/mutation": "data/certified/ablation/mutation/*.jsonl",
}


def sha256_text(text: Any) -> str:
    """The project's hash rule (src/retrieval/novelty_memory.py): strip, encode
    UTF-8, digest. An empty field stays empty rather than hashing to the digest
    of the empty string, which would read as a value."""
    value = str(text or "").strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def load_rows() -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for campaign, pattern in SOURCES.items():
        for path in sorted(glob.glob(pattern)):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_campaign"] = campaign
                row["_source_file"] = path
                key = str(row.get("problem_id") or "")
                if key and (key not in rows or row.get("status") == "certified"):
                    rows[key] = row
    return rows


#: The benchmark seed sheets carry the real informal statement in `goal`,
#: keyed by a sheet id (`proofnet-3`) rather than by the theorem name the
#: corpus uses. Joining on the declared name recovers it for 70 of the 108
#: distinct parents; the seed CSVs the pipeline reads carry only the stand-in
#: `Prove the theorem <id>.`
def _unescape_statement(text: str) -> str:
    r"""Undo the rewriter's double escaping.

    The rewriter answers in JSON, where `\in` has to be written `\\in`. It
    escaped a second time, so the corpus received `\\in` and every LaTeX
    command rendered as italic letters — `\mathbb{R}` came out as `mathbbR`.
    It did the same to Unicode escapes, so `\u03b2` survived `json.loads` as
    six literal characters and reached LaTeX as an unknown command.

    A backslash pair before a letter, brace or punctuation is always that
    mistake here; `\\` as a LaTeX line break never appears in a one-sentence
    problem statement.
    """
    collapsed = re.sub(r"\\\\(?=[A-Za-z{}()\[\]!,;:.\-])", r"\\", str(text or ""))
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), collapsed)


def _benchmark_of() -> Dict[str, str]:
    """Theorem name to the benchmark its seed came from."""
    import re

    families = {"minif2f_v2": "miniF2F", "proofnet_verified": "ProofNet"}
    out: Dict[str, str] = {}
    for path in sorted(glob.glob("data/benchmarks/*/seeds_50_levels.csv")):
        family = families.get(Path(path).parent.name, Path(path).parent.name)
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                match = re.search(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)",
                                  f"{row.get('lean_goal') or ''} {row.get('solution') or ''}")
                if match:
                    out[match.group(1)] = family
    return out


def _benchmark_prose() -> Dict[str, str]:
    import re

    out: Dict[str, str] = {}
    for path in sorted(glob.glob("data/benchmarks/*/seeds_50_levels.csv")):
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = str(row.get("goal") or "").strip()
                match = re.search(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)",
                                  f"{row.get('lean_goal') or ''} {row.get('solution') or ''}")
                if match and text:
                    out[match.group(1)] = text
    return out


def load_seeds() -> Dict[str, dict]:
    seeds: Dict[str, dict] = {}
    for path in sorted(glob.glob("data/certified/run-a/seeds/*.csv")):
        if path.endswith(".pre_p6_fix"):
            continue
        with open(path, encoding="utf-8") as handle:
            for seed in csv.DictReader(handle):
                key = str(seed.get("id") or seed.get("problem_id") or "")
                if key:
                    seeds[key] = seed
    return seeds


def _ev(row: dict, key: str) -> dict:
    value = (row.get("quality_evidence") or {}).get(key)
    return value if isinstance(value, dict) else {}


def _redundant(evidence: dict) -> bool:
    """A redundancy finding, of either shape.

    Both are Lean accepting a proof of something strictly harder than what the
    row claims -- the child with a hypothesis deleted, or a parent with all of
    them gone. Neither is an opinion to be weighed against the judge's, so this
    reads the findings themselves rather than the `redundant` summary, which a
    stale record may carry from a probe that no longer runs.
    """
    return bool(evidence.get("free_hypotheses") or evidence.get("universal_parents"))


def _review(row: dict, run1: Optional[dict], run2: Optional[dict]) -> dict:
    """Every opinion this row received, in the order it received them."""
    passes: List[dict] = []
    born = _ev(row, "judge")
    if born.get("ran"):
        passes.append(
            {
                "pass": "at_generation",
                "quality": born.get("quality") or "",
                "verdict": born.get("verdict") or "",
                "failure": born.get("failure") or "",
                "reason": born.get("reason") or "",
                "model": born.get("judge_model") or "",
                # The generation-time judge could not see the plan, the siblings
                # already kept from the same parents, or which mutation tier the
                # slot was asked for. Its verdicts are not comparable with the
                # two below and are kept for the record, not for the decision.
                "saw": ["parents", "child"],
            }
        )
    for label, record in (("rejudge_1", run1), ("rejudge_2", run2)):
        if not record:
            continue
        passes.append(
            {
                "pass": label,
                "quality": record.get("new_quality") or "",
                "verdict": record.get("new_verdict") or "",
                "failure": record.get("new_failure") or "",
                "reason": record.get("new_reason") or "",
                "fix_scope": record.get("fix_scope") or "",
                "model": "gpt-5.6-luna",
                "saw": ["parents", "child", "plan", "kept_siblings", "operator_tier", "probe_results"],
            }
        )
    deciding = [p for p in passes if p["pass"].startswith("rejudge")]
    qualities = {p["quality"] for p in deciding}
    return {
        "passes": passes,
        "deciding_passes": len(deciding),
        "agreement": "unanimous" if len(qualities) == 1 else "split",
        "quality": sorted(qualities)[0] if len(qualities) == 1 else "split",
    }


def _checks(row: dict) -> dict:
    """What each gate decided, and what kind of thing decided it."""
    certificate = row.get("certificate") or {}
    audit = certificate.get("axiom_audit") or {}
    vacuity = _ev(row, "vacuity")
    dedup, redundancy = _ev(row, "dedup"), _ev(row, "redundancy")
    silent, alignment = _ev(row, "silent"), _ev(row, "alignment_evidence")
    pruned, comparator = _ev(row, "dead_hypotheses"), _ev(row, "comparator")
    preservation = (_ev(row, "silent") or {}).get("hypothesis_preservation") or {}
    return {
        "statement_type_check": {
            "by": "lean", "ran": bool(certificate.get("statement_checked")),
            "result": "type-checks" if certificate.get("statement_checked") else "not run",
        },
        "proof_accepted": {
            "by": "lean", "ran": bool(certificate.get("proof_accepted")),
            "result": "accepted, no sorry" if certificate.get("proof_accepted") else "not run",
        },
        "axiom_audit": {
            "by": "lean", "ran": bool(audit.get("ran")),
            "result": ("closed over " + ", ".join(audit.get("axioms") or []))
            if audit.get("passed") else ("disallowed: " + ", ".join(audit.get("disallowed") or [])
                                         if audit.get("ran") else "not run"),
        },
        "vacuity": {
            "by": "lean", "ran": bool(vacuity.get("measured")),
            "result": ("hypotheses are satisfiable" if not vacuity.get("vacuous") else "VACUOUS")
            if vacuity.get("measured") else "not run",
        },
        "dead_hypotheses": {
            "by": "lean", "ran": bool(pruned.get("measured")),
            "result": ("removed " + ", ".join(pruned.get("removed") or [])
                       if pruned.get("removed")
                       else "every hypothesis is load-bearing")
            if pruned.get("measured") else "not run",
        },
        "corpus_dedup": {
            "by": "hash", "ran": bool(dedup.get("checked")),
            "result": (f"unique against {dedup.get('corpus_size')} statements"
                       if not dedup.get("duplicate") else f"duplicate of {dedup.get('duplicate_of')}")
            if dedup.get("checked") else "not run",
        },
        # Asked of both operators, in two forms: a crossover is probed for a
        # parent that carries the child alone, a mutation for a child its parent
        # already gives. It was briefly marked crossover-only here, which
        # reported 35 mutation rows that had actually been probed as
        # inapplicable — the opposite mistake to the one that prompted the
        # scoping. Coverage is genuinely partial: the inline probe runs the
        # tactic ladder without a prover, so it measures only where the ladder
        # reaches.
        "redundancy": {
            "by": "lean",
            # A silent mutation is equivalent to its parent by construction, so
            # "the child adds nothing" is the operator succeeding. Reporting it
            # as an unrun check made the coverage line read 152/153 for a row
            # nothing had failed to measure.
            "applies": str(row.get("operator_variant") or "") != "mutation_silent",
            "ran": bool(redundancy.get("measured")),
            "result": (redundancy.get("why") or
                       ("hypotheses Lean could drop: "
                        + ", ".join(redundancy.get("free_hypotheses") or [])
                        if redundancy.get("free_hypotheses")
                        else "no hypothesis Lean could drop, no parent it could prove alone"))
            if redundancy.get("measured") else "not run",
        },
        "hypothesis_preservation": {
            "by": "parser",
            "applies": str(row.get("operator_variant") or "") == "mutation_silent",
            "ran": bool(preservation.get("measured")),
            "result": (("every hypothesis the parent had is still there"
                        if preservation.get("preserved")
                        else "changed: " + str(preservation.get("why") or ""))
                       if preservation.get("measured") else "not run"),
        },
        "comparator": {
            "by": "lean",
            # Only a real run counts as having run. Preparation is reported in
            # its own words so the distinction cannot be read as a pass.
            "ran": bool(comparator.get("kernel_replayed")),
            "result": ("statement, axioms and kernel acceptance independently replayed"
                       if comparator.get("kernel_replayed")
                       else ("workspace prepared and ready; kernel replay still pending "
                             "(comparator needs Linux)"
                             if comparator.get("prepared") else "not run")),
        },
        "goal_roundtrip": {
            "by": "model",
            # A check that errored has not run. Treating a non-empty evidence
            # dict as "ran" counted six rows whose round-trip crashed toward
            # 166/166, and rendered them as `MISMATCH:` with nothing after it --
            # a failure to run displayed as the strongest negative verdict the
            # check can give.
            "ran": alignment.get("equivalent") is not None,
            "result": ("elaborated goal reads back as the stated problem"
                       if alignment.get("equivalent")
                       else "MISMATCH: " + "; ".join(map(str, alignment.get("mismatches") or []))
                       if alignment.get("equivalent") is False
                       else f"check did not complete ({alignment.get('status') or 'no result'})"
                       if alignment else "not run"),
            "elaborated_goal": alignment.get("elaborated_goal") or "",
            "read_back_as": alignment.get("informalized_statement") or "",
            "rationale": alignment.get("rationale") or "",
        },
    }


def _plan(row: dict) -> dict:
    card = row.get("operator_card") or {}
    return {
        "operator_goal": card.get("operator_goal") or card.get("goal") or "",
        "fusion_mechanism": card.get("fusion_mechanism") or "",
        "fusion_goal": card.get("fusion_goal") or "",
        "parent_roles": card.get("parent_roles") or {},
        "constraints": card.get("constraints") or [],
        "avoid": card.get("avoid") or [],
        "planned_op_type": row.get("planned_op_type") or "",
        "planned_operator_variant": row.get("planned_operator_variant") or "",
    }


_BENCHMARK_BY_NAME: Dict[str, str] = {}


def _benchmark_for(problem_id: str) -> str:
    from src.orchestration.problem_ids import roots_of

    families = {_BENCHMARK_BY_NAME.get(root, "") for root in roots_of(problem_id)}
    families.discard("")
    if not families:
        return "unknown"
    return sorted(families)[0] if len(families) == 1 else "mixed"


def _record(row: dict, run1: Optional[dict], run2: Optional[dict], parents: Dict[str, dict]) -> dict:
    certificate = row.get("certificate") or {}
    problem_id = str(row.get("problem_id") or "")
    return {
        "problem_id": problem_id,
        "campaign": row.get("_campaign") or "",
        # Empty on every row the pipeline writes; recovered from the seed sheets
        # by the root the id names. A crossover whose roots come from both is
        # `mixed`, which is a real category and not a gap.
        "benchmark": _benchmark_for(problem_id),
        "op_type": row.get("op_type") or "",
        "operator_variant": row.get("operator_variant") or "",
        "generation": row.get("generation"),
        "slot": row.get("slot"),
        "family": row.get("family") or "",
        "llm_model": row.get("llm_model") or "",
        "statement": row.get("statement") or "",
        "formal_statement": row.get("formal_statement") or "",
        "lean_code": row.get("lean_code") or "",
        "lean_header": row.get("lean_header") or "",
        "parents": [
            {
                "parent_id": pid,
                "formal_statement": str((parents.get(pid) or {}).get("formal_statement") or ""),
                "statement": str((parents.get(pid) or {}).get("statement") or ""),
                "contribution": str((row.get("parent_contributions") or {}).get(pid) or
                                    (_ev(row, "parent_contribution") or {}).get(pid) or ""),
            }
            for pid in (row.get("parent_ids") or [])
        ],
        "ancestor_ids": row.get("ancestor_ids") or [],
        # Counted from the operator chain the id now carries. This read
        # `count("__theorem_gen")`, which the id migration deleted: after it
        # every mutation lineage reported depth 0 and the three-step chains
        # vanished from the composition table.
        "lineage_depth": max(len(chain_of(problem_id)) - 1, 0),
        "plan": _plan(row),
        "certificate": {
            # `kernel_replayed` is claimed only when comparator says so.
            "level": ("kernel_replayed" if _ev(row, "comparator").get("kernel_replayed")
                      else (certificate.get("level") or "")),
            "verifier": certificate.get("verifier") or "",
            "lean_toolchain": certificate.get("lean_toolchain") or "",
            "mathlib_revision": certificate.get("mathlib_revision") or "",
            "axioms": (certificate.get("axiom_audit") or {}).get("axioms") or [],
        },
        "review": _review(row, run1, run2),
        "checks": _checks(row),
        # Computed here, from the strings this row actually ships. Reading
        # `row.get("statement_sha256")` and falling back to "" was the bug: no
        # upstream stage ever set it, so every released row carried two empty
        # audit hashes while `dedup_fingerprint`, computed inline, was fine.
        # Hashing the certified record instead would be worse than empty --- 129
        # statements are rewritten after certification, so the digest would
        # authenticate text nobody received.
        "hashes": {
            "statement_sha256": sha256_text(row.get("statement")),
            "formal_statement_sha256": sha256_text(row.get("formal_statement")),
            "dedup_fingerprint": fingerprint(row.get("formal_statement") or ""),
        },
    }


#: State that lives *after* this script and does not survive re-running it:
#:
#:   - rows withdrawn by hand from the release file
#:   - `reproducible` certificates, granted by
#:     scripts/faithfulness/kernel/check_export_reproducible.py
#:
#: Both are layered onto the written release, and this exporter rebuilds that
#: file from the campaign sources, so a re-run reverts them without saying so.
#: On 2026-08-23 that turned 535 rows with 535 top-rung certificates back into
#: 537 with none. Rebuild after any re-run: restore the pre-upgrade file and
#: re-apply the three `export_reproducible_*.json` reports, or re-run the
#: reproducibility check.


def _refuse_to_discard(output: Path, final: list, force: bool) -> None:
    """Stop if re-exporting would throw away state that lives after this script.

    Two things are layered onto the written release and are not rebuilt here:
    rows withdrawn by hand, and `reproducible` certificates granted by the
    two-platform export check. A re-run silently reverted both once -- 535 rows
    carrying 535 top-rung certificates became 537 carrying none -- and the
    warning at the top of this file did not prevent it, because a warning does
    not stop someone who runs the command without reading it.

    So it refuses instead. `--force` is there for the case where the exporter's
    output really is meant to replace what is on disk.
    """
    if force or not output.is_file():
        return
    existing = [json.loads(line) for line in
                output.read_text(encoding="utf-8").splitlines() if line.strip()]
    upgraded = [r for r in existing
                if (r.get("certificate") or {}).get("level") == "reproducible"]
    incoming = {r["problem_id"] for r in final}
    withdrawn = [r for r in existing if r["problem_id"] not in incoming]
    if not upgraded and not withdrawn:
        return
    print(f"\nrefusing to overwrite {output}", file=sys.stderr)
    if upgraded:
        print(f"  {len(upgraded)} of {len(existing)} rows there carry a "
              f"`reproducible` certificate, which this script does not rebuild.",
              file=sys.stderr)
    if withdrawn:
        print(f"  {len(withdrawn)} row(s) there are absent from this export and "
              f"were most likely withdrawn by hand:", file=sys.stderr)
        for row in withdrawn[:4]:
            print(f"      {row['problem_id']}", file=sys.stderr)
    print("\n  To rebuild that state afterwards: keep a copy of the current file,"
          "\n  re-run this with --force, then re-apply the reproducibility reports"
          "\n  (data/release/export_reproducible_*.json) or re-run"
          "\n  scripts/faithfulness/kernel/check_export_reproducible.py.",
          file=sys.stderr)
    raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, default=Path("data/release/rejudged.json"))
    parser.add_argument("--run2", type=Path, default=Path("data/release/rejudged_run2.json"))
    parser.add_argument("--output", type=Path, default=Path("data/release/eml_v1_release.jsonl"))
    parser.add_argument("--rejected-output", type=Path, default=Path("data/release/eml_v1_rejected.jsonl"))
    parser.add_argument("--prune", type=Path, default=Path("data/release/hypothesis_prune.json"),
                        help="Hypotheses Lean confirmed the proof does not need, removed from the "
                             "rows they were found in. Produced by check_dead_hypotheses.py for "
                             "this corpus; newer runs prune during certification instead.")
    parser.add_argument("--comparator", type=Path, default=Path("data/release/comparator_batch.json"),
                        help="Preparation dry-run from prepare_comparator_batch.py: whether each "
                             "workspace was built and whether comparator could consume it.")
    parser.add_argument("--rewrites", type=Path, default=Path("data/release/statement_rewrites.json"),
                        help="Statements rewritten in the register of the seed they descend from. "
                             "Only rewrites the goal round-trip accepted are applied; the rest "
                             "keep the original, because an accurate ugly statement beats a "
                             "readable one that describes a different theorem.")
    parser.add_argument("--roundtrip", type=Path, default=Path("data/release/goal_roundtrip.json"),
                        help="Goal round-trip verdicts. A row whose prose does not describe the "
                             "goal Lean elaborated is dropped: it type-checks and proves, and it "
                             "states a different problem than the one it is released as.")
    parser.add_argument("--redundancy-scan", type=Path,
                        default=Path("data/release/redundancy_scan.json"),
                        help="Corpus-wide redundancy scan, prover on. A row Lean can close with "
                             "one of its own hypotheses deleted, or one of whose parents Lean "
                             "proves outright, is dropped. The judge may have called it strong; "
                             "the gate does not weigh that, because the finding is a proof.")
    parser.add_argument("--manual-exclusions", type=Path,
                        default=Path("data/release/manual_exclusions.json"),
                        help="Rows held back by hand, each with the reason on file. Read here "
                             "rather than left as a note because a decision that lives only in a "
                             "conversation is one the next export silently reverses.")
    parser.add_argument("--comparator-results", type=Path,
                        default=Path("data/release/comparator_results.json"),
                        help="Kernel verdicts from run_comparator_batch.sh on Linux. Merged on "
                             "top of the preparation record rather than replacing it, so a row "
                             "keeps both facts: that its workspace was ready, and what the kernel "
                             "then said about it.")
    parser.add_argument("--prune-judged-1", type=Path, default=Path("data/release/pruned_rejudged_1.json"))
    parser.add_argument("--prune-judged-2", type=Path, default=Path("data/release/pruned_rejudged_2.json"))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a release that carries `reproducible` "
                             "certificates or hand-withdrawn rows. Those are not "
                             "rebuilt by this script; see _refuse_to_discard.")
    args = parser.parse_args()

    rows = load_rows()
    parents = dict(rows)
    parents.update(load_seeds())
    _BENCHMARK_BY_NAME.update(_benchmark_of())
    prose = _benchmark_prose()
    for key, seed in parents.items():
        if key in prose and isinstance(seed, dict):
            seed = dict(seed)
            seed["statement"] = prose[key]
            parents[key] = seed
    print(f"benchmark prose recovered for {sum(1 for k in parents if k in prose)} seed(s)")
    run1 = {r["problem_id"]: r for r in json.loads(args.run1.read_text(encoding="utf-8"))}
    run2 = {r["problem_id"]: r for r in json.loads(args.run2.read_text(encoding="utf-8"))}

    # A pruned row is a different statement from the one the first two passes
    # judged, so it does not inherit their verdict: it was re-judged twice in
    # its pruned form and those verdicts replace the originals outright.
    prune: Dict[str, dict] = {}
    if args.prune.is_file():
        prune = json.loads(args.prune.read_text(encoding="utf-8"))
        for key, path in (("1", args.prune_judged_1), ("2", args.prune_judged_2)):
            if not path.is_file():
                continue
            target = run1 if key == "1" else run2
            for record in json.loads(path.read_text(encoding="utf-8")):
                target[record["problem_id"]] = record
        # The scan covered every row, so every row records the check. Only the
        # ones it changed carry a removal; a row with nothing to remove is a row
        # that passed, not a row that was skipped, and the coverage column would
        # say otherwise if only the pruned rows were marked measured.
        scanned = {}
        scan_path = args.prune.with_name("dead_hypotheses.json")
        if scan_path.is_file():
            scanned = {
                record["problem_id"]: [f["hypothesis"] for f in record["findings"]
                                       if f["verdict"] == "used silently"]
                for record in json.loads(scan_path.read_text(encoding="utf-8"))
            }
        for problem_id, row in rows.items():
            patch = prune.get(problem_id)
            if patch:
                row["formal_statement"] = patch["formal_statement"]
                row["lean_code"] = patch["lean_code"]
            # A row absent from the scan had no candidate to test -- verified
            # above -- which is a pass, not a gap.
            row.setdefault("quality_evidence", {})["dead_hypotheses"] = {
                "measured": True,
                "removed": (patch or {}).get("removed") or [],
                "used_silently": scanned.get(problem_id, []),
            }
        print(f"pruned rows applied: {len(prune)} "
              f"({sum(len(p['removed']) for p in prune.values())} hypotheses removed)")

    # Comparator is the only thing that lifts a row past `proof_checked`. Until
    # it has actually run on Linux, the file here records preparation only, and
    # the level must not move -- a workspace that is ready to be checked is not
    # a workspace that has been checked.
    comparator: Dict[str, dict] = {}
    if args.comparator.is_file():
        comparator = {r["problem_id"]: r for r in json.loads(args.comparator.read_text(encoding="utf-8"))}
    if args.comparator_results.is_file():
        for record in json.loads(args.comparator_results.read_text(encoding="utf-8")):
            entry = comparator.setdefault(record["problem_id"], {})
            entry["kernel_replayed"] = bool(record.get("kernel_replayed"))
            entry["returncode"] = record.get("returncode")
            entry["log"] = str(record.get("log") or "")[-1500:]
        replayed = sum(1 for r in comparator.values() if r.get("kernel_replayed"))
        ready = sum(1 for r in comparator.values()
                    if r.get("prepared") and r.get("challenge_elaborates") and r.get("name_matches"))
        print(f"comparator: {len(comparator)} rows · prepared-and-ready {ready} · kernel_replayed {replayed}")
        for problem_id, record in comparator.items():
            row = rows.get(problem_id)
            if row:
                row.setdefault("quality_evidence", {})["comparator"] = record

    # The round-trip is the only check that reads the *statement* rather than
    # the theorem, and on the released corpus it disagreed on 13 of 146: a
    # conclusion asserted for all reals where the Lean bounds it to [0,10], an
    # arithmetic-progression claim the hypotheses do not give. Those rows are
    # sound Lean and wrong problems, so they are dropped rather than flagged.
    roundtrip: Dict[str, dict] = {}
    if args.roundtrip.is_file():
        roundtrip = {r["problem_id"]: r for r in json.loads(args.roundtrip.read_text(encoding="utf-8"))}
        for problem_id, record in roundtrip.items():
            row = rows.get(problem_id)
            if row is not None:
                row.setdefault("quality_evidence", {})["alignment_evidence"] = record
        decided = [r for r in roundtrip.values() if r.get("equivalent") is not None]

    # The scan carries the prover; the inline probe runs the tactic ladder only,
    # so a row can pass during generation and fail here. Where both exist the
    # scan wins: it asked the same question with more behind it.
    redundancy_scan: Dict[str, dict] = {}
    if args.redundancy_scan.is_file():
        redundancy_scan = {
            r["problem_id"]: r
            for r in json.loads(args.redundancy_scan.read_text(encoding="utf-8"))
        }
        for problem_id, record in redundancy_scan.items():
            row = rows.get(problem_id)
            if row is not None and record.get("measured"):
                row.setdefault("quality_evidence", {})["redundancy"] = {
                    k: v for k, v in record.items()
                    if k in ("measured", "free_hypotheses", "universal_parents",
                             "redundant", "why", "settled_by")
                }
        print(f"redundancy scan: {sum(1 for r in redundancy_scan.values() if r.get('measured'))} measured, "
              f"{sum(1 for r in redundancy_scan.values() if _redundant(r))} with a finding")
        print(f"goal round-trip: {len(decided)} decided, "
              f"{sum(1 for r in decided if not r.get('equivalent'))} prose/goal mismatches")

    if args.rewrites.is_file():
        applied = 0
        for record in json.loads(args.rewrites.read_text(encoding="utf-8")):
            row = rows.get(record.get("problem_id"))
            if row is not None and record.get("gate") == "accepted" and record.get("after"):
                # The rewriter answers in JSON, where a LaTeX command has to be
                # written `\\in` to mean `\in`. It escaped a second time, so the
                # corpus received `\\in` and every command rendered as italic
                # letters -- `\mathbb{R}` came out as `mathbbR`. A backslash pair
                # before a letter or brace is always that mistake here; `\\` as a
                # LaTeX line break never appears in a one-sentence statement.
                row["statement"] = _unescape_statement(record["after"])
                row.setdefault("quality_evidence", {})["statement_rewrite"] = {
                    "applied": True,
                    "before": record.get("before") or "",
                    "gated_by": "goal_roundtrip",
                }
                applied += 1
        print(f"statement rewrites applied: {applied}")

    excluded: Dict[str, str] = {}
    if args.manual_exclusions.is_file():
        held = json.loads(args.manual_exclusions.read_text(encoding="utf-8"))
        excluded = {str(r["problem_id"]): str(held.get("why") or "") for r in held.get("rows") or []}
        print(f"manual exclusions on file: {len(excluded)}")

    judged = [pid for pid in run1 if pid in run2 and pid in rows]
    accepted, rejected = [], []
    for pid in judged:
        record = _record(rows[pid], run1[pid], run2[pid], parents)
        qualities = [run1[pid].get("new_quality"), run2[pid].get("new_quality")]
        verdicts = [run1[pid].get("new_verdict"), run2[pid].get("new_verdict")]
        # Both a mismatch and an undecided check hold the row back. The second
        # is not a defect in the row -- the check crashed -- but a release whose
        # coverage table reads 166/166 must not contain rows the check never
        # reached. Six such rows were retried and five then passed; what is left
        # is held, not shipped with a blank where its evidence should be.
        verdict = roundtrip.get(pid, {}).get("equivalent")
        if verdict is not True and pid in roundtrip:
            record["admission"] = {
                "admitted": False,
                "why_not": ("prose does not describe the elaborated goal" if verdict is False
                            else "goal round-trip never returned a verdict"),
                "detail": str(roundtrip[pid].get("rationale") or "")[:300],
            }
            rejected.append(record)
            continue
        # A redundancy finding ends the row here, whatever the judge thought of
        # it. Some of these read well and were called strong twice; the gate
        # does not weigh that, because on the other side is a Lean proof that
        # the row asserts less than it states. A standard that bends for a good
        # problem is not a standard, and the corpus can afford the loss.
        # A hand decision, ahead of the automated gates so it is never masked by
        # one of them agreeing for a different reason.
        if pid in excluded:
            record["admission"] = {
                "admitted": False,
                "why_not": "held back by hand: a binder that can be substituted away",
                "detail": excluded[pid][:400],
            }
            rejected.append(record)
            continue
        scanned = redundancy_scan.get(pid) or {}
        if _redundant(scanned):
            free = ", ".join(scanned.get("free_hypotheses") or [])
            universal = ", ".join(scanned.get("universal_parents") or [])
            record["admission"] = {
                "admitted": False,
                "why_not": "Lean proved it without a hypothesis it states",
                "detail": "; ".join(part for part in (
                    f"hypotheses Lean could delete: {free}" if free else "",
                    f"parents Lean proves with no hypotheses: {universal}" if universal else "",
                ) if part),
            }
            rejected.append(record)
            continue
        # Both halves of both passes. Quality alone admitted six rows the judge
        # had voted to reject -- two of them rejected by both passes -- because
        # `verdicts` was only consulted in the branch below, which quality
        # strong/strong never reached. The two answer different questions: how
        # good the problem is, and whether it should be in the corpus at all. A
        # `recall` rejection is precisely a row that is good and should not be
        # here, so reading only the first throws away the finding.
        if qualities == ["strong", "strong"] and verdicts == ["keep", "keep"]:
            record["admission"] = {"admitted": True, "why_not": ""}
            accepted.append(record)
            continue
        # Three ways to miss the bar, and they are not the same finding. A row
        # both passes reject is a defect the pipeline caught; a row they split
        # on is a row the judge could not decide, which is a fact about the
        # judge; a row both keep but neither calls strong is simply below the
        # bar this release sets. Collapsing them would overstate the first.
        if all(v != "keep" for v in verdicts):
            why = "rejected by both passes"
        elif any(v != "keep" for v in verdicts):
            why = "passes disagreed on keep/reject"
        else:
            why = "kept by both passes but not strong in both"
        record["admission"] = {"admitted": False, "why_not": why}
        rejected.append(record)
    print(f"judged {len(judged)}  admitted {len(accepted)}  held back {len(rejected)}")
    print("  held back: " + str(dict(collections.Counter(
        r["admission"]["why_not"] for r in rejected).most_common())))

    # One row per fingerprint. Deeper evidence wins; ties go to the shallower
    # lineage, which is the one a reader can trace back to a seed fastest.
    groups: Dict[str, list] = collections.defaultdict(list)
    for record in accepted:
        groups[record["hashes"]["dedup_fingerprint"]].append(record)
    final, dropped = [], 0
    for members in groups.values():
        members.sort(key=lambda r: (-sum(1 for c in r["checks"].values() if c["ran"]), r["lineage_depth"]))
        final.append(members[0])
        dropped += len(members) - 1
    print(f"alpha-equivalent duplicates dropped {dropped}  ->  release {len(final)}")

    _refuse_to_discard(args.output, final, args.force)

    for path, records in ((args.output, final), (args.rejected_output, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda r: r["problem_id"]):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  written {path}  ({len(records)} rows)")

    print("\nrelease composition")
    for field in ("campaign", "op_type", "operator_variant", "lineage_depth"):
        print(f"  {field:18s} {dict(sorted(collections.Counter(r[field] for r in final).items(), key=lambda x: str(x[0])))}")
    # Counted against the rows a check applies to, not the whole release. Two
    # checks are operator-scoped, and dividing them by 153 reported them as
    # partially run when they had covered everything they speak about.
    ran = collections.Counter(name for r in final for name, c in r["checks"].items() if c["ran"])
    scope = collections.Counter(name for r in final for name, c in r["checks"].items()
                                if c.get("applies", True))
    print("  checks that ran   " + ", ".join(
        f"{k} {v}/{scope[k]}" for k, v in sorted(ran.items())))
    print("  reasoning texts   " + str(sum(len(r["review"]["passes"]) for r in final)) + " across " + str(len(final)) + " rows")
    both = [r for r in rejected if r["admission"]["why_not"] == "rejected by both passes"]
    print("\nfailure modes among rows both passes rejected")
    print("  " + str(dict(collections.Counter(
        (r["review"]["passes"][-1].get("failure") or "unlabelled") for r in both).most_common())))


if __name__ == "__main__":
    main()
