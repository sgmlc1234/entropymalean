#!/usr/bin/env python3
"""Seed-lineage ablation: how often does the prover solve every root
seed used by a generated problem yet fail the EML-1 child?

This script answers the multi-hop question raised by the family-aware
generator. For each accepted treatment row, we recursively follow the
recorded parent graph back to upstream benchmark seeds, then ask:

  - Did the model solve every root seed in that lineage (under the
    real-name filter)?
  - If yes, did it solve the child?
  - When the child failed, how much effort did it sink — measured as
    average turns or tactic-step nodes per attempt, and total
    wall-clock seconds.

The output is the seed-lineage proof gap (LPG) table and the per-cell
case list used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Callable
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_BENCHMARKS = ("minif2f", "proofnet")
DIRECTION_ORDER = ("increase", "decrease", "unknown")
PREFERRED_MODELS = ("BFS-Prover-V2-7B", "Goedel-Prover-V2-8B")


def theorem_name(formal: str) -> str:
    if not formal:
        return ""
    m = re.search(r"\btheorem\s+(\w+)", formal)
    return m.group(1) if m else ""


def _sha(text: object) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _formal_hash(row: dict) -> str:
    hashes = row.get("hashes") if isinstance(row.get("hashes"), dict) else {}
    return str(
        row.get("formal_statement_sha256")
        or hashes.get("formal_statement_sha256")
        or _sha(row.get("formal_statement") or row.get("lean_code"))
    )


def load_control_and_mapping(input_dir: Path):
    """Returns (control_ids_per_bench, problem_id→theorem_name)."""
    control_ids: dict[str, set[str]] = defaultdict(set)
    mapping: dict[tuple[str, str], str] = {}
    for bench in MAIN_BENCHMARKS:
        csv_path = input_dir / f"{bench}_control.csv"
        if csv_path.exists():
            for row in csv.DictReader(csv_path.open()):
                pid = row.get("id") or row.get("problem_id")
                fs = row.get("formal_statement") or row.get("formal_prefix")
                if pid:
                    control_ids[bench].add(pid)
                tn = theorem_name(fs)
                if pid and tn:
                    mapping[(bench, pid)] = tn
        jsonl_path = input_dir / f"{bench}_treatment.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r.get("eval_problem_id") or r.get("problem_id")
                tn = theorem_name(r.get("formal_statement") or r.get("lean_code"))
                if pid and tn:
                    mapping[(bench, pid)] = tn
    return control_ids, mapping


def load_treatment_eval_ids(input_dir: Path) -> dict[tuple[str, str, str], str]:
    """Map accepted-ledger rows to their evaluation ids.

    ``prepare_campaign_inputs.py`` may append a stable hash suffix in
    ``eval_problem_id`` when multiple accepted rows share the same public
    ``problem_id``. LPG must join accepted metadata to evaluated proof rows
    through that id, not through the raw id alone.
    """
    out: dict[tuple[str, str, str], str] = {}
    for bench in MAIN_BENCHMARKS:
        jsonl_path = input_dir / f"{bench}_treatment.jsonl"
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_id = str(row.get("problem_id") or "").strip()
            eval_id = str(row.get("eval_problem_id") or raw_id).strip()
            if raw_id and eval_id:
                out[(bench, raw_id, _formal_hash(row))] = eval_id
    return out


def real_pass(row: dict, mapping: dict) -> bool:
    if not row.get("pass_at_k"):
        return False
    tn = mapping.get((row["benchmark"].lower(), row["problem_id"]), "")
    if not tn:
        return row["pass_at_k"]
    needle = f"theorem {tn}"
    for a in row.get("attempts", []):
        if a.get("success") and needle in (a.get("final_proof") or ""):
            return True
    return False


def _parent_ids(row: dict) -> list[str]:
    """Return parent ids from the accepted-row public schema."""
    ids: list[str] = []
    for parent in row.get("parents") or []:
        parent_id = str(parent.get("parent_id") or "").strip()
        if parent_id:
            ids.append(parent_id)
    return list(dict.fromkeys(ids))


def _entropy_direction(row: dict) -> str:
    curation = row.get("curation") if isinstance(row.get("curation"), dict) else {}
    direction = str(curation.get("entropy_direction") or "").strip().lower()
    return direction if direction in {"increase", "decrease"} else "unknown"


def _new_direction_stat() -> dict:
    return {
        "candidates": 0,
        "seed_evaluable": 0,
        "seed_solved": 0,
        "seed_unsolved": 0,
        "child_solved": 0,
        "lpg_fail": 0,
        "rescue": 0,
        "missing_seed_evaluation": [],
        "lpg_cases": [],
        "rescue_cases": [],
    }


def _case_record(
    *,
    row: dict,
    op_type: str,
    bench: str,
    model: str,
    entropy_direction: str,
    root_seed_ids: list[str],
) -> dict:
    return {
        "op_type": op_type,
        "benchmark": bench,
        "model": model,
        "entropy_direction": entropy_direction,
        "problem_id": row["problem_id"],
        "root_seed_ids": root_seed_ids,
        "n_root_seeds": len(root_seed_ids),
    }


def _bump_direction_stats(
    stats: dict,
    *,
    row: dict,
    op_type: str,
    bench: str,
    model: str,
    entropy_direction: str,
    root_seed_ids: list[str],
    seed_rows_available: bool,
    seeds_ok: bool,
    child_passed: bool,
) -> None:
    stats["candidates"] += 1
    case = _case_record(
        row=row,
        op_type=op_type,
        bench=bench,
        model=model,
        entropy_direction=entropy_direction,
        root_seed_ids=root_seed_ids,
    )
    if not seed_rows_available:
        stats["missing_seed_evaluation"].append(case)
        return
    stats["seed_evaluable"] += 1
    if child_passed:
        stats["child_solved"] += 1
    if seeds_ok:
        stats["seed_solved"] += 1
        if not child_passed:
            stats["lpg_fail"] += 1
            stats["lpg_cases"].append(case)
    else:
        stats["seed_unsolved"] += 1
        if child_passed:
            stats["rescue"] += 1
            stats["rescue_cases"].append(case)


def _merge_direction_stats(dst: dict, src: dict) -> None:
    for key in (
        "candidates",
        "seed_evaluable",
        "seed_solved",
        "seed_unsolved",
        "child_solved",
        "lpg_fail",
        "rescue",
    ):
        dst[key] += src[key]
    for key in ("missing_seed_evaluation", "lpg_cases", "rescue_cases"):
        dst[key].extend(src[key])


def _finalize_direction_summary(stats_by_direction: dict) -> list[dict]:
    rows = []
    for direction in DIRECTION_ORDER:
        stat = stats_by_direction.get(direction)
        if not stat:
            continue
        seed_solved = stat["seed_solved"]
        seed_unsolved = stat["seed_unsolved"]
        rows.append({
            "entropy_direction": direction,
            "candidates": stat["candidates"],
            "seed_evaluable": stat["seed_evaluable"],
            "seed_solved": seed_solved,
            "seed_unsolved": seed_unsolved,
            "child_solved": stat["child_solved"],
            "lpg_fail": stat["lpg_fail"],
            "lpg_rate": round(stat["lpg_fail"] / seed_solved, 6) if seed_solved else None,
            "rescue": stat["rescue"],
            "rescue_rate": round(stat["rescue"] / seed_unsolved, 6) if seed_unsolved else None,
            "missing_seed_evaluation": stat["missing_seed_evaluation"],
            "lpg_cases": stat["lpg_cases"],
            "rescue_cases": stat["rescue_cases"],
        })
    return rows


def build_parent_graph(accepted: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Map ``(benchmark, problem_id)`` to all recorded immediate parents.

    The accepted ledger may contain multiple accepted rows with the same
    ``problem_id`` but different theorem surfaces. For LPG we need the seed
    lineage used by that id, so unioning parent ids is conservative and keeps
    duplicate accepted variants from under-counting ancestry.
    """
    graph: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in accepted:
        bench = str(row.get("benchmark") or "").lower()
        problem_id = str(row.get("problem_id") or "").strip()
        if not bench or not problem_id:
            continue
        graph[(bench, problem_id)].update(_parent_ids(row))
    return graph


_GEN_SUFFIX_RE = re.compile(r"__(?:theorem_)?gen\d+(?:_[A-Za-z0-9_]+)?$")


def _strip_generation_suffix(problem_id: str) -> str:
    """Remove trailing generated-id suffixes while preserving seed ids."""
    current = problem_id
    while True:
        stripped = _GEN_SUFFIX_RE.sub("", current)
        if stripped == current:
            return current
        current = stripped


def _infer_synthetic_seed_roots(problem_id: str, control_ids: set[str]) -> set[str]:
    """Best-effort root recovery for generated ids absent from the ledger.

    Older generated rows sometimes cite an intermediate parent that is not in
    the final accepted ledger. The id convention still exposes the root seeds:
    crossovers are joined with ``__x__`` and mutations append
    ``__theorem_genN`` or ``__genN_*``. This parser is deliberately limited to
    ids whose recovered leaves are present in the benchmark control set.
    """
    problem_id = str(problem_id or "").strip()
    if not problem_id:
        return set()
    if problem_id in control_ids:
        return {problem_id}

    if "__x__" in problem_id:
        roots: set[str] = set()
        for part in problem_id.split("__x__"):
            part_roots = _infer_synthetic_seed_roots(part, control_ids)
            if not part_roots:
                return set()
            roots.update(part_roots)
        return roots

    stripped = _strip_generation_suffix(problem_id)
    if stripped != problem_id:
        return _infer_synthetic_seed_roots(stripped, control_ids)

    # Last-resort prefix match for legacy one-parent ids with custom suffixes.
    matches = [
        seed_id
        for seed_id in control_ids
        if problem_id.startswith(f"{seed_id}__")
    ]
    if matches:
        return {max(matches, key=len)}
    return set()


def make_root_seed_resolver(
    control_ids: dict[str, set[str]],
    parent_graph: dict[tuple[str, str], set[str]],
) -> Callable[..., tuple[frozenset[str], frozenset[str]]]:
    """Return ``resolve(bench, problem_id) -> (roots, unresolved)``.

    ``roots`` is the upstream benchmark seed set used by the generated row.
    ``unresolved`` contains generated ids whose lineage could not be traced to
    benchmark control seeds, so callers can exclude them from LPG.
    """

    @lru_cache(maxsize=None)
    def resolve(
        bench: str,
        problem_id: str,
        stack: tuple[str, ...] = (),
    ) -> tuple[frozenset[str], frozenset[str]]:
        bench = bench.lower()
        problem_id = str(problem_id or "").strip()
        bench_control = control_ids.get(bench, set())
        if not problem_id:
            return frozenset(), frozenset({"<empty>"})
        if problem_id in bench_control:
            return frozenset({problem_id}), frozenset()
        if problem_id in stack:
            return frozenset(), frozenset({problem_id})

        parents = parent_graph.get((bench, problem_id))
        if parents:
            roots: set[str] = set()
            unresolved: set[str] = set()
            next_stack = stack + (problem_id,)
            for parent_id in parents:
                parent_roots, parent_unresolved = resolve(bench, parent_id, next_stack)
                roots.update(parent_roots)
                unresolved.update(parent_unresolved)
            return frozenset(roots), frozenset(unresolved)

        inferred = _infer_synthetic_seed_roots(problem_id, bench_control)
        if inferred:
            return frozenset(inferred), frozenset()
        return frozenset(), frozenset({problem_id})

    return resolve


def effort_proxy(row: dict) -> tuple[float, float, str]:
    """Return (mean per-attempt effort, total elapsed seconds, label).

    Tree-search rows expose ``nodes_explored`` per attempt; whole-proof
    rows expose ``turns_used``. The mean is across all K attempts so
    failed cells that never grew get the floor value (typically 1).
    """
    atts = row.get("attempts", []) or []
    if not atts:
        return (0.0, row.get("total_elapsed_seconds", 0.0), "n/a")
    first = atts[0]
    if "nodes_explored" in first:
        eff = sum(a.get("nodes_explored", 0) for a in atts) / len(atts)
        return (eff, row.get("total_elapsed_seconds", 0.0), "nodes/attempt")
    eff = sum(a.get("turns_used", 0) for a in atts) / len(atts)
    return (eff, row.get("total_elapsed_seconds", 0.0), "turns/attempt")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--campaign-dir", type=Path,
        default=ROOT / "data/evaluation/campaign_2026-05-20-Q8"
    )
    p.add_argument("--input-dir", type=Path, default=Path("/tmp/eml_campaign"))
    p.add_argument(
        "--accepted", type=Path,
        default=ROOT / "data/evaluation/treatment_inventory/final_curated/accepted.jsonl"
    )
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "data/evaluation/campaign_2026-05-20-Q8/parent_child_ablation.json"
    )
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model labels to aggregate. By default, use the "
            "models actually present in --campaign-dir, ordered by the main "
            "paper panel."
        ),
    )
    args = p.parse_args()

    control_ids, mapping = load_control_and_mapping(args.input_dir)
    treatment_eval_ids = load_treatment_eval_ids(args.input_dir)

    # Index per-(bench, model, pid) row.
    row_lookup: dict[tuple[str, str, str], dict] = {}
    for bench in MAIN_BENCHMARKS:
        path = args.campaign_dir / f"{bench}_proof.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            row_lookup[(bench, r["model"], r["problem_id"])] = r
    accepted = [json.loads(l) for l in args.accepted.open() if l.strip()]
    parent_graph = build_parent_graph(accepted)
    resolve_root_seeds = make_root_seed_resolver(control_ids, parent_graph)
    if args.models:
        models = [item.strip() for item in args.models.split(",") if item.strip()]
    else:
        present_models = {model for _, model, _ in row_lookup}
        models = [m for m in PREFERRED_MODELS if m in present_models]
        models.extend(sorted(present_models.difference(models)))
    rows_out = []
    global_direction_stats: dict[tuple[str, str], dict] = defaultdict(_new_direction_stat)
    for op_type in ("crossover", "mutation"):
        for bench in MAIN_BENCHMARKS:
            for model in models:
                cands = [
                    r for r in accepted
                    if r["benchmark"] == bench and r["op_type"] == op_type
                    and r.get("parents")
                ]
                cands_with_roots = []
                unresolved_lineage = []
                for r in cands:
                    roots, unresolved = resolve_root_seeds(bench, r["problem_id"])
                    roots_set = set(roots)
                    unresolved_set = set(unresolved)
                    if roots_set and not unresolved_set:
                        cands_with_roots.append((r, sorted(roots_set)))
                    else:
                        unresolved_lineage.append(
                            {
                                "problem_id": r["problem_id"],
                                "root_seed_ids": sorted(roots_set),
                                "unresolved_parent_ids": sorted(unresolved_set),
                            }
                        )
                evaluable_cands = []
                missing_child_evaluation = []
                for r, root_seed_ids in cands_with_roots:
                    trt_pid = treatment_eval_ids.get(
                        (bench, r["problem_id"], _formal_hash(r)),
                        r["problem_id"],
                    )
                    if (bench, model, trt_pid) in row_lookup:
                        evaluable_cands.append((r, root_seed_ids))
                    else:
                        missing_child_evaluation.append({
                            "problem_id": trt_pid,
                            "root_seed_ids": root_seed_ids,
                            "n_root_seeds": len(root_seed_ids),
                        })
                qualified = 0
                qualified_cases = []
                ablation = []
                direction_stats: dict[str, dict] = defaultdict(_new_direction_stat)
                for r, root_seed_ids in evaluable_cands:
                    entropy_direction = _entropy_direction(r)
                    seed_rows_available = all(
                        (bench, model, seed_id) in row_lookup
                        for seed_id in root_seed_ids
                    )
                    trt_pid = treatment_eval_ids.get(
                        (bench, r["problem_id"], _formal_hash(r)),
                        r["problem_id"],
                    )
                    child_row = row_lookup[(bench, model, trt_pid)]
                    child_passed = real_pass(child_row, mapping)
                    seeds_ok = (
                        seed_rows_available
                        and all(real_pass(row_lookup[(bench, model, seed_id)], mapping) for seed_id in root_seed_ids)
                    )
                    _bump_direction_stats(
                        direction_stats[entropy_direction],
                        row=r,
                        op_type=op_type,
                        bench=bench,
                        model=model,
                        entropy_direction=entropy_direction,
                        root_seed_ids=root_seed_ids,
                        seed_rows_available=seed_rows_available,
                        seeds_ok=seeds_ok,
                        child_passed=child_passed,
                    )
                    if seed_rows_available:
                        _bump_direction_stats(
                            global_direction_stats[(model, entropy_direction)],
                            row=r,
                            op_type=op_type,
                            bench=bench,
                            model=model,
                            entropy_direction=entropy_direction,
                            root_seed_ids=root_seed_ids,
                            seed_rows_available=True,
                            seeds_ok=seeds_ok,
                            child_passed=child_passed,
                        )
                    else:
                        _bump_direction_stats(
                            global_direction_stats[(model, entropy_direction)],
                            row=r,
                            op_type=op_type,
                            bench=bench,
                            model=model,
                            entropy_direction=entropy_direction,
                            root_seed_ids=root_seed_ids,
                            seed_rows_available=False,
                            seeds_ok=False,
                            child_passed=child_passed,
                        )
                        continue
                    if not seeds_ok:
                        continue
                    qualified += 1
                    qualified_cases.append({
                        "problem_id": trt_pid,
                        "entropy_direction": entropy_direction,
                        "root_seed_ids": root_seed_ids,
                        "n_root_seeds": len(root_seed_ids),
                        "child_real_pass": child_passed,
                    })
                    if child_passed:
                        continue
                    eff, elapsed, label = effort_proxy(child_row)
                    immediate_parent_ids = _parent_ids(r)
                    ablation.append({
                        "problem_id": trt_pid,
                        "immediate_parent_ids": immediate_parent_ids,
                        "n_parents": len(immediate_parent_ids),
                        "root_seed_ids": root_seed_ids,
                        "n_root_seeds": len(root_seed_ids),
                        "entropy_direction": entropy_direction,
                        "effort": round(eff, 2),
                        "effort_label": label,
                        "elapsed_s": round(elapsed, 1),
                    })
                rows_out.append({
                    "op_type": op_type,
                    "benchmark": bench,
                    "model": model,
                    "candidates": len(evaluable_cands),
                    "lineage_candidates": len(cands_with_roots),
                    "raw_candidates": len(cands),
                    "unresolved_lineage": unresolved_lineage,
                    "missing_child_evaluation": missing_child_evaluation,
                    "entropy_direction_summary": _finalize_direction_summary(direction_stats),
                    "qualified": qualified,
                    "qualified_cases": qualified_cases,
                    "ablation_cases": ablation,
                    "n_ablation": len(ablation),
                    "lpg_rate": (
                        round(len(ablation) / qualified, 6) if qualified else None
                    ),
                })

    # Pretty print
    print(f"{'op':10} {'bench':12} {'model':22}  {'cands':>5} {'qual':>5}  {'fail':>5}  effort  elapsed")
    print("-" * 90)
    for r in rows_out:
        if r["ablation_cases"]:
            avg_eff = sum(c["effort"] for c in r["ablation_cases"]) / len(r["ablation_cases"])
            avg_el = sum(c["elapsed_s"] for c in r["ablation_cases"]) / len(r["ablation_cases"])
            label = r["ablation_cases"][0]["effort_label"]
            eff_str = f"{avg_eff:.1f} {label}"
            el_str = f"{avg_el:.0f}s"
        else:
            eff_str = "—"
            el_str = "—"
        print(f"{r['op_type']:10} {r['benchmark']:12} {r['model'][:22]:22}  {r['candidates']:>5} {r['qualified']:>5}  {r['n_ablation']:>5}  {eff_str:>20}  {el_str}")

    entropy_direction_rows = []
    for model in models:
        model_stats = {
            direction: global_direction_stats[(model, direction)]
            for direction in DIRECTION_ORDER
            if (model, direction) in global_direction_stats
        }
        for row in _finalize_direction_summary(model_stats):
            row = {"model": model, **row}
            entropy_direction_rows.append(row)

    print(f"\n{'model':22} {'direction':9} {'cands':>5} {'seed-ok':>7} {'LPG fail':>8} {'rescue':>6}")
    print("-" * 66)
    for r in entropy_direction_rows:
        print(
            f"{r['model'][:22]:22} {r['entropy_direction']:9} "
            f"{r['candidates']:>5} {r['seed_solved']:>7} "
            f"{r['lpg_fail']:>8} {r['rescue']:>6}"
        )

    args.output.write_text(json.dumps({
        "metric": "seed_lineage_proof_gap",
        "definition": (
            "For each treatment row, recursively trace parent_id lineage to "
            "upstream benchmark control seeds. Candidates are rows with a "
            "resolved root-seed lineage and an evaluated child result for the "
            "model. Count a row as qualified when all root seeds are solved "
            "under the real-name filter; LPG is the fraction of qualified rows "
            "whose generated child is not solved. The same pass also stratifies "
            "by curation.entropy_direction and counts lineage rescue cases, "
            "where at least one root seed is not solved but the generated child is solved."
        ),
        "rows": rows_out,
        "entropy_direction_rows": entropy_direction_rows,
    }, indent=2))
    print(f"\nwrote → {args.output}")


if __name__ == "__main__":
    main()
