#!/usr/bin/env python3
"""Apply exact-theorem-name hallucination filter to a campaign output and
recompute Pass@K.

Why: Goedel-Prover-V2 (and any whole-proof model) sometimes returns a
syntactically valid `theorem foo := by ...` that uses a *different*
theorem name than the one we asked it to prove. Lean happily compiles
the unrelated theorem, the verifier reports `complete=True`, and the
campaign records a false PASS. The filter is used for the main-results
miniF2F and ProofNet campaign outputs.

Filter rule: a row counts as a REAL pass iff at least one successful
attempt's `final_proof` contains the exact substring `theorem <T>`
where `T` is the theorem name we wanted. Anything else is hallucination
(re-tagged FAIL).

Outputs a side-by-side raw-vs-real summary per (benchmark, model, arm)
and writes a JSON next to the campaign dir for downstream tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
MAIN_BENCHMARKS = ("minif2f", "proofnet")


def extract_theorem_name(formal: str) -> str:
    if not formal:
        return ""
    m = re.search(r"\btheorem\s+(\w+)", formal)
    return m.group(1) if m else ""


def build_id_to_name(input_dir: Path) -> Dict[Tuple[str, str], str]:
    """(benchmark, problem_id) → target theorem name, scanned from the
    per-benchmark control CSV and treatment JSONL files used at launch."""
    mapping: Dict[Tuple[str, str], str] = {}
    for bench in MAIN_BENCHMARKS:
        csv_path = input_dir / f"{bench}_control.csv"
        if csv_path.exists():
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    pid = row.get("id") or row.get("problem_id")
                    fs = row.get("formal_statement") or row.get("formal_prefix")
                    name = extract_theorem_name(fs)
                    if pid and name:
                        mapping[(bench, pid)] = name
        jsonl_path = input_dir / f"{bench}_treatment.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r.get("eval_problem_id") or r.get("problem_id")
                fs = r.get("formal_statement") or r.get("lean_code")
                name = extract_theorem_name(fs)
                if pid and name:
                    mapping[(bench, pid)] = name
    return mapping


def real_pass(row: dict, target_name: str) -> bool:
    """A row truly passes iff at least one successful attempt's
    final_proof matches the target theorem name."""
    if not row.get("pass_at_k"):
        return False
    if not target_name:
        return row["pass_at_k"]  # no target → fall back to raw (rare)
    needle = f"theorem {target_name}"
    for a in row.get("attempts", []):
        if not a.get("success"):
            continue
        proof = a.get("final_proof") or ""
        if needle in proof:
            return True
    return False


def summarize(campaign_dir: Path, mapping: Dict[Tuple[str, str], str]) -> dict:
    """Returns per-(benchmark, model, arm) raw vs real Pass@K counts."""
    cells: dict = defaultdict(lambda: {"n": 0, "raw_pass": 0, "real_pass": 0,
                                       "hallucinations": []})
    bench_map = {"minif2f": "miniF2F", "proofnet": "proofnet"}
    for bench_lower, bench_label in bench_map.items():
        path = campaign_dir / f"{bench_lower}_proof.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            target = mapping.get((bench_lower, row["problem_id"]), "")
            key = (bench_label, row["model"], row["arm"])
            c = cells[key]
            c["n"] += 1
            if row["pass_at_k"]:
                c["raw_pass"] += 1
                if real_pass(row, target):
                    c["real_pass"] += 1
                else:
                    c["hallucinations"].append({
                        "problem_id": row["problem_id"],
                        "target_name": target,
                    })
    return cells


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "  -  "
    return f"{num/den*100:5.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path,
                   default=ROOT / "data/evaluation/campaign_2026-05-20-Q8")
    p.add_argument("--input-dir", type=Path, default=Path("/tmp/eml_campaign"))
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path (default: campaign_dir/hallucination_filter.json)")
    args = p.parse_args()

    mapping = build_id_to_name(args.input_dir)
    print(f"problem_id → theorem_name mapping: {len(mapping)} entries\n")

    cells = summarize(args.campaign_dir, mapping)

    # Pretty table
    print(f"{'benchmark':12} {'model':22} {'arm':10} {'N':>4}  "
          f"{'raw P@K':>8} {'real P@K':>9}  {'halluc':>7}")
    print("-" * 80)
    by_arm = sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    out_rows = []
    for (bench, model, arm), c in by_arm:
        h = len(c["hallucinations"])
        line = (f"{bench:12} {model[:22]:22} {arm:10} {c['n']:>4}  "
                f"{fmt_pct(c['raw_pass'], c['n']):>8} "
                f"{fmt_pct(c['real_pass'], c['n']):>9}  "
                f"{h:>3}/{c['raw_pass']:<3}")
        print(line)
        out_rows.append({
            "benchmark": bench,
            "model": model,
            "arm": arm,
            "n": c["n"],
            "raw_pass": c["raw_pass"],
            "real_pass": c["real_pass"],
            "hallucinations": c["hallucinations"],
        })

    # Drops (control vs treatment) using real pass
    print(f"\n{'='*80}\nDrop control − treatment (real pass)\n{'='*80}")
    by_model_bench = defaultdict(dict)
    for (bench, model, arm), c in cells.items():
        by_model_bench[(bench, model)][arm] = (c["real_pass"], c["n"])
    for (bench, model), arms in by_model_bench.items():
        if "control" in arms and "treatment" in arms:
            cp, cn = arms["control"]
            tp, tn = arms["treatment"]
            cr = cp / cn * 100 if cn else 0
            tr = tp / tn * 100 if tn else 0
            print(f"  {bench:12} {model[:22]:22}  ctrl={cr:5.1f}%  trt={tr:5.1f}%  "
                  f"drop={cr-tr:+5.1f}pp")

    output_path = args.output or (args.campaign_dir / "hallucination_filter.json")
    output_path.write_text(json.dumps({"rows": out_rows}, indent=2))
    print(f"\nwrote → {output_path}")


if __name__ == "__main__":
    main()
