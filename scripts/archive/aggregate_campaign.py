#!/usr/bin/env python3
"""Aggregate per-benchmark summaries into a single campaign-level report.

Produces:
- ``cells``: flat list of (benchmark, model, control acc, treatment acc, drop,
  CI, Pass@3 drop)
- ``per_benchmark``: avg drop across models
- ``per_model``: avg drop across benchmarks
- LaTeX-ready table fragments under ``latex_tables``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


def _load(paths: List[Path]) -> List[Dict]:
    out: List[Dict] = []
    for p in paths:
        out.extend(json.loads(p.read_text())["cells"])
    return out


def _per_benchmark(cells: List[Dict]) -> List[Dict]:
    by_bench: Dict[str, List[Dict]] = {}
    for cell in cells:
        by_bench.setdefault(cell["benchmark"], []).append(cell)
    out = []
    for bench, group in sorted(by_bench.items()):
        out.append(
            {
                "benchmark": bench,
                "avg_control": mean(c["control_accuracy"] for c in group),
                "avg_treatment": mean(c["treatment_accuracy"] for c in group),
                "avg_drop_pp": mean(c["drop_pp"] for c in group),
                "avg_pass_at_3_drop_pp": mean(c["pass_at_3_drop_pp"] for c in group),
                "rows_control": group[0]["control_rows"],
                "rows_treatment": group[0]["treatment_rows"],
                "models": [c["model"] for c in group],
            }
        )
    return out


def _per_model(cells: List[Dict]) -> List[Dict]:
    by_model: Dict[str, List[Dict]] = {}
    for cell in cells:
        by_model.setdefault(cell["model"], []).append(cell)
    out = []
    for model, group in sorted(by_model.items()):
        out.append(
            {
                "model": model,
                "avg_control": mean(c["control_accuracy"] for c in group),
                "avg_treatment": mean(c["treatment_accuracy"] for c in group),
                "avg_drop_pp": mean(c["drop_pp"] for c in group),
                "avg_pass_at_3_drop_pp": mean(c["pass_at_3_drop_pp"] for c in group),
                "benchmarks": [c["benchmark"] for c in group],
            }
        )
    return out


def _latex_table_per_cell(cells: List[Dict]) -> str:
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Benchmark & Model & Control & Treatment & Drop (pp) & 95\\% CI & P@3 drop & N (ctl/trt) \\\\",
        "\\midrule",
    ]
    for cell in sorted(cells, key=lambda c: (c["benchmark"], c["model"])):
        ci = cell["drop_ci_pp"]
        lines.append(
            f"{cell['benchmark']} & {cell['model']} & "
            f"{cell['control_accuracy']*100:.1f}\\% & "
            f"{cell['treatment_accuracy']*100:.1f}\\% & "
            f"{cell['drop_pp']:+.1f} & "
            f"[{ci[0]:+.1f}, {ci[1]:+.1f}] & "
            f"{cell['pass_at_3_drop_pp']:+.1f} & "
            f"{cell['control_rows']}/{cell['treatment_rows']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    cells = _load(args.summaries)
    report = {
        "cells": cells,
        "per_benchmark": _per_benchmark(cells),
        "per_model": _per_model(cells),
        "latex_tables": {
            "per_cell": _latex_table_per_cell(cells),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
