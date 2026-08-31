r"""Recompute the reported Pass@3 figures from the episode records themselves.

The tables and the abstract carry numbers that were correct for the cell as it
stood when they were written, and a cell that is still playing moves under them:
Goedel's \proofb{} treatment read 6.3% at 1545 episodes and 7.0% at 1595. This
recomputes every figure from the records and prints them in the form the tables
use, so the paper is edited from one source rather than from whichever partial
count was on screen. Pass@3 is the row-level indicator that at least one of the
three repeats closed the row, which is what the bootstrap resamples.
"""
import argparse, collections, glob, json, sys
from pathlib import Path

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to `scripts/` itself, which exists, so the import only fails when
    the command is run from somewhere other than the repository root."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


sys.path.insert(0, str(_repo_root()))
from src.evaluation.bootstrap_ci import bootstrap_drop_ci

BENCH = (("minif2f_v2", "miniF2F"), ("proofnet_verified", "ProofNet"))


def _resolve(cell_dir):
    """Cell paths in the config are repo-relative, so they only resolve when the
    command is run from the repository root. Anchor them instead of trusting the
    working directory: the failure otherwise is an empty glob, which reads as a
    cell with no episodes rather than as a path that was never found."""
    if not cell_dir:
        return None
    p = Path(cell_dir)
    return p if p.is_absolute() else _repo_root() / p


def pass3(cell_dir):
    """Row-level Pass@3 indicators, per benchmark."""
    out = collections.defaultdict(list)
    cell_dir = _resolve(cell_dir)
    if not cell_dir:
        return out
    files = [f for f in glob.glob(f"{cell_dir}/episodes_*.jsonl")
             if "before-replay" not in f]
    by_seed = collections.defaultdict(list)
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_seed[row["seed"]].append(row)
    # Sorted, not file order: the bootstrap resamples indices, so the order the
    # episodes happen to sit in decides which draws a seeded generator makes. A
    # replayed cell rewrites that order without changing a single result.
    for seed in sorted(by_seed):
        episodes = by_seed[seed]
        out[episodes[0].get("benchmark")].append(
            1.0 if any(e.get("success") for e in episodes) else 0.0
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_repo_root() / "config/exam_cells.json"))
    ap.add_argument("--models", default="")
    ap.add_argument("--latex", default="",
                    help="emit LaTeX row bodies for a table: `drops` for "
                         "tables/main_drops.tex, `baseline` for "
                         "tables/slm_baseline.tex. A cell with no treatment arm "
                         "prints tbd in the drop columns rather than a blank, so "
                         "an unfinished row is visible in the built PDF.")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # The panel's reporting order. Read it from the config's groups so a cell
    # added or withdrawn mid-campaign cannot leave this list behind -- it once
    # named a withdrawn model and omitted two live ones.
    # Every group read from the config, none of them spelled out here. The
    # middle group was fixed once and the two ends were left literal, so
    # adding a frontier cell put it in --plan and in the run but not in the
    # table -- the same silence, one group over.
    groups = cfg.get("groups") or {}
    order = [m for g in ("lean_provers", "reasoning_slms", "frontier_llms")
             for m in (groups.get(g) or [])]
    wanted = [m for m in (args.models.split(",") if args.models else order) if m]

    for model in wanted:
        c = pass3(cfg["controls"].get(model))
        t = pass3(cfg["treatments"].get(model))
        if not c:
            continue
        print(f"\n{model}")
        for key, label in BENCH:
            ca, ta = c.get(key, []), t.get(key, [])
            if not ca:
                continue
            cm = 100.0 * sum(ca) / len(ca)
            if not ta:
                print(f"  {label:9s} ctrl {cm:5.1f}%  (n={len(ca)})   treatment not run")
                continue
            tm = 100.0 * sum(ta) / len(ta)
            drop, lo, hi = bootstrap_drop_ci(ca, ta)
            print(f"  {label:9s} {cm:5.1f}/{tm:4.1f}  drop {drop:5.1f}  "
                  f"[{lo:.1f},{hi:.1f}]   n={len(ca)}/{len(ta)}")
    if args.latex:
        print(f"\n% --- {args.latex} rows ---")
        emit(args.latex, cfg, wanted)


LABEL = {"bfs": "BFS-V2-7B", "goedel": "Goedel-V2-8B", "pythagoras": "Pythagoras-4B",
         "leanstral": "Leanstral-1.5", "muse": "Muse-Glimmer-30B",
         "qwen3_14b": "Qwen3-14B", "nemotron": "Nemotron-3-nano",
         "qwen36": "Qwen3.6-35B-A3B", "gptoss": "gpt-oss-20b",
         "grok": "Grok-4.6", "luna": r"\texttt{gpt-5.6-luna}",
         "nemotron_nano_9b": "Nemotron-nano-9B",
         "gemini_flash": "gemini-3.7-flash",
         # The paper names the model, not OpenRouter's routing tier: `:free`
         # and `-contributor` are billing routes to one owner-operated
         # endpoint, not different weights. The tier is recorded in
         # config/exam_cells.json and in Appendix G's serving column.
         "muse_spark": "Muse-Spark-1.2",
         "qwen38": "Qwen3.8-27B (withdrawn)"}
TBD = r"\textsc{tbd}"


def emit(kind, cfg, wanted):
    from src.evaluation.bootstrap_ci import bootstrap_drop_ci
    for model in wanted:
        c, t = pass3(cfg["controls"].get(model)), pass3(cfg["treatments"].get(model))
        if model not in LABEL:
            # A cell can be added to the config and never reach the table:
            # emitting nothing for an unknown key is how that happens silently.
            # Say so instead -- adding a model means adding its display name.
            print(f"% !! {model}: no entry in LABEL, add one", file=sys.stderr)
        if not c:
            # No control episodes yet. Still emit the row: a cell that has been
            # declared but not run belongs in the table as `tbd`, and dropping
            # it makes an unrun cell indistinguishable from one that was never
            # part of the panel.
            width = 6 if kind == "drops" else 2
            print(f"  {LABEL.get(model, model)} & " + " & ".join([TBD] * width) + r" \\")
            continue
        cells = []
        for key, _ in BENCH if kind == "drops" else (("minif2f_v2", ""), ("proofnet_verified", "")):
            ca, ta = c.get(key, []), t.get(key, [])
            if kind == "baseline":
                cells.append(f"${100*sum(ca)/len(ca):.1f}$" if ca else TBD)
                continue
            if not ca or not ta:
                cells += [TBD, TBD, TBD]
                continue
            drop, lo, hi = bootstrap_drop_ci(ca, ta)
            cells += [f"{100*sum(ca)/len(ca):.1f}/{100*sum(ta)/len(ta):.1f}",
                      f"{drop:.1f}", f"$[{lo:.1f},{hi:.1f}]$"]
        print(f"  {LABEL.get(model, model)} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
