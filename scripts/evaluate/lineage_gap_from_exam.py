r"""Compute the lineage proof gap (LPG) from the exam cells themselves.

The figures in the paper's LPG table were produced by an earlier pipeline that
read `{bench}_control.csv` and `{bench}_treatment.jsonl` out of a scratch
directory. That directory does not survive a reboot, and the panel has since
moved to per-cell episode files under `data/evaluation/exam/`, so the table
could no longer be regenerated from the data it describes. This reads the cells
directly, which is also what lets a cell that is still playing be re-read as it
grows -- Goedel's treatment arm went from 1,362 to 1,605 episodes after the
table was last written.

For treatment row j: A_j is the set of control seeds its lineage reaches, S_j is
"the model solved every seed in A_j at Pass@3", R_j is "the model solved j".
On Q = {j : A_j non-empty and S_j},

    LPG = |{j in Q : not R_j}| / |Q|

with a Wilson 95% interval. Rescue is the complementary diagnostic, not S_j and
R_j, reported as a count over the rows whose lineage the model could not fully
prove. Conditioning on S_j is the whole point: a model cannot register a gap on
a lineage it could not prove in the first place, so low capability cannot
inflate the number.

Lineage comes from the recorded id convention, parsed by the resolver in
analyze_parent_child_ablation.py rather than a second copy of it here.
"""
import argparse, collections, glob, importlib.util, json, math
from pathlib import Path

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


ROOT = _repo_root()
BENCH = (("proofnet_verified", "ProofNet"), ("minif2f_v2", "miniF2F"))


def _resolver():
    spec = importlib.util.spec_from_file_location(
        "pca", ROOT / "scripts/evaluate/analyze_parent_child_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._infer_synthetic_seed_roots


def wilson(k, n, z=1.96):
    """Wilson score interval, in percent. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - half) / d, 100 * (c + half) / d)


def _resolve(cell_dir):
    """Cell paths in the config are repo-relative, so they only resolve when the
    command is run from the repository root. Anchor them instead of trusting the
    working directory: the failure otherwise is an empty glob, which reads as a
    cell with no episodes rather than as a path that was never found."""
    if not cell_dir:
        return None
    p = Path(cell_dir)
    return p if p.is_absolute() else ROOT / p


def pass3(cell_dir):
    """{problem_id: bool} -- did any of the repeats close this row?"""
    out = {}
    cell_dir = _resolve(cell_dir)
    if not cell_dir:
        return out
    by_seed = collections.defaultdict(list)
    for path in glob.glob(f"{cell_dir}/episodes_*.jsonl"):
        if "before-replay" in path:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_seed[row["seed"]].append(row)
    for seed, episodes in by_seed.items():
        out[seed] = any(e.get("success") for e in episodes)
    return out


def seed_key(name):
    """Episode seeds carry the row name; the ledger keys on problem_id."""
    return name


def newcombe_diff(k1, n1, k2, n2, z=1.96):
    """Wilson-based interval for p1 - p2, in percentage points.

    Newcombe's method 10: build each proportion's Wilson interval and combine
    the two, which keeps coverage when either rate is near 0 or 1. A normal
    approximation on the difference does not, and both LPG rates here sit far
    from a half.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0)
    def wil(k, n):
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((c - h) / d, (c + h) / d)
    l1, u1 = wil(k1, n1)
    l2, u2 = wil(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    # No further z here: the Wilson bounds already carry it, and multiplying
    # again widened ProofNet's contrast from [5.5,31.9] to [-7.5,44.4], turning
    # a separation into a non-result.
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (100 * lo, 100 * hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config/exam_cells.json"))
    ap.add_argument("--seeds", default=str(ROOT / "data/evaluation/exam/seeds_all100.jsonl"))
    ap.add_argument("--release", default=str(ROOT / "data/evaluation/exam/release537_playable.jsonl"))
    ap.add_argument("--models", default="",
                    help="comma-separated labels; empty means the whole panel in "
                         "reporting order, read from the config's groups")
    ap.add_argument("--latex", action="store_true", help="emit the table body rows")
    ap.add_argument("--contrast", default="",
                    help="two model labels, `a,b`: report LPG(a) - LPG(b) with a "
                         "Newcombe interval, which is the between-model comparison "
                         "the paper makes on ProofNet")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    roots_of = _resolver()

    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    control_ids = collections.defaultdict(set)
    seed_name = {}
    for s in seeds:
        pid = s.get("problem_id") or s["name"]
        control_ids[s["benchmark"]].add(pid)
        seed_name[(s["benchmark"], pid)] = s["name"]

    release = [json.loads(l) for l in open(args.release, encoding="utf-8") if l.strip()]

    table = {}
    # Every group read from the config, none of them spelled out here. The
    # middle group was fixed once and the two ends were left literal, so
    # adding a frontier cell put it in --plan and in the run but not in the
    # table -- the same silence, one group over.
    groups = cfg.get("groups") or {}
    order = [m for g in ("lean_provers", "reasoning_slms", "frontier_llms")
             for m in (groups.get(g) or [])]
    for model in (args.models.split(",") if args.models else order):
        model = model.strip()
        if not model:
            continue
        ctrl = pass3(cfg["controls"].get(model))
        trt = pass3(cfg["treatments"].get(model))
        if not ctrl or not trt:
            print(f"\n{model}: control {len(ctrl)} rows, treatment {len(trt)} rows -- skipped")
            continue
        print(f"\n{model}")
        for key, label in BENCH:
            q = fail = rescue = unqualified = 0
            for row in release:
                if row["benchmark"] != key:
                    continue
                if row["name"] not in trt:
                    continue
                roots = roots_of(row["problem_id"], control_ids[key])
                if not roots:
                    continue
                names = [seed_name.get((key, r)) for r in roots]
                if any(n is None or n not in ctrl for n in names):
                    continue
                solved_all_roots = all(ctrl[n] for n in names)
                child = trt[row["name"]]
                if solved_all_roots:
                    q += 1
                    if not child:
                        fail += 1
                else:
                    unqualified += 1
                    if child:
                        rescue += 1
            if q == 0:
                print(f"  {label:9s} Q=0")
                continue
            lo, hi = wilson(fail, q)
            print(f"  {label:9s} Q/Fail {q}/{fail}  LPG {100*fail/q:.1f}%  "
                  f"[{lo:.1f},{hi:.1f}]  rescue {rescue}/{unqualified}")
            table[(model, key)] = (q, fail, rescue, unqualified)


    if args.contrast:
        a, b = args.contrast.split(",")
        print(f"\n{a} minus {b}")
        for key, label in BENCH:
            if (a, key) not in table or (b, key) not in table:
                continue
            qa, fa, *_ = table[(a, key)]
            qb, fb, *_ = table[(b, key)]
            lo, hi = newcombe_diff(fa, qa, fb, qb)
            print(f"  {label:9s} {100*fa/qa - 100*fb/qb:+.1f}pp  [{lo:.1f},{hi:.1f}]")


if __name__ == "__main__":
    main()
