"""Check every certified crossover for a parent it did not need.

One-sided by construction: a parent is reported redundant only when Lean has
accepted a proof that does without it. Failing to find such a proof clears
nothing, so the scan can discard rows without discarding anything it merely
failed to understand.
"""

from __future__ import annotations

import argparse, asyncio, csv, glob, json, os, sys
from pathlib import Path

sys.path.insert(0, ".")
from src.certification.redundancy import check_parents, brief
from src.evaluation.lean_verifier import verify_lean_proof
from src.utils.codex_cli import call_codex_cli


def load(root: Path):
    rows = []
    for path in sorted(glob.glob(str(root / "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                row["_g"] = Path(path).stem
                rows.append(row)
    return rows


async def prover(system: str, user: str) -> str:
    reply = await call_codex_cli(
        model=os.getenv("REDUNDANCY_PROVER_MODEL", "gpt-5.6-terra"),
        system=system, user=user, timeout_seconds=300,
    )
    return reply.raw_text


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/certified/run-a"))
    ap.add_argument("--output", type=Path, default=Path("data/cache/crossover_redundancy.json"))
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    rows = load(args.root)
    index = {}
    for row in rows:
        pid = row.get("problem_id")
        if pid and (pid not in index or row.get("status") == "certified"):
            index[pid] = row
    for path in glob.glob("data/certified/run-a/seeds/*.csv"):
        with open(path, encoding="utf-8") as handle:
            for seed in csv.DictReader(handle):
                key = seed.get("id") or seed.get("problem_id")
                if key and key not in index:
                    index[key] = seed

    targets = [r for r in rows if r.get("op_type") == "crossover" and r.get("status") == "certified"]
    # The two rows already shown by hand to lean on one parent go first: if the
    # scan cannot reproduce those, it is not working and the rest is noise.
    known = {("minif2f_g06", 5, 4), ("minif2f_g06", 3, 5)}
    targets.sort(key=lambda r: (r["_g"], r.get("generation"), r.get("slot")) not in known)

    out = []
    for i, row in enumerate(targets, 1):
        pack = [
            {"name": str(p), "statement": str((index.get(str(p)) or {}).get("formal_statement") or "")}
            for p in (row.get("parent_ids") or [])
        ]
        evidence = await check_parents(
            verify_lean_proof,
            row.get("lean_header") or "import Mathlib",
            row.get("formal_statement") or "",
            pack, prover=prover, timeout=args.timeout,
        )
        rec = {
            "problem_id": row.get("problem_id"), "group": row["_g"],
            "generation": row.get("generation"), "slot": row.get("slot"),
            "judge_quality": ((row.get("quality_evidence") or {}).get("judge") or {}).get("quality"),
            **evidence, "brief": brief(evidence),
        }
        out.append(rec)
        mark = "REDUNDANT" if rec["redundant"] else "ok       "
        print(f"[{i}/{len(targets)}] {mark} {rec['group']} gen{rec['generation']}s{rec['slot']} "
              f"judge={rec['judge_quality']}  {rec['brief'][:90]}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    bad = [r for r in out if r["redundant"]]
    print(f"\nscanned {len(out)}  redundant {len(bad)} ({len(bad)/max(1,len(out)):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
