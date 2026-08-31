#!/usr/bin/env python3
"""Export what each model did on each problem, attempt by attempt.

A Pass@3 number says a model solved a problem or did not. It does not say
whether the model was one identifier away or never produced a proof body at all,
and those are different failures. The episodes record the difference -- each
attempt carries Lean's verdict on it -- so this puts that on the page.

Split per problem rather than inlined. The whole trace set is 19 MB across 749
problems; inlining it would take the working page from 3 MB to 22 MB and make
every reader pay for the one problem they are looking at. One file per problem
is fetched when a row is opened.

What an attempt can carry:

    status    solved | rejected | accepted | sketch_only | error
    message   Lean's verdict -- "Unknown identifier `n`", "All goals closed."
    tokens    what that attempt spent
    lines     proof body lines, so a truncated sketch is visible as one

`solved_code` is the proof that closed the goal, and it is the only attempt text
the episodes keep: a rejected attempt leaves its verdict but not its body. The
page says so rather than implying the body was withheld.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


ROOT = _repo_root()


def slug(seed: str) -> str:
    """A filename for a problem id, short and collision-free.

    Problem ids run past 90 characters and carry `/` in no case but `.` in many;
    a hash keeps the filenames uniform and the directory listable."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=pathlib.Path,
                    default=ROOT / "data/evaluation/exam")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "site/traces")
    ap.add_argument("--max-code-chars", type=int, default=20000,
                    help="truncate a proof longer than this, marking that it was")
    args = ap.parse_args()

    files = [p for p in sorted(args.episodes.glob("*/episodes_*closed_book.jsonl"))
             if "before-replay" not in p.name]
    if not files:
        raise SystemExit(f"no episode files under {args.episodes}")

    by_seed: dict[str, list] = collections.defaultdict(list)
    episodes = 0
    for path in files:
        cell = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episodes += 1
            code = (row.get("solved_code") or "").strip()
            truncated_code = len(code) > args.max_code_chars
            record = {
                "cell": cell,
                "model": row.get("model"),
                "arm": row.get("tag"),
                "attempt": row.get("attempt"),
                "outcome": row.get("outcome"),
                "tokens": row.get("tokens_used"),
                "budget": row.get("token_budget"),
                "seconds": row.get("elapsed_seconds"),
                "log": [
                    {"status": a.get("status"), "message": a.get("message"),
                     "tokens": a.get("tokens"), "lines": a.get("body_lines"),
                     "truncated": bool(a.get("truncated"))}
                    for a in (row.get("attempt_log") or []) if isinstance(a, dict)
                ],
            }
            if code:
                record["code"] = code[: args.max_code_chars]
                record["code_truncated"] = truncated_code
            by_seed[row.get("seed")].append(record)

    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for seed, records in by_seed.items():
        # Group by model so the page can show one block per model rather than a
        # flat list of attempts whose owner has to be read off each row.
        models: dict[str, list] = collections.defaultdict(list)
        for r in records:
            models[r["model"] or "unknown"].append(r)
        for group in models.values():
            group.sort(key=lambda r: (r.get("arm") or "", r.get("attempt") or 0))
        name = slug(seed)
        (args.out / f"{name}.json").write_text(
            json.dumps({"seed": seed, "models": models}, ensure_ascii=False,
                       separators=(",", ":")), encoding="utf-8")
        solved = {m for m, g in models.items() if any(r["outcome"] == "solved" for r in g)}
        index[seed] = {"f": name, "n": len(records),
                       "models": sorted(models), "solved_by": sorted(solved)}

    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    total = sum(p.stat().st_size for p in args.out.glob("*.json"))
    print(f"  {episodes} episodes over {len(by_seed)} problems")
    print(f"  {len(list(args.out.glob('*.json')))} files, {total/1048576:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
