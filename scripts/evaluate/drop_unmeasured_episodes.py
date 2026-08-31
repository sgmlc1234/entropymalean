#!/usr/bin/env python3
"""Remove episodes that measured nothing, so `--resume` will play them again.

`generator_empty` says the serving stack returned no answer. That is not a
result, and leaving the row in place is worse than having no row at all:
`run_seed_exam.py --resume` skips any `(seed, attempt)` already on file, so an
episode recorded as empty is never retried, and the seed's pass@3 is computed
over two real attempts while reporting three. The cell reads as harder than it
is, and nothing in the summary shows why.

So the repair is to delete exactly those lines and re-run the same command with
`--resume`, which replays the holes and nothing else.

Deliberately narrow. Only the two outcomes that mean *nothing was measured* are
dropped — `generator_empty` from the whole-proof player and `sampler_empty` from
the tactic search. Never a `token_budget` or `attempts_exhausted` row, which are
answers, however bad. A tool that could delete failures on request would
eventually be pointed at them.

Read-only unless `--write` is passed; with `--write` the original is kept
alongside as `<name>.before-replay.jsonl` so the discarded rows remain
inspectable.

Usage:
  python3 scripts/evaluate/drop_unmeasured_episodes.py \
    data/evaluation/exam/treatment_luna/episodes_luna_closed_book.jsonl --write
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DROP = {"generator_empty", "sampler_empty", "verifier_error"}

# An attempt Lean actually judged. `no_code`, `error` and `over_budget` are the
# model's or the budget's doing and are answers; these four are the verifier
# having seen something.
JUDGED = {"rejected", "accepted", "solved", "sketch_only"}


def unjudged_after_transport_failure(row: dict) -> bool:
    """The generator failed and Lean never judged one attempt of this episode.

    The harness files such an episode under `attempts_exhausted`, which reads
    as a model that tried and failed, so the outcome filter above never sees
    it. The pair of conditions is what makes this safe to delete: an episode
    that reached the verifier even once is an answer and is left alone, and an
    episode with no transport failure recorded is left alone whatever its
    outcome.
    """
    if not (row.get("generator_empty") or row.get("generator_error")):
        return False
    log = row.get("attempt_log")
    if log is None:                       # tactic-step player: records `steps`
        return False
    return not any(a.get("status") in JUDGED for a in log)


def backup_path(path: Path) -> Path:
    """A backup name that never overwrites an earlier one."""
    stem = path.with_suffix("")
    candidate = stem.with_suffix(".before-replay.jsonl")
    n = 2
    while candidate.exists():
        candidate = stem.with_suffix(f".before-replay{n}.jsonl")
        n += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--also-unjudged", action="store_true",
        help=(
            "additionally drop episodes that recorded a generator failure and "
            "whose every attempt went unjudged by Lean. These are filed as "
            "`attempts_exhausted`, so the outcome filter misses them, but "
            "nothing was measured in them either"
        ))
    parser.add_argument(
        "--desync-against", type=Path, default=None,
        help=(
            "an exam-rows file the pre-flight accepted. With it, "
            "`statement_invalid` episodes whose row appears there are dropped "
            "too: a row that elaborated during pre-flight did not become "
            "unplayable, so that verdict came from another problem. Without "
            "it, `statement_invalid` is never dropped, because replaying a "
            "genuinely unelaborable row only reproduces the same refusal"
        ))
    args = parser.parse_args()

    playable = set()
    if args.desync_against and args.desync_against.is_file():
        playable = {json.loads(l)["name"]
                    for l in args.desync_against.read_text(encoding="utf-8").splitlines()
                    if l.strip()}

    for path in args.files:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        def unmeasured(r):
            if r.get("outcome") in DROP:
                return True
            if args.also_unjudged and unjudged_after_transport_failure(r):
                return True
            return (r.get("outcome") == "statement_invalid"
                    and r.get("seed") in playable)
        keep = [r for r in rows if not unmeasured(r)]
        dropped = [r for r in rows if unmeasured(r)]
        if not dropped:
            print(f"{path.name}: {len(rows)} episodes, none unmeasured")
            continue

        by_attempt = collections.Counter(r.get("attempt") for r in dropped)
        print(f"{path.name}: dropping {len(dropped)} of {len(rows)} — {dict(sorted(by_attempt.items()))}")
        for row in dropped[:10]:
            print(f"    a{row.get('attempt')} {str(row.get('seed'))[:52]}")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped) - 10} more")

        if args.write:
            backup = backup_path(path)
            path.rename(backup)
            with path.open("w", encoding="utf-8") as handle:
                for row in keep:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    wrote {len(keep)} episodes; original kept as {backup.name}")
            print("    now re-run the cell's original command with --resume")

    if not args.write:
        print("\n(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
