#!/usr/bin/env python3
"""Find episodes that record an environment failure as a model failure.

A cell's summary reports pass@k and nothing about whether the run was healthy
while it was measured. It cannot: a generator that stops answering produces
episodes in the same shape as a model that cannot prove the theorem, and the
summary counts them the same way. One 342-episode run lost 266 episodes that
way — the serving stack died at episode 76 and the run played to completion,
filing every subsequent episode as `attempts_exhausted`.

Three signatures, none of which need the run's logs:

`dead_generator`   a run of consecutive episodes that generated no tokens.
                   A live model occasionally returns nothing; a dead one
                   returns nothing until the run ends, so the tail matters
                   more than the count.
`no_token_episodes` the same thing scattered rather than contiguous — worth
                   seeing even when it never becomes a run.
`speed_cliff`      the median episode duration collapsing partway through.
                   Catches a generator that answers with something cheap and
                   useless rather than with nothing at all, which no
                   token-count test would see.

Read-only. Prints a table and, with `--json`, writes the findings.

Usage:
  python3 scripts/analysis/audit_exam_episodes.py
  python3 scripts/analysis/audit_exam_episodes.py --root data/evaluation/exam --json /tmp/audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

#: Consecutive zero-token episodes that stop looking like bad luck. Matches
#: `EMPTY_RUN_ABORT` in `run_seed_exam.py`, which now ends a run at this count.
DEAD_RUN = 3

#: How much slower the first half has to be before the drop is called a cliff.
#: A generator that keeps answering varies by a factor of two or so across a
#: seed set; an order of magnitude is not variance.
CLIFF_RATIO = 8.0


def episode_rows(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def counts_tokens(rows: List[Dict[str, Any]]) -> bool:
    """Whether this file's token counts mean anything.

    Not every cell records them. The codex-driven player never returns a token
    count, so all 300 of its episodes read as zero-token — including 26 that
    spent six minutes and four attempts on a real proof. Older runs predate the
    field entirely. Testing tokens on those files does not find a dead
    generator, it invents one, so the test only runs where at least one episode
    proves the field is wired.
    """
    return any(int(r.get("tokens_used") or 0) > 0 for r in rows)


def longest_zero_token_tail(rows: List[Dict[str, Any]]) -> int:
    """Length of the zero-token run that reaches the end of the file.

    A dead generator is not merely a long run of empty episodes, it is one that
    is still going when the file stops. A run in the middle that recovers is a
    hiccup and is reported separately.
    """
    tail = 0
    for row in reversed(rows):
        if int(row.get("tokens_used") or 0) == 0 and not row.get("success"):
            tail += 1
        else:
            break
    return tail


def speed_cliff(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Median seconds before and after the file's midpoint, if they diverge."""
    if len(rows) < 20:
        return None
    half = len(rows) // 2
    early = [r.get("elapsed_seconds") or 0 for r in rows[:half]]
    late = [r.get("elapsed_seconds") or 0 for r in rows[half:]]
    a, b = statistics.median(early), statistics.median(late)
    if b <= 0 or a <= 0 or a / b < CLIFF_RATIO:
        return None
    return {"median_first_half": round(a, 1), "median_second_half": round(b, 1)}


def audit(path: Path) -> Dict[str, Any]:
    rows = episode_rows(path)
    if not rows:
        return {"file": str(path), "episodes": 0, "findings": []}
    tokens_wired = counts_tokens(rows)
    zero = (
        [r for r in rows if int(r.get("tokens_used") or 0) == 0 and not r.get("success")]
        if tokens_wired else []
    )
    tail = longest_zero_token_tail(rows) if tokens_wired else 0
    cliff = speed_cliff(rows)
    findings = []
    if tail >= DEAD_RUN:
        findings.append(
            {
                "kind": "dead_generator",
                "episodes_lost": tail,
                "share": round(tail / len(rows), 3),
                "detail": (
                    f"the last {tail} of {len(rows)} episodes generated no tokens; "
                    f"they record the serving stack, not the model"
                ),
            }
        )
    scattered = len(zero) - tail
    if scattered > 0:
        findings.append(
            {
                "kind": "no_token_episodes",
                "episodes": scattered,
                "share": round(scattered / len(rows), 3),
                "detail": "zero-token episodes outside the trailing run",
            }
        )
    if cliff:
        findings.append({"kind": "speed_cliff", **cliff,
                         "detail": "episode duration collapsed partway through"})
    return {
        "file": str(path),
        "episodes": len(rows),
        "model": rows[0].get("model") or rows[0].get("player"),
        "arm": rows[0].get("arm"),
        # Recorded so a clean report is not mistaken for a checked one: where
        # tokens are not wired, only the timing test ran.
        "token_test_ran": tokens_wired,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/evaluation/exam"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    reports = [
        audit(Path(p))
        for p in sorted(glob.glob(str(args.root / "**" / "episodes_*.jsonl"), recursive=True))
    ]
    flagged = [r for r in reports if r["findings"]]

    for report in reports:
        mark = "!!" if any(f["kind"] == "dead_generator" for f in report["findings"]) else \
               " ?" if report["findings"] else "  "
        name = report["file"].replace(str(args.root) + "/", "")
        partial = "" if report.get("token_test_ran") else "  (timing test only — no token counts)"
        print(f"{mark} {name:62s} {report['episodes']:5d} ep{partial}")
        for finding in report["findings"]:
            print(f"     - {finding['kind']}: {finding['detail']}")

    print(
        f"\n{len(reports)} files, {len(flagged)} with findings, "
        f"{sum(1 for r in flagged if any(f['kind'] == 'dead_generator' for f in r['findings']))} "
        f"with a dead generator."
    )
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
