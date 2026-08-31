#!/usr/bin/env python3
"""Gen-0 proof completion over a whole seed CSV, independent of pool size.

`run_pool_generation.py` completes proofs only for the seeds that make up the
initial pool, so asking it for proofs over a 50-row seed file with a pool of 5
silently returns 5. This runs the same completion over every row and writes the
seeds back with their proofs, which is what a seed file needs before it can be
bred from.

Usage:
  set -a; source .env; set +a
  GENERATION_PROVIDER=codex_cli GENERATION_MODEL=gpt-5.6-luna \
  python scripts/analysis/complete_seed_proofs.py \
    --input data/benchmarks/minif2f_v2/raw/seeds_50.csv \
    --output data/benchmarks/minif2f_v2/raw/seeds_50_proved.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolved one level short after the
    move -- to a directory that exists, so nothing raised."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.certifier import CertificationInput  # noqa: E402
from src.certification.generation import default_generation_config  # noqa: E402
from src.evaluation.lean_repl_verifier import (  # noqa: E402
    close_global_repl_verifier,
)
from src.orchestration.pool_generation import (  # noqa: E402
    _complete_generation_zero_proofs,
    _generation_zero_summary,
)

_SORRY = re.compile(r"(?<![A-Za-z_])(sorry|admit)(?![A-Za-z_])")
# `native_decide` discharges a goal by running compiled code, which pulls
# `Lean.ofReduceBool` into the axiom closure and so can never earn
# `proof_checked` under our allowlist. Accepting one here yields a seed that
# cannot be certified and whose every mutation inherits the same shortcut —
# four of them reached a finished seed set before this gate existed.
_NATIVE = re.compile(r"(?<![A-Za-z_])native_decide(?![A-Za-z_])")


def uncertifiable(code: str) -> str:
    """Why this proof can never be certified, or empty if it can."""
    if not code.strip():
        return "empty"
    if _SORRY.search(code):
        return "contains sorry/admit"
    if _NATIVE.search(code):
        return "uses native_decide (Lean.ofReduceBool)"
    return ""


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-k", type=int, default=6)
    parser.add_argument("--proof-turns", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="seeds per reported batch; bounds what one wedged run costs",
    )
    parser.add_argument("--max-seed-seconds", type=float, default=900.0)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep proofs already present in --output and only attempt the rest",
    )
    return parser.parse_args()


def load_seeds(path: Path) -> List[Dict[str, Any]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


async def _run() -> None:
    args = _parse()
    rows = load_seeds(args.input)
    proved: Dict[str, str] = {}
    if args.resume and args.output.is_file():
        for row in load_seeds(args.output):
            code = (row.get("lean_code") or "").strip()
            if not uncertifiable(code):
                proved[str(row["id"])] = code
        print(f"resuming: {len(proved)} already proved")

    todo = [r for r in rows if str(r["id"]) not in proved]
    print(f"seeds: {len(rows)} · attempting {len(todo)}")

    pool = [
        CertificationInput(
            id=str(row["id"]),
            statement=str(row.get("statement") or ""),
            answer=str(row.get("answer") or ""),
            metadata={
                "formal_statement": str(row.get("formal_statement") or ""),
                "lean_header": str(row.get("lean_header") or ""),
                "informal_proof": str(row.get("informal_proof") or ""),
                "split": str(row.get("split") or ""),
                "problem_style": "theorem_proof",
            },
        )
        for row in todo
    ]

    config = default_generation_config(args.model, None)
    import os as _os
    import time as _time
    print(
        f"model: {config.model} · verifier: {_os.getenv('LEAN_VERIFIER', 'file')} "
        f"· K={args.proof_k} T={args.proof_turns} parallel={args.max_parallel} "
        f"cap={args.max_seed_seconds:.0f}s/seed batch={args.batch_size}",
        flush=True,
    )

    # Run in batches rather than one 49-wide call. The underlying routine only
    # reports when the whole pool is done, so a single call is a black box for
    # however long it takes — and when it wedges (concurrent subprocess spawns
    # can leave the loop waiting on a child that already exited) there is no
    # way to tell a hang from slow progress. Batching bounds what a wedge costs
    # to one batch, and prints often enough to see the difference.
    completed: List[Any] = []
    started = _time.monotonic()
    for offset in range(0, len(pool), args.batch_size):
        batch = pool[offset : offset + args.batch_size]
        batch_started = _time.monotonic()
        # A wall-clock ceiling over the whole batch, on top of the per-seed cap
        # inside. The inner cap guards a slow proof; it does not fire when the
        # event loop itself stops making progress, which is what happens when
        # several codex subprocesses are spawned concurrently — the children
        # exit, the parent keeps waiting, and the run sits at 0% CPU forever.
        # Twice the theoretical worst case, so only a wedge trips it.
        ceiling = args.max_seed_seconds * (len(batch) / max(1, args.max_parallel) + 1) * 2
        try:
            done = await asyncio.wait_for(
                _complete_generation_zero_proofs(
                    batch,
                    config=config,
                    proof_k=args.proof_k,
                    proof_turns=args.proof_turns,
                    max_seed_seconds=args.max_seed_seconds,
                    max_parallel=args.max_parallel,
                ),
                timeout=ceiling,
            )
        except asyncio.TimeoutError:
            print(
                f"  batch wedged past {ceiling:.0f}s ceiling — abandoning it and "
                f"continuing; these seeds stay unproved and can be retried",
                flush=True,
            )
            done = batch
        except Exception as error:
            print(f"  batch failed: {type(error).__name__}: {error}", flush=True)
            done = batch
        completed.extend(done)
        won = sum(
            1
            for item in done
            if not uncertifiable(str((item.metadata or {}).get("lean_code") or ""))
        )
        print(
            f"[{offset + len(batch):3d}/{len(pool)}] +{won}/{len(batch)} proved "
            f"· batch {_time.monotonic() - batch_started:.0f}s "
            f"· total {(_time.monotonic() - started) / 60:.1f}m",
            flush=True,
        )
    await close_global_repl_verifier()

    summary = _generation_zero_summary(pool, completed)
    failures: List[Dict[str, Any]] = []
    for item in completed:
        meta = item.metadata or {}
        code = str(meta.get("lean_code") or "").strip()
        blocker = uncertifiable(code)
        if not blocker:
            proved[item.id] = code
            continue
        if blocker.startswith("uses native_decide"):
            failures.append(
                {"seed_id": item.id, "reason": blocker, "class": "uncertifiable_proof"}
            )
            continue
        # A run that proves nothing has to say why: the summary counts failures
        # but keeps the reason in a nested packet the scalar report drops, which
        # is how a crash and an honest no-proof looked identical from outside.
        packet = meta.get("gen0_failure_packet") or {}
        failures.append(
            {
                "seed_id": item.id,
                "reason": str(
                    packet.get("failure_reason")
                    or packet.get("gen0_exception")
                    or packet.get("failure_signature")
                    or meta.get("gen0_status")
                    or "unknown"
                )[:400],
                "class": str(packet.get("failure_class") or "")[:120],
            }
        )
    if failures:
        print(f"\nfailed {len(failures)} seed(s); first reasons:", flush=True)
        for failure in failures[:5]:
            print(f"  {failure['seed_id']}: [{failure['class']}] {failure['reason']}", flush=True)

    fields = list(rows[0].keys())
    for extra in ("lean_code", "gen0_proof_completed"):
        if extra not in fields:
            fields.append(extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            code = proved.get(str(row["id"]), "")
            writer.writerow(
                {**row, "lean_code": code, "gen0_proof_completed": bool(code)}
            )

    report = {
        "input": str(args.input),
        "seeds": len(rows),
        "proved": len(proved),
        "rate": round(len(proved) / max(1, len(rows)), 3),
        "model": config.model,
        "proof_k": args.proof_k,
        "proof_turns": args.proof_turns,
        "gen0_summary": {
            k: v for k, v in summary.items() if not isinstance(v, (list, dict))
        },
        "failures": failures,
        "unproved": [str(r["id"]) for r in rows if str(r["id"]) not in proved],
    }
    (args.output.parent / f"{args.output.stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in report.items() if k not in ("unproved", "failures")}, indent=2
    ))
    print(f"proved {len(proved)}/{len(rows)} -> {args.output}")


if __name__ == "__main__":
    asyncio.run(_run())
