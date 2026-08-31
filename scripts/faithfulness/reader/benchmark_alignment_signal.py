#!/usr/bin/env python3
"""Score the goal_roundtrip alignment signal against ProofNet-Verified labels.

An et al. ("Ground False", AI4Math @ ICML 2026) audited all 367 ProofNet
formalizations and, for every item they call unfaithful, either proved the
negation in Lean or had a human adjudicate a unanimous three-judge verdict.
That gives something rare: ground truth for a faithfulness detector that was
NOT produced by the detector's own family of methods.

This script replays our automated signal over the same items and reports
precision/recall against those labels. The task is binary — does the
formalization faithfully encode the natural-language claim? — with the
audit's ``faithful`` as the negative class and
``stronger``/``weaker``/``incomparable`` as the positive (unfaithful) class.
Items the audit blames on the benchmark itself (``nl_ambiguous``,
``nl_wrong``) are excluded: there is no correct formalization to detect.

Default lineage is the original ProofNet ported to Lean 4.28 (194/367
unfaithful — a near-balanced task); ``--lineage sharp`` scores the cleaner
ProofNet# instead.

Usage:
  set -a; source .env; set +a
  GENERATION_PROVIDER=codex_cli GENERATION_MODEL=gpt-5.5 \
  python scripts/faithfulness/reader/benchmark_alignment_signal.py --limit 60 \
    --output data/evaluation/alignment_benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.alignment import elaborated_goal_alignment  # noqa: E402
from src.certification.generation import default_generation_config  # noqa: E402

PNV_ROOT = REPO_ROOT / "references" / "ProofNet-Verified"

UNFAITHFUL = {"stronger", "weaker", "incomparable"}
EXCLUDED = {"nl_ambiguous", "nl_wrong"}

LINEAGES = {
    "original": (
        PNV_ROOT / "data" / "proofnet" / "proofnet-4.28.jsonl",
        PNV_ROOT / "error_taxonomy" / "proofnet" / "results.jsonl",
    ),
    "sharp": (
        PNV_ROOT / "data" / "proofnet-verified.jsonl",
        PNV_ROOT / "error_taxonomy" / "proofnet#" / "results.jsonl",
    ),
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", choices=sorted(LINEAGES), default="original")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/evaluation/alignment_benchmark"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="0 = all items; else stratified sample"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--lean-timeout", type=float, default=240.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip items already present in the output predictions file",
    )
    return parser.parse_args()


def load_items(lineage: str) -> List[Dict[str, Any]]:
    data_path, tax_path = LINEAGES[lineage]
    rows = {
        f"proofnet-{row['index']}": row
        for row in (
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    taxonomy = {
        record["stem"]: record
        for record in (
            json.loads(line)
            for line in tax_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    items = []
    for stem, row in rows.items():
        audit = taxonomy.get(stem)
        if not audit:
            continue
        verdict = str(audit.get("q2_faithfulness"))
        if verdict in EXCLUDED:
            continue
        header = str(row.get("header") or "import Mathlib")
        helper = str(row.get("helper") or "").strip()
        items.append(
            {
                "stem": stem,
                "name": row["name"],
                "statement_nl": str(row.get("informal_stmt") or ""),
                "formal_statement": str(row.get("formal_stmt") or ""),
                "lean_header": f"{header}\n\n{helper}" if helper else header,
                "label_faithfulness": verdict,
                "label_provable": str(audit.get("q3_provability")),
                "label_error_type": str(audit.get("q4_error_type")),
                "label_unfaithful": verdict in UNFAITHFUL,
            }
        )
    return items


def stratified_sample(items, limit, seed):
    if limit <= 0 or limit >= len(items):
        return items
    buckets = defaultdict(list)
    for item in items:
        buckets[item["label_faithfulness"]].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked, index = [], 0
    while len(picked) < limit:
        added = False
        for key in sorted(buckets):
            if index < len(buckets[key]) and len(picked) < limit:
                picked.append(buckets[key][index])
                added = True
        if not added:
            break
        index += 1
    return picked


def score(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if r["prediction"] in {"faithful", "unfaithful"}]
    tp = sum(1 for r in scored if r["prediction"] == "unfaithful" and r["label_unfaithful"])
    fp = sum(1 for r in scored if r["prediction"] == "unfaithful" and not r["label_unfaithful"])
    fn = sum(1 for r in scored if r["prediction"] == "faithful" and r["label_unfaithful"])
    tn = sum(1 for r in scored if r["prediction"] == "faithful" and not r["label_unfaithful"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )
    by_verdict = defaultdict(lambda: {"n": 0, "detected": 0})
    for record in scored:
        bucket = by_verdict[record["label_faithfulness"]]
        bucket["n"] += 1
        if record["prediction"] == "unfaithful":
            bucket["detected"] += 1
    by_error = defaultdict(lambda: {"n": 0, "detected": 0})
    for record in scored:
        if not record["label_unfaithful"]:
            continue
        bucket = by_error[record["label_error_type"]]
        bucket["n"] += 1
        if record["prediction"] == "unfaithful":
            bucket["detected"] += 1
    false_labels = [r for r in scored if r["label_provable"] == "false"]
    return {
        "scored": len(scored),
        "unscorable": len(records) - len(scored),
        "unscorable_reasons": dict(
            Counter(
                r["signal_status"] for r in records if r["prediction"] == "no_signal"
            )
        ),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(scored) if scored else None,
        "recall_on_provably_false": (
            sum(1 for r in false_labels if r["prediction"] == "unfaithful")
            / len(false_labels)
            if false_labels
            else None
        ),
        "provably_false_n": len(false_labels),
        "by_label_verdict": {k: dict(v) for k, v in sorted(by_verdict.items())},
        "by_error_type": {k: dict(v) for k, v in sorted(by_error.items())},
    }


async def _run() -> None:
    args = _parse()
    config = default_generation_config(args.model, args.temperature)
    items = stratified_sample(load_items(args.lineage), args.limit, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / f"predictions_{args.lineage}.jsonl"

    done = set()
    if args.resume and predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
        items = [item for item in items if item["stem"] not in done]
    print(
        f"lineage={args.lineage} items={len(items)} "
        f"(resumed {len(done)}) model={config.model}"
    )

    semaphore = asyncio.Semaphore(max(1, args.max_parallel))
    write_lock = asyncio.Lock()

    async def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            signal = await elaborated_goal_alignment(
                statement_nl=item["statement_nl"],
                formal_statement=item["formal_statement"],
                lean_header=item["lean_header"],
                config=config,
                lean_timeout=args.lean_timeout,
            )
        if signal.get("status") == "ok" and signal.get("equivalent") is not None:
            prediction = "faithful" if signal["equivalent"] else "unfaithful"
        else:
            prediction = "no_signal"
        record = {
            **{k: v for k, v in item.items() if k != "lean_header"},
            "prediction": prediction,
            "signal_status": signal.get("status"),
            "mismatches": signal.get("mismatches"),
            "rationale": signal.get("rationale"),
            "informalized": signal.get("informalized_statement"),
        }
        async with write_lock:
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        hit = "·"
        if prediction != "no_signal":
            hit = "✓" if (prediction == "unfaithful") == item["label_unfaithful"] else "✗"
        print(
            f"{hit} {item['name'][:44]:46s} label={item['label_faithfulness']:13s} "
            f"pred={prediction}"
        )
        return record

    await asyncio.gather(*(evaluate(item) for item in items))

    records = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "lineage": args.lineage,
        "model": config.model,
        "items": len(records),
        **score(records),
    }
    (args.output / f"summary_{args.lineage}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "by_error_type"}, indent=2))


if __name__ == "__main__":
    asyncio.run(_run())
