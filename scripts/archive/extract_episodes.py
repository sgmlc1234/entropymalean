"""Assemble one record per slot from the LangSmith trace and the output rows.

A generated row on its own says whether it certified. It does not say what the
orchestrator was asked to do, what it decided and on what evidence, what the
worker was told, or why the judge kept or discarded the result. Those four live
in four different places, and reading them together is the only way to ask
whether a disappointing row was a bad plan, a bad execution, or a good row
thrown away.

So the episode is the unit: plan -> generation -> certification -> judgment,
joined on the slot. The trace supplies the first, second and fourth; the output
JSONL supplies the third and the artifact itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import Client

_SLOT = re.compile(r"slot[_.](\d+)")


def _slot_of(name: str) -> Optional[int]:
    match = _SLOT.search(str(name or ""))
    return int(match.group(1)) if match else None


def _all_spans(client: Client, project: str, trace_id: str) -> List[Any]:
    """Every span of one group's run. Paged, because a group makes thousands."""
    spans: List[Any] = []
    for span in client.list_runs(
        project_name=project, filter=f'eq(trace_id, "{trace_id}")', limit=None
    ):
        spans.append(span)
    return spans


def build(client: Client, project: str, trace_id: str, rows_path: Path) -> Dict[str, Any]:
    spans = _all_spans(client, project, trace_id)
    by_id = {s.id: s for s in spans}

    def ancestry(span: Any) -> List[Any]:
        chain, cur = [], span
        while cur is not None:
            chain.append(cur)
            cur = by_id.get(cur.parent_run_id) if cur.parent_run_id else None
        return chain

    def slot_and_generation(span: Any) -> tuple[Optional[int], Optional[int]]:
        slot = generation = None
        for node in ancestry(span):
            if slot is None:
                slot = _slot_of(node.name)
            match = re.match(r"generation\.(\d+)\.graph", str(node.name or ""))
            if match and generation is None:
                generation = int(match.group(1))
        return slot, generation

    plans = [s for s in spans if s.name == "planner.operator_cards"]
    episodes: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

    for span in spans:
        slot, generation = slot_and_generation(span)
        if slot is None or generation is None:
            continue
        key = (generation, slot)
        episode = episodes[key]
        episode.setdefault("generation", generation)
        episode.setdefault("slot", slot)
        if span.name.endswith(("theorem_certify.attempt_0", "theorem_certify.attempt_1")):
            episode.setdefault("plan", (span.inputs or {}).get("operator_card"))
            episode.setdefault("op_type", (span.inputs or {}).get("op_type"))
            episode.setdefault("parent_ids", (span.inputs or {}).get("parent_ids"))
            episode["certify"] = span.outputs or {}
        if span.name == "codex_cli_call" and (span.inputs or {}).get("prompt"):
            episode.setdefault("generator_calls", []).append(
                {
                    "prompt": (span.inputs or {}).get("prompt"),
                    "output": (span.outputs or {}),
                    "model": (span.inputs or {}).get("model"),
                }
            )
        if span.name == "problem_quality_judge":
            # Every judge span for the slot, in order. A slot that was rejected
            # and retried has two, and keeping only the last silently pairs one
            # attempt's verdict with the other attempt's row -- which read as a
            # pipeline inconsistency (`verdict=reject` on a certified row) until
            # the rows themselves were checked and found consistent.
            episode.setdefault("judge_spans", []).append(
                {"inputs": span.inputs or {}, "outputs": span.outputs or {}}
            )

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_slot: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("generation"), row.get("slot"))
        if key not in by_slot or row.get("status") == "certified":
            by_slot[key] = row
    for key, episode in episodes.items():
        row = by_slot.get(key)
        episode["row"] = row
        # The verdict that actually decided this row's status lives on the row.
        # The trace is the only place its prompt lives. Keep both, and do not
        # let the trace speak for the outcome.
        episode["judge"] = ((row or {}).get("quality_evidence") or {}).get("judge") or {}
        episode["attempts_judged"] = len(episode.get("judge_spans") or [])

    return {
        "trace_id": trace_id,
        "plan_spans": [
            {"inputs": p.inputs or {}, "outputs": p.outputs or {}} for p in plans
        ],
        "episodes": [episodes[k] for k in sorted(episodes)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    client = Client()
    data = build(client, os.getenv("LANGCHAIN_PROJECT", ""), args.trace_id, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    eps = data["episodes"]
    print(f"episodes={len(eps)}  plans={len(data['plan_spans'])}")
    print(f"  with generator prompt : {sum(1 for e in eps if e.get('generator_calls'))}")
    print(f"  with judge verdict    : {sum(1 for e in eps if e.get('judge'))}")
    print(f"  with output row       : {sum(1 for e in eps if e.get('row'))}")


if __name__ == "__main__":
    main()
