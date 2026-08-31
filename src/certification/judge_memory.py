"""Past judgments, kept so the judge can plan a retry from what has worked.

The judge's verdict is only half of its job; the other half is the brief that
tells the generator what to do differently. A brief written from the rubric
alone is advice in general. A brief written knowing that the same correction was
issued four times for this failure and never once produced an accepted child is
advice about this pipeline.

So the record carries the outcome, not just the verdict. This is the
retrospective half of reflective memory management: entries are reweighted by
what happened downstream rather than by how confident they looked when written.
An entry whose brief was followed by a rejected retry is evidence against that
corrective, and it is worth more than another restatement of the rubric.

Memories are kept apart by operator. The reasoning that settles a mutation —
does the proof still end on the parent's lemma — says nothing about whether two
parents met, and mixing them would dilute both pools. This is the opposite of
the choice made for planner lessons, where the two operators share most of their
failure catalogue; a *judgment* is specific to what it judged.

Retrieval degrades to nothing when the log is empty, which is its state until
the judge has run. Nothing here fabricates experience it does not have.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_LOG = Path(os.getenv("JUDGE_MEMORY_LOG", "data/cache/judge_memory.jsonl"))


def record_judgment(
    *,
    problem_id: str,
    op_type: str,
    verdict: Dict[str, Any],
    parent_statement: str,
    child_statement: str,
    judge_model: str = "",
    generator_model: str = "",
    log_path: Path = DEFAULT_LOG,
) -> None:
    """Append one judgment. Outcome is filled in later, if a retry happens.

    The judge and generator models are stored as plain provenance, so a later
    pass can ask whether verdicts moved when the models did. On the rows checked
    so far they did not: two different models returned the same verdict and the
    same failure name, which is what one expects from a reader given only the
    two Lean proofs and the measurements — it never sees the prompt, the
    operator card, or the plan the child was written against.
    """
    if not verdict.get("ran"):
        return
    entry = {
        "problem_id": str(problem_id or ""),
        "op_type": str(op_type or ""),
        "judge_model": str(judge_model or ""),
        "generator_model": str(generator_model or ""),
        "verdict": verdict.get("verdict", ""),
        "quality": verdict.get("quality", ""),
        "failure": verdict.get("failure", ""),
        "reason": verdict.get("reason", ""),
        "retry_plan": verdict.get("retry_plan", ""),
        "fix_scope": verdict.get("fix_scope", ""),
        "brief": verdict.get("retry_brief", ""),
        "parent_statement": str(parent_statement or "")[:600],
        "child_statement": str(child_statement or "")[:600],
        "retry_outcome": "",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def record_retry_outcome(
    problem_id: str, outcome: str, *, log_path: Path = DEFAULT_LOG
) -> None:
    """Attach what became of the retry this judgment triggered.

    Rewrites the log rather than appending a correction, because the entry is
    the unit that gets retrieved and a reader of it needs the outcome in hand.
    The file is small — one line per judged row — so this stays cheap.
    """
    if not log_path.is_file():
        return
    lines = log_path.read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("problem_id") == problem_id and not entry.get("retry_outcome"):
            entry["retry_outcome"] = str(outcome or "")
            lines[index] = json.dumps(entry, ensure_ascii=False)
            changed = True
            break
    if changed:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_judgments(
    op_type: str, *, log_path: Path = DEFAULT_LOG, limit: int = 400
) -> List[Dict[str, Any]]:
    """Judgments for one operator, newest last. Empty until the judge has run."""
    if not log_path.is_file():
        return []
    wanted = str(op_type or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not wanted or str(entry.get("op_type", "")).lower() == wanted:
            out.append(entry)
    return out[-limit:]


def _entry_text(entry: Dict[str, Any]) -> str:
    return " ".join(
        str(entry.get(field) or "")
        for field in ("child_statement", "failure", "reason")
    )


def similar_judgments(
    query_text: str,
    op_type: str,
    *,
    limit: int = 3,
    log_path: Path = DEFAULT_LOG,
    store: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """The closest past judgments for this operator, diversified.

    Entries whose retry outcome is known are preferred: a case that shows what
    the correction produced is worth more to a judge planning the next one than
    a case that only shows the verdict.
    """
    entries = load_judgments(op_type, log_path=log_path)
    if not entries:
        return []
    from src.retrieval.memory_search import EmbeddingStore, search_memory

    store = store or EmbeddingStore()
    resolved = [e for e in entries if e.get("retry_outcome")]
    pool = resolved + [e for e in entries if not e.get("retry_outcome")]
    prior = [1.0] * len(resolved) + [0.5] * (len(pool) - len(resolved))
    return search_memory(
        query_text,
        pool,
        limit=limit,
        prior=prior,
        store=store,
    )


def format_precedents(entries: Sequence[Dict[str, Any]]) -> str:
    """Past judgments as a prompt block, or empty when there are none."""
    if not entries:
        return ""
    lines = []
    for entry in entries:
        outcome = entry.get("retry_outcome") or "retry outcome unknown"
        lines.append(
            f"- Judged {entry.get('verdict', '?')}"
            + (f" ({entry.get('failure')})" if entry.get("failure") else "")
            + f": {str(entry.get('reason') or '')[:200]}\n"
            f"  Its retry: {outcome}"
        )
    return (
        "Earlier judgments on comparable children, with what the correction "
        "produced. A corrective that has repeatedly failed here is not worth "
        "issuing again in the same words:\n" + "\n".join(lines) + "\n"
    )
