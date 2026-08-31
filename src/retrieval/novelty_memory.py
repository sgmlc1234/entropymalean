"""Compact novelty-memory retrieval for pool generation.

This module intentionally stays lexical and auditable. It builds small problem
cards from accepted ledgers and run-local JSONL rows, ranks top-K neighbors by
content-token Jaccard, and returns delta contracts/gate verdicts that callers
can inject into planner and worker prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from src.utils.lean_templates import detect_family


DEFAULT_ACCEPTED_LEDGER_PATH = Path("data/evaluation/treatment_inventory/final_curated/accepted.jsonl")
DEFAULT_ACCEPTED_TOP_K = 4
DEFAULT_RUN_LOCAL_TOP_K = 3

_TOKEN_RE = re.compile(r"[^0-9a-zA-Z_]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "is",
        "are",
        "be",
        "with",
        "that",
        "this",
        "these",
        "those",
        "as",
        "at",
        "by",
        "let",
        "find",
        "compute",
        "determine",
        "prove",
        "show",
        "given",
        "suppose",
        "all",
        "any",
        "some",
        "each",
        "every",
        "if",
        "then",
        "such",
        "where",
        "which",
        "from",
        "problem",
        "solution",
        "answer",
        "theorem",
        "lemma",
        "import",
        "mathlib",
    }
)


def _compact(text: Any, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def _sha256_text(text: Any) -> str:
    value = str(text or "").strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def novelty_tokens(text: Any, *, min_len: int = 3) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.split(str(text or "").lower())
        if token and len(token) >= min_len and not token.isdigit() and token not in _STOPWORDS
    }


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def canonical_theorem_surface(text: Any) -> str:
    surface = str(text or "")
    surface = surface.split(":= by", 1)[0]
    surface = re.sub(r"\b(theorem|lemma)\s+[A-Za-z0-9_'.]+", r"\1 _", surface)
    return re.sub(r"\s+", " ", surface).strip().lower()


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return list(parsed) if isinstance(parsed, list) else [parsed]
    return [] if value in (None, "") else [value]


def _ints(text: Any) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", str(text or ""))]


def _params_from_statement(family: str, statement: Any) -> dict[str, Any]:
    nums = _ints(statement)
    if family in {"gcd", "gcd_divisor_sum"} and len(nums) >= 2:
        return {"a": nums[0], "b": nums[1]}
    if family == "units_digit" and len(nums) >= 2:
        return {"base": nums[0], "exp": nums[1]}
    if family == "divisor_sum" and nums:
        return {"n": nums[-1]}
    if family == "divisor_sum_mod" and len(nums) >= 2:
        return {"n": nums[0], "a": nums[1]}
    if family == "stars_and_bars" and nums:
        var_count = max(2, len(set(re.findall(r"x_(\d+)", str(statement or "")))))
        return {"vars": var_count, "sum": nums[-1]}
    if family == "arithmetic_series" and len(nums) >= 3:
        return {"n_terms": nums[0], "first": nums[1], "diff": nums[2] - nums[1]}
    if family == "modular_congruence" and len(nums) >= 2:
        return {"a": nums[0], "m": nums[1]}
    return {}


def _row_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return dict(row.model_dump())
    if isinstance(row, dict):
        return dict(row)
    return {}


def _hashes(row: dict[str, Any], *, statement: str, formal_surface: str) -> dict[str, str]:
    nested = _json_dict(row.get("hashes"))
    statement_hash = str(row.get("statement_sha256") or nested.get("statement_sha256") or "")
    formal_hash = str(
        row.get("formal_statement_sha256") or nested.get("formal_statement_sha256") or ""
    )
    answer_hash = str(row.get("answer_sha256") or nested.get("answer_sha256") or "")
    return {
        "statement_sha256": statement_hash or _sha256_text(statement),
        "formal_statement_sha256": formal_hash or _sha256_text(formal_surface),
        "answer_sha256": answer_hash or _sha256_text(row.get("answer")),
    }


def _numeric_signature(family: str, row: dict[str, Any], statement: str) -> str:
    if family == "theorem_proof":
        return ""
    params = (
        _json_dict(row.get("generated_params"))
        or _json_dict(row.get("required_params"))
        or _json_dict(row.get("params"))
        or _params_from_statement(family, statement)
    )
    if not params:
        return ""
    return json.dumps({"family": family, "params": params}, ensure_ascii=False, sort_keys=True)


def _target_summary(row: dict[str, Any], *, family: str, formal_surface: str) -> str:
    if family == "theorem_proof":
        surface = canonical_theorem_surface(formal_surface or row.get("formal_statement") or row.get("lean_code"))
        if ":" in surface:
            return _compact(surface.rsplit(":", 1)[-1], 180)
        return _compact(surface, 180)
    signature = _numeric_signature(family, row, str(row.get("statement") or ""))
    if signature:
        return signature
    return _compact(row.get("answer"), 120)


def row_to_novelty_card(row: Any, *, source_kind: str = "run_local", source_file: str = "") -> dict[str, Any]:
    data = _row_dict(row)
    metadata = _json_dict(data.get("metadata"))
    if metadata:
        data = {**metadata, **data}
    statement = str(data.get("statement") or "")
    formal_surface = str(data.get("formal_statement") or data.get("lean_code") or "")
    family = str(data.get("family") or data.get("target_family") or detect_family(statement) or "unknown")
    if (
        family != "theorem_proof"
        and (
            data.get("target_style") == "theorem_proof"
            or data.get("certification_route") == "theorem_prover"
        )
    ):
        family = "theorem_proof"
    curation = _json_dict(data.get("curation"))
    quality_evidence = _json_dict(data.get("quality_evidence"))
    entropy_direction = str(
        curation.get("entropy_direction")
        or data.get("entropy_direction")
        or _json_dict(quality_evidence.get("entropy_direction")).get("direction")
        or ""
    )
    quality_flags = list(_json_list(data.get("quality_flags")))[:8]
    hashes = _hashes(data, statement=statement, formal_surface=formal_surface)
    numeric_signature = _numeric_signature(family, data, statement)
    theorem_surface = canonical_theorem_surface(formal_surface) if family == "theorem_proof" else ""
    text = " ".join(
        str(part or "")
        for part in (
            statement,
            formal_surface,
            data.get("proof_plan"),
            data.get("solution"),
            data.get("answer"),
            family,
            entropy_direction,
            json.dumps(quality_evidence.get("accepted_proxy") or {}, ensure_ascii=False),
        )
    )
    problem_id = str(
        data.get("problem_id")
        or data.get("id")
        or data.get("source_problem_id")
        or hashes["statement_sha256"][:12]
    )
    return {
        "problem_id": problem_id,
        "source_kind": source_kind,
        "source_file": source_file,
        "generation": data.get("generation"),
        "benchmark": data.get("benchmark"),
        "op_type": data.get("op_type"),
        "family": family,
        "target_style": data.get("target_style") or ("theorem_proof" if family == "theorem_proof" else "numeric_answer"),
        "statement_excerpt": _compact(statement, 260),
        "formal_surface_excerpt": _compact(formal_surface, 260),
        "target_summary": _target_summary(data, family=family, formal_surface=formal_surface),
        "entropy_direction": entropy_direction,
        "quality_flags": quality_flags,
        "curation_rationale": _compact(curation.get("rationale") or data.get("curation_rationale"), 220),
        "hashes": hashes,
        "numeric_signature": numeric_signature,
        "theorem_surface": theorem_surface,
        "_tokens": sorted(novelty_tokens(text)),
    }


def load_jsonl_cards(path: Optional[Path], *, source_kind: str) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    cards: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            cards.append(row_to_novelty_card(row, source_kind=source_kind, source_file=str(path)))
    return cards


def cards_from_rows(rows: Iterable[Any], *, source_kind: str) -> list[dict[str, Any]]:
    return [
        row_to_novelty_card(row, source_kind=source_kind)
        for row in rows or []
        if _row_dict(row)
    ]


def _same_root_lineage(left: str, right: str) -> bool:
    def root(value: str) -> str:
        text = str(value or "")
        return text.split("__", 1)[0] if "__" in text else text

    return bool(left and right and root(left) == root(right))


def retrieve_similar_cards(
    query: Any,
    cards: Iterable[dict[str, Any]],
    *,
    k: int = 3,
    min_score: float = 0.03,
) -> list[dict[str, Any]]:
    query_card = row_to_novelty_card(query, source_kind="query")
    query_tokens = set(query_card.get("_tokens") or [])
    if not query_tokens:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards or []:
        if card.get("problem_id") == query_card.get("problem_id"):
            continue
        token_score = jaccard_similarity(query_tokens, set(card.get("_tokens") or []))
        if token_score <= 0:
            continue
        score = token_score
        if card.get("family") == query_card.get("family"):
            score += 0.08
        if card.get("target_style") == query_card.get("target_style"):
            score += 0.03
        same_lineage = _same_root_lineage(str(query_card.get("problem_id") or ""), str(card.get("problem_id") or ""))
        if same_lineage:
            score -= 0.02
        if score >= min_score:
            display = {key: value for key, value in card.items() if key != "_tokens"}
            display.update(
                {
                    "_retrieval_similarity": round(score, 3),
                    "_token_jaccard": round(token_score, 3),
                    "_same_lineage": same_lineage,
                }
            )
            scored.append((score, display))
    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("source_kind") or ""),
            str(item[1].get("problem_id") or ""),
        )
    )
    return [item for _score, item in scored[:k]]


def exact_blockers(query: Any, cards: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    query_card = row_to_novelty_card(query, source_kind="query")
    query_hashes = dict(query_card.get("hashes") or {})
    blockers: list[dict[str, Any]] = []
    for card in cards or []:
        card_hashes = dict(card.get("hashes") or {})
        for key in ("statement_sha256", "formal_statement_sha256"):
            if query_hashes.get(key) and query_hashes.get(key) == card_hashes.get(key):
                blockers.append(
                    {
                        "kind": key,
                        "matched_problem_id": card.get("problem_id", ""),
                        "source_kind": card.get("source_kind", ""),
                    }
                )
        if query_card.get("theorem_surface") and query_card.get("theorem_surface") == card.get("theorem_surface"):
            blockers.append(
                {
                    "kind": "canonical_theorem_surface",
                    "matched_problem_id": card.get("problem_id", ""),
                    "source_kind": card.get("source_kind", ""),
                }
            )
        if query_card.get("numeric_signature") and query_card.get("numeric_signature") == card.get("numeric_signature"):
            blockers.append(
                {
                    "kind": "numeric_family_params",
                    "matched_problem_id": card.get("problem_id", ""),
                    "source_kind": card.get("source_kind", ""),
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for blocker in blockers:
        key = (str(blocker.get("kind")), str(blocker.get("matched_problem_id")))
        if key not in seen:
            seen.add(key)
            deduped.append(blocker)
    return deduped


def _pool_exact_blockers(
    pool_rows: Iterable[Any],
    cards: Iterable[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for idx, row in enumerate(pool_rows or []):
        row_data = _row_dict(row)
        metadata = _json_dict(row_data.get("metadata"))
        query_id = str(
            row_data.get("problem_id")
            or row_data.get("id")
            or metadata.get("problem_id")
            or metadata.get("id")
            or f"pool_{idx}"
        )
        for blocker in exact_blockers(row, cards):
            blockers.append({"query_problem_id": query_id, **blocker})
            if len(blockers) >= limit:
                return blockers
    return blockers


def evaluate_candidate_novelty(
    candidate: Any,
    cards: Iterable[dict[str, Any]],
    *,
    k: int = 3,
) -> dict[str, Any]:
    all_cards = list(cards or [])
    candidate_card = row_to_novelty_card(candidate)
    blockers = exact_blockers(candidate, all_cards)
    matches = retrieve_similar_cards(candidate, all_cards, k=k)
    if blockers:
        return {
            "verdict": "near_duplicate",
            "matched_card_id": blockers[0].get("matched_problem_id", ""),
            "exact_blockers": blockers[:5],
            "matched_cards": matches,
            "gate_cards": matches,
            "reason": "Exact novelty-memory blocker matched an accepted or run-local card.",
        }
    top = matches[0] if matches else {}
    same_target = bool(top and top.get("target_summary") == candidate_card.get("target_summary"))
    if top and top.get("family") == candidate_card.get("family") and same_target and float(top.get("_token_jaccard") or 0.0) >= 0.9:
        return {
            "verdict": "near_duplicate",
            "matched_card_id": top.get("problem_id", ""),
            "exact_blockers": [],
            "matched_cards": matches,
            "gate_cards": matches,
            "reason": "Lexical top match has the same family and target semantics.",
        }
    if top and float(top.get("_retrieval_similarity") or 0.0) >= 0.12:
        return {
            "verdict": "structural_overlap",
            "matched_card_id": top.get("problem_id", ""),
            "exact_blockers": [],
            "matched_cards": matches,
            "gate_cards": matches,
            "reason": "Top novelty-memory match shares family or surface terms; keep only if the candidate has a new target role.",
        }
    return {
        "verdict": "novel",
        "matched_card_id": "",
        "exact_blockers": [],
        "matched_cards": matches,
        "gate_cards": matches,
        "reason": "No close novelty-memory match retrieved.",
    }


def build_memory_delta_contract(item: dict[str, Any], cards: Iterable[dict[str, Any]], *, k: int = 3) -> dict[str, Any]:
    if item.get("op_type") == "survivor":
        return {}
    query = {
        "problem_id": f"slot_{item.get('slot', 0)}_plan",
        "statement": " ".join(
            str(value or "")
            for value in (
                item.get("goal"),
                item.get("operator_goal"),
                item.get("reasoning_goal"),
                item.get("variation_axis"),
                " ".join(item.get("constraints") or []),
            )
        ),
        "formal_statement": "",
        "family": item.get("target_family") or "",
        "target_style": item.get("target_style") or "",
    }
    matches = retrieve_similar_cards(query, cards, k=k)
    similar_ids = [str(card.get("problem_id") or "") for card in matches if card.get("problem_id")]
    summaries = [
        _compact(card.get("target_summary") or card.get("statement_excerpt"), 160)
        for card in matches
        if card.get("target_summary") or card.get("statement_excerpt")
    ]
    target_style = str(item.get("target_style") or "")
    if target_style == "theorem_proof" or item.get("target_family") == "theorem_proof":
        delta = (
            "Change the final theorem obligation or proof role; do not recreate a matched "
            "formal target, helper-only theorem, AP affine drift, or Lean-surface paraphrase."
        )
    else:
        delta = (
            "Use different canonical params and a different target role; do not recreate the "
            "same family+params, same final target semantics, or parameter-shift-only variant."
        )
    return {
        "similar_card_ids": similar_ids[:k],
        "must_not_repeat": summaries[:k],
        "required_distinguishing_delta": delta,
        "allowed_overlap": (
            "Same family/domain is allowed only when a new object, target role, or consumed checkpoint is explicit."
        ),
        "novelty_rationale": (
            "Closest accepted/run-local analogues are evidence, not parents; the child must name what changes."
            if matches
            else "No close accepted/run-local analogue retrieved; still avoid exact parent or sibling recreation."
        ),
    }


def build_novelty_memory_pack(
    pool_rows: Iterable[Any],
    *,
    accepted_ledger_path: Optional[Path] = None,
    run_rows: Optional[Iterable[Any]] = None,
    accepted_top_k: int = DEFAULT_ACCEPTED_TOP_K,
    run_local_top_k: int = DEFAULT_RUN_LOCAL_TOP_K,
) -> dict[str, Any]:
    accepted_path = accepted_ledger_path or DEFAULT_ACCEPTED_LEDGER_PATH
    accepted_cards = load_jsonl_cards(accepted_path, source_kind="accepted")
    run_cards = cards_from_rows(run_rows or [], source_kind="run_local")
    pool_rows_list = list(pool_rows or [])
    query = {
        "problem_id": "planner_pool_query",
        "statement": "\n".join(str(getattr(row, "statement", "") or _row_dict(row).get("statement") or "") for row in pool_rows_list),
        "formal_statement": "\n".join(
            str((getattr(row, "metadata", {}) or {}).get("formal_statement") or _row_dict(row).get("formal_statement") or "")
            for row in pool_rows_list
        ),
        "family": "",
    }
    accepted_neighbors = retrieve_similar_cards(query, accepted_cards, k=accepted_top_k)
    run_neighbors = retrieve_similar_cards(query, run_cards, k=run_local_top_k)
    accepted_blockers = _pool_exact_blockers(pool_rows_list, accepted_cards, limit=8)
    run_local_blockers = _pool_exact_blockers(pool_rows_list, run_cards, limit=8)
    return {
        "enabled": True,
        "accepted_ledger_path": str(accepted_path),
        "accepted_card_count": len(accepted_cards),
        "run_local_card_count": len(run_cards),
        "cards": accepted_cards + run_cards,
        "planner_view": {
            "exact_blockers": {
                "accepted": accepted_blockers,
                "run_local": run_local_blockers,
            },
            "soft_neighbors": {
                "accepted": accepted_neighbors,
                "run_local": run_neighbors,
            },
            "accepted_neighbors": accepted_neighbors,
            "run_local_neighbors": run_neighbors,
            "instructions": [
                "Closest accepted/run-local analogues are evidence, not parents.",
                "Exact blockers identify surfaces or numeric signatures that must not be recreated.",
                "Do not recreate matched target semantics.",
                "For every generated slot, name the required distinguishing delta.",
            ],
        },
    }


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    is_theorem = card.get("target_style") == "theorem_proof" or card.get("family") == "theorem_proof"
    return {
        "problem_id": card.get("problem_id"),
        "source_kind": card.get("source_kind"),
        "generation": card.get("generation"),
        "benchmark": card.get("benchmark"),
        "op_type": card.get("op_type"),
        "family": card.get("family"),
        "target_style": card.get("target_style"),
        "statement_excerpt": card.get("statement_excerpt"),
        "formal_surface_excerpt": card.get("formal_surface_excerpt") if is_theorem else "",
        "target_summary": card.get("target_summary"),
        "entropy_direction": card.get("entropy_direction"),
        "quality_flags": list(card.get("quality_flags") or [])[:6],
        "curation_rationale": card.get("curation_rationale"),
        "_retrieval_similarity": card.get("_retrieval_similarity"),
        "_token_jaccard": card.get("_token_jaccard"),
        "_same_lineage": card.get("_same_lineage"),
    }


def format_novelty_memory_pack(pack: dict[str, Any]) -> str:
    if not pack or not pack.get("enabled"):
        return "NoveltyMemoryPack: disabled"
    view = dict(pack.get("planner_view") or {})
    payload = {
        "accepted_ledger_path": pack.get("accepted_ledger_path"),
        "accepted_card_count": pack.get("accepted_card_count", 0),
        "run_local_card_count": pack.get("run_local_card_count", 0),
        "instructions": view.get("instructions", []),
        "exact_blockers": view.get("exact_blockers", {"accepted": [], "run_local": []}),
        "soft_neighbors": {
            "accepted": [compact_card(card) for card in view.get("soft_neighbors", {}).get("accepted", view.get("accepted_neighbors", []))],
            "run_local": [compact_card(card) for card in view.get("soft_neighbors", {}).get("run_local", view.get("run_local_neighbors", []))],
        },
    }
    return "NoveltyMemoryPack:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
