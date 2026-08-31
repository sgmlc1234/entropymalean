"""Small LeanSearch client used as premise context for theorem workers.

The client is intentionally thin: hosted LeanSearch is treated as a hint
provider, and every returned name is checked against the local Lake project
before it is shown to a model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import langsmith as ls


DEFAULT_ENDPOINT = "https://leansearch.net/search"
DEFAULT_LIMIT = 8
DEFAULT_CACHE_DIR = Path("data/cache/leansearch")
DEFAULT_MAX_QUERIES_PER_PROBLEM = 3

_THROTTLE_LOCK = asyncio.Lock()
_LAST_REQUEST_AT = 0.0


@dataclass
class PremiseCandidate:
    name: str
    module_name: str = ""
    kind: str = ""
    signature: str = ""
    type: str = ""
    informal_name: str = ""
    informal_description: str = ""
    value: str = ""

    def prompt_name(self) -> str:
        return self.name.strip()

    def to_digest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module_name": self.module_name,
            "kind": self.kind,
            "signature": self.signature[:500],
            "type": self.type[:500],
            "informal_name": self.informal_name[:200],
        }


@dataclass
class PremisePack:
    query: str
    validated_candidates: List[PremiseCandidate] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "leansearch"
    cache_hit: bool = False
    retrieval_error: str = ""

    def digest(self) -> Dict[str, Any]:
        return {
            "query": self.query[:500],
            "validated_count": len(self.validated_candidates),
            "rejected_count": len(self.rejected_candidates),
            "source": self.source,
            "cache_hit": self.cache_hit,
            "retrieval_error": self.retrieval_error[:500],
            "candidates": [candidate.to_digest() for candidate in self.validated_candidates[:8]],
        }


def leansearch_enabled(disabled: bool = False) -> bool:
    env_disabled = os.getenv("LEANSEARCH_DISABLED", "").strip().lower()
    return not disabled and env_disabled not in {"1", "true", "yes", "y"}


def max_queries_per_problem() -> int:
    raw = os.getenv("LEANSEARCH_MAX_QUERIES_PER_PROBLEM", str(DEFAULT_MAX_QUERIES_PER_PROBLEM))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_QUERIES_PER_PROBLEM


def build_statement_query(statement: str, formal_statement: str) -> str:
    return "\n".join(
        part.strip()
        for part in (
            "Find relevant Mathlib lemmas for proving this Lean theorem.",
            f"Natural statement: {statement}",
            f"Formal statement: {formal_statement}",
        )
        if part and part.strip()
    )


def should_retrieve_for_diagnostics(diagnostics: str) -> bool:
    lowered = diagnostics.lower()
    triggers = (
        "unknown identifier",
        "unknown constant",
        "invalid field",
        "failed to synthesize",
        "type mismatch",
        "application type mismatch",
        "unsolved goals",
        "rewrite",
        "did not find an occurrence",
    )
    return any(trigger in lowered for trigger in triggers)


def build_diagnostic_query(statement: str, formal_statement: str, diagnostics: str) -> str:
    return "\n".join(
        part.strip()
        for part in (
            "Find Mathlib lemmas or theorem names that fix these Lean proof diagnostics.",
            f"Natural statement: {statement}",
            f"Formal statement: {formal_statement}",
            f"Lean diagnostics: {diagnostics[:1800]}",
        )
        if part and part.strip()
    )


def format_premise_pack(pack: PremisePack) -> str:
    if pack.retrieval_error and not pack.validated_candidates:
        return (
            "LeanSearch PremisePack (hints only; local Lean verifier is authority)\n"
            f"Query: {pack.query[:500]}\n"
            f"Retrieval error: {pack.retrieval_error[:500]}\n"
        )
    lines = [
        "LeanSearch PremisePack (hints only; local Lean verifier is authority)",
        f"Query: {pack.query[:500]}",
        "Use these candidates as possible proof vocabulary. Do not invent names; local Lean verification decides.",
    ]
    for index, candidate in enumerate(pack.validated_candidates, start=1):
        statement = candidate.signature or candidate.type or candidate.value
        why = candidate.informal_description or candidate.informal_name or "Matches the theorem/query terms."
        lines.extend(
            [
                f"Candidate {index}:",
                f"Formal name: {candidate.prompt_name()}",
                f"Formal statement: {statement[:1200] or 'not_available'}",
                f"Informal name: {candidate.informal_name[:300] or 'not_available'}",
                f"Why it may help: {why[:500]}",
            ]
        )
    return "\n".join(lines) + "\n"


def _cache_dir(cache_dir: Optional[Path]) -> Path:
    path = Path(os.getenv("LEANSEARCH_CACHE_DIR") or cache_dir or DEFAULT_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _query_cache_path(query: str, *, endpoint: str, limit: int, cache_dir: Path) -> Path:
    digest = hashlib.sha256(f"{endpoint}\0{limit}\0{query}".encode("utf-8")).hexdigest()
    return cache_dir / f"query_{digest}.json"


def _check_cache_path(name: str, *, cache_dir: Path) -> Path:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return cache_dir / f"check_{digest}.json"


def _http_post_json(endpoint: str, payload: Dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "entropy-mag-leansearch/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


async def _throttled_post(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: float,
    http_post_json: Optional[Callable[[str, Dict[str, Any], float], Any]] = None,
) -> Any:
    global _LAST_REQUEST_AT
    async with _THROTTLE_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        post = http_post_json or _http_post_json
        try:
            return await asyncio.to_thread(post, endpoint, payload, timeout)
        finally:
            _LAST_REQUEST_AT = time.monotonic()


def _candidate_name(value: Any) -> str:
    if isinstance(value, list):
        return ".".join(str(part) for part in value if str(part).strip())
    return str(value or "").strip()


def normalize_leansearch_response(payload: Any) -> List[PremiseCandidate]:
    rows = payload
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]
    candidates: List[PremiseCandidate] = []
    for item in rows if isinstance(rows, list) else []:
        result = item.get("result") if isinstance(item, dict) else None
        if result is None and isinstance(item, dict):
            result = item
        if not isinstance(result, dict):
            continue
        name = _candidate_name(result.get("name"))
        if not name:
            continue
        candidates.append(
            PremiseCandidate(
                name=name,
                module_name=_candidate_name(result.get("module_name")),
                kind=str(result.get("kind") or ""),
                signature=str(result.get("signature") or ""),
                type=str(result.get("type") or ""),
                informal_name=str(result.get("informal_name") or ""),
                informal_description=str(result.get("informal_description") or result.get("docstring") or ""),
                value=str(result.get("value") or ""),
            )
        )
    return candidates


def _safe_check_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*(\.[A-Za-z_][A-Za-z0-9_'.]*)*", name))


async def validate_candidate_name(
    name: str,
    *,
    repo_root: Path,
    cache_dir: Path,
    timeout: float = 45.0,
) -> tuple[bool, str]:
    check_path = _check_cache_path(name, cache_dir=cache_dir)
    if check_path.exists():
        try:
            cached = json.loads(check_path.read_text(encoding="utf-8"))
            return bool(cached.get("ok")), str(cached.get("error") or "")
        except (OSError, json.JSONDecodeError):
            pass
    if not _safe_check_name(name):
        error = "unsafe Lean name syntax"
        check_path.write_text(json.dumps({"ok": False, "error": error}), encoding="utf-8")
        return False, error
    lake = shutil.which("lake")
    if lake is None:
        return False, "lake executable not found"
    with tempfile.NamedTemporaryFile("w", suffix=".lean", encoding="utf-8", delete=False) as tmp:
        tmp.write("import Mathlib\n")
        tmp.write(f"#check {name}\n")
        tmp_path = Path(tmp.name)
    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            [lake, "env", "lean", str(tmp_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        ok = completed.returncode == 0
        error = (completed.stderr or completed.stdout or "")[:1000]
        check_path.write_text(json.dumps({"ok": ok, "error": error}), encoding="utf-8")
        return ok, error
    except subprocess.TimeoutExpired:
        return False, f"#check timeout after {timeout}s"
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


async def retrieve_premise_pack(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    endpoint: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    disabled: bool = False,
    phase: str = "initial",
    http_post_json: Optional[Callable[[str, Dict[str, Any], float], Any]] = None,
    validator: Optional[Callable[[PremiseCandidate], Any]] = None,
) -> PremisePack:
    if not leansearch_enabled(disabled):
        return PremisePack(query=query, source="disabled", retrieval_error="LeanSearch disabled")
    endpoint = endpoint or os.getenv("LEANSEARCH_ENDPOINT") or DEFAULT_ENDPOINT
    limit = max(1, int(limit or DEFAULT_LIMIT))
    cache = _cache_dir(cache_dir)
    repo_root = Path(repo_root or Path.cwd())
    cache_path = _query_cache_path(query, endpoint=endpoint, limit=limit, cache_dir=cache)
    cache_hit = cache_path.exists()
    with ls.trace(
        name=f"leansearch.retrieve.{phase}",
        run_type="retriever",
        inputs={"query": query[:1000], "limit": limit, "endpoint": endpoint, "cache_hit": cache_hit},
        tags=["leansearch", phase],
    ) as retrieve_run:
        try:
            if cache_hit:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                payload = await _throttled_post(
                    endpoint,
                    {"query": [query], "num_results": limit},
                    timeout=30.0,
                    http_post_json=http_post_json,
                )
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            candidates = normalize_leansearch_response(payload)[:limit]
        except Exception as exc:
            pack = PremisePack(
                query=query,
                source=endpoint,
                cache_hit=cache_hit,
                retrieval_error=f"{type(exc).__name__}: {exc}",
            )
            retrieve_run.end(outputs=pack.digest())
            return pack
        retrieve_run.end(outputs={"candidate_count": len(candidates), "cache_hit": cache_hit})

    validated: List[PremiseCandidate] = []
    rejected: List[Dict[str, Any]] = []
    with ls.trace(
        name="leansearch.validate_candidates",
        run_type="tool",
        inputs={"candidate_names": [candidate.name for candidate in candidates], "repo_root": str(repo_root)},
        tags=["leansearch", "validation"],
    ) as validate_run:
        for candidate in candidates:
            if validator is not None:
                maybe_ok = validator(candidate)
                ok = await maybe_ok if asyncio.iscoroutine(maybe_ok) else bool(maybe_ok)
                error = "" if ok else "validator rejected candidate"
            else:
                ok, error = await validate_candidate_name(candidate.name, repo_root=repo_root, cache_dir=cache)
            if ok:
                validated.append(candidate)
            else:
                rejected.append({"name": candidate.name, "error": error[:500]})
        validate_run.end(outputs={"validated_count": len(validated), "rejected": rejected[:8]})
    return PremisePack(
        query=query,
        validated_candidates=validated,
        rejected_candidates=rejected,
        source=endpoint,
        cache_hit=cache_hit,
    )
