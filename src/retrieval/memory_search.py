"""Retrieve planner memory by meaning, and keep the retrieved set diverse.

The previous retriever was a hand-weighted linear score over metadata: +16 for
coming from a curated file, +10 for a dataset hint, +8 for lexical term overlap,
and so on. Measured on three seed groups drawn from different areas — algebra,
number theory, induction — it returned 6 of the same 8 cards to all three, every
card from a single file. The dominant weight was `is_curated`, worth twice the
entire content signal, so one file answered every query regardless of what was
being asked.

Two changes, in the order the literature puts them.

*Embed first.* The retriever had no semantic signal at all; lexical overlap
capped at +8 cannot distinguish a Sylow theorem from a congruence. Reranking
fixes good candidates in the wrong order, which is not the failure here — the
candidates themselves were wrong.

*Then diversify.* MMR trades relevance against redundancy through one
parameter, needs no model, and costs nothing at retrieval time. It is the
direct answer to eight cards from one file.

Embeddings are cached on disk by content hash: the corpus grows by a few
thousand rows per campaign and re-embedding it on every planner call would cost
more than the planning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

DEFAULT_EMBED_URL = os.getenv(
    "MEMORY_EMBED_URL", "http://100.77.209.48:5678/v1/embeddings"
)
DEFAULT_EMBED_MODEL = os.getenv(
    "MEMORY_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
)
DEFAULT_CACHE = Path(os.getenv("MEMORY_EMBED_CACHE", "data/cache/memory_embeddings"))

#: Relevance-versus-redundancy in MMR. 0.7 keeps relevance dominant while still
#: refusing near-duplicates; the classic formulation and most implementations
#: sit in 0.5-0.8. Not tuned here — the measurable claim is that any λ < 1
#: breaks the single-file monopoly, and this value should be swept once yield
#: data exists.
DEFAULT_LAMBDA = 0.7


def _key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class EmbeddingStore:
    """Embeddings with a content-addressed disk cache."""

    def __init__(
        self,
        url: str = DEFAULT_EMBED_URL,
        model: str = DEFAULT_EMBED_MODEL,
        cache_dir: Path = DEFAULT_CACHE,
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._memo: Dict[str, List[float]] = {}

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _load(self, key: str) -> Optional[List[float]]:
        if key in self._memo:
            return self._memo[key]
        path = self._cache_path(key)
        if path.is_file():
            try:
                vector = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            self._memo[key] = vector
            return vector
        return None

    def _store(self, key: str, vector: Sequence[float]) -> None:
        self._memo[key] = list(vector)
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(vector)), encoding="utf-8")

    def embed(self, texts: Sequence[str], *, batch: int = 32) -> List[Optional[List[float]]]:
        """Vectors for `texts`, cached. Returns None per text on failure.

        A retrieval that cannot embed must degrade to the metadata score rather
        than return nothing, so failures are reported per item instead of
        raising: an unreachable embedding server should cost ranking quality,
        not the run.
        """
        out: List[Optional[List[float]]] = [None] * len(texts)
        pending: List[Tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            clean = " ".join(str(text or "").split())[:2000]
            if not clean:
                continue
            key = _key(clean, self.model)
            cached = self._load(key)
            if cached is not None:
                out[index] = cached
            else:
                pending.append((index, clean, key))

        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            try:
                response = httpx.post(
                    self.url,
                    json={"model": self.model, "input": [c[1] for c in chunk]},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json().get("data") or []
            except Exception:
                continue
            for (index, _text, key), item in zip(chunk, data):
                vector = item.get("embedding")
                if vector:
                    self._store(key, vector)
                    out[index] = list(vector)
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr_select(
    query: Sequence[float],
    candidates: Sequence[Tuple[Any, Optional[Sequence[float]]]],
    *,
    limit: int,
    lam: float = DEFAULT_LAMBDA,
    prior: Optional[Sequence[float]] = None,
) -> List[Any]:
    """Maximal Marginal Relevance over embedded candidates.

    `prior` optionally supplies a metadata score per candidate, normalised by
    the caller and blended into relevance. Keeping it lets the operator-type and
    style signals survive rather than being discarded for embeddings alone —
    they are weak but not worthless, and the failure being fixed is monoculture,
    not the presence of metadata.
    """
    remaining = list(range(len(candidates)))
    chosen: List[int] = []
    relevance = []
    for index, (_item, vector) in enumerate(candidates):
        base = cosine(query, vector) if vector else 0.0
        if prior is not None and index < len(prior):
            base = 0.5 * base + 0.5 * float(prior[index])
        relevance.append(base)

    while remaining and len(chosen) < limit:
        best_index, best_score = remaining[0], float("-inf")
        for index in remaining:
            redundancy = 0.0
            vector = candidates[index][1]
            if vector:
                for picked in chosen:
                    other = candidates[picked][1]
                    if other:
                        redundancy = max(redundancy, cosine(vector, other))
            score = lam * relevance[index] - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        chosen.append(best_index)
        remaining.remove(best_index)
    return [candidates[i][0] for i in chosen]

# ── Query construction ───────────────────────────────────────────────────────
#
# The retriever it replaces had no query text at all: it compared card metadata
# against a bag of terms harvested from the pool, and that bag was drawn from
# `lean_code` as well as the statements. Tactic names and header tokens
# dominated it — `aesop`, `bigoperators`, `exact`, `apply`, `maxheartbeats`
# appeared in every pool — so three groups from algebra, number theory and
# induction produced queries whose embeddings sat at cosine 0.83-0.86 of each
# other. Asking the same question of every pool is why the same six cards came
# back to all three.
#
# What a planner is actually asking is "what went wrong when this kind of
# mathematics met this operator", so the query carries exactly that: the
# concepts the *statements* are about, the shape of what they claim, and which
# operator is being planned. Proof scripts are excluded — they describe how the
# parent was closed, which is not what is being looked up.

_NAMESPACE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\.[A-Za-z_][A-Za-z0-9_']*)+)\b")
_BARE_TYPE = re.compile(r"\b(Finset|Set|Polynomial|Matrix|Real|Complex|Rat|ZMod|GaussianInt|Subgroup|Sylow|Prime|Odd|Even|Irrational)\b")
#: Relation symbols name the *kind* of claim: a divisibility problem and an
#: inequality problem can share every namespace and still be unlike.
_RELATIONS = {
    "∣": "divisibility", "≡": "congruence", "≤": "inequality", "<": "inequality",
    "≥": "inequality", ">": "inequality", "∈": "membership", "⊆": "inclusion",
    "∑": "summation", "∏": "product", "√": "roots", "!": "factorial",
}
#: Identifiers that name one problem and match nothing else. Leaving them in
#: makes a pool similar only to itself.
_PROBLEM_ID = re.compile(r"\b(mathd|amc|aime|imo|induction|algebra|numbertheory)[a-z0-9_]*\b", re.I)


def _statement_of(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("formal_statement") or "")
    meta = getattr(row, "metadata", None) or {}
    return str(meta.get("formal_statement") or "")


def _prose_of(row: Any) -> str:
    """The human sentence, under whichever name this corpus stores it."""
    if isinstance(row, dict):
        for field in ("statement_nl", "informal_statement", "goal", "statement"):
            value = row.get(field)
            if value:
                return str(value)
        return ""
    return str(getattr(row, "statement", "") or "")


def build_query(
    rows: Sequence[Any],
    *,
    op_type: str = "",
    generation: Optional[int] = None,
) -> str:
    """The text a planner's memory lookup is actually asking about."""
    concepts: List[str] = []
    kinds: List[str] = []
    prose: List[str] = []
    for row in rows:
        statement = _statement_of(row).split(":= by", 1)[0]
        for name in _NAMESPACE.findall(statement) + _BARE_TYPE.findall(statement):
            if name not in concepts:
                concepts.append(name)
        for symbol, label in _RELATIONS.items():
            if symbol in statement and label not in kinds:
                kinds.append(label)
        text = _PROBLEM_ID.sub("", _prose_of(row)).strip()
        if text:
            prose.append(" ".join(text.split())[:180])

    # The operator and generation are deliberately *not* embedded. They are the
    # same string for every slot of a kind, so putting them in the vector moves
    # all queries toward a common point: adding the prefix raised the mean
    # inter-group cosine from 0.746 back to 0.847, undoing the discrimination
    # the concepts had won. They belong in the metadata filter, which is where
    # the operator layer already applies them.
    parts = []
    if concepts:
        parts.append("Mathematical objects: " + ", ".join(concepts[:20]) + ".")
    if kinds:
        parts.append("Claims of the form: " + ", ".join(kinds) + ".")
    if prose:
        parts.append("Problems: " + " | ".join(prose[:5]))
    return " ".join(parts)

def card_text(card: Any) -> str:
    """What a memory card is *about*, for embedding.

    The lesson alone is too short and too formulaic to separate cards — many
    begin "Avoid ..." and end alike. Pairing it with the surface it was learned
    from gives the vector something to hold on to.
    """
    get = card.get if isinstance(card, dict) else (lambda k, d=None: getattr(card, k, d))
    parts = [
        str(get("lesson", "") or ""),
        str(get("goal", "") or ""),
        str(get("reasoning_signature", "") or ""),
        " ".join(str(f) for f in (get("quality_flags", []) or [])),
    ]
    surface = get("raw_surface", {}) or {}
    if isinstance(surface, dict):
        parts.append(_PROBLEM_ID.sub("", str(surface.get("formal_statement") or ""))[:400])
    return " ".join(part for part in parts if part.strip())


def rerank_with_llm(
    query: str,
    cards: Sequence[Any],
    *,
    limit: int,
    call: Any,
    model: str,
) -> List[Any]:
    """Reorder a shortlist by asking a model which entries actually apply.

    Runs last and on a shortlist only. A reranker repairs good candidates in the
    wrong order; it cannot repair the wrong candidates, which is what the
    embedding step and the query rewrite are for. Any failure returns the input
    order, because a ranking that silently empties on a transport error is worse
    than an imperfect one.
    """
    if len(cards) <= limit:
        return list(cards)
    listing = "\n".join(
        f"[{index}] {card_text(card)[:300]}" for index, card in enumerate(cards)
    )
    prompt = (
        f"A planner is about to create new problems in this setting:\n\n{query[:1200]}\n\n"
        f"Past lessons available:\n{listing}\n\n"
        f"Which {limit} lessons bear on *this* setting? A lesson about an area or "
        f"operator that cannot arise here is useless however well written. "
        f'Return JSON only: {{"keep": [indices, most useful first]}}'
    )
    try:
        raw = call(prompt=prompt, model=model)
        match = re.search(r"\{.*\}", str(raw or ""), re.S)
        order = json.loads(match.group(0))["keep"] if match else []
        picked = [cards[i] for i in order if isinstance(i, int) and 0 <= i < len(cards)]
    except Exception:
        return list(cards)[:limit]
    if not picked:
        return list(cards)[:limit]
    for card in cards:
        if len(picked) >= limit:
            break
        if card not in picked:
            picked.append(card)
    return picked[:limit]


def search_memory(
    query_text: str,
    cards: Sequence[Any],
    *,
    limit: int = 8,
    prior: Optional[Sequence[float]] = None,
    store: Optional[EmbeddingStore] = None,
    shortlist: int = 24,
    lam: float = DEFAULT_LAMBDA,
    reranker: Optional[Any] = None,
    rerank_model: str = "",
) -> List[Any]:
    """Embed, diversify, then optionally rerank — in that order.

    Falls back to the caller's `prior` ordering whenever embedding is
    unavailable, so an unreachable embedding host degrades ranking rather than
    the run.
    """
    if not cards:
        return []
    store = store or EmbeddingStore()
    vectors = store.embed([query_text] + [card_text(c) for c in cards])
    query_vector, card_vectors = vectors[0], vectors[1:]
    if not query_vector or not any(card_vectors):
        ordered = sorted(
            range(len(cards)),
            key=lambda i: (prior[i] if prior and i < len(prior) else 0.0),
            reverse=True,
        )
        return [cards[i] for i in ordered[:limit]]

    pairs = list(zip(cards, card_vectors))
    picked = mmr_select(query_vector, pairs, limit=min(shortlist, len(pairs)), lam=lam, prior=prior)
    if reranker and rerank_model and len(picked) > limit:
        return rerank_with_llm(
            query_text, picked, limit=limit, call=reranker, model=rerank_model
        )
    return picked[:limit]

