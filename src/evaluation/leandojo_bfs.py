"""Best-first proof search backed by LeanDojo + LM Studio.

Workshop-budget port of ByteDance-Seed/BFS-Prover-V2/src/search:

* The reference uses Ray + vLLM; we use plain asyncio + an
  ``AsyncOpenAI`` client pointed at LM Studio's
  ``/v1/completions`` endpoint.
* The reference ranks the priority queue by ``cumulative_logprob`` from
  vLLM. LM Studio's llama.cpp backend does not return logprobs for the
  completions endpoint, so every expansion stamps ``logprob=0.0`` and
  the priority queue degenerates to pure BFS (FIFO with a max-heap
  tie-break). We log this limitation explicitly in the resulting
  ``SearchResult.proof_stats`` so the paper can disclose it.
* Tactic filters and prompt format match BFS-V2 verbatim
  (``"{state}:::"`` prompt, drop ``sorry``/``admit``/``native_decide``,
  drop buggy ``rcases``/``cases'``/``simpa`` patterns).

Exit conditions per (theorem, attempt):
  - Proof found (root.status == Status.PROVED)
  - Priority queue empty (all reachable states explored)
  - Wall-clock timeout
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lean_dojo import (
    Dojo,
    DojoCrashError,
    DojoTacticTimeoutError,
    LeanError,
    ProofFinished,
    ProofGivenUp,
    TacticState,
    Theorem,
)

from src.evaluation.leandojo_tree import (
    Edge,
    ErrorNode,
    InternalNode,
    Node,
    ProofFinishedNode,
    Status,
    extract_proof_data,
    validate_proof,
)

logger = logging.getLogger(__name__)

BFS_SEPARATOR = ":::"
BANNED_SUBSTRINGS = ("sorry", "admit", "native_decide")
BUGGY_KEYWORDS = ("rcases", "cases'", "simpa")


# ---------------------------------------------------------------------------
#  Tactic filtering (same as bfs_step_prover, kept local for clarity)
# ---------------------------------------------------------------------------


def is_buggy_tactic(tactic: str) -> bool:
    if not tactic:
        return True
    if any(s in tactic for s in BANNED_SUBSTRINGS):
        return True
    if any(k in tactic for k in BUGGY_KEYWORDS) and "?_" in tactic:
        return True
    if "simpa" in tactic and (
        " _" in tactic or "_ " in tactic or "_," in tactic or ",_" in tactic
    ):
        return True
    return False


# ---------------------------------------------------------------------------
#  Tactic generator wrapping LM Studio
# ---------------------------------------------------------------------------


class LeanDojoTacticGenerator:
    """LM Studio AsyncOpenAI wrapper that returns BFS-V2-style suggestions.

    ``generate(state_pp, n)`` issues ``n`` parallel ``completions.create``
    calls (LM Studio ignores the OpenAI ``n`` field), filters, dedupes,
    and returns ``List[Tuple[tactic, logprob]]`` where ``logprob`` is
    always ``0.0`` because llama.cpp does not expose token logprobs.
    """

    def __init__(
        self,
        client,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = list(stop or [BFS_SEPARATOR, "\n\n"])
        self.request_timeout = request_timeout

    async def _one_call(self, prompt: str) -> Optional[str]:
        try:
            response = await asyncio.wait_for(
                self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                ),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("tactic gen exception: %s", exc)
            return None
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        text = getattr(choices[0], "text", "") or ""
        return text.strip()

    async def generate(self, tactic_state: str, n: int) -> List[Tuple[str, float]]:
        prompt = f"{tactic_state}{BFS_SEPARATOR}"
        results = await asyncio.gather(*[self._one_call(prompt) for _ in range(n)])
        seen: set = set()
        out: List[Tuple[str, float]] = []
        for text in results:
            if not text or text in seen or is_buggy_tactic(text):
                continue
            seen.add(text)
            out.append((text, 0.0))
        return out


# ---------------------------------------------------------------------------
#  Search result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    full_name: str
    file_path: str
    status: Status
    proof: Optional[Tuple[Tuple[str, str], ...]]
    total_attempts: int
    tactic_time: float
    model_time: float
    total_time: float
    total_nodes: int
    explored_nodes: int
    proof_stats: Optional[Tuple[Tuple[float, float], ...]] = field(default=None)
    proof_validation_passed: Optional[bool] = field(default=None)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "full_name": self.full_name,
            "file_path": self.file_path,
            "status": self.status.value,
            "proof": list(self.proof) if self.proof else None,
            "tactic_time": self.tactic_time,
            "model_time": self.model_time,
            "total_time": self.total_time,
            "total_nodes": self.total_nodes,
            "explored_nodes": self.explored_nodes,
            "total_attempts": self.total_attempts,
            "proof_validation_passed": self.proof_validation_passed,
            "proof_stats": list(self.proof_stats) if self.proof_stats else None,
        }


# ---------------------------------------------------------------------------
#  Best-first search loop (asyncio port of BFS-V2's BestFirstSearch)
# ---------------------------------------------------------------------------


class _BestFirstSearch:
    def __init__(
        self,
        *,
        dojo: Dojo,
        root: InternalNode,
        generator: LeanDojoTacticGenerator,
        n_sampling: int,
        timeout_s: float,
        tactic_timeout_s: float,
    ) -> None:
        self.dojo = dojo
        self.root = root
        self.generator = generator
        self.n_sampling = n_sampling
        self.timeout_s = timeout_s
        self.tactic_timeout_s = tactic_timeout_s

        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.nodes: Dict[TacticState, Node] = {root.state: root}
        # We never reach `inf` because we always push the root first.
        self.priority_queue.put_nowait((-root.priority, root))

        # Profiling.
        self.tactic_time = 0.0
        self.model_time = 0.0
        self.explored_nodes = 0
        self.total_time = 0.0

    async def _run_tactic(
        self, node: InternalNode, tactic: str
    ) -> Tuple[Any, float]:
        wall = time.time()
        try:
            response = self.dojo.run_tac(node.state, tactic)
        except DojoTacticTimeoutError as exc:
            response = LeanError(f"tactic timeout: {exc}")
        except DojoCrashError as exc:
            response = LeanError(f"dojo crash: {exc}")
        except Exception as exc:  # pragma: no cover
            response = LeanError(f"tactic exception: {type(exc).__name__}: {exc}")
        elapsed = time.time() - wall
        self.tactic_time += elapsed
        return response, elapsed

    @staticmethod
    def _wrap(
        response: Any, src: InternalNode, tactic: str, time_s: float, logprob: float
    ) -> Tuple[Edge, Node]:
        depth = src.depth + 1
        if isinstance(response, (ProofFinished, ProofGivenUp)):
            dst: Node = ProofFinishedNode(inner=response, depth=depth)
        elif isinstance(response, LeanError):
            dst = ErrorNode(inner=response, depth=depth)
        elif isinstance(response, TacticState):
            dst = InternalNode(
                state=response,
                depth=depth,
                cumulative_logprob=logprob + src.cumulative_logprob,
            )
        else:
            raise ValueError(f"unknown response type: {type(response).__name__}")
        edge = Edge(tactic=tactic, src=src, dst=dst, time=time_s, logprob=logprob)
        return edge, dst

    async def run(self) -> None:
        started = time.monotonic()
        while True:
            if self.priority_queue.empty():
                logger.info("queue empty — search exhausted")
                break
            self.total_time = time.monotonic() - started
            if self.total_time > self.timeout_s:
                logger.info("search timeout after %.1fs", self.total_time)
                break
            if self.root.status == Status.PROVED:
                logger.info("proof found at total_time=%.1fs", self.total_time)
                break

            _, node = await self.priority_queue.get()
            if not isinstance(node, InternalNode) or node.is_explored:
                continue

            # Sample candidate tactics.
            model_started = time.time()
            suggestions = await self.generator.generate(node.state.pp, self.n_sampling)
            self.model_time += time.time() - model_started
            if not suggestions:
                node.out_edges = []  # mark explored with no children → FAILED
                continue

            out_edges: List[Edge] = []
            for tactic, logprob in suggestions:
                response, elapsed = await self._run_tactic(node, tactic)
                if response in self.nodes:
                    existing = self.nodes[response]
                    edge = Edge(
                        tactic=tactic,
                        src=node,
                        dst=existing,
                        time=elapsed,
                        logprob=logprob,
                    )
                else:
                    edge, dst = self._wrap(response, node, tactic, elapsed, logprob)
                    self.nodes[response] = dst
                    if isinstance(dst, InternalNode):
                        self.priority_queue.put_nowait((-dst.priority, dst))
                out_edges.append(edge)
                if isinstance(edge.dst, InternalNode):
                    edge.dst.in_edges.append(edge)
                if isinstance(edge.dst, ProofFinishedNode):
                    break
            node.out_edges = out_edges
            self.explored_nodes += 1
        self.total_time = time.monotonic() - started


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------


async def prove_with_leandojo_bfs(
    *,
    theorem: Theorem,
    generator: LeanDojoTacticGenerator,
    K: int = 3,
    timeout_per_attempt_s: float = 300.0,
    n_sampling: int = 16,
    tactic_timeout_s: float = 10.0,
    validate: bool = True,
) -> SearchResult:
    """Run up to ``K`` independent BFS attempts; stop at first success.

    Each attempt opens a fresh ``Dojo`` session (BFS-V2 convention — the
    session holds Lean process state that cannot be shared across
    parallel attempts).
    """
    cumulative_tactic_time = 0.0
    cumulative_model_time = 0.0
    cumulative_total_time = 0.0
    cumulative_nodes = 0
    cumulative_explored = 0
    last_status = Status.OPEN
    last_proof: Optional[Tuple[Tuple[str, str], ...]] = None
    last_proof_stats: Optional[Tuple[Tuple[float, float], ...]] = None
    proof_valid: Optional[bool] = None

    attempt_used = 0
    for attempt in range(1, K + 1):
        attempt_used = attempt
        try:
            with Dojo(theorem, timeout=int(timeout_per_attempt_s) + 60) as (
                dojo,
                init_state,
            ):
                root = InternalNode(
                    state=init_state, depth=0, cumulative_logprob=0.0
                )
                bfs = _BestFirstSearch(
                    dojo=dojo,
                    root=root,
                    generator=generator,
                    n_sampling=n_sampling,
                    timeout_s=timeout_per_attempt_s,
                    tactic_timeout_s=tactic_timeout_s,
                )
                await bfs.run()
            cumulative_tactic_time += bfs.tactic_time
            cumulative_model_time += bfs.model_time
            cumulative_total_time += bfs.total_time
            cumulative_nodes += len(bfs.nodes)
            cumulative_explored += bfs.explored_nodes
            last_status = root.status

            if root.status == Status.PROVED:
                proof_edges = root.extract_proof()
                last_proof, last_proof_stats = extract_proof_data(proof_edges)
                if validate and last_proof:
                    proof_valid = validate_proof(
                        theorem, [t for _, t in last_proof]
                    )
                break  # Pass@K satisfied
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "dojo init failed for %s on attempt %d: %s",
                theorem.full_name,
                attempt,
                exc,
            )

    return SearchResult(
        full_name=theorem.full_name,
        file_path=str(theorem.file_path),
        status=last_status,
        proof=last_proof,
        proof_stats=last_proof_stats,
        total_attempts=attempt_used,
        tactic_time=cumulative_tactic_time,
        model_time=cumulative_model_time,
        total_time=cumulative_total_time,
        total_nodes=cumulative_nodes,
        explored_nodes=cumulative_explored,
        proof_validation_passed=proof_valid,
    )
