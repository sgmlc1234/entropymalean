"""Proof tree dataclasses for the LeanDojo BFS evaluator.

Direct port of ByteDance-Seed/BFS-Prover-V2/src/search/proof_tree.py to
our codebase: same node/edge structure, same status semantics, same
``InternalNode`` priority + distance-to-proof bookkeeping. The only
changes are:

* ``loguru.logger`` replaced with the standard ``logging`` module so we
  do not pull a new dependency just for two log calls.
* No Ray decorators (the BFS-V2 reference only uses Ray on the prover
  level, not on the tree itself).

The contract:
    Status: PROVED / FAILED / OPEN.
    Node: abstract base — InternalNode, ProofFinishedNode, ErrorNode.
    Edge: a tactic application from src to dst with timing + logprob.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Iterable, List, Optional, Tuple, Union

from lean_dojo import (
    Dojo,
    LeanError,
    ProofFinished,
    ProofGivenUp,
    TacticState,
    Theorem,
)

logger = logging.getLogger(__name__)


class Status(Enum):
    PROVED = "Proved"
    FAILED = "Failed"
    OPEN = "Open"


class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status: ...

    @property
    @abstractmethod
    def distance_to_proof(self) -> float: ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool: ...


@dataclass
class ProofFinishedNode(Node):
    inner: Union[ProofFinished, ProofGivenUp]
    depth: int
    status = Status.PROVED
    distance_to_proof = 0
    is_terminal = True


@dataclass
class ErrorNode(Node):
    inner: LeanError
    depth: int
    status = Status.FAILED
    distance_to_proof = math.inf
    is_terminal = True


@total_ordering
@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """Non-terminal node in the search tree.

    Equality / hash is purely on ``state`` (BFS-V2 convention) so two
    nodes representing the same proof state at different depths are
    deduplicated by the search.
    """

    state: TacticState = field(compare=True)
    depth: int
    cumulative_logprob: float = field(compare=False, repr=False)

    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    is_terminal = False  # type: ignore[override]

    # --- explored / out_edges ---

    @property
    def out_edges(self) -> Optional[List["Edge"]]:
        return self._out_edges

    @out_edges.setter
    def out_edges(self, edges: Iterable["Edge"]) -> None:
        if self.is_explored:
            raise RuntimeError("Node is already explored.")
        self._out_edges = list(edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    @property
    def is_explored(self) -> bool:
        return self._out_edges is not None

    # --- status propagation ---

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s: Status) -> None:
        self._status = s

    def _recompute_status(self) -> None:
        assert self.is_explored and self._out_edges is not None
        if self._status != Status.OPEN:
            return
        if any(edge.dst.status == Status.PROVED for edge in self._out_edges):
            self._status = Status.PROVED
        if all(edge.dst.status == Status.FAILED for edge in self._out_edges):
            self._status = Status.FAILED
        if self._status != Status.OPEN:
            for edge in self.in_edges:
                edge.src._recompute_status()

    # --- distance to proof ---

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self) -> None:
        if self._out_edges:
            distance = min(edge.distance_to_proof() for edge in self._out_edges)
        else:
            distance = math.inf
        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()

    # --- priority queue ordering (max-heap on priority) ---

    @property
    def priority(self) -> float:
        return self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority

    # --- proof extraction ---

    def extract_proof(self) -> Optional[List["Edge"]]:
        if self.status != Status.PROVED:
            return None
        assert self.is_explored and self._out_edges
        proving_edge = min(self._out_edges, key=Edge.distance_to_proof)
        if proving_edge.dst.is_terminal:
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return [proving_edge]
        assert isinstance(proving_edge.dst, InternalNode)
        child = proving_edge.dst.extract_proof()
        assert child is not None
        return [proving_edge, *child]


@dataclass
class Edge:
    tactic: str
    src: InternalNode = field(repr=False)
    dst: Node = field(repr=False)
    time: float = field(repr=False)
    logprob: float = field(repr=False)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def validate_proof(
    theorem: Theorem, trajectory: List[str], *, focus_mode: bool = False
) -> bool:
    """Replay the tactic sequence in a fresh Dojo session, return True iff
    the final state is ``ProofFinished``.

    Mirrors BFS-V2's ``validate_proof`` (proof_tree.py:253). Used after a
    successful search to confirm the proof is reproducible.
    """
    started = time.time()
    state = None
    try:
        with Dojo(theorem, timeout=600) as (dojo, init):
            state = init
            for idx, tactic in enumerate(trajectory):
                tac = f"focus {tactic}" if focus_mode else tactic
                state = dojo.run_tac(state, tac)
                if idx != len(trajectory) - 1 and not isinstance(state, TacticState):
                    logger.warning(
                        "validate_proof: replay '%s' returned %s at step %d",
                        tactic,
                        type(state).__name__,
                        idx,
                    )
                    return False
    except Exception as exc:  # pragma: no cover - depends on Dojo runtime
        logger.warning("validate_proof exception: %s", exc)
        return False
    logger.info(
        "validate_proof took %.2fs for %s, result=%s",
        time.time() - started,
        theorem.full_name,
        type(state).__name__,
    )
    return isinstance(state, ProofFinished)


def extract_proof_data(
    proof_edges: Optional[List[Edge]], *, focus_mode: bool = False
) -> Tuple[
    Optional[Tuple[Tuple[str, str], ...]],
    Optional[Tuple[Tuple[float, float], ...]],
]:
    """Convert a successful edge sequence into (state, tactic) and
    (time, logprob) tuples ready for JSONL serialisation.
    """
    if not proof_edges:
        return None, None
    proof = tuple(
        (
            getattr(e.src.state, "pp1", None) if focus_mode else e.src.state.pp,
            e.tactic,
        )
        for e in proof_edges
    )
    stats = tuple((e.time, e.logprob) for e in proof_edges)
    return proof, stats
