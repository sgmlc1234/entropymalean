#!/usr/bin/env python3
"""End-to-end smoke for the LeanDojo BFS arm.

Loads the (just-finished) miniF2F-Lean4 trace, picks the first traced
theorem, opens a Dojo session, runs a single tactic, and prints the
state. Verifies the four layers we'll lean on for the campaign:

1. ``TracedRepo.load_from_disk`` reads the trace cache,
2. ``Theorem`` construction + ``Dojo.__enter__`` succeed,
3. ``Dojo.run_tac`` returns a usable response,
4. our ``LeanDojoTacticGenerator`` produces non-empty candidates.

Run with:
    python scripts/archive/smoke_leandojo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root on path so ``import src.evaluation...`` resolves
# when this script is invoked from ``scripts/``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lean_dojo import (
    Dojo,
    LeanGitRepo,
    ProofFinished,
    TacticState,
    Theorem,
    TracedRepo,
    get_traced_repo_path,
)

from src.evaluation.leandojo_bfs import (
    LeanDojoTacticGenerator,
    prove_with_leandojo_bfs,
)
from src.evaluation.model_runner import _client_for, ModelConfig


REPO_URL = "https://github.com/yangky11/miniF2F-lean4"
REPO_COMMIT = "5746b7d6c47855ce1294bed87329618ff7f1bc31"


async def main() -> None:
    repo = LeanGitRepo(REPO_URL, REPO_COMMIT)
    cache_path = get_traced_repo_path(repo)
    print(f"trace cache: {cache_path}")
    traced = TracedRepo.load_from_disk(cache_path)
    theorems = list(traced.get_traced_theorems())
    print(f"traced theorems: {len(theorems)}")

    # Pick something likely-easy for a smoke (any theorem; we just want
    # the round-trip to work). Skip stub theorems with no full_name —
    # those came from our fail-soft AST patch and don't represent real
    # provable goals.
    real_theorems = [t for t in theorems if t.theorem.full_name]
    print(f"non-stub theorems: {len(real_theorems)}/{len(theorems)}")
    target = next(
        (t for t in real_theorems if "mathd_numbertheory" in t.theorem.full_name),
        real_theorems[0] if real_theorems else theorems[0],
    )
    thm = target.theorem
    print(f"smoke target: {thm.full_name} in {thm.file_path}")

    # Layer 2 + 3: Dojo session + one tactic.
    with Dojo(thm, timeout=120) as (dojo, state):
        print(f"init state goals: {len(state.goals) if hasattr(state, 'goals') else '?'}")
        candidate = "norm_num"  # cheap probe; mostly fails harmlessly
        response = dojo.run_tac(state, candidate)
        print(f"run_tac('{candidate}') → {type(response).__name__}")
        if isinstance(response, ProofFinished):
            print("  ✓ proof closed by norm_num (lucky!)")
        elif isinstance(response, TacticState):
            print(f"  ∘ progress, new goals={len(response.goals)}")
        else:
            print(f"  ✗ no progress: {str(response)[:120]}")

    # Layer 4: model sampler smoke (1-shot, n=2).
    model = ModelConfig(
        label="BFS-Prover-V2-7B",
        provider_slug=os.environ.get("EVAL_BFS_PROVER_SLUG", "bfs-prover-eval"),
        backend="lm_studio",
        paradigm="completion",
        temperature=0.7,
        max_tokens=128,
        stop=[":::", "\n\n"],
    )
    client = _client_for(model)
    generator = LeanDojoTacticGenerator(
        client, model=model.provider_slug, temperature=0.7, max_tokens=128
    )
    state_pp = str(theorems[0].theorem.full_name)  # benign filler
    suggestions = await generator.generate(state_pp, n=2)
    print(f"sampler n=2 suggestions: {suggestions}")

    # Final layer: end-to-end BFS on the smoke target, tiny budget.
    print()
    print("running prove_with_leandojo_bfs (K=1, n_sampling=4, timeout=60s)...")
    result = await prove_with_leandojo_bfs(
        theorem=thm,
        generator=generator,
        K=1,
        timeout_per_attempt_s=60.0,
        n_sampling=4,
        tactic_timeout_s=10.0,
        validate=False,
    )
    print(f"  status={result.status.value}  attempts={result.total_attempts}")
    print(f"  total_time={result.total_time:.1f}s  nodes={result.total_nodes}")
    print(f"  explored={result.explored_nodes}  proof={'yes' if result.proof else 'no'}")
    if result.proof:
        for state_pp, tactic in result.proof:
            print(f"    tactic: {tactic!r}")

    close = getattr(client, "close", None)
    if close is not None:
        maybe = close()
        if asyncio.iscoroutine(maybe):
            await maybe


if __name__ == "__main__":
    asyncio.run(main())
