"""Tests for the seed-lineage proof-gap aggregation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_parent_child_ablation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_parent_child_ablation", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_seed_resolver_follows_multihop_accepted_lineage():
    ablation = _load_script_module()
    accepted = [
        {
            "benchmark": "minif2f",
            "problem_id": "child",
            "parents": [{"parent_id": "mid"}],
        },
        {
            "benchmark": "minif2f",
            "problem_id": "child",
            "parents": [{"parent_id": "seed_c"}],
        },
        {
            "benchmark": "minif2f",
            "problem_id": "mid",
            "parents": [{"parent_id": "seed_a"}, {"parent_id": "seed_b"}],
        },
    ]
    graph = ablation.build_parent_graph(accepted)
    resolve = ablation.make_root_seed_resolver(
        {"minif2f": {"seed_a", "seed_b", "seed_c"}},
        graph,
    )

    roots, unresolved = resolve("minif2f", "child")

    assert set(roots) == {"seed_a", "seed_b", "seed_c"}
    assert set(unresolved) == set()


def test_root_seed_resolver_recovers_synthetic_crossover_and_mutation_ids():
    ablation = _load_script_module()
    resolve = ablation.make_root_seed_resolver(
        {"minif2f": {"seed_a", "seed_b", "seed_c"}},
        {},
    )

    crossover_roots, crossover_unresolved = resolve(
        "minif2f",
        "seed_a__theorem_gen1__x__seed_b__gen2_filtered_sum",
    )
    mutation_roots, mutation_unresolved = resolve(
        "minif2f",
        "seed_c__theorem_gen1__theorem_gen2",
    )
    unknown_roots, unknown_unresolved = resolve("minif2f", "unknown__theorem_gen1")
    partial_roots, partial_unresolved = resolve(
        "minif2f",
        "seed_a__x__unknown__theorem_gen1",
    )

    assert set(crossover_roots) == {"seed_a", "seed_b"}
    assert set(crossover_unresolved) == set()
    assert set(mutation_roots) == {"seed_c"}
    assert set(mutation_unresolved) == set()
    assert set(unknown_roots) == set()
    assert set(unknown_unresolved) == {"unknown__theorem_gen1"}
    assert set(partial_roots) == set()
    assert set(partial_unresolved) == {"seed_a__x__unknown__theorem_gen1"}


def test_entropy_direction_stats_separate_gap_and_rescue_cases():
    ablation = _load_script_module()
    row = {
        "problem_id": "child",
        "curation": {"entropy_direction": "decrease"},
    }
    stats = ablation._new_direction_stat()

    ablation._bump_direction_stats(
        stats,
        row=row,
        op_type="mutation",
        bench="minif2f",
        model="ModelA",
        entropy_direction="decrease",
        root_seed_ids=["seed_a"],
        seed_rows_available=True,
        seeds_ok=False,
        child_passed=True,
    )
    summary = ablation._finalize_direction_summary({"decrease": stats})

    assert summary[0]["seed_unsolved"] == 1
    assert summary[0]["seed_solved"] == 0
    assert summary[0]["rescue"] == 1
    assert summary[0]["rescue_rate"] == 1.0
    assert summary[0]["lpg_rate"] is None
