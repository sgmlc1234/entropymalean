import json

from src.retrieval.novelty_memory import (
    build_novelty_memory_pack,
    cards_from_rows,
    evaluate_candidate_novelty,
    exact_blockers,
    format_novelty_memory_pack,
    load_jsonl_cards,
    retrieve_similar_cards,
)


def test_novelty_memory_jaccard_ranks_matching_family_and_exact_blockers(tmp_path):
    accepted_path = tmp_path / "accepted.jsonl"
    rows = [
        {
            "problem_id": "accepted_gcd",
            "family": "gcd",
            "statement": "Find GCD(84, 126).",
            "answer": "42",
            "generated_params": {"a": 84, "b": 126},
        },
        {
            "problem_id": "accepted_units",
            "family": "units_digit",
            "statement": "Find the units digit of 7^2026.",
            "answer": "9",
            "generated_params": {"base": 7, "exp": 2026},
        },
    ]
    accepted_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    cards = load_jsonl_cards(accepted_path, source_kind="accepted")

    query = {
        "problem_id": "candidate",
        "family": "gcd",
        "statement": "Find GCD(84, 126).",
        "answer": "42",
        "generated_params": {"a": 84, "b": 126},
    }

    matches = retrieve_similar_cards(query, cards, k=2)
    blockers = exact_blockers(query, cards)

    assert matches[0]["problem_id"] == "accepted_gcd"
    assert any(blocker["kind"] == "statement_sha256" for blocker in blockers)
    assert any(blocker["kind"] == "numeric_family_params" for blocker in blockers)


def test_structural_overlap_is_not_hard_rejected():
    cards = cards_from_rows(
        [
            {
                "problem_id": "accepted_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "generated_params": {"a": 84, "b": 126},
            }
        ],
        source_kind="accepted",
    )
    candidate = {
        "problem_id": "candidate",
        "family": "gcd",
        "statement": "Find GCD(96, 64).",
        "answer": "32",
        "generated_params": {"a": 96, "b": 64},
    }

    assessment = evaluate_candidate_novelty(candidate, cards, k=1)

    assert assessment["verdict"] == "structural_overlap"
    assert assessment["exact_blockers"] == []
    assert assessment["gate_cards"][0]["problem_id"] == "accepted_gcd"


def test_novelty_memory_pack_separates_exact_blockers_and_soft_neighbors(tmp_path):
    accepted_path = tmp_path / "accepted.jsonl"
    accepted = {
        "problem_id": "accepted_gcd",
        "family": "gcd",
        "statement": "Find GCD(84, 126).",
        "answer": "42",
        "generated_params": {"a": 84, "b": 126},
    }
    accepted_path.write_text(json.dumps(accepted, ensure_ascii=False) + "\n", encoding="utf-8")

    pack = build_novelty_memory_pack(
        [
            {
                "id": "seed_gcd",
                "family": "gcd",
                "statement": "Find GCD(84, 126).",
                "answer": "42",
                "generated_params": {"a": 84, "b": 126},
            }
        ],
        accepted_ledger_path=accepted_path,
        run_rows=[],
    )
    rendered = format_novelty_memory_pack(pack)

    assert pack["planner_view"]["exact_blockers"]["accepted"][0]["query_problem_id"] == "seed_gcd"
    assert pack["planner_view"]["soft_neighbors"]["accepted"][0]["problem_id"] == "accepted_gcd"
    assert "exact_blockers" in rendered
    assert "soft_neighbors" in rendered
