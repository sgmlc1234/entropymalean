import asyncio

from src.retrieval.jixia_memory import analyze_lean_code_for_memory, jixia_toolchain_matches
from src.retrieval.leansearch import (
    format_premise_pack,
    normalize_leansearch_response,
    retrieve_premise_pack,
    should_retrieve_for_diagnostics,
)


def test_leansearch_response_normalizes_compact_candidates():
    payload = [
        [
            {
                "result": {
                    "module_name": ["Mathlib", "Data", "Rat"],
                    "kind": "theorem",
                    "name": ["Irrational", "add_ratCast"],
                    "signature": "theorem Irrational.add_ratCast ...",
                    "type": "Irrational x -> Irrational (x + q)",
                    "informal_name": "irrational add rational",
                    "informal_description": "adding a rational preserves irrationality",
                    "value": "",
                }
            }
        ]
    ]

    candidates = normalize_leansearch_response(payload)

    assert candidates[0].name == "Irrational.add_ratCast"
    assert candidates[0].module_name == "Mathlib.Data.Rat"
    assert candidates[0].informal_name == "irrational add rational"


def test_leansearch_cache_and_validation_are_mockable(tmp_path):
    calls = {"network": 0, "validator": 0}

    def fake_post(endpoint, payload, timeout):
        calls["network"] += 1
        return [[{"result": {"name": ["Nat", "gcd_comm"], "signature": "Nat.gcd_comm"}}]]

    def validator(candidate):
        calls["validator"] += 1
        return candidate.name == "Nat.gcd_comm"

    async def run():
        first = await retrieve_premise_pack(
            "gcd commutativity",
            endpoint="https://example.test/search",
            cache_dir=tmp_path,
            repo_root=tmp_path,
            http_post_json=fake_post,
            validator=validator,
        )
        second = await retrieve_premise_pack(
            "gcd commutativity",
            endpoint="https://example.test/search",
            cache_dir=tmp_path,
            repo_root=tmp_path,
            http_post_json=fake_post,
            validator=validator,
        )
        return first, second

    first, second = asyncio.run(run())

    assert calls["network"] == 1
    assert calls["validator"] == 2
    assert first.validated_candidates[0].name == "Nat.gcd_comm"
    assert second.cache_hit is True
    assert "Formal name: Nat.gcd_comm" in format_premise_pack(second)


def test_leansearch_diagnostic_trigger_terms():
    assert should_retrieve_for_diagnostics("invalid field add_rat for type Irrational")
    assert should_retrieve_for_diagnostics("failed to synthesize OfNat")
    assert not should_retrieve_for_diagnostics("complete")


def test_jixia_toolchain_mismatch_skips(tmp_path):
    repo = tmp_path / "repo"
    jixia = tmp_path / "jixia"
    repo.mkdir()
    jixia.mkdir()
    (repo / "lean-toolchain").write_text("leanprover/lean4:v4.30.0-rc2", encoding="utf-8")
    (jixia / "lean-toolchain").write_text("leanprover/lean4:v4.29.0", encoding="utf-8")

    assert not jixia_toolchain_matches(repo, jixia)
    result = analyze_lean_code_for_memory(
        "theorem t : True := by trivial",
        repo_root=repo,
        jixia_dir=jixia,
    )
    assert result["reason"] == "jixia_skipped_toolchain_mismatch"

