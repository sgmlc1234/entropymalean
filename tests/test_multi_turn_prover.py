"""Unit tests for the multi-turn proof evaluator.

These tests stub both the model call and the Lean verifier so the loop
logic can be exercised without external dependencies.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from src.evaluation.dataset import EvalRow
from src.evaluation.lean_verifier import (
    LeanMessage,
    LeanVerifyResult,
    contains_sorry,
    extract_lean_block,
    parse_lean_messages,
)
from src.evaluation.model_runner import ModelConfig
from src.evaluation.multi_turn_prover import (
    ModelTurnResponse,
    _default_model_call,
    prove_with_refinement,
)
from src.evaluation.proof_orchestrator import (
    _build_protocol_metadata,
    _has_bfs_candidate_evidence,
    _validate_trusted_row,
    summarize_proof_jsonl,
)


# ---------- lean_verifier helpers ----------


def test_extract_lean_block_returns_last_fence():
    text = (
        "before\n```lean4\nfoo\n```\nthen\n```lean4\nbar := by rfl\n```\nafter"
    )
    assert extract_lean_block(text).strip() == "bar := by rfl"


def test_extract_lean_block_falls_back_to_raw():
    assert extract_lean_block("just text") == "just text"


def test_parse_lean_messages_collects_errors_and_warnings():
    payload = (
        "/tmp/file.lean:3:7: error: unknown identifier 'foo'\n"
        "/tmp/file.lean:4:1: warning: unused variable 'h'\n"
        "other noise\n"
        "/tmp/file.lean:10:2: error: type mismatch\n"
    )
    msgs = parse_lean_messages(payload)
    severities = [m.severity for m in msgs]
    assert severities == ["error", "warning", "error"]
    assert msgs[0].line == 3 and msgs[0].column == 7
    assert "unknown identifier" in msgs[0].body


def test_contains_sorry_ignores_comments():
    assert contains_sorry("theorem x : True := sorry") is True
    assert contains_sorry("-- sorry\ntheorem x : True := trivial") is False
    assert contains_sorry("/- sorry -/\ntheorem x : True := trivial") is False


def test_lean_verify_result_summary_complete():
    r = LeanVerifyResult(ok=True, complete=True)
    assert r.summary() == "complete"


def test_lean_verify_result_summary_pass_but_sorry():
    r = LeanVerifyResult(ok=True, complete=False)
    summary = r.summary()
    assert "sorry" in summary or "failed" in summary


def test_lean_verify_result_summary_errors_truncated():
    errs = [LeanMessage("error", i, 0, f"err-{i}") for i in range(10)]
    r = LeanVerifyResult(ok=False, complete=False, errors=errs)
    summary = r.summary(max_errors=3)
    assert "err-0" in summary and "err-2" in summary
    assert "7 more" in summary


# ---------- multi-turn loop ----------


_DUMMY_MODEL = ModelConfig(label="MockModel", provider_slug="mock/model")


def _make_model_responses(*texts: str):
    """Return a stub model_call that emits the given texts in order."""
    queue = list(texts)
    call_count = {"value": 0}

    async def stub(config, system, user, temperature, max_tokens):
        call_count["value"] += 1
        txt = queue.pop(0) if queue else ""
        return ModelTurnResponse(raw_text=txt, elapsed_seconds=0.001)

    stub.call_count = call_count
    return stub


def _make_verifier(*results: bool):
    """Verifier stub that returns LeanVerifyResult with complete=results[i]."""
    queue = list(results)

    async def stub(code, *, timeout=120.0, extra_env=None):
        ok = bool(queue.pop(0)) if queue else False
        return LeanVerifyResult(
            ok=ok,
            complete=ok,
            errors=[] if ok else [LeanMessage("error", 1, 0, "stub error")],
            verify_time=0.001,
        )

    return stub


def _proof(text: str) -> str:
    return f"```lean4\n{text}\n```"


def test_loop_succeeds_on_first_turn():
    model_call = _make_model_responses(_proof("theorem t : True := trivial"))
    verifier = _make_verifier(True)
    record = asyncio.run(
        prove_with_refinement(
            benchmark="miniF2F",
            arm="control",
            problem_id="t1",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=3,
            T_max=4,
        )
    )
    assert record.pass_at_k is True
    assert record.min_turns_to_success == 1
    assert len(record.attempts) == 1
    assert record.attempts[0].turns[0].verify.complete is True


def test_loop_refines_to_success_on_turn_2():
    model_call = _make_model_responses(
        _proof("theorem t : True := bad"),
        _proof("theorem t : True := trivial"),
    )
    verifier = _make_verifier(False, True)
    record = asyncio.run(
        prove_with_refinement(
            benchmark="miniF2F",
            arm="treatment",
            problem_id="t2",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=3,
            T_max=4,
        )
    )
    assert record.pass_at_k is True
    assert record.min_turns_to_success == 2
    assert len(record.attempts) == 1
    assert len(record.attempts[0].turns) == 2


def test_loop_injects_initial_and_reflection_premise_context():
    seen_users: List[str] = []

    async def model_call(config, system, user, temperature, max_tokens):
        seen_users.append(user)
        if len(seen_users) == 1:
            return ModelTurnResponse(raw_text=_proof("theorem t : True := bad"), elapsed_seconds=0.001)
        return ModelTurnResponse(raw_text=_proof("theorem t : True := trivial"), elapsed_seconds=0.001)

    verifier = _make_verifier(False, True)

    async def premise_provider(**kwargs):
        phase = kwargs["phase"]
        return {
            "prompt_block": f"Formal name: Test.{phase}",
            "digest": {"phase": phase, "validated_count": 1},
        }

    record = asyncio.run(
        prove_with_refinement(
            benchmark="miniF2F",
            arm="treatment",
            problem_id="t-premise",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=2,
            premise_context_provider=premise_provider,
        )
    )

    assert record.pass_at_k is True
    assert "Formal name: Test.initial" in seen_users[0]
    assert "Formal name: Test.reflection" in seen_users[1]
    assert record.attempts[0].turns[0].premise_pack["phase"] == "initial"
    assert record.attempts[0].turns[1].premise_pack["phase"] == "reflection"


def test_loop_falls_back_to_next_attempt_when_all_turns_fail():
    # First attempt: 4 turns all fail. Second attempt: succeeds on turn 1.
    model_call = _make_model_responses(
        _proof("bad1"),
        _proof("bad2"),
        _proof("bad3"),
        _proof("bad4"),
        _proof("good"),
    )
    verifier = _make_verifier(False, False, False, False, True)
    record = asyncio.run(
        prove_with_refinement(
            benchmark="putnambench",
            arm="treatment",
            problem_id="p1",
            statement="...",
            formal_prefix="theorem p : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=3,
            T_max=4,
        )
    )
    assert record.pass_at_k is True
    assert len(record.attempts) == 2
    assert record.min_turns_to_success == 1  # second attempt succeeded on turn 1


def test_loop_exhausts_all_attempts_and_reports_failure():
    model_call = _make_model_responses(*[_proof("bad")] * 12)  # K=3 * T_max=4
    verifier = _make_verifier(*([False] * 12))
    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="control",
            problem_id="x",
            statement="...",
            formal_prefix="theorem x : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=3,
            T_max=4,
        )
    )
    assert record.pass_at_k is False
    assert len(record.attempts) == 3
    assert all(len(a.turns) == 4 for a in record.attempts)


def test_loop_handles_empty_lean_block():
    model_call = _make_model_responses("no fence here at all", _proof("theorem t : True := trivial"))
    verifier = _make_verifier(True)  # only used once
    record = asyncio.run(
        prove_with_refinement(
            benchmark="miniF2F",
            arm="control",
            problem_id="empty",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=2,
            T_max=2,
        )
    )
    # First turn: fence-less response falls back to raw text => verifier sees
    # "no fence here at all" which our stub still routes to the next result.
    # We accept either outcome but the loop must not crash.
    assert isinstance(record.pass_at_k, bool)


def test_goedel_v2_uses_fenced_code_and_original_statement():
    seen_codes: List[str] = []
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel",
        prompt_style="goedel_v2",
    )
    model_call = _make_model_responses(
        "Proof plan first.\n\n```lean4\n"
        "theorem changed_name : False := by\n"
        "  trivial\n"
        "```\n"
    )

    async def verifier(code, *, timeout=120.0, extra_env=None):
        seen_codes.append(code)
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="control",
            problem_id="goedel-stmt",
            statement="True",
            formal_prefix="theorem original_name : True := by",
            model_config=model,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=1,
        )
    )

    assert record.pass_at_k is True
    assert "theorem original_name : True := by" in seen_codes[0]
    assert "changed_name" not in seen_codes[0]
    assert "Proof plan" not in seen_codes[0]


def test_goedel_v1_uses_fenced_code_and_original_statement():
    seen_codes: List[str] = []
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel",
        prompt_style="goedel_v1",
    )
    model_call = _make_model_responses(
        "```lean4\n"
        "theorem unrelated_easy_theorem : False := by\n"
        "  trivial\n"
        "```\n"
    )

    async def verifier(code, *, timeout=120.0, extra_env=None):
        seen_codes.append(code)
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="control",
            problem_id="goedel-v1-stmt",
            statement="True",
            formal_prefix="theorem original_name : True := by",
            model_config=model,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=1,
        )
    )

    assert record.pass_at_k is True
    assert "theorem original_name : True := by" in seen_codes[0]
    assert "unrelated_easy_theorem" not in seen_codes[0]


def test_goedel_v2_accepts_statement_only_prefix():
    seen_prompts: List[str] = []
    seen_codes: List[str] = []
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel",
        prompt_style="goedel_v2",
    )

    async def model_call(config, system, user, temperature, max_tokens):
        seen_prompts.append(user)
        return ModelTurnResponse(
            raw_text=(
                "Plan.\n"
                "```lean4\n"
                "theorem generated_name : True := by\n"
                "  trivial\n"
                "```\n"
            )
        )

    async def verifier(code, *, timeout=120.0, extra_env=None):
        seen_codes.append(code)
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="minif2f",
            arm="treatment",
            problem_id="statement-only",
            statement="True",
            formal_prefix="theorem original_name : True",
            model_config=model,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=1,
        )
    )

    assert record.pass_at_k is True
    assert "theorem original_name : True := by sorry" in seen_prompts[0]
    assert "theorem original_name : True := by" in seen_codes[0]
    assert "generated_name" not in seen_codes[0]


def test_goedel_v2_rejects_unfenced_prose_before_verifier():
    verifier_called = {"value": False}
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel",
        prompt_style="goedel_v2",
    )
    model_call = _make_model_responses("### Proof plan only\nNo code block.")

    async def verifier(code, *, timeout=120.0, extra_env=None):
        verifier_called["value"] = True
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="control",
            problem_id="goedel-no-fence",
            statement="True",
            formal_prefix="theorem original_name : True := by",
            model_config=model,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=1,
        )
    )

    assert record.pass_at_k is False
    assert verifier_called["value"] is False
    assert "no fenced lean code" in record.attempts[0].turns[0].verify.summary()
    summary = record.to_summary()
    diagnostic = summary["attempts"][0]["turn_diagnostics"][0]
    assert diagnostic["candidate_proof"] == ""
    assert diagnostic["raw_model_text"] == "### Proof plan only\nNo code block."
    assert diagnostic["finish_reason"] is None
    assert diagnostic["outcome"] == "extraction_error"
    assert summary["attempts"][0]["terminated_reason"] == "turn_budget_exhausted"


def test_model_timeout_is_recorded_as_refinement_feedback():
    seen_prompts: List[str] = []

    async def model_call(config, system, user, temperature, max_tokens):
        seen_prompts.append(user)
        if len(seen_prompts) == 1:
            return ModelTurnResponse(
                raw_text="",
                finish_reason="timeout",
                error="timeout after 240s",
            )
        return ModelTurnResponse(raw_text=_proof("theorem t : True := trivial"))

    verifier = _make_verifier(True)
    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="gen0",
            problem_id="timeout",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=2,
        )
    )

    first_summary = record.attempts[0].turns[0].verify.summary()
    assert "model_timeout" in first_summary
    assert "model_timeout" in seen_prompts[1]
    assert record.pass_at_k is True
    diagnostic = record.to_summary()["attempts"][0]["turn_diagnostics"][0]
    assert diagnostic["finish_reason"] == "timeout"
    assert diagnostic["outcome"] == "model_timeout"


def test_goedel_v2_rejects_native_decide_before_verifier():
    verifier_called = {"value": False}
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel",
        prompt_style="goedel_v2",
    )
    model_call = _make_model_responses(
        _proof("theorem t : True := by\n  native_decide")
    )

    async def verifier(code, *, timeout=120.0, extra_env=None):
        verifier_called["value"] = True
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="control",
            problem_id="native-decide",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=model,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=1,
        )
    )

    assert record.pass_at_k is False
    assert verifier_called["value"] is False
    diagnostic = record.to_summary()["attempts"][0]["turn_diagnostics"][0]
    assert "native_decide" in diagnostic["summary"]
    assert diagnostic["outcome"] == "policy_rejection"


def test_protocol_metadata_pins_budget_decoding_and_toolchain(tmp_path: Path):
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.30.0-rc2")
    (tmp_path / "lake-manifest.json").write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "mathlib",
                        "rev": "mathlib-test-revision",
                    }
                ]
            }
        )
    )
    model = ModelConfig(
        label="Goedel",
        provider_slug="goedel-local",
        backend="lm_studio",
        prompt_style="goedel_v2",
        first_temperature=1.0,
        refine_temperature=0.0,
        top_p=0.95,
        max_tokens=6144,
        seed=30,
    )

    protocol = _build_protocol_metadata(
        tmp_path,
        model,
        K=1,
        T_max=2,
        S_max=6,
        n_per_step=8,
        n_parallel=1,
        bfs_tree_search=False,
        bfs_tree_max_nodes=64,
        lean_timeout=300.0,
        model_timeout=240.0,
        max_tokens=None,
        comparator_enabled=False,
        run_purpose="smoke",
    )

    assert protocol["run_purpose"] == "smoke"
    assert protocol["budget"]["K"] == 1
    assert protocol["budget"]["T_max"] == 2
    assert protocol["decoding"] == {
        "first_temperature": 1.0,
        "refine_temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 6144,
        "base_seed": 30,
        "seed_schedule": "base_seed + zero_based_model_call_index",
    }
    assert protocol["provenance"]["lean_toolchain"] == (
        "leanprover/lean4:v4.30.0-rc2"
    )
    assert protocol["provenance"]["mathlib_revision"] == (
        "mathlib-test-revision"
    )


def test_default_model_call_forwards_top_p_and_seed():
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="proof"),
                        finish_reason="stop",
                    )
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    config = ModelConfig(
        label="Goedel",
        provider_slug="goedel-local",
        top_p=0.95,
        seed=31,
    )

    response = asyncio.run(
        _default_model_call(
            config,
            "",
            "prompt",
            1.0,
            6144,
            client=client,
            timeout_seconds=10.0,
        )
    )

    assert response.finish_reason == "stop"
    assert captured["top_p"] == 0.95
    assert captured["seed"] == 31
    assert captured["temperature"] == 1.0
    assert captured["max_tokens"] == 6144


def test_trusted_row_rejects_axiom_declarations():
    row = EvalRow(
        problem_id="bad-axiom",
        benchmark="proofnet",
        arm="control",
        statement="True",
        gold_answer=None,
        formal_statement="theorem target : True := by",
        lean_header="import Mathlib\naxiom shortcut : False",
    )

    with pytest.raises(ValueError, match="axiom/constant declaration"):
        _validate_trusted_row(row)


def test_verifier_exception_is_recorded_as_refinement_feedback():
    seen_prompts: List[str] = []

    async def model_call(config, system, user, temperature, max_tokens):
        seen_prompts.append(user)
        return ModelTurnResponse(raw_text=_proof("theorem t : True := trivial"))

    async def verifier(code, *, timeout=120.0, extra_env=None):
        if len(seen_prompts) == 1:
            raise RuntimeError("lean crashed")
        return LeanVerifyResult(ok=True, complete=True)

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet",
            arm="gen0",
            problem_id="verifier-exception",
            statement="True",
            formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL,
            model_call=model_call,
            verifier=verifier,
            K=1,
            T_max=2,
        )
    )

    assert "lean_verify_exception" in record.attempts[0].turns[0].verify.summary()
    assert "lean_verify_exception" in seen_prompts[1]
    assert record.pass_at_k is True


# ---------- summarizer ----------


def test_bfs_candidate_evidence_handles_tree_and_step_records():
    assert _has_bfs_candidate_evidence(
        {
            "paradigm": "tree_search",
            "attempts": [{"expansion_diagnostics": [{"candidates": []}]}],
        }
    ) is False
    assert _has_bfs_candidate_evidence(
        {
            "paradigm": "step_level",
            "attempts": [
                {
                    "step_diagnostics": [
                        {"candidates": [{"tactic": "rfl"}]}
                    ]
                }
            ],
        }
    ) is True
    assert _has_bfs_candidate_evidence(
        {"paradigm": "chat", "attempts": []}
    ) is True


def test_summarize_proof_jsonl_computes_pass_at_k_and_turn_distribution(tmp_path: Path):
    eval_jsonl = tmp_path / "eval.jsonl"
    records = [
        {
            "benchmark": "miniF2F",
            "arm": "control",
            "problem_id": "c1",
            "model": "ModelA",
            "provider_slug": "x/y",
            "pass_at_k": True,
            "min_turns_to_success": 1,
            "total_elapsed_seconds": 1.0,
            "attempts": [],
        },
        {
            "benchmark": "miniF2F",
            "arm": "control",
            "problem_id": "c2",
            "model": "ModelA",
            "provider_slug": "x/y",
            "pass_at_k": False,
            "min_turns_to_success": None,
            "total_elapsed_seconds": 5.0,
            "attempts": [],
            "had_system_error": True,
        },
        {
            "benchmark": "miniF2F",
            "arm": "treatment",
            "problem_id": "t1",
            "model": "ModelA",
            "provider_slug": "x/y",
            "pass_at_k": True,
            "min_turns_to_success": 2,
            "total_elapsed_seconds": 3.0,
            "attempts": [],
        },
        {
            "benchmark": "miniF2F",
            "arm": "treatment",
            "problem_id": "t2",
            "model": "ModelA",
            "provider_slug": "x/y",
            "pass_at_k": False,
            "min_turns_to_success": None,
            "total_elapsed_seconds": 7.0,
            "attempts": [],
        },
        {
            "benchmark": "miniF2F",
            "arm": "treatment",
            "problem_id": "t3",
            "model": "ModelA",
            "provider_slug": "x/y",
            "pass_at_k": False,
            "min_turns_to_success": None,
            "total_elapsed_seconds": 4.0,
            "attempts": [],
        },
    ]
    eval_jsonl.write_text("\n".join(json.dumps(r) for r in records))
    summary_json = tmp_path / "summary.json"
    payload = summarize_proof_jsonl(
        eval_jsonl, summary_json, bootstrap_iterations=100
    )

    cells = {(c["benchmark"], c["model"], c["arm"]): c for c in payload["cells"]}
    control = cells[("miniF2F", "ModelA", "control")]
    treatment = cells[("miniF2F", "ModelA", "treatment")]
    assert control["pass_at_k"] == pytest.approx(0.5)
    assert treatment["pass_at_k"] == pytest.approx(1 / 3)
    assert control["rows_with_system_error"] == 1
    assert control["clean_rows"] == 1
    assert control["clean_pass_at_k"] == pytest.approx(1.0)
    assert control["turn_distribution"][1] == 1
    assert treatment["turn_distribution"][2] == 1

    drops = {(d["benchmark"], d["model"]): d for d in payload["drops"]}
    drop_cell = drops[("miniF2F", "ModelA")]
    assert drop_cell["control_pass_at_k"] == pytest.approx(0.5)
    assert drop_cell["treatment_pass_at_k"] == pytest.approx(1 / 3)
    assert drop_cell["drop_pp"] == pytest.approx(
        (0.5 - 1 / 3) * 100, rel=1e-6
    )


def test_summarize_proof_jsonl_counts_only_latest_resumed_record(tmp_path: Path):
    """``resume`` appends retries; the superseded record must not be counted.

    ``_completed_cells`` re-runs BFS cells whose earlier record shows no
    generated candidates, so the JSONL can legitimately contain two records
    for the same (benchmark, model, arm, problem_id). Only the later record
    reflects a real evaluation.
    """
    eval_jsonl = tmp_path / "eval.jsonl"
    stale = {
        "benchmark": "miniF2F",
        "arm": "treatment",
        "problem_id": "t1",
        "model": "BFS-Prover-V2-7B",
        "provider_slug": "x/y",
        "paradigm": "step_level",
        "pass_at_k": False,
        "min_turns_to_success": None,
        "total_elapsed_seconds": 2.0,
        "attempts": [],
        "had_system_error": True,
    }
    retried = dict(
        stale,
        pass_at_k=True,
        min_turns_to_success=1,
        had_system_error=False,
    )
    other = {
        "benchmark": "miniF2F",
        "arm": "treatment",
        "problem_id": "t2",
        "model": "BFS-Prover-V2-7B",
        "provider_slug": "x/y",
        "paradigm": "step_level",
        "pass_at_k": False,
        "min_turns_to_success": None,
        "total_elapsed_seconds": 3.0,
        "attempts": [],
    }
    eval_jsonl.write_text(
        "\n".join(json.dumps(r) for r in [stale, other, retried])
    )

    payload = summarize_proof_jsonl(
        eval_jsonl, tmp_path / "summary.json", bootstrap_iterations=100
    )

    cells = {(c["benchmark"], c["model"], c["arm"]): c for c in payload["cells"]}
    cell = cells[("miniF2F", "BFS-Prover-V2-7B", "treatment")]
    assert cell["rows"] == 2  # t1 counted once (latest), plus t2
    assert cell["pass_at_k"] == pytest.approx(0.5)
    assert cell["rows_with_system_error"] == 0
    assert cell["clean_rows"] == 2


def test_crashed_provider_call_is_resampled_not_charged_to_the_turn_budget():
    """A call that never happened must not spend a turn.

    Observed in a traced run: `codex exec exited with code 1` returned zero
    characters on 8 of 29 calls, each consuming a turn, so two attempts spent
    their tails on nothing. With one turn of budget the loop must still reach
    the proof by resampling the crash.
    """
    calls = {"n": 0}

    async def model_call(config, system, user, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] <= 2:
            return ModelTurnResponse(
                raw_text="",
                finish_reason="error",
                error="codex exec exited with code 1; stderr_tail=",
            )
        return ModelTurnResponse(raw_text=_proof("theorem t : True := trivial"))

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet", arm="gen0", problem_id="crash",
            statement="True", formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL, model_call=model_call,
            verifier=_make_verifier(True), K=1, T_max=1,
        )
    )

    assert record.pass_at_k is True, "resampling should have reached the proof"
    assert calls["n"] == 3, "both crashes resampled, then the real answer"
    turn = record.attempts[0].turns[0]
    assert len(turn.discarded_responses) == 2
    assert all("failed_call" in d["reason"] for d in turn.discarded_responses)
    # The waste stays visible next to the turn count it would otherwise hide in.
    assert record.to_summary()["attempts"][0]["discarded_response_count"] == 2


def test_a_short_answer_is_still_a_real_attempt() -> None:
    """Only failed calls are resampled — never answers we merely dislike.

    Rerolling a reply because it looks unpromising would discard evidence and
    let the prover sample until it got a verdict it liked.
    """
    calls = {"n": 0}

    async def model_call(config, system, user, temperature, max_tokens):
        calls["n"] += 1
        return ModelTurnResponse(raw_text=_proof("theorem t : True := by\n  norm_num"))

    record = asyncio.run(
        prove_with_refinement(
            benchmark="proofnet", arm="gen0", problem_id="short",
            statement="True", formal_prefix="theorem t : True := by",
            model_config=_DUMMY_MODEL, model_call=model_call,
            verifier=_make_verifier(False), K=1, T_max=1,
        )
    )

    assert calls["n"] == 1, "a genuine answer must not be resampled"
    assert record.attempts[0].turns[0].discarded_responses == []
