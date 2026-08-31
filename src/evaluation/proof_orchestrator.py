"""Pass@K aggregator for the multi-turn proof evaluator.

Pairs with ``multi_turn_prover.prove_with_refinement`` the same way
``orchestrator.py`` paired with ``model_runner.run_model_panel``.

Output:

- ``<bench>_proof_eval.jsonl``: one record per (problem, model)
  (each record contains all K attempts and all turns within them);
- ``<bench>_proof_summary.json``: per-(benchmark, model) cell with
  Pass@K, turn distribution, sorry rate, and bootstrap CIs over row
  indices.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.evaluation.bfs_step_prover import (
    StepProblemAttempt,
    default_tactic_sampler,
    prove_bfs_step,
)
from src.evaluation.bfs_tree_prover import (
    TreeProblemAttempt,
    prove_bfs_tree,
)
from src.evaluation.bootstrap_ci import bootstrap_ci, bootstrap_drop_ci
from src.evaluation.dataset import EvalRow, ensure_auto_implicit_false
from src.evaluation.lean_comparator import (
    LeanComparatorGate,
    validate_comparator_runtime,
)
from src.evaluation.lean_verifier import verify_lean_proof
from src.evaluation.lean_repl_verifier import (
    close_global_repl_verifier,
    verify_lean_proof_repl,
)
from src.evaluation.model_runner import (
    MODEL_PANEL,
    ModelConfig,
    _client_for,
)
from src.evaluation.multi_turn_prover import (
    DEFAULT_HEADER,
    ProblemAttempt,
    _default_model_call,
    prove_with_refinement,
)


def _select_verifier():
    """Return the verifier coroutine the orchestrator should use.

    Toggle via the ``LEAN_VERIFIER`` env var:
    - ``LEAN_VERIFIER=repl`` (default for chat-paradigm campaigns since
      2026-05-20): use the persistent ``lake exe repl`` (50–500× faster
      after warm-up; mirrors the official Goedel-Prover-V2 pipeline).
    - ``LEAN_VERIFIER=file`` (legacy): spawn ``lake env lean`` per
      candidate. Slower (5–50 s per verify cold) but no pexpect
      dependency.
    """
    pick = os.getenv("LEAN_VERIFIER", "repl").lower()
    if pick == "repl":
        return verify_lean_proof_repl
    return verify_lean_proof


def _build_protocol_metadata(
    repo_root: Path,
    model: ModelConfig,
    *,
    K: int,
    T_max: int,
    S_max: int,
    n_per_step: int,
    n_parallel: int,
    bfs_tree_search: bool,
    bfs_tree_max_nodes: int,
    lean_timeout: float,
    model_timeout: float,
    max_tokens: Optional[int],
    comparator_enabled: bool,
    run_purpose: str,
) -> Dict[str, Any]:
    """Return the immutable budget, decoding, and version manifest for a row."""
    toolchain_path = repo_root / "lean-toolchain"
    lean_toolchain = (
        toolchain_path.read_text().strip()
        if toolchain_path.is_file()
        else None
    )
    mathlib_revision = None
    manifest_path = repo_root / "lake-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            mathlib_revision = next(
                (
                    package.get("rev")
                    for package in manifest.get("packages") or []
                    if package.get("name") == "mathlib"
                ),
                None,
            )
        except (OSError, json.JSONDecodeError):
            mathlib_revision = None

    repo_commit = None
    repo_dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        repo_commit = commit.stdout.strip()
        repo_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    first_temperature = (
        model.first_temperature
        if model.first_temperature is not None
        else 1.0
    )
    refine_temperature = (
        model.refine_temperature
        if model.refine_temperature is not None
        else 0.0
    )
    return {
        "schema_version": "eml-proof-eval-v2",
        "run_purpose": run_purpose,
        "model": {
            "label": model.label,
            "provider_slug": model.provider_slug,
            "backend": model.backend,
            "paradigm": model.paradigm,
            "prompt_style": model.prompt_style,
        },
        "budget": {
            "K": K,
            "T_max": T_max,
            "S_max": S_max,
            "n_per_step": n_per_step,
            "n_parallel": n_parallel,
            "bfs_tree_search": bfs_tree_search,
            "bfs_tree_max_nodes": bfs_tree_max_nodes,
            "early_stop_on_success": True,
        },
        "decoding": {
            "first_temperature": first_temperature,
            "refine_temperature": refine_temperature,
            "top_p": (
                model.top_p
                if model.top_p is not None
                else (0.95 if model.paradigm == "chat" else 1.0)
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else model.max_tokens
            ),
            "base_seed": model.seed,
            "seed_schedule": (
                "base_seed + zero_based_model_call_index"
                if model.seed is not None
                else None
            ),
        },
        "verification": {
            "backend": os.getenv("LEAN_VERIFIER", "repl").lower(),
            "lean_timeout_seconds": lean_timeout,
            "model_timeout_seconds": model_timeout,
            "comparator_enabled": comparator_enabled,
        },
        "provenance": {
            "repo_commit": repo_commit,
            "repo_dirty": repo_dirty,
            "lean_toolchain": lean_toolchain,
            "mathlib_revision": mathlib_revision,
        },
    }


_AXIOM_DECLARATION_RE = re.compile(
    r"(?m)^[ \t]*(?:axiom|constant)[ \t]+[A-Za-z_]"
)
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)


def _validate_trusted_row(row: EvalRow) -> None:
    """Reject benchmark inputs that can inject an admitted shortcut."""
    trusted_source = "\n".join(
        [row.lean_header or "", row.formal_statement or ""]
    )
    without_blocks = _BLOCK_COMMENT_RE.sub("", trusted_source)
    without_comments = "\n".join(
        line.split("--", 1)[0] for line in without_blocks.splitlines()
    )
    if _AXIOM_DECLARATION_RE.search(without_comments):
        raise ValueError(
            f"{row.benchmark}/{row.problem_id}: trusted benchmark source "
            "contains an axiom/constant declaration"
        )


def _comparator_workspace_name(
    row: EvalRow,
    model: ModelConfig,
    formal_prefix: str,
) -> str:
    identity = "\0".join(
        [row.benchmark, row.arm, row.problem_id, model.label, formal_prefix]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{row.benchmark}-{row.problem_id}-{model.label}",
    ).strip("_.-")
    return f"{readable[:80] or 'problem'}-{digest}"


async def _run_cell(
    row: EvalRow,
    model: ModelConfig,
    client,
    *,
    K: int,
    T_max: int,
    S_max: int,
    n_per_step: int,
    n_parallel: int,
    bfs_tree_search: bool,
    bfs_tree_max_nodes: int,
    lean_timeout: float,
    model_timeout: float,
    max_tokens: Optional[int],
    formal_prefix_override: Optional[str],
    comparator_enabled: bool = False,
    comparator_bin: str = "comparator",
    comparator_work_root: Optional[Path] = None,
    comparator_mathlib_dir: Optional[Path] = None,
    comparator_timeout: float = 600.0,
    comparator_lean_toolchain: Optional[str] = None,
):
    """Evaluate a single (problem, model) cell under the K-attempt protocol.

    Dispatches on ``model.paradigm``:
    - ``"chat"`` (default): whole-proof + verifier-feedback refinement loop.
    - ``"completion"``: tactic-step BFS-V2-aligned multi-candidate search.
    """
    # Priority order for the Lean prefix shown to the model:
    # 1) explicit override (e.g. a JSONL with a freshly rendered theorem),
    # 2) the EvalRow's own formal_statement (native miniF2F / PutnamBench
    #    rows and EMG-2 certified treatment rows carry one),
    # 3) fall back to the natural-language statement so the model is at
    #    least informed even when no native Lean prefix exists.
    formal_prefix = (
        formal_prefix_override
        or row.formal_statement
        or row.statement
    )
    header = ensure_auto_implicit_false(row.lean_header or DEFAULT_HEADER)
    verifier = _select_verifier()
    if comparator_enabled:
        if comparator_work_root is None or comparator_mathlib_dir is None:
            raise ValueError(
                "comparator_work_root and comparator_mathlib_dir are required "
                "when comparator is enabled"
            )
        verifier = LeanComparatorGate(
            fast_verifier=verifier,
            header=header,
            formal_prefix=formal_prefix,
            workspace_dir=(
                comparator_work_root
                / _comparator_workspace_name(row, model, formal_prefix)
            ),
            mathlib_dir=comparator_mathlib_dir,
            lean_toolchain=comparator_lean_toolchain
            or "leanprover/lean4:v4.30.0-rc2",
            comparator_bin=comparator_bin,
            comparator_timeout=comparator_timeout,
        )

    if getattr(model, "paradigm", "chat") == "completion":
        model_call_index = 0

        async def sampler(config: ModelConfig, prompt: str, n: int):
            nonlocal model_call_index
            scheduled = config
            if config.seed is not None:
                scheduled = replace(
                    config,
                    seed=config.seed + model_call_index,
                )
                model_call_index += n
            return await default_tactic_sampler(
                scheduled, client, prompt, n, timeout_seconds=model_timeout,
            )

        if bfs_tree_search:
            return await prove_bfs_tree(
                benchmark=row.benchmark,
                arm=row.arm,
                problem_id=row.problem_id,
                statement=row.statement,
                formal_prefix=formal_prefix,
                header=header,
                model_config=model,
                tactic_sampler=sampler,
                verifier=verifier,
                K=K,
                n_per_step=n_per_step,
                max_nodes=bfs_tree_max_nodes,
                timeout_per_attempt_s=lean_timeout * bfs_tree_max_nodes,
                lean_timeout=lean_timeout,
            )

        return await prove_bfs_step(
            benchmark=row.benchmark,
            arm=row.arm,
            problem_id=row.problem_id,
            statement=row.statement,
            formal_prefix=formal_prefix,
            header=header,
            model_config=model,
            tactic_sampler=sampler,
            verifier=verifier,
            K=K,
            S_max=S_max,
            n_per_step=n_per_step,
            lean_timeout=lean_timeout,
        )

    model_call_index = 0

    async def call(config, system, user, temperature, mtokens):
        nonlocal model_call_index
        scheduled = config
        if config.seed is not None:
            scheduled = replace(
                config,
                seed=config.seed + model_call_index,
            )
            model_call_index += 1
        return await _default_model_call(
            scheduled,
            system,
            user,
            temperature,
            mtokens,
            client=client,
            timeout_seconds=model_timeout,
        )

    proof_max_tokens = max_tokens if max_tokens is not None else model.max_tokens
    return await prove_with_refinement(
        benchmark=row.benchmark,
        arm=row.arm,
        problem_id=row.problem_id,
        statement=row.statement,
        formal_prefix=formal_prefix,
        header=header,
        model_config=model,
        model_call=call,
        verifier=verifier,
        K=K,
        T_max=T_max,
        n_parallel=n_parallel,
        max_tokens=proof_max_tokens,
        lean_timeout=lean_timeout,
    )


def _has_bfs_candidate_evidence(record: Dict[str, Any]) -> bool:
    """Return whether a BFS row contains at least one generated tactic.

    A zero-candidate BFS row means the remote model call or extraction path
    failed before the prover actually searched. Treating such a row as
    complete makes ``--resume`` permanently skip the problem, which biases the
    evaluation as a false negative.
    """
    if (
        record.get("model") != "BFS-Prover-V2-7B"
        and record.get("paradigm") not in {"step_level", "tree_search"}
    ):
        return True
    for attempt in record.get("attempts") or []:
        for diag in attempt.get("expansion_diagnostics") or []:
            if diag.get("candidates"):
                return True
        for step in attempt.get("step_diagnostics") or []:
            if step.get("candidates"):
                return True
    return False


def _has_system_error(value: Any) -> bool:
    """Recursively detect explicit verifier/model infrastructure failures."""
    if isinstance(value, dict):
        if value.get("system_error"):
            return True
        return any(_has_system_error(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_system_error(item) for item in value)
    return False


def _completed_cells(jsonl_path: Path) -> set:
    """Return the set of ``(problem_id, model_label)`` cells already in the
    JSONL — used by ``run_proof_evaluation`` when ``resume=True`` to skip
    work the previous run already finished.
    """
    if not jsonl_path.exists():
        return set()
    seen: set = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("problem_id")
            model = r.get("model")
            if pid and model and _has_bfs_candidate_evidence(r):
                seen.add((pid, model))
    return seen


async def run_proof_evaluation(
    rows: Sequence[EvalRow],
    output_jsonl: Path,
    *,
    K: int = 3,
    T_max: int = 4,
    S_max: int = 6,
    n_per_step: int = 8,
    n_parallel: int = 1,
    bfs_tree_search: bool = False,
    bfs_tree_max_nodes: int = 64,
    max_parallel_cells: int = 4,
    models: Optional[Sequence[ModelConfig]] = None,
    lean_timeout: float = 300.0,
    model_timeout: float = 180.0,
    max_tokens: Optional[int] = None,
    formal_prefix_lookup: Optional[Dict[str, str]] = None,
    resume: bool = False,
    comparator_enabled: bool = False,
    comparator_bin: str = "comparator",
    comparator_work_root: Optional[Path] = None,
    comparator_mathlib_dir: Optional[Path] = None,
    comparator_timeout: float = 600.0,
    run_purpose: str = "evaluation",
) -> Path:
    """Run multi-turn proof eval over the (rows × models) grid.

    Each ``(row, model)`` produces a single JSONL record. Whole-proof
    models bundle the K attempts × T_max turns inside one
    ``ProblemAttempt``; tactic-step models bundle the K attempts ×
    S_max steps × ``n_per_step`` candidates inside one
    ``StepProblemAttempt``. The orchestrator caps concurrent cells via a
    semaphore so the Lean verifier stays responsive, and opens one
    backend-specific client per ``ModelConfig`` so mixed panels work.

    With ``resume=True`` the orchestrator appends to an existing JSONL
    and skips every ``(problem_id, model_label)`` cell already present —
    use this to recover from a mid-campaign crash without re-running the
    cells that already completed. Defaults to ``False`` so explicit opt-in
    is required.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if run_purpose not in {"smoke", "pilot", "paper", "evaluation"}:
        raise ValueError(f"unsupported run_purpose: {run_purpose}")
    if run_purpose == "paper" and not comparator_enabled:
        raise RuntimeError(
            "paper runs require --lean-comparator so exact-statement, "
            "axiom-policy, and kernel checks are enforced"
        )
    for row in rows:
        _validate_trusted_row(row)
    repo_root = Path(__file__).resolve().parents[2]
    lean_toolchain_path = repo_root / "lean-toolchain"
    lean_toolchain = lean_toolchain_path.read_text().strip()
    if comparator_enabled:
        comparator_runtime = validate_comparator_runtime(comparator_bin)
        comparator_bin = comparator_runtime["comparator"]
        comparator_work_root = (
            comparator_work_root
            or output_jsonl.parent / "comparator-workspaces"
        ).expanduser().resolve()
        comparator_mathlib_dir = (
            comparator_mathlib_dir
            or repo_root / ".lake" / "packages" / "mathlib"
        ).expanduser().resolve()
        if not comparator_mathlib_dir.is_dir():
            raise RuntimeError(
                "comparator mathlib checkout does not exist: "
                f"{comparator_mathlib_dir}; run `lake update` or pass "
                "--comparator-mathlib-dir"
            )
        comparator_work_root.mkdir(parents=True, exist_ok=True)
    panel = list(models or MODEL_PANEL)
    protocols = {
        model.label: _build_protocol_metadata(
            repo_root,
            model,
            K=K,
            T_max=T_max,
            S_max=S_max,
            n_per_step=n_per_step,
            n_parallel=n_parallel,
            bfs_tree_search=bfs_tree_search,
            bfs_tree_max_nodes=bfs_tree_max_nodes,
            lean_timeout=lean_timeout,
            model_timeout=model_timeout,
            max_tokens=max_tokens,
            comparator_enabled=comparator_enabled,
            run_purpose=run_purpose,
        )
        for model in panel
    }
    semaphore = asyncio.Semaphore(max_parallel_cells)
    clients: Dict[str, Any] = {m.label: _client_for(m) for m in panel}
    started = time.monotonic()
    written = 0

    already_done: set = _completed_cells(output_jsonl) if resume else set()
    if resume and already_done:
        print(
            f"resume: skipping {len(already_done)} already-completed "
            f"(problem, model) cells in {output_jsonl}"
        )
    open_mode = "a" if resume and output_jsonl.exists() else "w"

    async def _one(row: EvalRow, model: ModelConfig) -> Dict[str, Any]:
        formal_prefix = (
            formal_prefix_lookup.get(row.problem_id)
            if formal_prefix_lookup
            else None
        )
        async with semaphore:
            record = await _run_cell(
                row,
                model,
                clients[model.label],
                K=K,
                T_max=T_max,
                S_max=S_max,
                n_per_step=n_per_step,
                n_parallel=n_parallel,
                bfs_tree_search=bfs_tree_search,
                bfs_tree_max_nodes=bfs_tree_max_nodes,
                lean_timeout=lean_timeout,
                model_timeout=model_timeout,
                max_tokens=max_tokens,
                formal_prefix_override=formal_prefix,
                comparator_enabled=comparator_enabled,
                comparator_bin=comparator_bin,
                comparator_work_root=comparator_work_root,
                comparator_mathlib_dir=comparator_mathlib_dir,
                comparator_timeout=comparator_timeout,
                comparator_lean_toolchain=lean_toolchain,
            )
        summary = record.to_summary()
        summary["verification_policy"] = (
            "fast+leanprover/comparator"
            if comparator_enabled
            else f"fast:{os.getenv('LEAN_VERIFIER', 'repl').lower()}"
        )
        summary["had_system_error"] = (
            _has_system_error(summary)
            or not _has_bfs_candidate_evidence(summary)
        )
        protocol = dict(protocols[model.label])
        trusted_prefix = str(summary.get("formal_prefix") or "")
        trusted_header = ensure_auto_implicit_false(
            row.lean_header or DEFAULT_HEADER
        )
        protocol["target"] = {
            "formal_prefix_sha256": hashlib.sha256(
                trusted_prefix.encode("utf-8")
            ).hexdigest(),
            "header_sha256": hashlib.sha256(
                trusted_header.encode("utf-8")
            ).hexdigest(),
            "auto_implicit_false": "set_option autoImplicit false"
            in trusted_header,
            "exact_statement_gate": (
                "leanprover/comparator"
                if comparator_enabled
                else (
                    "original_statement_body_splice"
                    if model.prompt_style in {"goedel_v1", "goedel_v2"}
                    else "fast_verifier_only"
                )
            ),
        }
        summary["protocol"] = protocol
        return summary

    try:
        with output_jsonl.open(open_mode) as out:
            tasks = [
                _one(row, model)
                for row in rows
                for model in panel
                if (row.problem_id, model.label) not in already_done
            ]
            for fut in asyncio.as_completed(tasks):
                record = await fut
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
    finally:
        for cli in clients.values():
            close = getattr(cli, "close", None)
            if close is not None:
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
        if os.getenv("LEAN_VERIFIER", "repl").lower() == "repl":
            await close_global_repl_verifier()

    elapsed = time.monotonic() - started
    print(
        f"proof-eval done cells={written} elapsed={elapsed:.1f}s -> {output_jsonl}"
    )
    return output_jsonl


# ---------------------------------------------------------------------------
#  Summarisation
# ---------------------------------------------------------------------------


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _dedup_records(path: Path) -> List[Dict[str, Any]]:
    """Keep the latest record per evaluation cell.

    ``resume`` appends to the JSONL, and ``_completed_cells`` intentionally
    re-runs cells whose earlier record shows no generated candidates (a model
    or infrastructure failure). The superseded record stays in the file, so
    summaries must count only the last occurrence of each
    ``(benchmark, model, arm, problem_id)`` cell.
    """
    deduped: Dict[Any, Dict[str, Any]] = {}
    unkeyed = 0
    for record in _iter_records(path):
        problem_id = record.get("problem_id")
        if problem_id is None:
            unkeyed += 1
            key: Any = ("__unkeyed__", unkeyed)
        else:
            key = (
                record.get("benchmark"),
                record.get("model"),
                record.get("arm"),
                problem_id,
            )
        deduped[key] = record
    return list(deduped.values())


def summarize_proof_jsonl(
    eval_jsonl: Path,
    summary_json: Path,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Pass@K, turn distribution, sorry rate, and CIs per cell."""
    by_cell: Dict[str, Dict[str, Any]] = {}
    verification_policies: set[str] = set()
    protocols: Dict[str, Dict[str, Any]] = {}
    for r in _dedup_records(eval_jsonl):
        verification_policies.add(
            str(r.get("verification_policy") or "legacy-unspecified")
        )
        protocol = r.get("protocol")
        if isinstance(protocol, dict):
            key_protocol = json.dumps(protocol, sort_keys=True)
            protocols[key_protocol] = protocol
        key = f"{r['benchmark']}::{r['model']}::{r['arm']}"
        cell = by_cell.setdefault(
            key,
            {
                "benchmark": r["benchmark"],
                "model": r["model"],
                "arm": r["arm"],
                "rows": 0,
                "passes": 0,
                "rows_with_system_error": 0,
                "turn_counts": {1: 0, 2: 0, 3: 0, 4: 0, ">=5": 0},
                "row_acc": [],  # 1.0 if pass else 0.0, used for bootstrap
                "clean_row_acc": [],
                "turn_outcome_counts": {},
            },
        )
        cell["rows"] += 1
        cell["row_acc"].append(1.0 if r["pass_at_k"] else 0.0)
        if r.get("had_system_error"):
            cell["rows_with_system_error"] += 1
        else:
            cell["clean_row_acc"].append(1.0 if r["pass_at_k"] else 0.0)
        if r["pass_at_k"]:
            cell["passes"] += 1
            min_turn = r.get("min_turns_to_success") or 1
            key_turn = min_turn if min_turn <= 4 else ">=5"
            cell["turn_counts"][key_turn] = cell["turn_counts"].get(key_turn, 0) + 1
        for attempt in r.get("attempts") or []:
            for diagnostic in attempt.get("turn_diagnostics") or []:
                outcome = diagnostic.get("outcome")
                if outcome:
                    counts = cell["turn_outcome_counts"]
                    counts[outcome] = counts.get(outcome, 0) + 1

    summary_cells: List[Dict[str, Any]] = []
    for cell in by_cell.values():
        n = max(cell["rows"], 1)
        pass_rate = cell["passes"] / n
        mean, lo, hi = bootstrap_ci(
            cell["row_acc"],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        summary_cells.append(
            {
                "benchmark": cell["benchmark"],
                "model": cell["model"],
                "arm": cell["arm"],
                "rows": cell["rows"],
                "clean_rows": len(cell["clean_row_acc"]),
                "rows_with_system_error": cell["rows_with_system_error"],
                "pass_at_k": pass_rate,
                "pass_at_k_ci": [lo, hi],
                "clean_pass_at_k": (
                    sum(cell["clean_row_acc"]) / len(cell["clean_row_acc"])
                    if cell["clean_row_acc"]
                    else None
                ),
                "turn_distribution": cell["turn_counts"],
                "turn_outcome_counts": cell["turn_outcome_counts"],
            }
        )

    # also compute control-vs-treatment drop per (benchmark, model) for
    # the Pass@K metric, mirroring the legacy orchestrator's contract.
    drops: List[Dict[str, Any]] = []
    by_bm: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for c in summary_cells:
        bm = by_bm.setdefault(c["benchmark"], {}).setdefault(c["model"], {})
        # We need raw per-row accuracies, recompute from `by_cell`.
        key = f"{c['benchmark']}::{c['model']}::{c['arm']}"
        bm[c["arm"]] = by_cell[key]["row_acc"]
    for benchmark, models in by_bm.items():
        for model, arms in models.items():
            control = arms.get("control") or []
            treatment = arms.get("treatment") or []
            if not control or not treatment:
                continue
            mean, lo, hi = bootstrap_drop_ci(
                control,
                treatment,
                iterations=bootstrap_iterations,
                seed=seed,
            )
            drops.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "control_pass_at_k": sum(control) / len(control),
                    "treatment_pass_at_k": sum(treatment) / len(treatment),
                    "drop_pp": mean,
                    "drop_ci_pp": [lo, hi],
                    "control_rows": len(control),
                    "treatment_rows": len(treatment),
                }
            )

    payload = {
        "source_jsonl": str(eval_jsonl),
        "cells": sorted(
            summary_cells, key=lambda c: (c["benchmark"], c["model"], c["arm"])
        ),
        "drops": sorted(drops, key=lambda d: (d["benchmark"], d["model"])),
        "config": {
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
            "verification_policies": sorted(verification_policies),
            "protocols": list(protocols.values()),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2))
    return payload
