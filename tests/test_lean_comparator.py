"""Tests for the optional final Lean comparator gate."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.evaluation.lean_comparator import (
    ComparatorProcessResult,
    LeanComparatorGate,
    declaration_name,
    prepare_comparator_workspace,
)
from src.evaluation.lean_verifier import LeanVerifyResult

# A realistic miniF2F-style target: explicit binders (a variable and a
# hypothesis) before the colon. The pre-fix Submission bridge failed to
# elaborate for every such theorem, so binder-carrying fixtures are the
# regression guard here.
BINDER_PREFIX = (
    "theorem target (n : Nat) (h : 0 < n) : n + 0 = n := by"
)
BINDER_CANDIDATE = (
    "set_option autoImplicit false\n"
    "theorem target (n : Nat) (h : 0 < n) : n + 0 = n := by\n"
    "  exact Nat.add_zero n\n"
)


def test_declaration_name_includes_trusted_namespace():
    assert declaration_name(
        "theorem target (n : Nat) : n = n := by",
        "import Mathlib\nnamespace Bench\nnamespace Algebra",
    ) == "Bench.Algebra.target"


def test_declaration_name_pops_dotted_namespace_blocks():
    header = (
        "import Mathlib\n"
        "namespace Bench.Algebra\n"
        "end Bench.Algebra\n"
        "namespace Live"
    )
    assert declaration_name(
        "theorem target : True := by", header
    ) == "Live.target"


def test_declaration_name_ignores_section_blocks():
    header = (
        "import Mathlib\n"
        "section Helpers\n"
        "end Helpers\n"
        "namespace Bench\n"
        "section\n"
        "end"
    )
    assert declaration_name(
        "theorem target : True := by", header
    ) == "Bench.target"


def test_prepare_comparator_workspace_official_layout(tmp_path: Path):
    """Solution.lean must be the candidate verbatim — no Submission bridge."""
    mathlib = tmp_path / "mathlib"
    mathlib.mkdir()
    workspace = tmp_path / "workspace"
    candidate = (
        "import Mathlib\n"
        "set_option autoImplicit false\n"
        f"{BINDER_CANDIDATE}"
    )

    prepared = prepare_comparator_workspace(
        workspace,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        candidate_code=candidate,
        mathlib_dir=mathlib,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
    )

    assert prepared.theorem_name == "target"
    challenge = (workspace / "Challenge.lean").read_text()
    assert f"{BINDER_PREFIX}\n  sorry" in challenge
    solution = (workspace / "Solution.lean").read_text()
    assert solution == candidate.strip() + "\n"
    assert "Submission" not in solution
    assert not (workspace / "Submission.lean").exists()
    lakefile = (workspace / "lakefile.toml").read_text()
    assert 'defaultTargets = ["Challenge", "Solution"]' in lakefile
    assert "Submission" not in lakefile
    assert str(mathlib.resolve()) in lakefile
    config = json.loads((workspace / "config.json").read_text())
    assert config["theorem_names"] == ["target"]
    assert config["permitted_axioms"] == [
        "propext",
        "Quot.sound",
        "Classical.choice",
    ]
    assert (workspace / "lean-toolchain").read_text().strip().endswith(
        "v4.30.0-rc2"
    )


def test_prepare_comparator_workspace_prepends_header_when_no_imports(
    tmp_path: Path,
):
    mathlib = tmp_path / "mathlib"
    mathlib.mkdir()
    workspace = tmp_path / "workspace"
    header = "import Mathlib\nset_option autoImplicit false"

    prepare_comparator_workspace(
        workspace,
        header=header,
        formal_prefix=BINDER_PREFIX,
        candidate_code=BINDER_CANDIDATE,
        mathlib_dir=mathlib,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
    )

    solution = (workspace / "Solution.lean").read_text()
    assert solution.startswith(header)
    assert BINDER_CANDIDATE.strip() in solution


def test_prepare_comparator_workspace_removes_stale_submission(tmp_path: Path):
    mathlib = tmp_path / "mathlib"
    mathlib.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Submission.lean").write_text("-- stale bridge artifact\n")

    prepare_comparator_workspace(
        workspace,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        candidate_code=BINDER_CANDIDATE,
        mathlib_dir=mathlib,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
    )

    assert not (workspace / "Submission.lean").exists()


@pytest.mark.skipif(
    shutil.which("lean") is None, reason="lean toolchain not on PATH"
)
def test_generated_workspace_elaborates_binder_theorem(tmp_path: Path):
    """Challenge and Solution must both be valid Lean for binder theorems.

    Runs the real ``lean`` binary on the generated files (Mathlib-free header
    so elaboration stays fast). This is the end-to-end regression test for
    the pre-fix bridge, which produced a Solution.lean that failed to
    elaborate for every theorem with explicit binders.
    """
    mathlib = tmp_path / "mathlib"
    mathlib.mkdir()
    workspace = tmp_path / "workspace"
    repo_toolchain = (
        Path(__file__).resolve().parents[1] / "lean-toolchain"
    ).read_text().strip()

    prepare_comparator_workspace(
        workspace,
        header="set_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        candidate_code=BINDER_CANDIDATE,
        mathlib_dir=mathlib,
        lean_toolchain=repo_toolchain,
    )

    for module in ("Challenge.lean", "Solution.lean"):
        result = subprocess.run(
            ["lean", module],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"{module} failed to elaborate:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def test_prepare_reuses_parent_project_pinned_packages(tmp_path: Path):
    packages = tmp_path / ".lake" / "packages"
    mathlib = packages / "mathlib"
    mathlib.mkdir(parents=True)
    (tmp_path / "lake-manifest.json").write_text(
        json.dumps(
            {
                "version": "1.2.0",
                "packagesDir": ".lake/packages",
                "packages": [
                    {
                        "type": "git",
                        "name": "mathlib",
                        "scope": "",
                        "rev": "pinned",
                        "manifestFile": "lake-manifest.json",
                        "inherited": False,
                        "configFile": "lakefile.lean",
                    },
                    {
                        "type": "git",
                        "name": "batteries",
                        "scope": "leanprover-community",
                        "rev": "also-pinned",
                        "manifestFile": "lake-manifest.json",
                        "inherited": True,
                        "configFile": "lakefile.toml",
                    },
                    {
                        "type": "git",
                        "name": "unrelated",
                        "scope": "",
                        "rev": "unused",
                        "manifestFile": "lake-manifest.json",
                        "inherited": False,
                        "configFile": "lakefile.toml",
                    },
                ],
            }
        )
    )
    workspace = tmp_path / "workspace"

    prepare_comparator_workspace(
        workspace,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix="theorem target : True := by",
        candidate_code="import Mathlib\ntheorem target : True := by trivial",
        mathlib_dir=mathlib,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
    )

    manifest = json.loads((workspace / "lake-manifest.json").read_text())
    assert manifest["packagesDir"] == ".lake/packages"
    assert [package["name"] for package in manifest["packages"]] == [
        "mathlib",
        "batteries",
    ]
    assert manifest["packages"][0]["type"] == "path"
    assert manifest["packages"][0]["dir"] == str(mathlib)


def test_gate_skips_comparator_for_incomplete_fast_verdict(tmp_path: Path):
    calls: list[str] = []

    async def fast(code: str, *, timeout: float):
        calls.append("fast")
        return LeanVerifyResult(ok=False, complete=False)

    async def prime(workspace: Path, timeout: float):
        calls.append("prime")
        return ComparatorProcessResult(0, "", "")

    async def compare(workspace: Path, timeout: float):
        calls.append("compare")
        return ComparatorProcessResult(0, "Your solution is okay!", "")

    gate = LeanComparatorGate(
        fast_verifier=fast,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        workspace_dir=tmp_path / "workspace",
        mathlib_dir=tmp_path,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
        prime_runner=prime,
        comparator_runner=compare,
    )

    verdict = asyncio.run(gate(BINDER_CANDIDATE))

    assert verdict.complete is False
    assert calls == ["fast"]


def test_gate_requires_comparator_acceptance_for_complete_candidate(tmp_path: Path):
    calls: list[str] = []

    async def fast(code: str, *, timeout: float):
        calls.append("fast")
        return LeanVerifyResult(ok=True, complete=True, verify_time=0.1)

    async def prime(workspace: Path, timeout: float):
        calls.append("prime")
        return ComparatorProcessResult(0, "", "")

    async def reject(workspace: Path, timeout: float):
        calls.append("compare")
        return ComparatorProcessResult(1, "", "declarations do not match")

    gate = LeanComparatorGate(
        fast_verifier=fast,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        workspace_dir=tmp_path / "workspace",
        mathlib_dir=tmp_path,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
        prime_runner=prime,
        comparator_runner=reject,
    )

    verdict = asyncio.run(gate("import Mathlib\n" + BINDER_CANDIDATE))

    assert verdict.ok is False
    assert verdict.complete is False
    assert verdict.system_error is None
    assert "comparator rejected" in verdict.summary()
    assert calls == ["fast", "prime", "compare"]


def test_gate_accepts_only_after_fast_and_comparator_pass(tmp_path: Path):
    calls: list[str] = []
    solution_seen: list[str] = []

    async def fast(code: str, *, timeout: float):
        calls.append("fast")
        return LeanVerifyResult(ok=True, complete=True, verify_time=0.1)

    async def prime(workspace: Path, timeout: float):
        calls.append("prime")
        return ComparatorProcessResult(0, "", "")

    async def accept(workspace: Path, timeout: float):
        calls.append("compare")
        solution_seen.append((workspace / "Solution.lean").read_text())
        return ComparatorProcessResult(0, "Your solution is okay!", "")

    gate = LeanComparatorGate(
        fast_verifier=fast,
        header="import Mathlib\nset_option autoImplicit false",
        formal_prefix=BINDER_PREFIX,
        workspace_dir=tmp_path / "workspace",
        mathlib_dir=tmp_path,
        lean_toolchain="leanprover/lean4:v4.30.0-rc2",
        prime_runner=prime,
        comparator_runner=accept,
    )

    candidate = "import Mathlib\n" + BINDER_CANDIDATE
    verdict = asyncio.run(gate(candidate))

    assert verdict.ok is True
    assert verdict.complete is True
    assert "Your solution is okay!" in verdict.raw_stdout
    assert calls == ["fast", "prime", "compare"]
    # The comparator must have judged the candidate itself, not a bridge.
    assert solution_seen == [candidate.strip() + "\n"]
