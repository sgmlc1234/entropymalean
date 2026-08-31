"""Optional final-proof gate backed by ``leanprover/comparator``.

Search still uses the fast REPL/file verifier for partial candidates. Only a
candidate that the fast verifier considers complete reaches comparator, which
then checks the exact trusted Challenge statement, the axiom allowlist, and
Lean kernel acceptance.

Comparator's sandbox uses Linux Landlock through ``landrun``. The gate is
therefore intended for a Linux runner/CI job; macOS can prepare and unit-test
the workspaces but cannot provide the hardened execution guarantee.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence

from src.evaluation.lean_verifier import LeanMessage, LeanVerifyResult


DEFAULT_PERMITTED_AXIOMS = (
    "propext",
    "Quot.sound",
    "Classical.choice",
)

_DECLARATION_NAME_RE = re.compile(
    r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)"
)
_NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)\s*(?:--.*)?$"
)
_SECTION_RE = re.compile(
    r"^\s*section(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*(?:--.*)?$"
)
_END_RE = re.compile(
    r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*(?:--.*)?$"
)
_IMPORT_LINE_RE = re.compile(r"(?m)^\s*import\b")


@dataclass(frozen=True)
class ComparatorWorkspace:
    path: Path
    theorem_name: str


@dataclass(frozen=True)
class ComparatorProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def declaration_name(formal_prefix: str, header: str = "") -> str:
    """Return the fully qualified target name under trusted namespaces.

    Mirrors Lean's scope stack: ``namespace A.B`` pushes one scope per
    component, ``section`` pushes an anonymous or named scope, and
    ``end X.Y`` pops one scope per component (a bare ``end`` pops one).
    """
    match = _DECLARATION_NAME_RE.search(formal_prefix or "")
    if match is None:
        raise ValueError("comparator requires a named theorem or lemma target")
    local_name = match.group(1)
    if "." in local_name:
        return local_name

    scopes: list[tuple[str, Optional[str]]] = []
    for line in (header or "").splitlines():
        namespace = _NAMESPACE_RE.match(line)
        if namespace:
            for component in namespace.group(1).split("."):
                scopes.append(("namespace", component))
            continue
        section = _SECTION_RE.match(line)
        if section:
            scopes.append(("section", section.group(1)))
            continue
        end = _END_RE.match(line)
        if end and scopes:
            popped = len((end.group(1) or "").split(".")) if end.group(1) else 1
            del scopes[-min(popped, len(scopes)):]
    namespaces = [name for kind, name in scopes if kind == "namespace"]
    return ".".join([*namespaces, local_name])


def _proof_with_body(formal_prefix: str, body: str) -> str:
    prefix = (formal_prefix or "").rstrip()
    if re.search(r":=\s*by\s*$", prefix):
        return f"{prefix}\n  {body}"
    if re.search(r":=\s*$", prefix):
        return f"{prefix} by\n  {body}"
    return f"{prefix} := by\n  {body}"


def _toml_string(value: object) -> str:
    # JSON string quoting is valid TOML basic-string quoting.
    return json.dumps(str(value), ensure_ascii=False)


def _git_revision(directory: Path) -> str:
    """The checked-out revision of a package directory, or "" if unknowable.

    Returns empty rather than raising: a workspace that cannot name its mathlib
    is still a valid `kernel_replayed` bundle, and the reproducibility check is
    what refuses to grant the higher rung.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_shared_manifest(workspace: Path, mathlib: Path) -> bool:
    """Point Lake at the trusted project's already-pinned package checkouts."""
    packages_dir = mathlib.parent
    project_manifest = packages_dir.parent.parent / "lake-manifest.json"
    if not project_manifest.is_file():
        return False
    source = json.loads(project_manifest.read_text())
    packages = []
    for package in source.get("packages") or []:
        name = package.get("name")
        if name == "mathlib":
            # Every other package here carries the revision it was resolved at;
            # mathlib carried only a path, because Lake's `path` package form
            # has no `rev` field. That made the bundle the one thing it must
            # not be -- silent about the dependency that matters most. A reader
            # replaying it got "whatever is at this directory", and the
            # `reproducible` rung is precisely the promise that every revision
            # is recorded. Lake ignores the extra keys (checked: `lake env
            # comparator` still returns 0 with them present), so the pin is
            # written down where the replay reads it rather than only in the
            # certificate beside it.
            entry = {
                "type": "path",
                "scope": "",
                "name": "mathlib",
                "manifestFile": package.get(
                    "manifestFile", "lake-manifest.json"
                ),
                "inherited": False,
                "dir": str(mathlib),
                "configFile": package.get("configFile", "lakefile.lean"),
            }
            revision = _git_revision(mathlib)
            if revision:
                entry["rev"] = revision
                entry["url"] = "https://github.com/leanprover-community/mathlib4"
            packages.append(entry)
        elif package.get("inherited"):
            packages.append(package)
            source_dir = packages_dir / str(name)
            shared_dir = workspace / ".lake" / "packages" / str(name)
            if source_dir.is_dir() and not shared_dir.exists():
                shared_dir.parent.mkdir(parents=True, exist_ok=True)
                shared_dir.symlink_to(source_dir, target_is_directory=True)
    manifest = {
        "version": source.get("version", "1.2.0"),
        "packagesDir": ".lake/packages",
        "packages": packages,
        "name": "eml_comparator_workspace",
        "lakeDir": ".lake",
        "fixedToolchain": False,
    }
    (workspace / "lake-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return True


def prepare_comparator_workspace(
    workspace_dir: Path,
    *,
    header: str,
    formal_prefix: str,
    candidate_code: str,
    mathlib_dir: Path,
    lean_toolchain: str,
    permitted_axioms: Sequence[str] = DEFAULT_PERMITTED_AXIOMS,
    enable_nanoda: bool = False,
) -> ComparatorWorkspace:
    """Materialize a controlled Challenge/Solution workspace.

    This follows the official ``leanprover/comparator`` layout: ``Challenge``
    is derived only from the trusted row (statement closed with ``sorry``),
    and ``Solution`` is the model's candidate file verbatim. Comparator
    itself guarantees — via ``lean4export`` — that the ``theorem_names``
    declaration in Solution proves the *same statement* as Challenge, uses
    only ``permitted_axioms``, and is accepted by the Lean kernel. A model
    that renames the theorem fails the ``theorem_names`` lookup and a model
    that alters the statement fails the type comparison, so no additional
    bridging declaration is needed (a ``by exact`` bridge would in fact fail
    to elaborate for any theorem with explicit binders, because the binders
    are already introduced into the tactic-block context).
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    mathlib = Path(mathlib_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    theorem_name = declaration_name(formal_prefix, header)

    challenge = (
        f"{header.rstrip()}\n\n"
        f"{_proof_with_body(formal_prefix, 'sorry')}\n"
    )
    solution_code = (candidate_code or "").strip()
    if not _IMPORT_LINE_RE.search(solution_code):
        # Candidates produced by the evaluation loop always carry the row
        # header (imports, opens, set_options); keep a defensive fallback so
        # a bare proof file still elaborates inside the workspace.
        solution_code = f"{header.rstrip()}\n\n{solution_code}"
    solution = solution_code + "\n"
    lakefile = (
        'name = "eml_comparator_workspace"\n'
        'defaultTargets = ["Challenge", "Solution"]\n\n'
        "[leanOptions]\n"
        "autoImplicit = false\n\n"
        "[[require]]\n"
        'name = "mathlib"\n'
        f"path = {_toml_string(mathlib)}\n\n"
        "[[lean_lib]]\n"
        'name = "Challenge"\n\n'
        "[[lean_lib]]\n"
        'name = "Solution"\n'
    )
    config = {
        "challenge_module": "Challenge",
        "solution_module": "Solution",
        "theorem_names": [theorem_name],
        "permitted_axioms": list(permitted_axioms),
        "enable_nanoda": bool(enable_nanoda),
    }

    (workspace / "Challenge.lean").write_text(challenge)
    (workspace / "Solution.lean").write_text(solution)
    # Drop any Submission.lean left behind by the pre-fix bridge layout so a
    # reused workspace cannot build stale model code.
    (workspace / "Submission.lean").unlink(missing_ok=True)
    (workspace / "lakefile.toml").write_text(lakefile)
    (workspace / "lean-toolchain").write_text(lean_toolchain.strip() + "\n")
    (workspace / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    _write_shared_manifest(workspace, mathlib)
    return ComparatorWorkspace(path=workspace, theorem_name=theorem_name)


def _resolve_executable(value: str | Path) -> Optional[str]:
    text = str(value)
    if "/" in text or os.sep in text:
        path = Path(text).expanduser().resolve()
        return str(path) if path.is_file() else None
    return shutil.which(text)


def validate_comparator_runtime(
    comparator_bin: str | Path = "comparator",
    *,
    lake_bin: str | Path = "lake",
) -> dict[str, str]:
    """Fail fast unless the hardened comparator runtime is available."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Lean comparator hardened mode requires Linux: landrun uses "
            "Landlock and is not available on macOS"
        )
    required = {
        "lake": lake_bin,
        "comparator": comparator_bin,
        "landrun": "landrun",
        "lean4export": "lean4export",
        "systemd-run": "systemd-run",
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for label, value in required.items():
        executable = _resolve_executable(value)
        if executable is None:
            missing.append(f"{label} ({value})")
        else:
            resolved[label] = executable
    if missing:
        raise RuntimeError(
            "missing Lean comparator runtime executable(s): "
            + ", ".join(missing)
        )
    return resolved


async def _run_process(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
) -> ComparatorProcessResult:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        return ComparatorProcessResult(
            returncode=-1,
            stdout="",
            stderr=f"timed out after {timeout}s",
            timed_out=True,
        )
    return ComparatorProcessResult(
        returncode=process.returncode or 0,
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
    )


FastVerifier = Callable[..., Awaitable[LeanVerifyResult]]
ProcessRunner = Callable[[Path, float], Awaitable[ComparatorProcessResult]]


class LeanComparatorGate:
    """Fast verifier followed by comparator for complete candidates only."""

    def __init__(
        self,
        *,
        fast_verifier: FastVerifier,
        header: str,
        formal_prefix: str,
        workspace_dir: Path,
        mathlib_dir: Path,
        lean_toolchain: str,
        comparator_bin: str | Path = "comparator",
        lake_bin: str | Path = "lake",
        systemd_run_bin: str | Path = "systemd-run",
        comparator_timeout: float = 600.0,
        permitted_axioms: Sequence[str] = DEFAULT_PERMITTED_AXIOMS,
        enable_nanoda: bool = False,
        prime_runner: Optional[ProcessRunner] = None,
        comparator_runner: Optional[ProcessRunner] = None,
    ) -> None:
        self.fast_verifier = fast_verifier
        self.header = header
        self.formal_prefix = formal_prefix
        self.workspace_dir = Path(workspace_dir)
        self.mathlib_dir = Path(mathlib_dir)
        self.lean_toolchain = lean_toolchain
        self.comparator_bin = str(comparator_bin)
        self.lake_bin = str(lake_bin)
        self.systemd_run_bin = str(systemd_run_bin)
        self.comparator_timeout = comparator_timeout
        self.permitted_axioms = tuple(permitted_axioms)
        self.enable_nanoda = enable_nanoda
        self._lock = asyncio.Lock()
        self._primed = False

        async def default_prime(
            workspace: Path, timeout: float
        ) -> ComparatorProcessResult:
            # A controlled manifest that points at the parent project's pinned,
            # pre-built packages needs no network/update step. This also avoids
            # cloning Mathlib's transitive dependencies once per problem.
            if (workspace / "lake-manifest.json").is_file():
                return ComparatorProcessResult(0, "", "")
            return await _run_process(
                [self.lake_bin, "update"], cwd=workspace, timeout=timeout
            )

        async def default_compare(
            workspace: Path, timeout: float
        ) -> ComparatorProcessResult:
            return await _run_process(
                [
                    self.systemd_run_bin,
                    "--property=RestrictAddressFamilies=~AF_UNIX",
                    "--user",
                    "--pipe",
                    "--wait",
                    "--collect",
                    f"--working-directory={workspace}",
                    f"--setenv=PATH={os.environ.get('PATH', '')}",
                    self.lake_bin,
                    "env",
                    self.comparator_bin,
                    "config.json",
                ],
                cwd=workspace,
                timeout=timeout,
            )

        self._prime_runner = prime_runner or default_prime
        self._comparator_runner = comparator_runner or default_compare

    async def __call__(
        self,
        code: str,
        *,
        timeout: float = 300.0,
        extra_env: Optional[dict] = None,
        **kwargs,
    ) -> LeanVerifyResult:
        fast = await self.fast_verifier(
            code,
            timeout=timeout,
            **({"extra_env": extra_env} if extra_env else {}),
        )
        if not fast.complete:
            return fast

        started = time.monotonic()
        async with self._lock:
            try:
                prepare_comparator_workspace(
                    self.workspace_dir,
                    header=self.header,
                    formal_prefix=self.formal_prefix,
                    candidate_code=code,
                    mathlib_dir=self.mathlib_dir,
                    lean_toolchain=self.lean_toolchain,
                    permitted_axioms=self.permitted_axioms,
                    enable_nanoda=self.enable_nanoda,
                )
            except Exception as exc:
                return LeanVerifyResult(
                    ok=False,
                    complete=False,
                    verify_time=fast.verify_time
                    + time.monotonic()
                    - started,
                    system_error=(
                        "comparator workspace preparation failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if not self._primed:
                try:
                    prime = await self._prime_runner(
                        self.workspace_dir, self.comparator_timeout
                    )
                except Exception as exc:
                    return LeanVerifyResult(
                        ok=False,
                        complete=False,
                        verify_time=fast.verify_time
                        + time.monotonic()
                        - started,
                        system_error=(
                            "comparator workspace initialization failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                if prime.timed_out:
                    return LeanVerifyResult(
                        ok=False,
                        complete=False,
                        verify_time=fast.verify_time + time.monotonic() - started,
                        system_error=(
                            "comparator workspace `lake update` timed out "
                            f"after {self.comparator_timeout}s"
                        ),
                        raw_stdout=prime.stdout,
                        raw_stderr=prime.stderr,
                    )
                if prime.returncode != 0:
                    return LeanVerifyResult(
                        ok=False,
                        complete=False,
                        verify_time=fast.verify_time + time.monotonic() - started,
                        system_error=(
                            "comparator workspace `lake update` failed: "
                            + (prime.stderr or prime.stdout).strip()[-2000:]
                        ),
                        raw_stdout=prime.stdout,
                        raw_stderr=prime.stderr,
                    )
                self._primed = True

            try:
                compared = await self._comparator_runner(
                    self.workspace_dir, self.comparator_timeout
                )
            except Exception as exc:
                return LeanVerifyResult(
                    ok=False,
                    complete=False,
                    verify_time=fast.verify_time
                    + time.monotonic()
                    - started,
                    system_error=(
                        "Lean comparator execution failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        elapsed = fast.verify_time + time.monotonic() - started
        if compared.timed_out:
            return LeanVerifyResult(
                ok=False,
                complete=False,
                verify_time=elapsed,
                system_error=(
                    f"Lean comparator timed out after {self.comparator_timeout}s"
                ),
                raw_stdout=compared.stdout,
                raw_stderr=compared.stderr,
            )
        if compared.returncode != 0:
            detail = (compared.stderr or compared.stdout).strip()[-2000:]
            return LeanVerifyResult(
                ok=False,
                complete=False,
                errors=[
                    LeanMessage(
                        severity="error",
                        line=None,
                        column=None,
                        body=f"comparator rejected candidate: {detail}",
                    )
                ],
                verify_time=elapsed,
                raw_stdout=compared.stdout,
                raw_stderr=compared.stderr,
            )
        return replace(
            fast,
            verify_time=elapsed,
            raw_stdout=(
                fast.raw_stdout
                + ("\n" if fast.raw_stdout and compared.stdout else "")
                + compared.stdout
            ),
            raw_stderr=(
                fast.raw_stderr
                + ("\n" if fast.raw_stderr and compared.stderr else "")
                + compared.stderr
            ),
        )
