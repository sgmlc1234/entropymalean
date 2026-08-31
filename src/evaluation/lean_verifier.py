"""Lean 4 verifier wrapper for the multi-turn proof evaluator.

We model the verifier contract on Goedel-Prover's ``Lean4ServerScheduler``
(see references/eval_refs/goedel/verifier.py): a single function takes a
Lean source string and returns a structured result describing whether
the file type-checked, what the diagnostics were, and how long the check
took.

For the AI4Math workshop budget we use a plain ``lake env lean`` invocation
against the project under ``lean/`` rather than the ``lake exe repl``
server-mode interface. This keeps the wrapper standalone and avoids
forcing the workshop submission archive to ship a mathlib4 sub-process
manager. If the user has a faster repl-based scheduler available, this
module exposes the same dataclass surface so it can be swapped in.

The wrapper also handles three extraction tasks that every Lean prover
pipeline ends up duplicating:

- ``extract_lean_block(text)``: pull a ```lean4 ...``` block out of an
  LLM response, falling back to the raw text when no fence is present;
- ``parse_lean_messages(payload)``: turn the verifier's stderr/stdout
  into structured ``LeanMessage`` records (file, line, severity, body)
  so the refinement prompt can quote them;
- ``contains_sorry(code)``: regex-match ``sorry`` outside comments, used
  to distinguish ``complete`` from ``pass`` in line with Goedel.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import signal
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Identity cache
#
# Caches (code → LeanVerifyResult) keyed by SHA-1 of the verified source.
# Used by the tree-search prover where K=3 attempts often re-issue the same
# (formal_prefix, tactics, header) triple across attempts — the verifier
# would otherwise pay full whole-file elaboration cost twice. Bounded LRU
# (~512 entries) to avoid unbounded memory growth across a campaign.
#
# Only verdicts with no ``system_error`` are cached (timeouts and process
# crashes are transient and should be retried). Concurrent calls for the
# same key cooperate via ``_inflight`` futures: the first caller does the
# real verify; siblings await the same future and never spawn duplicate
# Lean processes.
# ---------------------------------------------------------------------------

_CACHE_MAX = 512
_cache: "OrderedDict[str, 'LeanVerifyResult']" = OrderedDict()
_inflight: "dict[str, asyncio.Future]" = {}
_cache_lock = asyncio.Lock()
_cache_hits = 0
_cache_misses = 0


def _hash_code(code: str, timeout: float) -> str:
    """Cache key — includes timeout so a longer-budget retry doesn't pull
    a stale short-budget timeout verdict from the cache."""
    h = hashlib.sha1()
    h.update(code.encode("utf-8"))
    h.update(f"|t={timeout}".encode("utf-8"))
    return h.hexdigest()


def get_cache_stats() -> "dict[str, int]":
    """Returns running hit/miss counters; useful for campaign logs."""
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "size": len(_cache),
    }


def clear_cache() -> None:
    """Drop all cached verdicts. Tests use this to isolate runs."""
    global _cache_hits, _cache_misses
    _cache.clear()
    _inflight.clear()
    _cache_hits = 0
    _cache_misses = 0


# ---------------------------------------------------------------------------
#  Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LeanMessage:
    """One parsed compiler diagnostic."""

    severity: str            # 'error' | 'warning' | 'info'
    line: Optional[int]
    column: Optional[int]
    body: str

    def short(self) -> str:
        prefix = (
            f"{self.severity}"
            + (f" @ line {self.line}" if self.line is not None else "")
        )
        return f"{prefix}: {self.body.strip()}"


@dataclass
class LeanVerifyResult:
    """Outcome of one ``lake env lean`` call on a candidate proof."""

    ok: bool                          # True iff no errors
    complete: bool                    # ok AND no sorry / 'failed' warnings
    errors: List[LeanMessage] = field(default_factory=list)
    warnings: List[LeanMessage] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    verify_time: float = 0.0
    system_error: Optional[str] = None  # subprocess crash / timeout

    def summary(self, max_errors: int = 5) -> str:
        """Compact textual digest of the diagnostics for prompt refinement."""
        if self.system_error:
            return f"[verifier system error] {self.system_error}"
        if self.complete:
            return "complete"
        if self.ok and not self.complete:
            return "pass (parsed) but proof contains `sorry` or `failed` warning"
        head = self.errors[:max_errors]
        body = "\n".join(m.short() for m in head)
        suffix = (
            f"\n... ({len(self.errors) - max_errors} more error(s) omitted)"
            if len(self.errors) > max_errors
            else ""
        )
        return body + suffix or "no diagnostics captured"


# ---------------------------------------------------------------------------
#  Output parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(
    r"```lean4?\s*\n(?P<body>.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)

# Lean stderr format: `path/to/file.lean:LINE:COL: severity: message`.
# Newer toolchains tag the severity with a diagnostic name, e.g.
# `error(lean.synthInstanceFailed): ...` — accept both shapes.
_LEAN_DIAG_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<severity>error|warning|info)(?:\((?P<diag_name>[^)\n]+)\))?:\s*(?P<body>.+?)$",
    flags=re.MULTILINE,
)

_SORRY_RE = re.compile(r"(?<![A-Za-z_])sorry(?![A-Za-z_])")
_FAILED_WARN_RE = re.compile(r"(?:declaration uses 'sorry'|failed)", re.IGNORECASE)


def extract_lean_block(text: str) -> str:
    """Pull the last ```lean4``` (or ```lean```) block out of a model reply.

    Falls back to the trimmed raw text if no fence is present, which is the
    convention Goedel-Prover follows in ``quick_start.py``.
    """
    if not text:
        return ""
    matches = list(_FENCE_RE.finditer(text))
    if matches:
        return matches[-1].group("body").rstrip()
    return text.strip()


def parse_lean_messages(payload: str) -> List[LeanMessage]:
    """Parse Lean's ``file:line:col: severity: body`` lines into messages.

    Multi-line bodies (e.g. ``error: unsolved goals`` followed by the goal
    display) are folded into the preceding diagnostic: any line that does not
    itself start a new diagnostic continues the current message. This keeps
    goal states and expected/actual type dumps available to retry feedback.
    """
    if not payload:
        return []
    messages: List[LeanMessage] = []
    current_body: Optional[List[str]] = None

    def _flush(match: "re.Match[str]", body_lines: List[str]) -> None:
        messages.append(
            LeanMessage(
                severity=match.group("severity"),
                line=int(match.group("line")),
                column=int(match.group("col")),
                body="\n".join(body_lines).strip(),
            )
        )

    open_match: Optional["re.Match[str]"] = None
    for line in payload.splitlines():
        m = _LEAN_DIAG_RE.match(line)
        if m:
            if open_match is not None and current_body is not None:
                _flush(open_match, current_body)
            open_match = m
            current_body = [m.group("body")]
        elif open_match is not None and current_body is not None:
            current_body.append(line)
    if open_match is not None and current_body is not None:
        _flush(open_match, current_body)
    return messages


def contains_sorry(code: str) -> bool:
    """True if ``sorry`` appears as a Lean tactic / term (not inside a comment)."""
    # strip line comments and block comments before scanning
    stripped = re.sub(r"--.*?$", "", code, flags=re.MULTILINE)
    stripped = re.sub(r"/-.*?-/", "", stripped, flags=re.DOTALL)
    return bool(_SORRY_RE.search(stripped))


# ---------------------------------------------------------------------------
#  Verifier entry point
# ---------------------------------------------------------------------------


def _default_workspace() -> Optional[Path]:
    """Return the local Lean 4 project that owns the EMG-2 templates, if any."""
    env_value = os.getenv("LEAN_PROJECT_DIR")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        return candidate if candidate.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "lean"):
        if any(
            (candidate / config).exists()
            for config in ("lakefile.lean", "lakefile.toml")
        ):
            return candidate
    return None


def _resolve_lean_invocation() -> Tuple[str, List[str], Optional[Path]]:
    """Choose between ``lake env lean`` (preferred) and a bare ``lean`` call.

    Returns ``(executable, prefix_args, workspace)``. The workspace is the
    directory the subprocess should ``cwd`` into so mathlib4 / EMG-2 lakefile
    is resolved.
    """
    workspace = _default_workspace()
    lake = shutil.which(os.getenv("LAKE_PATH", "lake"))
    if workspace is not None and lake is not None and (workspace / "lakefile.lean").exists():
        return lake, ["env", "lean"], workspace
    if workspace is not None and lake is not None and (workspace / "lakefile.toml").exists():
        return lake, ["env", "lean"], workspace
    lean = shutil.which(os.getenv("LEAN_PATH", "lean"))
    if lean is None:
        raise FileNotFoundError(
            "neither `lake` nor `lean` is on PATH; install Lean 4 or set LEAN_PATH"
        )
    return lean, [], workspace


def _lean_resource_args() -> List[str]:
    """Resource flags for local Lean checks.

    Defaults are tuned for the local workstation class we use for theorem
    route pilots, but each value can be overridden or disabled by env vars.
    """
    args: List[str] = []
    memory_mb = os.getenv("LEAN_MEMORY_MB", "32768").strip()
    threads = os.getenv("LEAN_THREADS", "6").strip()
    heartbeats = os.getenv("LEAN_MAX_HEARTBEATS", "2000000").strip()
    if memory_mb and memory_mb != "0":
        args.extend(["-M", memory_mb])
    if threads and threads != "0":
        args.extend(["-j", threads])
    if heartbeats and heartbeats != "0":
        args.extend(["-D", f"maxHeartbeats={heartbeats}"])
    return args


async def verify_lean_proof(
    code: str,
    *,
    timeout: float = 300.0,
    extra_env: Optional[dict] = None,
    use_cache: bool = True,
) -> LeanVerifyResult:
    """Type-check a Lean 4 source string and return a structured outcome.

    The function is async so the multi-turn loop can run it concurrently with
    other rows, but the underlying Lean process is synchronous; concurrency
    must be capped at the orchestrator level.

    Caching: when ``use_cache=True`` (default), identical ``(code, timeout)``
    pairs hit an in-process LRU. Tree-search re-issues the same partial
    proofs across K attempts so the cache typically yields 30–60% hit rate.
    Set ``use_cache=False`` when intentionally re-verifying (e.g. retry
    after a transient timeout) or in tests that need isolation.
    """
    global _cache_hits, _cache_misses

    if use_cache and not extra_env:
        key = _hash_code(code, timeout)
        async with _cache_lock:
            cached = _cache.get(key)
            if cached is not None:
                # LRU touch
                _cache.move_to_end(key)
                _cache_hits += 1
                return cached
            pending = _inflight.get(key)
            if pending is not None:
                # Another coroutine is already verifying this same code —
                # await its result rather than starting a duplicate Lean
                # process. Saves ~80s per coalesced call when sibling
                # attempts converge on the same partial proof.
                _cache_hits += 1
                fut = pending
                # Release lock before awaiting so we don't block the
                # in-flight task's eventual store under the same lock.
                pass
            else:
                fut = asyncio.get_event_loop().create_future()
                _inflight[key] = fut
                _cache_misses += 1
                pending = None  # signal: we own this future
        if pending is not None:
            return await pending

        # We own the future — actually run the verify, store, broadcast.
        try:
            verdict = await _verify_lean_proof_uncached(
                code, timeout=timeout, extra_env=extra_env
            )
        except BaseException as exc:
            async with _cache_lock:
                if not fut.done():
                    fut.set_exception(exc)
                _inflight.pop(key, None)
            raise
        async with _cache_lock:
            # Only cache stable verdicts. Timeouts / process crashes are
            # transient — leave them out so a retry isn't sticky.
            if verdict.system_error is None:
                _cache[key] = verdict
                _cache.move_to_end(key)
                while len(_cache) > _CACHE_MAX:
                    _cache.popitem(last=False)
            if not fut.done():
                fut.set_result(verdict)
            _inflight.pop(key, None)
        return verdict

    return await _verify_lean_proof_uncached(
        code, timeout=timeout, extra_env=extra_env
    )


async def _verify_lean_proof_uncached(
    code: str,
    *,
    timeout: float = 300.0,
    extra_env: Optional[dict] = None,
) -> LeanVerifyResult:
    """Inner verifier — runs ``lake env lean`` without consulting the cache.
    Callers should normally go through :func:`verify_lean_proof`."""
    started = time.monotonic()
    try:
        executable, prefix_args, workspace = _resolve_lean_invocation()
    except FileNotFoundError as exc:
        return LeanVerifyResult(
            ok=False,
            complete=False,
            verify_time=time.monotonic() - started,
            system_error=str(exc),
        )

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    )
    temp_file.write(code)
    temp_file.flush()
    temp_file.close()

    try:
        cmd = [executable, *prefix_args, *_lean_resource_args(), temp_file.name]
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        # Spawn in a NEW process group so `os.killpg` can take down lake +
        # its lean grandchild together. Without this, `process.kill()`
        # only signals `lake`, leaving the actual `lean` worker orphaned
        # and continuing to hold ~2 GB of Mathlib OLEAN pages until the
        # OS slowly reclaims them — observed 2026-05-20 v5/v6 to leak
        # 8 GB+ over a handful of timeouts and ultimately stall the
        # campaign (lean child unaware of dead parent kept consuming CPU
        # mid-elaboration, lasting 10–60 min beyond lean_timeout).
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(workspace) if workspace else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Send SIGKILL to the entire process group (lake + lean +
            # any siblings). Bounded by a short reap timeout because
            # under heavy swap I/O even SIGKILL takes a moment to land.
            # If the group is still alive after 10s, log it and abandon
            # — OS will collect zombies eventually; the caller needs its
            # verdict to keep the search moving.
            pgid = process.pid  # we set group leader = process via setsid
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(pgid, signal.SIGKILL)  # try once more
                except (ProcessLookupError, PermissionError):
                    pass
            return LeanVerifyResult(
                ok=False,
                complete=False,
                verify_time=time.monotonic() - started,
                system_error=f"lean type-check timed out after {timeout}s",
            )
        stdout = stdout_b.decode("utf-8", errors="ignore")
        stderr = stderr_b.decode("utf-8", errors="ignore")
        diagnostics = parse_lean_messages(stdout + "\n" + stderr)
        errors = [d for d in diagnostics if d.severity == "error"]
        warnings = [d for d in diagnostics if d.severity == "warning"]
        ok = (process.returncode == 0) and not errors
        complete = ok and not contains_sorry(code) and not any(
            _FAILED_WARN_RE.search(w.body) for w in warnings
        )
        return LeanVerifyResult(
            ok=ok,
            complete=complete,
            errors=errors,
            warnings=warnings,
            raw_stdout=stdout,
            raw_stderr=stderr,
            verify_time=time.monotonic() - started,
        )
    except Exception as exc:  # pragma: no cover - rare process error
        return LeanVerifyResult(
            ok=False,
            complete=False,
            verify_time=time.monotonic() - started,
            system_error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
