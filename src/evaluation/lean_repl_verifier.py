"""Persistent ``lake exe repl`` verifier — port of the Goedel-Prover-V2
official ``lean_compiler/repl_scheduler.py`` to our async workflow.

Why this exists: ``verify_lean_proof`` in :mod:`lean_verifier` spawns a
fresh ``lake env lean`` subprocess for every candidate proof. Each
spawn pays the full ~5–50 s cost of loading Mathlib OLEAN files from
disk, which is the dominant bottleneck on a Mac laptop without
swap-heavy concurrency. The Goedel-Prover-V2 reference repo (cloned
under ``references/Goedel-Prover-V2``) instead drives the official
``lake exe repl`` interpreter via ``pexpect``, sending JSON commands
to a *persistent* REPL process — Mathlib is loaded once, every
subsequent verify runs at ~0.5 s (measured: 23.9 s import warm-up +
0.5 s per proof).

This module exposes the same :class:`LeanVerifyResult` contract as the
file-based verifier so callers can swap implementations transparently.

Concurrency: by default, a single REPL child process is serialized behind
an ``asyncio.Lock``. Set ``LEAN_REPL_POOL_SIZE=N`` to maintain a small
process-local pool for verifier-heavy BFS runs. Each REPL duplicates a
Mathlib environment in resident memory, so keep the default for
generation-bound whole-proof runs and raise it only on machines with
headroom.

Activation: callers wrap the standalone factory :func:`make_repl_verifier`
to produce a coroutine identical in signature to ``verify_lean_proof``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pexpect

from src.evaluation.lean_verifier import LeanMessage, LeanVerifyResult


# The default imports the Goedel-Prover-V2 reference uses. We send these
# once at REPL startup to obtain an ``env`` id; every subsequent verify
# reuses that env so the model header (``import Mathlib`` etc.) doesn't
# need to be re-elaborated.
DEFAULT_REPL_IMPORTS = (
    "import Mathlib\n"
    "import Aesop\n"
    f"set_option maxHeartbeats {os.getenv('LEAN_REPL_MAX_HEARTBEATS', '0')}\n"
    "open BigOperators Real Nat Topology Rat"
)

# Workspace must own a built Mathlib (``lake build`` already done).
# Our repo root carries the right lakefile.lean + lean-toolchain
# (v4.30.0-rc2) and a populated ``.lake/packages/``. The official repo's
# ``mathlib4`` submodule pins v4.9.0-rc1, which is too old for our
# campaign code paths — we reuse our own workspace instead.
DEFAULT_REPL_WORKSPACE = Path(__file__).resolve().parents[2]

DEFAULT_LAKE_PATH = os.path.expanduser(os.getenv("LAKE_PATH", "~/.elan/bin/lake"))

# bash startup banners (`chsh`, zsh notice, prompts like ``bash-3.2$``)
# show up in pexpect.before on the very first command and break JSON
# parsing. The REPL response itself is always a balanced ``{...}`` JSON
# object — find the LAST such block in the captured text and try that.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_repl_json(text: str) -> str:
    """Return the most likely JSON object in a REPL response, or empty.

    pexpect.before captures *everything* between the previous match and
    the ``\\n\\n`` delimiter, including the macOS bash banner on the
    first command. Rather than trying to enumerate every possible
    banner line, we just scan for the last balanced ``{...}`` JSON
    object — that's invariably the REPL's actual reply.
    """
    candidates = _JSON_OBJECT_RE.findall(text)
    return candidates[-1].strip() if candidates else ""


def _build_messages(raw: List[dict]) -> List[LeanMessage]:
    out: List[LeanMessage] = []
    for m in raw:
        out.append(
            LeanMessage(
                severity=m.get("severity", "info"),
                line=m.get("pos", {}).get("line"),
                column=m.get("pos", {}).get("column"),
                body=m.get("data", ""),
            )
        )
    return out


_FAILED_WARN_RE = re.compile(r"\bfailed\b", re.IGNORECASE)
_USES_SORRY_RE = re.compile(r"declaration uses .sorry.", re.IGNORECASE)
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+\S+.*$", re.MULTILINE)
_MAX_HEARTBEATS_LINE_RE = re.compile(r"^\s*set_option\s+maxHeartbeats\s+\S+.*$", re.MULTILINE)


def strip_repl_import_commands(code: str) -> str:
    """Remove Lean ``import`` commands before sending code into a reused env.

    ``lake exe repl`` accepts imports only when creating the base env. After
    a command is evaluated under ``env=N``, sending another ``import`` command
    inside the proof block fails with:

        invalid 'import' command, it must be used in the beginning of the file

    Our evaluator often assembles candidates as whole Lean files because the
    file-based verifier needs that shape. The persistent REPL verifier already
    loaded ``DEFAULT_REPL_IMPORTS`` once, so per-candidate import lines are
    redundant and must be stripped while preserving ``set_option``, ``open``,
    namespace commands, and the theorem body.
    """
    stripped = _IMPORT_LINE_RE.sub("", code or "")
    repl_heartbeats = os.getenv("LEAN_REPL_MAX_HEARTBEATS", "").strip()
    if repl_heartbeats and repl_heartbeats != "0":
        stripped = _MAX_HEARTBEATS_LINE_RE.sub("", stripped)
    lines = stripped.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


@dataclass
class _LeanReplState:
    """Holds the live REPL child + the env id of the pre-loaded imports."""
    child: pexpect.spawn
    env_id: Optional[int]
    started_at: float = 0.0
    proofs_handled: int = 0


class LeanReplVerifier:
    """Async-friendly wrapper around a single persistent ``lake exe repl``.

    Use :meth:`verify` as a drop-in for :func:`verify_lean_proof`. The
    REPL is lazily started on the first call and reused thereafter.
    Calls are serialized — the underlying REPL is single-threaded by
    nature, and serializing keeps Mathlib resident in one place.
    """

    def __init__(
        self,
        *,
        workspace: Path = DEFAULT_REPL_WORKSPACE,
        lake_path: str = DEFAULT_LAKE_PATH,
        imports: str = DEFAULT_REPL_IMPORTS,
        import_timeout: float = 180.0,
    ) -> None:
        self.workspace = Path(workspace)
        self.lake_path = lake_path
        self.imports = imports
        self.import_timeout = import_timeout
        self._state: Optional[_LeanReplState] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # REPL lifecycle
    # ------------------------------------------------------------------

    def _start_blocking(self) -> _LeanReplState:
        """Spawn the REPL and send the pre-cached imports.

        Blocks ~24 s on the first call (Mathlib load). Subsequent calls
        in the same process should NEVER hit this path because the
        verifier holds the child across the campaign lifetime.
        """
        child = pexpect.spawn(
            "/bin/bash",
            cwd=str(self.workspace),
            encoding="utf-8",
            maxread=1,
            echo=False,
            timeout=self.import_timeout,
        )
        child.sendline("stty -icanon 2>/dev/null")
        child.sendline(f"{self.lake_path} exe repl")
        state = _LeanReplState(child=child, env_id=None, started_at=time.monotonic())

        # Send imports → get back the env id.
        raw = self._send_json_blocking(
            state, {"cmd": self.imports}, timeout=self.import_timeout
        )
        try:
            parsed = json.loads(raw)
            state.env_id = parsed.get("env")
        except json.JSONDecodeError:
            # If parsing failed we're in an unknown state — abandon and
            # let the next call retry from scratch.
            child.terminate(force=True)
            raise RuntimeError(
                f"Lean REPL failed to return JSON on import: {raw[:200]!r}"
            )
        return state

    def _send_json_blocking(
        self, state: _LeanReplState, payload: dict, timeout: float
    ) -> str:
        """Send one JSON command and return the embedded JSON response."""
        cmd = json.dumps(payload, ensure_ascii=False)
        state.child.sendline(cmd)
        state.child.sendline("")  # second newline = end of input
        state.child.expect(["\r\n\r\n", "\n\n"], timeout=timeout)
        return _extract_repl_json(state.child.before or "")

    def _verify_blocking(self, code: str, timeout: float) -> LeanVerifyResult:
        """Run a single verify under the REPL — blocks the calling thread."""
        started = time.monotonic()
        if self._state is None:
            self._state = self._start_blocking()
        state = self._state
        code = strip_repl_import_commands(code)
        # Run the user's code in the imports env. The REPL semantic is:
        # `env=N` reuses imports N's namespace, the new declarations
        # extend it but don't pollute the original env.
        try:
            raw = self._send_json_blocking(
                state, {"cmd": code, "env": state.env_id}, timeout=timeout,
            )
        except pexpect.TIMEOUT:
            # Drop the REPL, exactly as the EOF path does. The timed-out
            # command's response is still on its way, and pexpect's buffer
            # still holds what was read so far; keeping the child means the
            # *next* call reads the previous command's verdict as its own.
            # That is not a lost measurement but a wrong one, and it is
            # invisible unless the stale verdict happens to land on a reset,
            # where it surfaces as a tactic error reported against a statement.
            # One 1245-episode BFS campaign produced 24 such boundary cases.
            self._state = None
            try:
                state.child.terminate(force=True)
            except Exception:  # pragma: no cover - best effort
                pass
            return LeanVerifyResult(
                ok=False, complete=False,
                verify_time=time.monotonic() - started,
                system_error=f"REPL timeout after {timeout}s",
            )
        except pexpect.EOF as exc:
            # REPL died — drop the state and let the next call respawn.
            self._state = None
            return LeanVerifyResult(
                ok=False, complete=False,
                verify_time=time.monotonic() - started,
                system_error=f"REPL EOF: {exc}",
            )
        except Exception as exc:  # pragma: no cover
            self._state = None
            return LeanVerifyResult(
                ok=False, complete=False,
                verify_time=time.monotonic() - started,
                system_error=f"REPL exception: {type(exc).__name__}: {exc}",
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return LeanVerifyResult(
                ok=False, complete=False,
                verify_time=time.monotonic() - started,
                system_error=f"REPL JSON decode error: {exc} (raw head: {raw[:120]!r})",
            )

        all_msgs = parsed.get("messages", [])
        errors = _build_messages([m for m in all_msgs if m.get("severity") == "error"])
        warnings = _build_messages([m for m in all_msgs if m.get("severity") == "warning"])
        sorries = parsed.get("sorries", [])

        ok = not errors
        complete = (
            ok
            and not sorries
            and not any(
                _USES_SORRY_RE.search(w.body) or _FAILED_WARN_RE.search(w.body)
                for w in warnings
            )
        )
        state.proofs_handled += 1
        return LeanVerifyResult(
            ok=ok,
            complete=complete,
            errors=errors,
            warnings=warnings,
            raw_stdout=raw,
            raw_stderr="",
            verify_time=time.monotonic() - started,
        )

    async def verify(
        self,
        code: str,
        *,
        timeout: float = 180.0,
        extra_env: Optional[dict] = None,
        use_cache: bool = True,  # accepted for signature compatibility
    ) -> LeanVerifyResult:
        """Async wrapper — serializes REPL access via ``self._lock``."""
        async with self._lock:
            started = time.monotonic()
            fut = asyncio.get_event_loop().run_in_executor(
                None, self._verify_blocking, code, timeout
            )
            try:
                wall_timeout = timeout + min(5.0, max(0.2, timeout * 0.1))
                return await asyncio.wait_for(
                    asyncio.shield(fut), timeout=wall_timeout
                )
            except asyncio.TimeoutError:
                # pexpect enforces timeout while waiting for a REPL reply, but
                # we have also seen PTY writes block when a previous command
                # leaves the child busy and not reading. Guard the whole
                # executor call so one wedged REPL cannot stall a campaign.
                if self._state is not None and self._state.child is not None:
                    try:
                        self._state.child.terminate(force=True)
                    except Exception:
                        pass
                self._state = None
                def _consume_late_result(done):
                    try:
                        done.exception()
                    except BaseException:
                        pass

                fut.add_done_callback(_consume_late_result)
                return LeanVerifyResult(
                    ok=False,
                    complete=False,
                    verify_time=time.monotonic() - started,
                    system_error=f"REPL wall-time timeout after {timeout}s",
                )

    async def close(self) -> None:
        async with self._lock:
            if self._state is not None and self._state.child is not None:
                try:
                    self._state.child.terminate(force=True)
                except Exception:
                    pass
            self._state = None


class LeanReplVerifierPool:
    """Small async pool of independent persistent REPL verifiers.

    The individual REPL instances are still single-threaded, but separate
    instances can elaborate different candidates concurrently. We allocate
    verifiers eagerly and start their child processes lazily on first use.
    """

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._verifiers = [LeanReplVerifier() for _ in range(self.size)]
        self._queue: Optional[asyncio.Queue[LeanReplVerifier]] = None
        self._queue_lock = asyncio.Lock()

    async def _get_queue(self) -> asyncio.Queue[LeanReplVerifier]:
        if self._queue is not None:
            return self._queue
        async with self._queue_lock:
            if self._queue is None:
                queue: asyncio.Queue[LeanReplVerifier] = asyncio.Queue()
                for verifier in self._verifiers:
                    queue.put_nowait(verifier)
                self._queue = queue
            return self._queue

    async def verify(
        self,
        code: str,
        *,
        timeout: float = 180.0,
        extra_env: Optional[dict] = None,
        use_cache: bool = True,
    ) -> LeanVerifyResult:
        queue = await self._get_queue()
        verifier = await queue.get()
        try:
            return await verifier.verify(
                code, timeout=timeout, extra_env=extra_env, use_cache=use_cache
            )
        finally:
            queue.put_nowait(verifier)

    async def close(self) -> None:
        await asyncio.gather(*(verifier.close() for verifier in self._verifiers))


def _repl_pool_size_from_env() -> int:
    raw = os.getenv("LEAN_REPL_POOL_SIZE", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


# Process-global singleton — callers grab one verifier or pool per Python process.
_global_verifier: Optional[LeanReplVerifier | LeanReplVerifierPool] = None


def get_global_repl_verifier() -> LeanReplVerifier | LeanReplVerifierPool:
    """Lazy global instance. Reused across all cells in a campaign."""
    global _global_verifier
    if _global_verifier is None:
        pool_size = _repl_pool_size_from_env()
        _global_verifier = (
            LeanReplVerifier()
            if pool_size == 1
            else LeanReplVerifierPool(pool_size)
        )
    return _global_verifier


async def close_global_repl_verifier() -> None:
    """Terminate the process-global REPL verifier/pool if it was created."""
    global _global_verifier
    if _global_verifier is None:
        return
    await _global_verifier.close()
    _global_verifier = None


async def verify_lean_proof_repl(
    code: str,
    *,
    timeout: float = 180.0,
    extra_env: Optional[dict] = None,
    use_cache: bool = True,
) -> LeanVerifyResult:
    """Module-level coroutine — drop-in replacement for
    :func:`src.evaluation.lean_verifier.verify_lean_proof`.

    Routes through the process-global :class:`LeanReplVerifier` so the
    REPL stays warm across cells. The ``extra_env`` and ``use_cache``
    parameters are accepted for signature compatibility but currently
    have no effect (the REPL has no per-call env override beyond the
    pre-loaded imports env, and caching is handled upstream by the
    file-based verifier; for the REPL the warm-up cost amortizes
    naturally over many calls).
    """
    return await get_global_repl_verifier().verify(
        code, timeout=timeout, extra_env=extra_env, use_cache=use_cache,
    )
