"""Small async wrapper for OAuth-backed ``codex exec`` calls.

This intentionally treats Codex CLI as a subprocess provider instead of
reading OAuth tokens from ``~/.codex/auth.json``. The CLI owns auth refresh and
we only consume its final assistant message.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CodexCliResponse:
    raw_text: str
    elapsed_seconds: float
    finish_reason: str = "stop"
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


def build_codex_exec_command(
    *,
    model: str,
    output_path: Path,
    sandbox: str = "read-only",
    cwd: Optional[Path] = None,
    output_schema_path: Optional[Path] = None,
) -> List[str]:
    """Return the non-interactive Codex command used by the provider wrapper."""
    command = [
        os.getenv("CODEX_CLI_BIN", "codex"),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--json",
        "--sandbox",
        sandbox,
        "--model",
        model,
        "--output-last-message",
        str(output_path),
    ]
    reasoning_effort = os.getenv("CODEX_CLI_MODEL_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if cwd is not None:
        command.extend(["--cd", str(cwd)])
    if output_schema_path is not None:
        command.extend(["--output-schema", str(output_schema_path)])
    command.append("-")
    return command


def build_codex_prompt(system: str, user: str, *, json_mode: bool = False) -> str:
    """Flatten chat-style messages into one Codex CLI prompt."""
    prompt = (
        "You are being used as a non-interactive model backend. "
        "Follow the task contract exactly and return only the requested artifact.\n\n"
        "<system>\n"
        f"{system.strip()}\n"
        "</system>\n\n"
        "<task>\n"
        f"{user.strip()}\n"
        "</task>\n"
    )
    if json_mode:
        prompt += (
            "\n<output_json>\n"
            "Return one JSON object only. Do not wrap it in Markdown.\n"
            "</output_json>\n"
        )
    return prompt


#: The provider reports an exhausted quota through the same non-zero exit as a
#: crashed process, so a run that has stopped being able to generate anything
#: looks exactly like 255 unlucky slots. It is not retryable and not a model
#: failure; the only correct response is to stop and come back later, so it is
#: given its own marker for callers to key on.
USAGE_LIMIT_MARKER = "usage_limit_reached"

_USAGE_LIMIT_SIGNS = (
    "usage limit",
    "hit your usage limit",
    "purchase more credits",
    "quota exceeded",
    "insufficient_quota",
)


def is_usage_limit(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(sign in lowered for sign in _USAGE_LIMIT_SIGNS)


#: How many bare failures in a row before the run is called exhausted.
#: An exhausted quota does not always announce itself: on 2026-08-07 the CLI
#: exited 1 with an empty stderr and a stdout holding nothing but
#: `thread.started`, so there was no text for the signs above to match. Six
#: ProofNet groups then ran to "completion" in three minutes each, every one of
#: their 35 generation slots failing identically, and the driver's halt guard —
#: which greps for the marker — never fired. The groups were saved as survivor
#: copies of their own seeds.
#:
#: Slot failures are independent; quota exhaustion is not. A short run of bare
#: exit-1s is bad luck, a long one is a wall, and counting them needs no
#: cooperation from the provider's error text.
_BARE_FAILURE_LIMIT = 5

_bare_failures = 0


def _note_call_outcome(returncode: int, stdout: str, stderr: str) -> bool:
    """Track consecutive contentless failures. True once they look terminal.

    A success of any kind resets the count, so a run that is merely struggling
    never trips this.
    """
    global _bare_failures
    bare = bool(returncode) and not str(stderr or "").strip() and len(str(stdout or "")) < 400
    _bare_failures = _bare_failures + 1 if bare else 0
    return _bare_failures >= _BARE_FAILURE_LIMIT


async def call_codex_cli(
    *,
    model: str,
    system: str,
    user: str,
    timeout_seconds: float,
    cwd: Optional[Path] = None,
    sandbox: str = "read-only",
    output_schema: Optional[dict] = None,
) -> CodexCliResponse:
    """Call ``codex exec`` and return the final assistant message."""
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", delete=False) as handle:
        output_path = Path(handle.name)
    schema_path: Optional[Path] = None
    use_output_schema = os.getenv("CODEX_CLI_USE_OUTPUT_SCHEMA", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if output_schema and use_output_schema:
        with tempfile.NamedTemporaryFile(
            prefix="codex-output-schema-",
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(output_schema, handle)
            schema_path = Path(handle.name)

    command = build_codex_exec_command(
        model=model,
        output_path=output_path,
        sandbox=sandbox,
        cwd=cwd,
        output_schema_path=schema_path,
    )
    prompt = build_codex_prompt(system, user, json_mode=output_schema is not None)
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    stdout_handle = None
    stderr_handle = None
    try:
        # Codex CLI may run nested commands that emit large JSON/log streams.
        # Use files instead of PIPEs so subprocess completion cannot be blocked
        # by an unread pipe or a delayed pipe EOF from a nested process.
        stdout_handle = tempfile.NamedTemporaryFile(prefix="codex-stdout-", delete=False)
        stderr_handle = tempfile.NamedTemporaryFile(prefix="codex-stderr-", delete=False)
        stdout_path = Path(stdout_handle.name)
        stderr_path = Path(stderr_handle.name)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        await asyncio.wait_for(
            process.communicate(prompt.encode("utf-8")),
            timeout=timeout_seconds,
        )
        stdout_handle.flush()
        stderr_handle.flush()
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        raw_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        error = None
        finish_reason = "stop"
        # Called on every outcome, not only failures: a success has to reset the
        # streak or the counter would only ever climb.
        exhausted = _note_call_outcome(process.returncode, stdout, stderr)
        if process.returncode:
            finish_reason = "error"
            error = (
                f"codex exec exited with code {process.returncode}; "
                f"stderr_tail={stderr[-900:]}; stdout_tail={stdout[-900:]}"
            )
            if is_usage_limit(stdout) or is_usage_limit(stderr) or exhausted:
                error = f"{USAGE_LIMIT_MARKER}: {error}"
        elif not raw_text.strip():
            finish_reason = "error"
            error = "codex exec produced no final assistant message"
        return CodexCliResponse(
            raw_text=raw_text,
            elapsed_seconds=time.monotonic() - started,
            finish_reason=finish_reason,
            error=error,
            stdout=stdout,
            stderr=stderr,
        )
    except asyncio.TimeoutError:
        if "process" in locals() and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                process.kill()
            await process.wait()
        stdout = (
            stdout_path.read_text(encoding="utf-8", errors="replace")
            if stdout_path is not None and stdout_path.exists()
            else ""
        )
        stderr = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path is not None and stderr_path.exists()
            else ""
        )
        return CodexCliResponse(
            raw_text="",
            elapsed_seconds=time.monotonic() - started,
            finish_reason="timeout",
            error=f"timeout after {timeout_seconds}s",
            stdout=stdout,
            stderr=stderr,
        )
    except Exception as exc:
        return CodexCliResponse(
            raw_text="",
            elapsed_seconds=time.monotonic() - started,
            finish_reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            output_path.unlink(missing_ok=True)
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)
            if stdout_path is not None:
                stdout_path.unlink(missing_ok=True)
            if stderr_path is not None:
                stderr_path.unlink(missing_ok=True)
        except Exception:
            pass


def call_codex_cli_sync(
    *,
    model: str,
    system: str,
    user: str,
    timeout_seconds: float,
    cwd: Optional[Path] = None,
    sandbox: str = "read-only",
    output_schema: Optional[dict] = None,
) -> CodexCliResponse:
    """Synchronous variant for LangGraph sync planner nodes."""
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", delete=False) as handle:
        output_path = Path(handle.name)
    schema_path: Optional[Path] = None
    use_output_schema = os.getenv("CODEX_CLI_USE_OUTPUT_SCHEMA", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if output_schema and use_output_schema:
        with tempfile.NamedTemporaryFile(
            prefix="codex-output-schema-",
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(output_schema, handle)
            schema_path = Path(handle.name)

    command = build_codex_exec_command(
        model=model,
        output_path=output_path,
        sandbox=sandbox,
        cwd=cwd,
        output_schema_path=schema_path,
    )
    prompt = build_codex_prompt(system, user, json_mode=output_schema is not None)
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    stdout_handle = None
    stderr_handle = None
    try:
        stdout_handle = tempfile.NamedTemporaryFile(prefix="codex-stdout-", mode="w+", encoding="utf-8", delete=False)
        stderr_handle = tempfile.NamedTemporaryFile(prefix="codex-stderr-", mode="w+", encoding="utf-8", delete=False)
        stdout_path = Path(stdout_handle.name)
        stderr_path = Path(stderr_handle.name)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                process.kill()
            process.wait()
            stdout_handle.flush()
            stderr_handle.flush()
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            return CodexCliResponse(
                raw_text="",
                elapsed_seconds=time.monotonic() - started,
                finish_reason="timeout",
                error=f"timeout after {timeout_seconds}s",
                stdout=stdout,
                stderr=stderr,
            )
        stdout_handle.flush()
        stderr_handle.flush()
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        raw_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        error = None
        finish_reason = "stop"
        # Called on every outcome, not only failures: a success has to reset the
        # streak or the counter would only ever climb.
        exhausted = _note_call_outcome(process.returncode, stdout, stderr)
        if process.returncode:
            finish_reason = "error"
            error = (
                f"codex exec exited with code {process.returncode}; "
                f"stderr_tail={stderr[-900:]}; stdout_tail={stdout[-900:]}"
            )
            if is_usage_limit(stdout) or is_usage_limit(stderr) or exhausted:
                error = f"{USAGE_LIMIT_MARKER}: {error}"
        elif not raw_text.strip():
            finish_reason = "error"
            error = "codex exec produced no final assistant message"
        return CodexCliResponse(
            raw_text=raw_text,
            elapsed_seconds=time.monotonic() - started,
            finish_reason=finish_reason,
            error=error,
            stdout=stdout,
            stderr=stderr,
        )
    except Exception as exc:
        return CodexCliResponse(
            raw_text="",
            elapsed_seconds=time.monotonic() - started,
            finish_reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            output_path.unlink(missing_ok=True)
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)
            if stdout_path is not None:
                stdout_path.unlink(missing_ok=True)
            if stderr_path is not None:
                stderr_path.unlink(missing_ok=True)
        except Exception:
            pass
