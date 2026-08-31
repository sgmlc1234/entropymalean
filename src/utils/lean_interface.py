"""Small Lean 4 subprocess wrapper used by the certification MVP."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import langsmith as ls


class LeanChecker:
    """Template-only Lean checker.

    This class intentionally does not call an LLM. Unsupported problem families
    return an empty theorem from ``autoformalize_with_meta`` and are handled as
    ``unsupported`` by the certification layer.
    """

    def __init__(self, timeout_sec: float = 300, lean_path: str = "lean"):
        self.timeout = timeout_sec
        self.lean_path = lean_path
        self.lean_version: Optional[str] = None
        self._health_check_cache: Optional[bool] = None
        self._health_check_timestamp: Optional[float] = None
        self._cache_ttl = 300
        self._health_check_lock = asyncio.Lock()

    @ls.traceable(name="lean_health_check", run_type="tool")
    async def health_check(self) -> bool:
        """Return True when the Lean binary is available and executable."""
        now = time.time()
        if (
            self._health_check_cache is not None
            and self._health_check_timestamp is not None
            and now - self._health_check_timestamp < self._cache_ttl
        ):
            return self._health_check_cache

        async with self._health_check_lock:
            now = time.time()
            if (
                self._health_check_cache is not None
                and self._health_check_timestamp is not None
                and now - self._health_check_timestamp < self._cache_ttl
            ):
                return self._health_check_cache

            try:
                process = await asyncio.create_subprocess_exec(
                    self.lean_path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
                self.lean_version = stdout.decode("utf-8", errors="ignore").strip() or None
                ok = process.returncode == 0 and b"Lean" in stdout
            except (asyncio.TimeoutError, FileNotFoundError, OSError):
                ok = False
                self.lean_version = None

            if ok:
                self._health_check_cache = True
                self._health_check_timestamp = now
            else:
                self._health_check_cache = None
                self._health_check_timestamp = None
        run_tree = ls.get_current_run_tree()
        if run_tree is not None:
            run_tree.metadata.update(
                {
                    "lean_path": self.lean_path,
                    "lean_version": self.lean_version,
                    "available": ok,
                }
            )
        return ok

    async def autoformalize_with_meta(
        self,
        statement: str,
        answer: str = "",
        model: Optional[str] = None,
        parent_lean_context: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a Lean theorem using only deterministic templates.

        ``model`` and ``parent_lean_context`` are accepted for old call-site
        compatibility but ignored.
        """
        from src.utils.lean_templates import (
            SUPPORTED_FAMILIES,
            detect_family,
            generate_lean_template,
            is_trivial_stub,
        )

        family = detect_family(statement)
        if family not in SUPPORTED_FAMILIES:
            return "", {
                "family": family,
                "theorem_name": None,
                "proof_method": "none",
                "params": {},
                "aligned": False,
                "alignment_note": f"unsupported family={family!r}",
                "lean_level": 0,
                "anti_stub_passed": False,
                "source": "unsupported",
            }

        template = generate_lean_template(statement, answer, family=family)
        if template is None or is_trivial_stub(template.lean_code):
            return "", {
                "family": family,
                "theorem_name": None,
                "proof_method": "none",
                "params": {},
                "aligned": False,
                "alignment_note": "template generation failed",
                "lean_level": 0,
                "anti_stub_passed": False,
                "source": "template",
            }

        lean_level = 3 if template.aligned else 2
        return template.lean_code, {
            "family": template.family,
            "theorem_name": template.theorem_name,
            "proof_method": template.proof_method,
            "params": template.params,
            "aligned": template.aligned,
            "alignment_note": template.alignment_note,
            "lean_level": lean_level,
            "anti_stub_passed": True,
            "source": "template",
        }

    async def autoformalize(self, statement: str, model: str = "") -> str:
        """Backward-compatible wrapper returning only the Lean code string."""
        lean_code, _ = await self.autoformalize_with_meta(statement, answer="", model=model)
        return lean_code

    @ls.traceable(name="lean_type_check", run_type="tool")
    async def type_check(self, lean_code: str) -> Tuple[bool, str]:
        """Run full Lean elaboration on ``lean_code``."""
        temp_file = None
        process = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
                f.write(lean_code)
                temp_file = f.name

            process = await asyncio.create_subprocess_exec(
                self.lean_path,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            run_tree = ls.get_current_run_tree()
            if run_tree is not None:
                run_tree.metadata.update(
                    {
                        "lean_path": self.lean_path,
                        "timeout_sec": self.timeout,
                        "exit_code": process.returncode,
                        "code_chars": len(lean_code),
                    }
                )
            if process.returncode == 0:
                return True, stdout.decode("utf-8", errors="ignore")
            return False, stderr.decode("utf-8", errors="ignore")
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()
            return False, "Lean type-check timeout"
        except Exception as exc:
            return False, f"Type-check error: {exc}"
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    async def check_syntax(self, lean_code: str) -> Tuple[bool, str]:
        """Compatibility wrapper; syntax is checked by full type checking."""
        return await self.type_check(lean_code)
