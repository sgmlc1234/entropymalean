"""Optional offline Jixia analysis hooks for planner memory.

Jixia is not part of the runtime proof loop. This module only records whether
the local Jixia checkout is compatible, and provides a narrow wrapper for
future offline enrichment jobs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _toolchain(path: Path) -> str:
    file = path / "lean-toolchain"
    try:
        return file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def jixia_toolchain_matches(repo_root: Path, jixia_dir: Path) -> bool:
    return bool(_toolchain(repo_root)) and _toolchain(repo_root) == _toolchain(jixia_dir)


def analyze_lean_code_for_memory(
    lean_code: str,
    *,
    repo_root: Path,
    jixia_dir: Optional[Path] = None,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    """Run Jixia only when its toolchain matches the current project.

    The current Jixia CLI surface is intentionally not assumed here. If a local
    checkout exposes an analyzer script later, this wrapper is the single place
    to wire it in without touching generation-time code.
    """
    jixia_dir = jixia_dir or (repo_root / "references" / "jixia")
    if not jixia_dir.exists():
        return {"status": "skipped", "reason": "jixia_not_found"}
    if not jixia_toolchain_matches(repo_root, jixia_dir):
        return {
            "status": "skipped",
            "reason": "jixia_skipped_toolchain_mismatch",
            "repo_toolchain": _toolchain(repo_root),
            "jixia_toolchain": _toolchain(jixia_dir),
        }
    if not lean_code.strip():
        return {"status": "skipped", "reason": "empty_lean_code"}

    input_path = jixia_dir / ".entropy_mag_jixia_input.lean"
    input_path.write_text(lean_code, encoding="utf-8")
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", str(input_path)],
            cwd=jixia_dir,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            input_path.unlink()
        except OSError:
            pass
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "typeReferences": [],
        "valueReferences": [],
        "line_goals": [],
    }


def jixia_digest(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {
        "status": value.get("status"),
        "reason": value.get("reason"),
        "typeReferences": list(value.get("typeReferences") or [])[:20],
        "valueReferences": list(value.get("valueReferences") or [])[:20],
        "line_goals": list(value.get("line_goals") or [])[:10],
    }
