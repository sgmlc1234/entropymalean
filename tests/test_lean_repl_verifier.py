import asyncio
import time

from src.evaluation.lean_verifier import LeanVerifyResult
from src.evaluation.lean_repl_verifier import strip_repl_import_commands
from src.evaluation.lean_repl_verifier import LeanReplVerifier


def test_strip_repl_import_commands_preserves_proof_surface():
    code = """import Mathlib
import Aesop
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem t : True := by
  trivial
"""

    stripped = strip_repl_import_commands(code)

    assert "import Mathlib" not in stripped
    assert "import Aesop" not in stripped
    assert stripped.startswith("set_option maxHeartbeats 400000")
    assert "open BigOperators Real Nat Topology Rat" in stripped
    assert "theorem t : True := by" in stripped


def test_repl_verify_has_outer_wall_timeout():
    verifier = LeanReplVerifier()

    def stuck_verify(code: str, timeout: float) -> LeanVerifyResult:
        time.sleep(1.0)
        return LeanVerifyResult(ok=True, complete=True)

    verifier._verify_blocking = stuck_verify  # type: ignore[method-assign]

    result = asyncio.run(verifier.verify("theorem t : True := by trivial", timeout=0.01))

    assert result.complete is False
    assert "REPL wall-time timeout" in result.summary()
