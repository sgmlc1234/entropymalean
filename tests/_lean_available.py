"""One question every Lean-touching test asks: is a built Mathlib here?

`shutil.which("lake")` is the wrong test. A checkout with the toolchain
installed but `lake exe cache get` not yet run has `lake` on PATH and no
`Mathlib.olean`, and every probe then fails with `unknown module prefix
'Mathlib'` --- reported as a failing proof rather than as a missing build.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATHLIB_OLEAN = ROOT / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean"


def mathlib_built() -> bool:
    return shutil.which("lake") is not None and MATHLIB_OLEAN.is_file()


SKIP_REASON = (
    "needs a built Mathlib: run `lake exe cache get && lake build` in the "
    "repository root (REVIEWERS.md §0.2)"
)
