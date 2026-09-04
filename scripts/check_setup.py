#!/usr/bin/env python3
"""Say, before anything runs, whether this checkout can run it.

Every Lean check in the release depends on one toolchain and one Mathlib
revision, and a checkout that has the toolchain but not the built Mathlib
fails every probe with `unknown module prefix 'Mathlib'` --- a message that
reads as a failing proof rather than as a missing build. Every hosted
evaluation cell depends on one credential, and a missing one used to surface
as a 401 from the wrong host. This script asks each of those questions
directly and prints the fix beside any answer that is no.

    python3 scripts/check_setup.py             # Lean, Mathlib, pins
    python3 scripts/check_setup.py --cells     # plus the credential each panel cell reads
    python3 scripts/check_setup.py --model leanstral   # one cell's credential

Exit status is the number of failed checks, so it can gate a script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATHLIB_OLEAN = ROOT / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"
MANIFEST = ROOT / "lake-manifest.json"
TOOLCHAIN = ROOT / "lean-toolchain"
CONFIG = ROOT / "config/exam_cells.json"
RELEASE = ROOT / "data/release/eml1_release.jsonl"
ENV = ROOT / ".env"

OK, NO = "  ok   ", "  FAIL "


def _release_pins() -> tuple[str, str]:
    """The toolchain and Mathlib revision every released row's certificate names."""
    for candidate in (RELEASE, ROOT / "data/release/eml2_release.jsonl"):
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                row = json.loads(next(l for l in handle if l.strip()))
            cert = row.get("certificate") or {}
            return str(cert.get("lean_toolchain") or ""), str(cert.get("mathlib_revision") or "")
    return "", ""


def _load_env() -> None:
    if not ENV.is_file():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def check_lean() -> list[tuple[bool, str, str]]:
    out = []
    lake = shutil.which("lake")
    out.append((lake is not None, "lake on PATH",
                "install elan: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"))
    want_tc = TOOLCHAIN.read_text(encoding="utf-8").strip() if TOOLCHAIN.is_file() else ""
    pin_tc, pin_ml = _release_pins()
    out.append((bool(want_tc) and want_tc == pin_tc,
                f"lean-toolchain ({want_tc or 'missing'}) matches the release certificate ({pin_tc or '?'})",
                "the checkout's lean-toolchain must equal the pin every certificate names"))
    if lake:
        try:
            got = subprocess.run(["lean", "--version"], capture_output=True, text=True,
                                 cwd=ROOT, timeout=60).stdout.strip()
        except Exception as error:  # noqa: BLE001
            got = f"({error})"
        # elan resolves the toolchain from lean-toolchain in the cwd, so a wrong
        # version here means the pinned toolchain is not installed yet.
        ver = want_tc.split(":")[-1].lstrip("v")
        out.append((ver in got, f"`lean --version` reports the pinned toolchain ({got})",
                    f"elan toolchain install {want_tc}"))
    rev = ""
    if MANIFEST.is_file():
        for pkg in json.loads(MANIFEST.read_text(encoding="utf-8")).get("packages", []):
            if pkg.get("name") == "mathlib":
                rev = str(pkg.get("rev") or "")
    out.append((bool(rev) and rev == pin_ml,
                f"lake-manifest pins Mathlib {rev[:10] or '?'} = certificate {pin_ml[:10] or '?'}",
                "do not `lake update`; the manifest must stay at the certified revision"))
    out.append((MATHLIB_OLEAN.is_file(), "Mathlib is built (Mathlib.olean present)",
                "lake exe cache get && lake build   (downloads prebuilt oleans, ~minutes; building from source takes hours)"))
    if lake and MATHLIB_OLEAN.is_file():
        # One real elaboration. Two things can go wrong that the checks above
        # cannot see: Mathlib imports but does not load, and lake prefixes every
        # run with `manifest out of date`, which the evaluation environment
        # would otherwise have to filter out of goal states.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as handle:
            handle.write("import Mathlib\nexample : (2 : Nat) + 2 = 4 := by norm_num\n")
            probe = handle.name
        try:
            run = subprocess.run(["lake", "env", "lean", probe], capture_output=True, text=True,
                                 cwd=ROOT, timeout=600)
            text = (run.stdout + run.stderr)
            out.append((run.returncode == 0 and "error" not in text.lower(),
                        "`import Mathlib` elaborates through `lake env lean`",
                        (text.strip().splitlines() or ["no output"])[-1][:120]))
            out.append(("manifest out of date" not in text,
                        "lake runs without `manifest out of date` warnings",
                        "the checked-out packages disagree with lake-manifest.json; re-run `lake exe cache get` "
                        "in a clean clone, and never `lake update` (it would move off the certified revision)"))
        except Exception as error:  # noqa: BLE001
            out.append((False, "`lake env lean` probe ran", str(error)[:120]))
        finally:
            Path(probe).unlink(missing_ok=True)
    return out


def check_cells(models: list[str] | None) -> list[tuple[bool, str, str]]:
    out = []
    if not CONFIG.is_file():
        return [(False, "config/exam_cells.json present", "restore it from the repository")]
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    panel = [m for g in ("lean_provers", "reasoning_slms", "frontier_llms") for m in cfg["groups"].get(g, [])]
    for model in (models or panel):
        budget = cfg["budgets"].get(model)
        if budget is None:
            out.append((False, f"{model}: in config", f"unknown model; known: {', '.join(panel)}"))
            continue
        url = str(budget.get("url") or "")
        local = not url.startswith("http") or any(h in url for h in ("127.0.0.1", "localhost"))
        if local:
            out.append((True, f"{model}: served locally ({budget.get('local_serving', {}).get('engine', 'llama.cpp')}), no credential", ""))
            continue
        name = budget.get("api_key_env") or "OPENROUTER_API_KEY"
        out.append((bool(os.environ.get(name)), f"{model}: ${name} set", f"add {name}=... to .env"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", action="store_true", help="also check the credential each panel cell reads")
    ap.add_argument("--model", action="append", default=[], help="check one cell's credential (repeatable)")
    args = ap.parse_args()
    _load_env()

    checks = check_lean()
    if args.cells or args.model:
        checks += check_cells(args.model or None)
    failed = 0
    for ok, what, fix in checks:
        print((OK if ok else NO) + what)
        if not ok:
            failed += 1
            if fix:
                print(f"         -> {fix}")
    print(f"\n{len(checks) - failed} of {len(checks)} checks pass.")
    if not failed:
        print("This checkout can verify the release. Hosted evaluation cells also need their credentials (--cells).")
    sys.exit(failed)


if __name__ == "__main__":
    main()
