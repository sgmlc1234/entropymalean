#!/usr/bin/env python3
"""Export each proof term on two platforms and require the digests to agree.

The levels below this one all run in one place, so they answer "does this hold
here" rather than "does this hold for anyone who runs it". The gap is not
hypothetical: a third of ProofNet-Verified fails to compile under our pin
despite shipping as verified, and nothing in that artifact would have predicted
it.

What makes the stronger claim checkable is that `lean4export` is deterministic
across operating systems. Given the same toolchain and the same package
revisions, the exported term for a theorem is byte-identical on macOS-aarch64
and linux-aarch64 — verified here, 34,392,833 bytes and the same SHA-256 on
both. So the digest is a compact certificate of the whole environment: a third
party who matches our published pins regenerates the export and compares one
hash, without needing our machine, our elaborator, or a 34 MB download.

The export must be produced the same way on every platform or the comparison is
meaningless. Elaborating `Solution.lean` under an explicit `LEAN_PATH` and
exporting from that module is the procedure; running it inside a Lake project
instead pulls a different set of modules into the environment and changes the
term's dependency closure — that difference alone accounted for 34 MB vs 20 MB
in an earlier run and looked exactly like platform nondeterminism.

Usage:
  python scripts/faithfulness/kernel/check_export_reproducible.py \
    --rows data/benchmarks/proofnet_verified/raw/seeds_50_rows.jsonl \
    --output data/benchmarks/proofnet_verified/raw/seeds_50_export_digests.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories.

    `parents[1]` encoded this file's depth under `scripts/`. When the tree was
    reorganised it resolved one level short -- to a directory that exists, so
    nothing raised and the script simply found no data.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.certification.levels import runtime_pins  # noqa: E402

HOST_EXPORT = REPO_ROOT / "references/lean4export/.lake/build/bin/lean4export"
#: The second platform. This was a local aarch64 Lima VM, reached with
#: `limactl shell`; it is now the x86-64 Ubuntu host that runs the comparator
#: batches. The pair is stronger for it -- the old one shared the Mac's silicon
#: and differed only in OS, while this one differs in OS *and* architecture *and*
#: machine, which is what "does it hold for anyone who runs it" is asking.
REMOTE_HOST = "user@linux-host.example"
#: Reuse the connection the user authenticated; this script never authenticates.
#: The socket's name changes each time it is re-opened, so it is discovered
#: rather than hard-coded: a stale path does not fail loudly, it makes every
#: remaining row a remote failure that reads as a digest mismatch.
def _live_socket() -> str:
    import subprocess as _sp
    for candidate in sorted(Path.home().joinpath(".ssh").glob("[0-9]" * 10 + "*"),
                            key=lambda q: -q.stat().st_mtime):
        if not candidate.is_socket():
            continue
        probe = _sp.run(["ssh", "-S", str(candidate), "-O", "check", REMOTE_HOST],
                        capture_output=True, text=True)
        if "Master running" in (probe.stdout + probe.stderr):
            return str(candidate)
    raise SystemExit(
        "no live ControlMaster socket for " + REMOTE_HOST + ".\n"
        "Open one first:  ssh -M -S ~/.ssh/<name> -fN " + REMOTE_HOST)
REMOTE_EXPORT = "~/lean4export-pinned/.lake/build/bin/lean4export"
REMOTE_PACKAGES = "~/comparator-toolchain/mathlib4/.lake/packages"
#: mathlib4 is the project on that host, not one of its packages.
REMOTE_MATHLIB_LIB = "$HOME/comparator-toolchain/mathlib4/.lake/build/lib/lean"
#: The exporter is part of the environment being compared and was missing from
#: the pin record: a different lean4export can serialise the same term
#: differently, which would read as a platform disagreement. Both sides run this
#: revision, and it is written into the report.
EXPORTER_REV = "12581a6b680d8478175596338eb2d53383a323e3"
TOOLCHAIN = (REPO_ROOT / "lean-toolchain").read_text().strip()


def _script(export_bin: str, packages_glob: str, workdir: str, toolchain: str,
            extra_lean_path: tuple = ()) -> str:
    """One recipe, run verbatim on both platforms.

    Written as a single shell program rather than as per-platform Python so the
    two runs cannot drift apart in ways that would show up as a digest
    mismatch and be mistaken for a real one.

    The toolchain is written into the working directory rather than assumed:
    `elan` resolves `lean` from the nearest `lean-toolchain`, so a bare temp
    directory silently gets whatever the machine's default is. That is how the
    first attempt failed — the host picked a different Lean and rejected our
    oleans with "incompatible header" — and a check for reproducibility that
    leans on an unpinned default has no business granting the level.
    """
    extra = "\n".join(f'LP="$LP:{path}"' for path in extra_lean_path)
    return f"""
set -e
cd {workdir}
printf '%s\n' {shlex.quote(toolchain)} > lean-toolchain
export PATH="$HOME/.elan/bin:$PATH"
LP=""
for d in {packages_glob}/*/.lake/build/lib/lean; do LP="$LP:$d"; done
{extra}
export LEAN_PATH="${{LP#:}}"
lean -o Solution.olean Solution.lean
export LEAN_PATH="$LEAN_PATH:{workdir}"
{export_bin} Solution -- "$DECL" > {workdir}/export.txt
"""


def digest_of(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def run_host(name: str, solution: str, workdir: Path, timeout: float) -> Optional[Dict[str, Any]]:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "Solution.lean").write_text(solution, encoding="utf-8")
    script = _script(
        str(HOST_EXPORT), str(REPO_ROOT / ".lake/packages"), str(workdir), TOOLCHAIN
    )
    proc = subprocess.run(
        ["bash", "-lc", script],
        env={"DECL": name, "HOME": str(Path.home()), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        capture_output=True, text=True, timeout=timeout,
    )
    out = workdir / "export.txt"
    if proc.returncode or not out.is_file():
        return None
    return {"digest": digest_of(out), "bytes": out.stat().st_size}


_SOCKET_CACHE: Optional[str] = None


def _socket() -> str:
    global _SOCKET_CACHE
    if _SOCKET_CACHE is None:
        _SOCKET_CACHE = _live_socket()
        print(f"  ControlMaster: {_SOCKET_CACHE}", flush=True)
    return _SOCKET_CACHE


def run_vm(name: str, solution: str, timeout: float) -> Optional[Dict[str, Any]]:
    # Unique per process: the fixed path meant a second invocation
    # silently overwrote a running batch's working directory.
    workdir = f"/tmp/xrepro-{os.getpid()}"
    script = (
        f"rm -rf {workdir} && mkdir -p {workdir} && "
        f"cat > {workdir}/Solution.lean <<'EOF_SOLUTION'\n{solution}\nEOF_SOLUTION\n"
        + f"DECL={shlex.quote(name)}\n"
        + _script(REMOTE_EXPORT, REMOTE_PACKAGES, workdir, TOOLCHAIN,
                  extra_lean_path=(REMOTE_MATHLIB_LIB,))
        + f"\nwc -c < {workdir}/export.txt\nsha256sum {workdir}/export.txt | cut -d' ' -f1\n"
    )
    proc = subprocess.run(
        ["ssh", "-S", _socket(), "-o", "BatchMode=yes", REMOTE_HOST,
         "bash", "-lc", shlex.quote(script)],
        capture_output=True, text=True, timeout=timeout,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode or len(lines) < 2:
        return None
    return {"digest": lines[-1], "bytes": int(lines[-2])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    results: List[Dict[str, Any]] = []
    consecutive_remote_failures = 0
    for index, row in enumerate(rows, 1):
        name = str(row.get("name"))
        solution = str(row.get("lean_code") or "")
        host = run_host(name, solution, Path("/tmp/xrepro-host") / name, args.timeout)
        vm = run_vm(name, solution, args.timeout)
        agree = bool(host and vm and host["digest"] == vm["digest"])
        results.append(
            {
                "name": name,
                "reproducible": agree,
                "export_digest": host["digest"] if agree else None,
                "export_bytes": host["bytes"] if agree else None,
                "macos_aarch64": host,
                "linux_x86_64": vm,
            }
        )
        # Name the three outcomes apart. They were all printed as DIFF, so a
        # dead SSH socket read as twelve consecutive digest mismatches -- the
        # strongest negative verdict this check can give -- for rows it never
        # managed to measure at all.
        if agree:
            mark = "SAME"
        elif not host:
            mark = "HOSTFAIL"
        elif not vm:
            mark = "REMOTEFAIL"
        else:
            mark = "DIFF"
        # A remote that has stopped answering will not start again on its own;
        # grinding through the rest turns a fixable interruption into hours of
        # unusable output. Consecutive failures end the run instead.
        consecutive_remote_failures = (
            consecutive_remote_failures + 1 if mark == "REMOTEFAIL" else 0
        )
        size = f"{host['bytes'] / 1048576:.1f}MB" if host else "-"
        print(
            f"[{mark}] {index:3d}/{len(rows)} {name[:40]:40s} {size:>8s} "
            f"{(host or {}).get('digest', '?')[:12]}",
            flush=True,
        )

        if index % 10 == 0 or index == len(rows):
            _write(args.output, rows, results)
        if consecutive_remote_failures >= 5:
            print(f"\n{consecutive_remote_failures} consecutive remote failures — "
                  f"the connection is gone, not the reproducibility. Stopping at "
                  f"{index}/{len(rows)}; results so far are in {args.output}.",
                  flush=True)
            break

    _write(args.output, rows, results)


def _write(output: Path, rows: list, results: List[Dict[str, Any]]) -> None:
    agreed = sum(1 for r in results if r["reproducible"])
    report = {
        "rows": len(rows),
        "reproducible": agreed,
        "rate": round(agreed / max(1, len(rows)), 3),
        "platforms": ["macos-aarch64", "linux-x86_64"],
        "pins": dict(runtime_pins(str(REPO_ROOT)), lean4export_revision=EXPORTER_REV),
        "mismatch_kinds": dict(
            Counter(
                "host_failed" if not r["macos_aarch64"]
                else "remote_failed" if not r["linux_x86_64"]
                else "digest_differs"
                for r in results if not r["reproducible"]
            )
        ),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Written whole each time rather than appended: a checkpoint that leaves a
    # half-written record is worse than none, because it still parses.
    tmp = output.with_suffix(output.suffix + ".part")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    if len(results) == len(rows):
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"reproducible {agreed}/{len(results)} measured -> {output}", flush=True)


if __name__ == "__main__":
    main()
