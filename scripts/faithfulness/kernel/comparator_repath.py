"""Point every prepared workspace at this machine's mathlib.

`prepare_comparator_workspace` writes the mathlib location as an absolute path,
resolved against the machine that built the batch. That machine was macOS and
the machine that can actually run comparator is Linux, so the path in all 146
`lakefile.toml` files is wrong the moment the bundle is copied.

Rewriting on arrival keeps the bundle portable: the server says where its own
mathlib lives and nothing else has to change. Both the lakefile and the pinned
manifest carry the path, and missing either leaves `lake` resolving half the
build against a directory that does not exist.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_LAKE_PATH = re.compile(r'path = "[^"]*"')
_MANIFEST_PATH = re.compile(r'"[^"]*[/\\]packages[/\\]mathlib"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", type=Path)
    parser.add_argument("mathlib", type=Path)
    args = parser.parse_args()

    mathlib = args.mathlib.expanduser().resolve()
    if not mathlib.is_dir():
        sys.exit(f"no such directory: {mathlib}")

    # Mathlib's own dependencies -- plausible, batteries, aesop, proofwidgets,
    # Qq, Cli, importGraph, LeanSearchClient -- are listed in the manifest as
    # git packages, so `lake` tries to clone all eight into every workspace. On
    # the first run it failed on the first one and every workspace reported a
    # comparator failure in under a second, which reads exactly like 146 rejected
    # proofs and was nothing of the kind. They are already built inside the
    # mathlib clone, so each workspace is pointed at those instead of fetching
    # its own copy: eight symlinks rather than 1168 clones.
    packages = mathlib / ".lake" / "packages"
    shared = sorted(p for p in packages.iterdir() if p.is_dir()) if packages.is_dir() else []

    changed = 0
    for lakefile in sorted(args.workspaces.glob("*/lakefile.toml")):
        lakefile.write_text(
            _LAKE_PATH.sub(f'path = "{mathlib}"', lakefile.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        manifest = lakefile.with_name("lake-manifest.json")
        if manifest.is_file():
            manifest.write_text(
                _MANIFEST_PATH.sub(f'"{mathlib}"', manifest.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        local = lakefile.parent / ".lake" / "packages"
        local.mkdir(parents=True, exist_ok=True)
        for package in shared:
            link = local / package.name
            # A leftover from a partial run is what produced "cannot create
            # directory: file exists"; replace whatever is there.
            if link.is_symlink() or link.exists():
                if link.is_symlink() or link.is_file():
                    link.unlink()
                else:
                    continue
            link.symlink_to(package, target_is_directory=True)
        changed += 1
    print(f"  repathed {changed} workspace(s) -> {mathlib}")
    print(f"  linked {len(shared)} shared package(s): {', '.join(p.name for p in shared) or 'none'}")
    if not changed:
        sys.exit(f"no workspaces found under {args.workspaces}")
    if not shared:
        print("  ! no packages found under the mathlib clone — lake will try to fetch them")


if __name__ == "__main__":
    main()
