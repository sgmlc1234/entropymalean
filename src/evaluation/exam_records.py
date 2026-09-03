"""Read the episode records behind a panel cell, from wherever they are.

Two layouts hold the same records. The working tree keeps a cell as
`data/evaluation/exam/<cell>/episodes_*.jsonl`, one directory per run, which
`.gitignore` excludes because that tree also holds ablations and aborted
runs. The release ships the twenty-six reported cells as
`data/evaluation/exam_evidence/<model>_<arm>.jsonl.gz` with a manifest of
SHA-256 digests. Every script that scores a cell reads through this module, so
a checkout that has only the bundle -- a reviewer's -- scores the same cells
the same way as the tree the bundle was cut from.

The raw directory wins when both are present, because it is the one still
being written to during a campaign; the bundle is a frozen copy of it.
"""

from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


ROOT = repo_root()
BUNDLE_DIR = ROOT / "data/evaluation/exam_evidence"


def read_episode_file(path: Path | str) -> list[dict]:
    """One episode file, plain or gzipped, as a list of rows."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bundle_path(model: str, arm: str) -> Path:
    """`arm` is `control` or `treatment`; the config's plural is accepted too."""
    arm = arm[:-1] if arm.endswith("s") else arm
    return BUNDLE_DIR / f"{model}_{arm}.jsonl.gz"


def raw_files(cell_dir: Optional[str | Path]) -> list[str]:
    """The scored files of a raw cell directory, `before-replay` backups excluded."""
    if not cell_dir:
        return []
    p = Path(cell_dir)
    if not p.is_absolute():
        p = ROOT / p
    return sorted(f for f in glob.glob(f"{p}/episodes_*.jsonl") if "before-replay" not in f)


def cell_episodes(cell_dir: Optional[str | Path], model: str = "", arm: str = "") -> list[dict]:
    """Every episode of one cell.

    Reads the raw directory when it holds scored files, otherwise the bundled
    copy named by `model` and `arm`. Returns an empty list when neither exists,
    which callers already treat as "cell not run"; it never guesses a file.
    """
    files = raw_files(cell_dir)
    if files:
        rows: list[dict] = []
        for f in files:
            rows.extend(read_episode_file(f))
        return rows
    if model and arm:
        bundled = bundle_path(model, arm)
        if bundled.exists():
            return read_episode_file(bundled)
    return []


def source_of(cell_dir: Optional[str | Path], model: str = "", arm: str = "") -> str:
    """Where `cell_episodes` would read from, for a script to print."""
    files = raw_files(cell_dir)
    if files:
        return ", ".join(str(Path(f).relative_to(ROOT)) for f in files)
    if model and arm and bundle_path(model, arm).exists():
        return str(bundle_path(model, arm).relative_to(ROOT))
    return "(no records)"


def episode_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand what a command line names -- files, gz files, or cell directories."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(Path(f) for f in raw_files(p))
        else:
            out.append(p)
    return out
