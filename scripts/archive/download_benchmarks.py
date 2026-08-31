#!/usr/bin/env python3
"""Snapshot-download the AI4Math 2026 benchmark suite into ``data/benchmarks/``.

Usage:
    # all benchmarks declared in config/benchmarks.yaml
    python scripts/archive/download_benchmarks.py

    # only one
    python scripts/archive/download_benchmarks.py miniF2F

    # custom config path
    python scripts/archive/download_benchmarks.py --config config/benchmarks.yaml

The script uses ``huggingface_hub.snapshot_download`` so the data is mirrored
read-only into the local directory; re-running the script is idempotent and
only re-fetches changed files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download

# Local imports: allow `python scripts/archive/download_benchmarks.py` from repo root.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from src.evaluation.benchmarks import (  # noqa: E402
    BenchmarkCatalog,
    BenchmarkSpec,
    load_catalog,
)


def _download_one(spec: BenchmarkSpec, repo_root: Path) -> Path:
    target = repo_root / spec.local_dir
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.name}] downloading {spec.hf_repo}@{spec.revision} -> {target}")
    snapshot_download(
        repo_id=spec.hf_repo,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=[
            # canonical parquet layouts
            "data/*.parquet",
            "*.parquet",
            "default/**/*.parquet",
            # original record formats some datasets ship on the main branch
            "*.jsonl",
            "*.json",
            "*.csv",
            "data/*.jsonl",
            "data/*.csv",
            # documentation / licensing
            "README*",
            "LICENSE*",
            "*.md",
        ],
    )
    return target


def _write_manifest(catalog: BenchmarkCatalog, names: List[str], repo_root: Path) -> None:
    """Emit a small JSON manifest summarizing what is available locally."""
    manifest = {}
    for name in names:
        spec = catalog.benchmarks[name]
        path = repo_root / spec.local_dir
        data_files = sorted(p.relative_to(path).as_posix() for p in path.glob("data/*.parquet"))
        manifest[name] = {
            "hf_repo": spec.hf_repo,
            "revision": spec.revision,
            "license": spec.license,
            "citation": spec.citation,
            "local_dir": str(spec.local_dir),
            "data_files": data_files,
            "splits": spec.splits,
            "schema": {
                "problem_id_col": spec.schema.problem_id_col,
                "statement_col": spec.schema.statement_col,
                "answer_extraction": spec.schema.answer_extraction,
                "quality_filter": spec.schema.quality_filter,
            },
        }
    out = repo_root / catalog.cache_root / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", help="subset to download")
    parser.add_argument(
        "--config",
        default="config/benchmarks.yaml",
        type=Path,
        help="catalog path (default: config/benchmarks.yaml)",
    )
    args = parser.parse_args()

    repo_root = _REPO
    catalog_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    catalog = load_catalog(catalog_path)
    names = args.benchmarks or sorted(catalog.benchmarks)
    unknown = [n for n in names if n not in catalog.benchmarks]
    if unknown:
        raise SystemExit(
            f"unknown benchmarks: {unknown}; known: {sorted(catalog.benchmarks)}"
        )

    for name in names:
        _download_one(catalog.benchmarks[name], repo_root)
    _write_manifest(catalog, names, repo_root)
    print("done.")


if __name__ == "__main__":
    main()
