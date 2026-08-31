#!/usr/bin/env python3
"""Fill the audit fields the exporter left empty, from the content it shipped.

`export_release.py` built the hash block by *reading* `statement_sha256` off the
upstream record and falling back to `""`:

    "statement_sha256": row.get("statement_sha256") or "",

Nothing upstream ever computed it, so all 417 rows shipped with two empty audit
hashes while `dedup_fingerprint`, which the same block computes inline, was
fine. The dataset card advertises the hashes as audit fields, so a reader who
opens the JSONL to check one finds nothing to check.

The hash has to be taken over the string the release actually ships, not the
certified record's: 129 statements were rewritten after certification, and a
hash of the pre-rewrite text would authenticate a statement no one received.

`family` and `llm_model` are also advertised and also absent; both are carried
by the certified record and are copied across.

Read-only unless `--write`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

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


REPO = _repo_root()
def sha256_text(text: Any) -> str:
    """The project's hash rule: strip, encode UTF-8, digest; empty stays empty."""
    value = str(text or "").strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def certified_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in glob.glob(str(REPO / "data/certified/**/*.jsonl"), recursive=True):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                pid = row.get("problem_id")
                if pid and pid not in index:
                    index[pid] = row
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path,
                        default=REPO / "data/release/eml1_release.jsonl")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.release.read_text(encoding="utf-8").splitlines() if l.strip()]
    cert = certified_index()
    changed = collections.Counter()
    unmatched = []

    for row in rows:
        h = row.setdefault("hashes", {})
        for field, key in (("statement", "statement_sha256"),
                           ("formal_statement", "formal_statement_sha256")):
            digest = sha256_text(row.get(field))
            if h.get(key) != digest:
                h[key] = digest
                changed[key] += 1
        source = cert.get(row.get("problem_id"))
        if source is None:
            unmatched.append(row.get("problem_id"))
            continue
        for key in ("family", "llm_model"):
            value = str(source.get(key) or "")
            if row.get(key) != value:
                row[key] = value
                changed[key] += 1

    print(f"{len(rows)} rows")
    for key, n in sorted(changed.items()):
        print(f"  {key:26s} set on {n}")
    if unmatched:
        print(f"  no certified record for {len(unmatched)}: {unmatched[:5]}")

    digests = [r["hashes"]["statement_sha256"] for r in rows]
    formal = [r["hashes"]["formal_statement_sha256"] for r in rows]
    print(f"\n  statement_sha256 distinct        {len(set(digests))}/{len(rows)}")
    print(f"  formal_statement_sha256 distinct {len(set(formal))}/{len(rows)}")
    print(f"  empty digests                    {sum(1 for d in digests + formal if not d)}")

    if not args.write:
        print("\n(dry run — pass --write to apply)")
        return
    with args.release.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.release}")


if __name__ == "__main__":
    main()
