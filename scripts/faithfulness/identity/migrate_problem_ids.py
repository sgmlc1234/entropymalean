"""Rewrite every stored identifier into the readable form, references included.

An identifier is only useful if everything that points at it moves with it, so
this is not a rename of one field. `parent_ids`, `ancestor_ids`,
`source_problem_id`, the judge's sibling records, the comparator workspace
directories and the release/report artifacts all carry ids, and a migration that
misses one silently orphans a lineage.

The chain has to be reconstructed rather than parsed, because the old form did
not record the operator anywhere. Each row knows its own `operator_variant` and
its parents, so the corpus is walked in dependency order: a row's chain is its
parent's chain plus its own code, and a row whose parent has not been resolved
yet waits for the next pass.

Nothing is written until every row has a new id and the mapping is one-to-one.
A collision would merge two problems into one, which is worse than a long name.
Backups are written beside each file and the full mapping is saved, so the whole
migration can be undone from the mapping alone.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.orchestration.problem_ids import chain_of, child_id, strip_suffix

#: Every field anywhere in a row that holds an id or a list of them.
ID_FIELDS = ("problem_id", "source_problem_id", "release_id")
ID_LIST_FIELDS = ("parent_ids", "ancestor_ids")


def load_rows(paths: Iterable[Path]) -> List[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_path"] = str(path)
                rows.append(row)
    return rows


#: sha256("")[:8]. Rows whose generation failed carry no statement, so the old
#: form fingerprinted the empty string and every one of them got this. Reusing
#: it would map several distinct failures onto one id.
EMPTY_FINGERPRINT = "e3b0c442"


def legacy_fingerprint(problem_id: str) -> str:
    """The 8-hex discriminator the old form put last, or a stable stand-in."""
    import hashlib

    tail = str(problem_id or "").rsplit("__", 1)[-1]
    if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail) and tail != EMPTY_FINGERPRINT:
        return tail
    return hashlib.sha256(str(problem_id).encode()).hexdigest()[:8]


def legacy_depth(problem_id: str) -> int:
    """How many generations a legacy id records, by counting its suffixes."""
    return str(problem_id or "").count("__theorem_gen")


def build_mapping(rows: List[dict], seeds: set) -> Dict[str, str]:
    """old id -> new id, resolved in dependency order."""
    # One id can appear on several rows -- a retry, a slot exception, the
    # certified result -- and only some of them record the operator. Taking the
    # last one wrote `mm` into eighteen chains whose rows knew their variant
    # perfectly well, so the most informative row wins instead.
    def rank(row: dict) -> tuple:
        return (
            1 if row.get("operator_variant") else 0,
            1 if row.get("status") == "certified" else 0,
            1 if row.get("parent_ids") else 0,
        )

    by_id: Dict[str, dict] = {}
    for row in rows:
        key = str(row.get("problem_id") or "")
        if key and (key not in by_id or rank(row) > rank(by_id[key])):
            by_id[key] = row
    mapping: Dict[str, str] = {seed: seed for seed in seeds}
    pending = {key for key in by_id if key not in mapping}
    while pending:
        progressed = False
        for old in sorted(pending):
            row = by_id[old]
            parents = [str(p) for p in (row.get("parent_ids") or []) if str(p or "").strip()]
            if not parents:
                # A survivor or a seed copy: it is its own row, so it keeps its id.
                mapping[old] = old
                pending.discard(old)
                progressed = True
                continue
            if any(p not in mapping for p in parents):
                continue
            # A parent that is still in legacy form carries its depth in its
            # suffixes but not its operators, so the chain is padded with `mm`
            # for each unknown ancestor. Dropping them instead would give a
            # third-generation row a one-link chain and collide it with a
            # first-generation sibling, which is what the first run did.
            new_id = child_id(
                [mapping[p] for p in parents],
                op_type=str(row.get("op_type") or ""),
                operator_variant=str(row.get("operator_variant") or ""),
                fingerprint=legacy_fingerprint(old),
            )
            missing = max(0, legacy_depth(old) - 1 - sum(
                len(chain_of(mapping[p])) for p in parents[:1]))
            if missing and len(parents) == 1:
                stem, chain, fingerprint = new_id.rsplit("__", 2)
                chain = ".".join(["mm"] * missing + chain.split("."))
                new_id = f"{stem}__{chain}__{fingerprint}"
            mapping[old] = new_id
            pending.discard(old)
            progressed = True
        if not progressed:
            # Parents outside the loaded set — map them through unchanged so the
            # rows that depend on them can still be resolved.
            orphan_parents = {
                p for key in pending for p in (by_id[key].get("parent_ids") or [])
                if str(p) not in mapping
            }
            if not orphan_parents:
                break
            for parent in orphan_parents:
                mapping[str(parent)] = str(parent)
    return mapping


def rewrite(value: Any, mapping: Dict[str, str]) -> Any:
    return mapping.get(str(value), value) if isinstance(value, str) else value


def apply_to_row(row: dict, mapping: Dict[str, str]) -> dict:
    out = dict(row)
    for field in ID_FIELDS:
        if field in out:
            out[field] = rewrite(out[field], mapping)
    for field in ID_LIST_FIELDS:
        if isinstance(out.get(field), list):
            out[field] = [rewrite(v, mapping) for v in out[field]]
    # Nested places an id hides: parent cards, contributions keyed by id, and
    # the release export's parent blocks.
    if isinstance(out.get("parents"), list):
        out["parents"] = [
            {**p, "parent_id": rewrite(p.get("parent_id"), mapping)} if isinstance(p, dict) else p
            for p in out["parents"]
        ]
    for field in ("parent_contributions", "semantic_parent_contribution"):
        if isinstance(out.get(field), dict):
            out[field] = {rewrite(k, mapping): v for k, v in out[field].items()}
    if isinstance(out.get("quality_evidence"), dict):
        evidence = dict(out["quality_evidence"])
        if isinstance(evidence.get("parent_contribution"), dict):
            evidence["parent_contribution"] = {
                rewrite(k, mapping): v for k, v in evidence["parent_contribution"].items()
            }
        out["quality_evidence"] = evidence
    out.pop("_path", None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="*", default=[
        Path("data/certified/run-a"),
        Path("data/certified/ablation"),
        Path("data/release"),
    ])
    parser.add_argument("--seeds", type=Path, default=Path("data/certified/run-a/seeds"))
    parser.add_argument("--mapping", type=Path, default=Path("data/release/id_migration.json"))
    parser.add_argument("--workspaces", type=Path, default=Path("data/release/comparator"))
    parser.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = parser.parse_args()

    import csv

    seeds = set()
    for path in sorted(args.seeds.glob("*.csv")):
        if path.name.endswith(".pre_p6_fix"):
            continue
        with path.open(encoding="utf-8") as handle:
            for seed in csv.DictReader(handle):
                if seed.get("id"):
                    seeds.add(str(seed["id"]))

    # The release and rejected files are derived from the campaign rows and use
    # a different shape for lineage (`parents: [{parent_id}]` rather than
    # `parent_ids`), so migrating them in place both misses their parents and
    # duplicates work. They are regenerated from the migrated sources instead.
    derived = {"eml_v1_release.jsonl", "eml_v1_rejected.jsonl"}
    paths = sorted({p for root in args.roots for p in root.rglob("*.jsonl")
                    if p.name not in derived})
    rows = load_rows(paths)
    print(f"{len(rows)} rows across {len(paths)} files · {len(seeds)} seeds")

    mapping = build_mapping(rows, seeds)
    generated = {old: new for old, new in mapping.items() if old != new}
    # A collision would merge two problems under one name, which is a worse
    # defect than the long ids this replaces, so it stops the migration.
    grouped = collections.defaultdict(list)
    for old_id, new_id in mapping.items():
        grouped[new_id].append(old_id)
    collisions = {new: sorted(olds) for new, olds in grouped.items() if len(olds) > 1}
    print(f"  ids rewritten: {len(generated)}   unchanged: {len(mapping) - len(generated)}")
    if collisions:
        print(f"  ! {len(collisions)} collisions — refusing to write")
        for new, olds in list(collisions.items())[:5]:
            print(f"      {new}\n        <- {olds}")
        raise SystemExit(1)

    lengths_before = [len(o) for o in generated]
    lengths_after = [len(n) for n in generated.values()]
    if generated:
        print(f"  id length: max {max(lengths_before)} -> {max(lengths_after)}, "
              f"mean {sum(lengths_before)//len(lengths_before)} -> {sum(lengths_after)//len(lengths_after)}")
        for old, new in list(sorted(generated.items(), key=lambda kv: -len(kv[0])))[:3]:
            print(f"    {old}\n    -> {new}")

    if not args.apply:
        print("\ndry run — pass --apply to write")
        return

    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    for path in paths:
        backup = path.with_suffix(path.suffix + ".pre_id_migration")
        if not backup.exists():
            shutil.copy2(path, backup)
        rewritten = [
            apply_to_row(json.loads(line), mapping)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rewritten) + "\n",
                        encoding="utf-8")
    print(f"  rewrote {len(paths)} files (backups: *.pre_id_migration)")

    # Side files keyed by problem_id: the two re-judge passes, the pruned-row
    # re-judgements, the hypothesis prune patch, the comparator preparation and
    # its kernel verdicts. Every one of them is an input to the release export,
    # so a mapping that stops at the JSONL leaves the export joining old keys
    # against new rows and silently dropping the evidence.
    for name, shape in (
        ("rejudged.json", "list"), ("rejudged_run2.json", "list"),
        ("pruned_rejudged_1.json", "list"), ("pruned_rejudged_2.json", "list"),
        ("comparator_batch.json", "list"), ("comparator_results.json", "list"),
        ("dead_hypotheses.json", "list"), ("hypothesis_prune.json", "dict"),
        # Added after a later run: these three were written before this list
        # existed and were left behind by it, so the export joined new rows
        # against old keys and reported the checks as never having run.
        ("goal_roundtrip.json", "list"), ("statement_rewrites.json", "list"),
        ("redundancy_scan.json", "list"),
    ):
        path = args.mapping.parent / name
        if not path.is_file():
            continue
        backup = path.with_suffix(path.suffix + ".pre_id_migration")
        if not backup.exists():
            shutil.copy2(path, backup)
        data = json.loads(path.read_text(encoding="utf-8"))
        if shape == "list":
            data = [{**r, "problem_id": mapping.get(str(r.get("problem_id")), r.get("problem_id"))}
                    for r in data]
        else:
            data = {mapping.get(str(k), k): v for k, v in data.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  remapped {name}")

    if args.workspaces.is_dir():
        moved = 0
        for directory in sorted(args.workspaces.iterdir()):
            new = mapping.get(directory.name)
            if directory.is_dir() and new and new != directory.name:
                directory.rename(directory.with_name(new))
                moved += 1
        print(f"  renamed {moved} comparator workspace(s)")
    print(f"  mapping: {args.mapping}")


if __name__ == "__main__":
    main()
