"""Regression tests for the public accepted-ledger schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "archive" / "normalize_accepted_ledger.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("normalize_accepted_ledger", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_row_preserves_fixed_nested_schema():
    normalizer = _load_script_module()
    row = {
        "problem_id": "child",
        "benchmark": "minif2f",
        "op_type": "mutation",
        "generation": 1,
        "slot": 2,
        "family": "theorem_proof",
        "statement": "A theorem.",
        "formal_statement": "theorem child : True := by trivial",
        "parents": [
            {
                "parent_id": "parent",
                "source": "manual",
                "statement": "Parent theorem.",
                "formal_statement": "theorem parent : True := by trivial",
            }
        ],
    }

    normalized = normalizer.normalize_row(row, {}, {})

    assert list(normalized.keys()) == normalizer.PUBLIC_FIELDS
    assert list(normalized["certificate"].keys()) == normalizer.CERTIFICATE_FIELDS
    assert list(normalized["curation"].keys()) == normalizer.CURATION_FIELDS
    assert list(normalized["provenance"].keys()) == normalizer.PROVENANCE_FIELDS
    assert list(normalized["hashes"].keys()) == normalizer.HASH_FIELDS
    assert list(normalized["parents"][0].keys()) == normalizer.PARENT_FIELDS
    assert normalized["parents"][0]["proof_idea"] == ""
    assert normalized["curation"]["snapshot"] is None
    assert normalized["provenance"]["source_file"] is None
