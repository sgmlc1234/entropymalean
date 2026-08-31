"""Small CLI-helper tests for the proof-evaluation launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.evaluation.model_runner import ModelConfig


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_proof_evaluation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_proof_evaluation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_models_filters_exact_labels_in_requested_order():
    cli = _load_script_module()
    panel = [
        ModelConfig(label="Goedel-Prover-V2-8B", provider_slug="goedel"),
        ModelConfig(label="BFS-Prover-V2-7B", provider_slug="bfs"),
    ]

    selected = cli._select_models("BFS-Prover-V2-7B,Goedel-Prover-V2-8B", panel)

    assert [model.label for model in selected] == [
        "BFS-Prover-V2-7B",
        "Goedel-Prover-V2-8B",
    ]


def test_select_models_rejects_unknown_label():
    cli = _load_script_module()
    panel = [ModelConfig(label="BFS-Prover-V2-7B", provider_slug="bfs")]

    with pytest.raises(SystemExit, match="unknown --models label"):
        cli._select_models("MissingModel", panel)
