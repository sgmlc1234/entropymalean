"""The panel is a declaration, and this pins it.

`config/exam_cells.json` names the thirteen models the paper reports, the
directory each cell's episodes live in, and the budget each cell runs under.
Every script that scores or replays a cell reads it. The failures this guards
against were all seen once: a `groups` entry with no cell behind it made
`run_panel` raise KeyError; a budget with no serving declaration made a replay
refuse a cell that had run for weeks; a credential variable named in the
config and absent from `.env.example` cost a reviewer a 401.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "exam_cells.json"
ENV_EXAMPLE = ROOT / ".env.example"

PANEL_GROUPS = ("lean_provers", "reasoning_slms", "frontier_llms")
SERVING_FORMS = ("provider", "first_party_serving", "local_serving")
EIGHT_BIT_OR_ABOVE = {"fp8", "bf16", "fp16", "int8", "q8_0", "fp32"}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel(cfg) -> list[str]:
    return [m for g in PANEL_GROUPS for m in cfg["groups"].get(g, [])]


def test_panel_has_thirteen_models_in_three_groups(cfg, panel):
    assert set(cfg["groups"]) == set(PANEL_GROUPS)
    assert len(panel) == 13
    assert len(set(panel)) == 13


def test_every_panel_model_has_both_cells_and_a_budget(cfg, panel):
    for model in panel:
        for registry in ("controls", "treatments", "budgets"):
            assert model in cfg[registry], f"{model} missing from {registry}"


def test_every_panel_budget_declares_one_serving_form(cfg, panel):
    """One form each. A first-party cell reached through a router may also
    carry a `provider` pin naming the maker -- that is a routing instruction,
    not a second serving claim -- but then it names no quantisation (the maker
    picks it) and no fallback (which would route away from the maker)."""
    for model in panel:
        budget = cfg["budgets"][model]
        forms = [f for f in SERVING_FORMS if budget.get(f)]
        if budget.get("first_party_serving") and budget.get("provider"):
            pin = budget["provider"]
            assert not pin.get("quantizations"), f"{model}: first-party cell declares quantisations"
            assert not pin.get("allow_fallbacks"), f"{model}: first-party cell allows fallbacks"
            forms.remove("provider")
        assert len(forms) == 1, f"{model} declares {forms or 'no serving form'}"


def test_pinned_providers_name_a_quantisation_at_or_above_eight_bits(cfg, panel):
    for model in panel:
        budget = cfg["budgets"][model]
        provider = budget.get("provider")
        if not provider or budget.get("first_party_serving"):
            continue
        quants = [str(q).lower() for q in provider.get("quantizations") or []]
        assert quants, f"{model} pins a provider without a quantisation"
        low = [q for q in quants if q not in EIGHT_BIT_OR_ABOVE]
        assert not low or cfg["budgets"][model].get("quantization_exception"), \
            f"{model} allows {low} below the floor with no recorded exception"
        assert not provider.get("allow_fallbacks"), f"{model} allows provider fallbacks"


def test_local_serving_names_engine_and_eight_bit_quantisation(cfg, panel):
    for model in panel:
        local = cfg["budgets"][model].get("local_serving")
        if not local:
            continue
        assert local.get("engine"), f"{model}: local_serving without engine"
        assert str(local.get("quantization", "")).lower() in EIGHT_BIT_OR_ABOVE, \
            f"{model}: local quantisation {local.get('quantization')!r} below the floor"


def test_named_credentials_are_documented(cfg, panel):
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M))
    for model in panel:
        name = cfg["budgets"][model].get("api_key_env")
        if name:
            assert name in documented, f"{model} reads ${name}, which .env.example does not list"


def test_cell_directories_are_repo_relative(cfg, panel):
    for model in panel:
        for registry in ("controls", "treatments"):
            path = cfg[registry][model]
            assert not Path(path).is_absolute(), f"{registry}/{model} is an absolute path"
            assert path.startswith("data/evaluation/exam/"), f"{registry}/{model} = {path}"


def test_treatment_and_control_share_the_proof_budget(cfg, panel):
    for model in panel:
        budget = cfg["budgets"][model]
        assert budget.get("token_budget") == 8192, f"{model}: token_budget {budget.get('token_budget')}"
        assert budget.get("max_attempts", 4) == 4 or budget.get("player") == "bfs", model
