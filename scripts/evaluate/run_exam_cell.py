#!/usr/bin/env python3
"""Run one exam cell from the declared budget, refusing to drift from its control.

A cell is a (model, corpus) pair. Everything that decides its result — token
budget, action ceiling, sampling, prefill — comes from `config/exam_cells.json`
rather than the command line, and the control cell it will be compared against
is checked for the same budget before a single episode is played.

That check is the whole point. The BFS control ran at `token_budget 16384` and
`max_actions 200`; its treatment ran at 8192 and 40, and 8% of treatment
episodes hit an action ceiling the control never reached. The drop that came
out carried the budget difference inside it, and the control had to be re-run.
Nothing in the pipeline objected, because the two cells were launched by hand
months apart and `max_actions` was not even recorded on an episode.

So this script does three things a shell line cannot:

  * resolves the budget from one declaration, so two cells cannot disagree
    unless someone edits the declaration;
  * reads the control's own episodes and compares the budget it actually ran
    with, not what anyone believes it ran with;
  * refuses on a mismatch, naming the fields, instead of producing a number
    that looks fine.

Also runs the pre-flight, because a corpus with rows the environment cannot
elaborate produces episodes shaped exactly like a dead server.

Usage:
  set -a; source .env; set +a
  python3 scripts/evaluate/run_exam_cell.py --model goedel --corpus release307 \
      --output data/evaluation/exam/release309_goedel \
      --whole-proof-url http://100.77.209.48:5678/v1 \
      --whole-proof-model goedel-prover-v2-8b

  # a corpus that is not in the config yet — new release rows, say
  python3 scripts/evaluate/run_exam_cell.py --model bfs --rows data/evaluation/exam/minif2f_new_rows.jsonl \
      --output data/evaluation/exam/minif2f_new_bfs
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def _repo_root() -> Path:
    """Walk up to the marker; do not count directories. `parents[1]` encoded
    this file's depth under `scripts/` and resolves one level short after a
    move -- to a directory that exists, so nothing raises."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[-1]


REPO_ROOT = _repo_root()
CONFIG = REPO_ROOT / "config" / "exam_cells.json"

#: Budget fields an episode records, and which therefore can be checked against
#: a control that already ran. `prefill` is compared through `prefill_used`.
CHECKED = ("token_budget", "max_tokens", "max_actions", "max_steps",
           "n_per_step", "temperature", "top_p", "prompt_style",
           "reasoning_headroom", "plan_outside_budget")

#: Fields whose absence from a cell's declaration means `False`.
BOOLEAN_FIELDS = frozenset({"plan_outside_budget"})

#: Quantisations at or above eight bits. `unknown` is absent on purpose: a
#: provider that declares nothing may be serving int4.
_EIGHT_BIT_OR_ABOVE = frozenset({"fp8", "bf16", "fp16", "int8", "q8_0", "fp32"})


def control_budget(control_dir: Path) -> Optional[Dict[str, Any]]:
    """What the control cell actually ran with, read off its own episodes.

    Older cells predate the fields being recorded; those return the subset they
    have, and the caller reports which fields could not be checked rather than
    treating silence as agreement.
    """
    found: Dict[str, Any] = {}
    # The summary and the episodes each hold part of the budget and neither
    # holds all of it — the summary had `max_actions` and no `token_budget`,
    # the episodes the reverse. Reading only one is what let a 200-vs-40 gap
    # pass unnoticed, so read both and let them fill each other in.
    for summary in sorted(control_dir.glob("summary_*.json")):
        block = (json.loads(summary.read_text(encoding="utf-8")) or {}).get("config") or {}
        found.update({k: v for k, v in block.items() if k in (*CHECKED, "prefill_used")})
        break
    for episodes in sorted(control_dir.glob("episodes_*.jsonl")):
        with episodes.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    found.update({k: row[k] for k in (*CHECKED, "prefill_used") if k in row})
                    break
        break
    return found or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="bfs | goedel | luna")
    parser.add_argument("--corpus", default=None, help="a key under `corpora`")
    parser.add_argument("--rows", type=Path, default=None,
                        help="an exam-rows file not in the config (new release rows)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--tag", default="")
    parser.add_argument("--llama-cpp", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--whole-proof-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--whole-proof-model", default="")
    parser.add_argument("--plans", type=Path, default=None)
    parser.add_argument("--arm", default="closed_book")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="only when the corpus has already been pre-flighted")
    parser.add_argument("--allow-control-mismatch", action="store_true",
                        help="run anyway. The result is not comparable to that control; "
                             "say so wherever it is reported")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.model not in config["budgets"]:
        raise SystemExit(f"unknown model {args.model!r}; known: {sorted(config['budgets'])}")
    budget = config["budgets"][args.model]

    if args.rows:
        rows = args.rows
    elif args.corpus:
        if args.corpus not in config["corpora"]:
            raise SystemExit(f"unknown corpus {args.corpus!r}; known: {sorted(config['corpora'])}")
        rows = REPO_ROOT / config["corpora"][args.corpus]["rows"]
    else:
        raise SystemExit("pass --corpus or --rows")
    if not Path(rows).is_file():
        raise SystemExit(f"rows file not found: {rows}")

    # ---- the control parity check -----------------------------------------
    control_dir = REPO_ROOT / config["controls"][args.model]
    control = control_budget(control_dir)
    if control is None:
        print(f"note: control {control_dir} has no episodes yet — nothing to compare against.", flush=True)
    else:
        mismatch, unchecked = {}, []
        # A flag the config does not mention is off, not unspecified. Reading
        # the bare `.get` as the declared value made an absent key compare
        # `None` against the `False` the episodes record, so every cell without
        # the key refused to resume against its own control.
        for field in CHECKED:
            declared = budget.get(field)
            if field in BOOLEAN_FIELDS:
                declared = bool(declared)
            if field not in control:
                unchecked.append(field)
            elif control[field] != declared:
                mismatch[field] = (control[field], declared)
        if "prefill_used" in control and control["prefill_used"] != bool(budget.get("prefill")):
            mismatch["prefill"] = (control["prefill_used"], bool(budget.get("prefill")))
        elif "prefill_used" not in control:
            unchecked.append("prefill")
        if unchecked:
            print(f"note: {control_dir.name} predates these fields, so they are unverified: "
                  f"{', '.join(unchecked)}")
        if mismatch:
            lines = "\n".join(f"    {k:14s} control={c!r}  this cell={t!r}"
                              for k, (c, t) in mismatch.items())
            message = (
                f"refusing to run: the budget differs from the control this cell will be\n"
                f"compared against ({control_dir}):\n{lines}\n"
                f"  Any drop measured against it would carry the budget difference inside it.\n"
                f"  Fix the declaration in {CONFIG.relative_to(REPO_ROOT)}, re-run the control at\n"
                f"  the declared budget, or pass --allow-control-mismatch and say so in the write-up."
            )
            if not args.allow_control_mismatch:
                raise SystemExit(message)
            print("WARNING " + message, flush=True)
        else:
            print(f"control parity OK against {control_dir.name}", flush=True)

    # ---- quantisation floor ------------------------------------------------
    # No cell is served below eight bits. The rule has to be checked here
    # because it is invisible afterwards: an fp4 host answers exactly like an
    # fp8 one, and the difference lands inside the drop rather than beside it.
    #
    # Two ways to satisfy it, and a cell must declare one. Pin a provider and
    # the quantisations it may use -- the case for open weights served by a
    # third party, which is where somebody else picks the number. Or declare
    # `first_party_serving`, for a model served by whoever made it: there is one
    # serving and nothing to choose between. The second used to be implicit in
    # the absence of a `provider` key, which is not the same thing --- an open
    # weight left unpinned looks identical in the config and is free to be
    # routed to fp4.
    provider = budget.get("provider") or {}
    local = budget.get("local_serving") or {}
    if local:
        # Served from this machine: there is no host to pin and no maker to
        # name, but the quantisation is still a fact about the cell and is
        # held to the same floor. `engine` is recorded so the appendix can say
        # which server produced the tokens (llama.cpp and LM Studio differ on
        # logprobs, which decides whether a tactic search is scored at all).
        quant = str(local.get("quantization") or "").lower()
        if not local.get("engine") or not quant:
            raise SystemExit(
                f"refusing to run: {args.model} declares local_serving without "
                f"`engine` and `quantization`.")
        if quant not in _EIGHT_BIT_OR_ABOVE:
            raise SystemExit(
                f"refusing to run: {args.model} is served locally at {quant}, below "
                f"the panel's eight-bit floor. Allowed: "
                f"{', '.join(sorted(_EIGHT_BIT_OR_ABOVE))}.")
        if str(budget.get("url", "")).startswith("http") and not any(
                h in str(budget.get("url")) for h in ("127.0.0.1", "localhost", "0.0.0.0")):
            raise SystemExit(
                f"refusing to run: {args.model} declares local_serving but its url "
                f"is not on this machine.")
    elif budget.get("first_party_serving"):
        if provider and provider.get("allow_fallbacks"):
            raise SystemExit(
                f"refusing to run: {args.model} claims first-party serving but "
                f"allows fallbacks, which can route it away from the maker."
            )
    elif provider:
        quants = [str(q).lower() for q in (provider.get("quantizations") or [])]
        low = [q for q in quants if q not in _EIGHT_BIT_OR_ABOVE]
        if not quants:
            raise SystemExit(
                f"refusing to run: {args.model} pins a provider but declares no "
                f"quantisation. Name one at or above eight bits, or set "
                f"`first_party_serving` if the endpoint is the model's maker."
            )
        if low:
            # An exception is allowed, but only written down. A cell that serves
            # below the floor is not wrong to run -- it is wrong to run without
            # the reader knowing -- so the reason is required here, printed on
            # every run, and carried into the harness table in the appendix.
            reason = str(budget.get("quantization_exception") or "").strip()
            if not reason:
                raise SystemExit(
                    f"refusing to run: {args.model} allows {', '.join(low)}, below "
                    f"the panel's eight-bit floor. Allowed: "
                    f"{', '.join(sorted(_EIGHT_BIT_OR_ABOVE))}. To run it anyway, "
                    f"set `quantization_exception` to the reason, which the "
                    f"appendix then reports."
                )
            print(
                f"NOTE {args.model} runs at {', '.join(low)}, below the eight-bit "
                f"floor: {reason}",
                flush=True,
            )
        if provider.get("allow_fallbacks"):
            raise SystemExit(
                f"refusing to run: {args.model} allows provider fallbacks, so the "
                f"quantisation it declares is not the one it is guaranteed."
            )
    else:
        raise SystemExit(
            f"refusing to run: {args.model} declares none of `provider` (a pinned "
            f"host with quantisations), `first_party_serving` (the maker's own "
            f"endpoint) or `local_serving` (engine and quantisation on this "
            f"machine). One of the three has to be said out loud --- silence "
            f"used to mean the second, and that let an open-weight cell run "
            f"unpinned."
        )

    # ---- pre-flight --------------------------------------------------------
    if not args.skip_preflight:
        print(f"pre-flighting {rows} …", flush=True)
        playable = Path(str(rows).replace(".jsonl", "")) .with_name(
            Path(rows).stem + "_playable.jsonl")
        if playable.is_file():
            print(f"  already pre-flighted: {playable}", flush=True)
            rows = playable
        else:
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "evaluate" / "preflight_exam_rows.py"),
                 "--rows", str(rows), "--output", str(playable),
                 "--report", str(playable.with_suffix(".preflight.json"))],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                raise SystemExit("pre-flight failed; not starting the cell")
            rows = playable

    # ---- launch ------------------------------------------------------------
    cmd: List[str] = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate" / "run_seed_exam.py"),
        "--rows", str(rows), "--output", str(args.output),
        "--arm", args.arm, "--player", budget["player"],
        "--model-label", args.model,
        "--token-budget", str(budget["token_budget"]),
        "--max-tokens", str(budget["max_tokens"]),
        "--max-actions", str(budget["max_actions"]),
        "--max-steps", str(budget["max_steps"]),
        "--n-per-step", str(budget["n_per_step"]),
        "--temperature", str(budget["temperature"]),
        "--top-p", str(budget["top_p"]),
        "--attempts", str(args.attempts),
        "--tag", args.tag or f"{args.model}-{args.corpus or Path(rows).stem}",
        "--resume",
    ]
    if budget.get("max_attempts"):
        cmd += ["--max-attempts", str(budget["max_attempts"])]
    if budget.get("prefill"):
        cmd += ["--prefill", budget["prefill"]]
    if budget.get("prompt_style"):
        cmd += ["--prompt-style", budget["prompt_style"]]
    if budget.get("provider"):
        cmd += ["--provider", json.dumps(budget["provider"])]
    if budget.get("reasoning_headroom"):
        cmd += ["--reasoning-headroom", str(budget["reasoning_headroom"])]
    if budget.get("episode_concurrency"):
        cmd += ["--episode-concurrency", str(budget["episode_concurrency"])]
    if budget.get("generate_timeout"):
        cmd += ["--generate-timeout", str(budget["generate_timeout"])]
    if budget.get("generate_retries") is not None:
        cmd += ["--generate-retries", str(budget["generate_retries"])]
    if budget.get("plan_outside_budget"):
        cmd += ["--plan-outside-budget"]
    if budget.get("prefill_prefix"):
        cmd += ["--prefill-prefix"]
    if budget.get("api_key_env"):
        cmd += ["--api-key-env", budget["api_key_env"]]
    if budget.get("extra_body"):
        cmd += ["--extra-body", json.dumps(budget["extra_body"])]
    if budget["player"] == "whole_proof":
        cmd += ["--whole-proof-url", args.whole_proof_url]
        if args.whole_proof_model:
            cmd += ["--whole-proof-model", args.whole_proof_model]
    if budget["player"] == "bfs":
        cmd += ["--llama-cpp", args.llama_cpp]
    if budget["player"] == "codex":
        cmd += ["--codex-model", budget["codex_model"]]
    if args.plans:
        cmd += ["--plans", str(args.plans)]

    print("\n" + " ".join(f"'{c}'" if "\n" in c else c for c in cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
