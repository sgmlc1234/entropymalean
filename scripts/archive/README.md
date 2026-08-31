# Archive

33 scripts, 6,732 lines. Nothing here is on a path anything calls, and nothing
here has been used for months.

**These are kept, not deleted, and the distinction is the point.** Several are
the only record of how something in the release was made. A script that ran once
is not dead if a released row came out of it — it is that row's provenance, and
the corpus is auditable only while that record exists.

## Dates

The refactor rewrote the repository-root idiom in every one of these files, so
every `mtime` now reads the day of that commit. The evidence for archiving them
is no longer visible on disk; it is in git, and it is written down here so it
survives the next thing that touches the files.

Last substantive change, ignoring the refactor commits:

| When | Count | What they are |
|---|---|---|
| 2026-05 | 16 | The first campaign and the LeanDojo evaluation harness |
| 2026-07 | 9 | The pre-`exam` evaluation generation, and the Goedel/BFS handoff drivers |
| unknown | 8 | Never committed before the checkpoint; predate version control here |

The eight with no history existed only on disk until commit `5e2eb04`. That is
the strongest argument for archiving rather than deleting: for those, this
directory *is* the record.

## What is in here

### Campaign drivers, superseded

`run_bfs_decrease_after_goedel.sh` · `run_goedel_decrease_8way.sh` ·
`run_goedel_decrease_treatment_only_resume.sh` ·
`proofnet_goedel_handoff_after_bfs.sh` ·
`proofnet_goedel_stmtfix_4way_handoff.sh` · `run_q8_expanded_resume.sh` ·
`run_full_campaign.sh` · `run_eml_campaign.sh` · `run_crossover_focus.sh` ·
`run_operator_ablation.sh`

1,855 lines, most of it near-identical wait-then-launch logic: hold until a lane
frees on another machine, start the next arm, poll. They were written one per
campaign because each waited on a different thing. `scripts/generate/run_eml1_main.sh`
replaced the pattern with one driver that skips groups that already have output.

### Evaluation harnesses, superseded

`run_evaluation.py` · `run_proof_evaluation.py` · `run_leandojo_eval.py` ·
`smoke_leandojo.py` · `prepare_leandojo_theorems.py` · `run_exam_grid.py` ·
`run_exam_smoke.py` · `reload_lmstudio_lean_eval_models.sh` ·
`serve_goedel_llama_server.sh`

Two generations of harness. The LeanDojo three are a different harness
altogether — the panel now plays through the exam environment in
`scripts/evaluate/`, and `provers.sh` at the repository root replaced the
serving scripts.

### One-off data work

`aggregate_campaign.py` · `apply_hallucination_filter.py` ·
`audit_accepted_statements.py` · `audit_generation_yield.py` ·
`build_accepted_gallery.py` · `certify_csv.py` · `download_benchmarks.py` ·
`normalize_accepted_ledger.py` · `repair_exam_row_context.py`

Each ran against a ledger or a benchmark snapshot that has since been folded
into what `data/` holds now. `build_accepted_gallery.py` and
`normalize_accepted_ledger.py` are 1,500 lines between them and produced the
review artifacts the current release format grew out of.

### Exam-row tooling, off the current path

`exam_rows_to_csv.py` · `exam_rows_to_game_csv.py` · `export_exam_demo.py` ·
`export_seed_gallery.py` · `extract_episodes.py`

Confirmed unused by the evaluation session before they were moved. The row
builders they sit next to — `build_minif2f_exam_rows.py`, `enrich_exam_rows.py` —
are **not** here: `release_to_exam_rows.py` imports them as modules, so they
stayed in `scripts/evaluate/` despite nothing invoking them. Not invoked is not
unused.

## Using something from here

They run, but their paths and assumptions are from when they were written. Two
things changed under them:

- The repository root idiom was rewritten during the refactor, so they resolve
  it by walking up to `pyproject.toml`. That part is current.
- Everything else — data paths, model names, ledger formats — is not. Read
  before running.

The classification that put them here, with the reasoning per script, is in
`../INVENTORY.md`.
