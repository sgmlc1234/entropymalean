# What each script is for

101 scripts, 20,655 lines. This classifies every one of them so the next step —
moving them — is a decision with evidence behind it rather than a guess.

**Nothing has been moved or deleted.** This file is the argument; the
reorganisation is separate and reversible (checkpoint commit `5e2eb04`).

Three buckets:

| Bucket | Meaning | Count |
|---|---|---|
| **Pipeline** | On a path something else calls: a driver, another script, or `REVIEWERS.md` | 43 |
| **Analysis** | Nothing calls it, but it produced a number or figure that is in the paper | 23 |
| **Historical** | Ran once, its output has been absorbed, nothing has called it in months | 35 |

The third bucket is the one that needs stating plainly: **a script that ran once
is not dead if a released row came out of it.** It is that row's provenance. So
"historical" here means *archive*, never *delete*.

---

## 1. Pipeline

### 1.1 Generation

| Script | What it does |
|---|---|
| `run_pool_generation.py` | One seed group through N generations. The unit of work. |
| `run_eml1_main.sh` | Drives every group in a campaign; skips groups that already have output |
| `run_proofnet_p.sh` | The ProofNet ten-generation campaign, and the shortest readable driver |
| `prepare_campaign_inputs.py` | Seed CSV → the shape the pool generator reads |

### 1.2 Faithfulness layer

This is the layer §3b of the paper describes, and it is organised the way that
section argues it should be: **each check sits with the faculty that can settle
the question it asks.** The grouping below is that argument made into
directories. A check placed with the wrong faculty is the failure mode — a judge
asked a syntactic question, or a parser asked a semantic one.

#### Questions with a determinate answer go to Lean

The answer does not depend on who is asking, so it is put to Lean as a
compilation whose outcome is the verdict.

| Script | Question it settles |
|---|---|
| `screen_vacuous_seeds.py` | Are the hypotheses jointly satisfiable, or does the theorem hold because nothing satisfies it? |
| `check_dead_hypotheses.py` | Remove each hypothesis and recompile: was it load-bearing? |
| `scan_release_redundancy.py` | Does one parent already prove this row outright? |
| `scan_crossover_redundancy.py` | The same question for crossovers, where one parent may be unused |
| `audit_axiom_closures.py` | Retroactive axiom audit of rows certified before the audit existed |
| `merge_certificate_level.py` | Folds the axiom verdict back into the certificate |

Two of these are weaker than their names suggest, and the difference is written
into `check_dead_hypotheses.py`: it asks whether *this proof* still compiles
without the hypothesis, not whether *the theorem* still holds without it. Those
come apart whenever a proof consumes a hypothesis it did not need.

#### Questions of identity go to a hash

Whether two rows state the same theorem is not a matter of degree, so it is not
a matter for a judge.

| Script | Question it settles |
|---|---|
| `export_release.py` | Alpha-normal collision against every earlier campaign, not just this run |
| `migrate_problem_ids.py` | Rebuilds identifiers when the id scheme changes, preserving the fingerprint |

The identifier is `{roots}__{operator chain}__{statement fingerprint}`, and the
fingerprint is a hash of the statement — so **two slots that produce the same
theorem must collide**, and that collision is what drops the duplicate. An id
collision is the mechanism working, not a bug.

#### Questions of meaning go to a reader, and are asked twice

| Script | Question it settles |
|---|---|
| `rejudge_corpus.py` | Does the child demand reasoning the parent's proof does not supply? Run twice; admission needs both. |
| `measure_judge_accuracy.py` | Scores the judge against hand labels — what one verdict is worth |
| `run_goal_roundtrip_batch.py` | Does the prose describe the goal Lean actually elaborated? |
| `run_alignment_audit.py` | The same audit, batched over rows certified before the check existed |
| `benchmark_alignment_signal.py` | Scores that signal against ProofNet-Verified labels |
| `rewrite_statements.py` | Rewrites prose in the seed's register, and gates the rewrite on the round-trip |

The round-trip's value comes from **three participants that cannot see each
other's work**: Lean elaborates, one model informalises the goal alone, a third
compares the two texts blind. Any change that lets one see another's input
destroys the check while leaving it looking like it ran.

#### Independence goes to a kernel that did not produce the proof

Everything above is decided by our own installation. This group is the part
whose value comes from *not* being ours.

| Script | Role |
|---|---|
| `fetch_lean_comparator.sh` | Pinned checkouts of comparator, lean4export, lean-eval |
| `comparator_setup.sh` | Installs the toolchain on a bare Ubuntu box, then verifies |
| `comparator_preflight.sh` | Does this machine have what comparator needs? Run before the batch. |
| `prepare_comparator_batch.py` | One portable workspace per row; runs anywhere |
| `comparator_repath.py` | Points prepared workspaces at *this* machine's Mathlib |
| `run_comparator_batch.sh` | Replays every workspace through the kernel. Linux only — the sandbox is Landlock. |
| `certify_seeds_replay.py` | The same, over the seed set |
| `verify_gt_replay.py` | Replays ground-truth proofs under our pin and records whether they hold |
| `check_export_reproducible.py` | Exports each term on two platforms and requires the digests to agree |

The last one is the top rung. It is also the most fragile to adapt, and two
mistakes are recorded in its docstring because both looked like platform
disagreement: exporting from inside a Lake project (which changes the term's
dependency closure), and running two different `lean4export` revisions.

### 1.3 Release assembly

| Script | What it does |
|---|---|
| `export_release.py` | Applies every gate and writes the release plus the rejected rows |
| `build_release_report.py` | Renders all rows as cards, with every judge's reasoning |
| `build_release_reports.sh` | Compiles those three documents with lualatex |
| `publish_release_page.py` | Injects the release into the working website |
| `backfill_release_metadata.py` | Fills fields added after rows were certified |

`export_release.py` reads only the campaigns in its `SOURCES` map, and
`rejudge_corpus.py` only those in `DEFAULT_INDEX`. **A campaign missing from
either contributes nothing, silently** — this has cost real rows twice.

### 1.4 Evaluation

Now in `scripts/evaluate/`, moved during a window when no cell was running.
`watch_openrouter_spend.py` moved here too: it was filed under analysis, but it
kills billed cells when the credit limit nears, which makes it an operational
part of this cluster rather than a measurement of it.

`run_panel.py` · `run_exam_cell.py` · `run_seed_exam.py` ·
`release_to_exam_rows.py` · `preflight_exam_rows.py` ·
`finalize_panel_numbers.py` · `lineage_gap_from_exam.py` ·
`analyze_parent_child_ablation.py` · `verify_solved_episodes.py` ·
`audit_repl_desync.py` · `drop_unmeasured_episodes.py` ·
`analyze_exam_arms.py` · `provers.sh` *(serves the two local provers)* · `check_setup.py` *(toolchain, Mathlib build and revision, per-cell credentials; run first)*

`analyze_parent_child_ablation.py` is on this list for a reason no reference
graph could find. `lineage_gap_from_exam.py:35` loads its lineage parser **by
path**:

```python
spec = importlib.util.spec_from_file_location(
    "pca", ROOT / "scripts/evaluate/analyze_parent_child_ablation.py")
```

That is not an import, so nothing static sees it, and moving the file breaks
Table 3 **at run time** rather than at analysis time. The reuse is deliberate:
two copies of the lineage rules would eventually disagree.

A sweep for the same pattern across `scripts/` and `src/` found no others — the
only other `__import__` calls pull in `json` and `re`. There are no scripts
whose path is assembled at run time.

Two of these are verification rather than measurement, and belong to the
faithfulness story even though they sit in the evaluation pipeline:
`verify_solved_episodes.py` re-elaborates the Lean of every episode scored as
solved, and `audit_repl_desync.py` counts episodes whose verdict was computed
for a different problem. That is the only direction of corruption that inflates
a score and the only one the records can still settle.

---

## 2. Analysis

Nothing calls these. Each produced a number or a figure that is in the paper or
on the website, which is why "unreferenced" cannot be the deletion criterion.

| Script | What it produced |
|---|---|
| `measure_topic_drift.py` | The topic-composition figures (§4, Appendix C) |
| `measure_topic_fit.py` | Tested whether topic fit predicts crossover success — it did not replicate |
| `measure_planner_variants.py` | What the planner actually chooses, by calling it |
| `plan_seed_groups.py`, `plan_minif2f_groups.py`, `plan_proofnet_groups.py` | Seed-group allocation for each campaign |
| `write_seed_groups.py`, `build_resume_seeds.py` | Materialise a plan as seed CSVs |
| `analyze_parent_child_ablation.py` | The seed-lineage ablation |
| `check_lineage_relation.py` | Asks Lean how a child relates to its parent |
| `classify_lean_failures.py` | Sorts `Unknown constant` failures by whether a palette can fix them |
| `audit_exam_episodes.py` | Finds episodes recording an environment failure as a model failure |
| `pnv_hygiene_check.py` | ProofNet parent hygiene against ProofNet-Verified |
| `select_minif2f_v2_seeds.py`, `select_proofnet_seeds.py` | Chose the two seed sets |
| `build_minif2f_exam_rows.py`, `build_pnv_exam_rows.py` | Built the control corpus — the only path back if it must be rebuilt |
| `assemble_minif2f_seed_set.py`, `complete_seed_proofs.py` | Built the miniF2F seeds and their Gen-0 proofs |
| `fill_hint_ladder.py`, `merge_hand_written_hints.py`, `generate_proof_plans.py` | The hint ladder and proof plans in the seed CSVs |
| `normalize_plan_numbering.py` | Strips self-written numbering from plan steps |
| `watch_openrouter_spend.py` | Stops billed cells before the credit limit |

`measure_topic_fit.py` is worth keeping for a reason that is not its output: it
records a hypothesis that **failed to replicate across three measurements**, and
that record is why the claim is not in the paper.

---

## 3. Historical

Ran once, nothing has called them in months, and their output has been absorbed
into artifacts that are now under version control. **Archive, do not delete** —
several are the provenance of released rows or of the seed sets.

**Campaign drivers, superseded** (90+ days, near-duplicates of one another):
`run_bfs_decrease_after_goedel.sh` · `run_goedel_decrease_8way.sh` ·
`run_goedel_decrease_treatment_only_resume.sh` ·
`proofnet_goedel_handoff_after_bfs.sh` ·
`proofnet_goedel_stmtfix_4way_handoff.sh` · `run_q8_expanded_resume.sh` ·
`run_full_campaign.sh` · `run_eml_campaign.sh` · `run_crossover_focus.sh` ·
`run_operator_ablation.sh`

These four alone are 1,100 lines of near-identical wait-then-launch logic.

**Superseded evaluation paths:** `run_evaluation.py` ·
`run_proof_evaluation.py` · `run_leandojo_eval.py` · `smoke_leandojo.py` ·
`prepare_leandojo_theorems.py` · `run_exam_grid.py` · `run_exam_smoke.py` ·
`reload_lmstudio_lean_eval_models.sh` · `serve_goedel_llama_server.sh`

**One-off data work:** `aggregate_campaign.py` · `apply_hallucination_filter.py`
· `audit_accepted_statements.py` · `audit_generation_yield.py` ·
`build_accepted_gallery.py` · `certify_csv.py` · `download_benchmarks.py` ·
`normalize_accepted_ledger.py` · `repair_exam_row_context.py`

**Exam-row builders, off-path:** `enrich_exam_rows.py` ·
`build_treatment_aid_columns.py` · `exam_rows_to_csv.py` ·
`exam_rows_to_game_csv.py` · `export_exam_demo.py` · `export_seed_gallery.py` ·
`extract_episodes.py`

The superseded evaluation drivers above are self-contained: the six shell
drivers all call `run_proof_evaluation.py`, and nothing outside that group calls
any of them. They move together or not at all.

**Confirmed off-path** by the evaluation session: the two live row files are
`seeds_all100.jsonl` (3 Aug, control) and `release535_playable.jsonl` (20 Aug,
treatment), and none of these produced either. The two that built the control
corpus are listed under Analysis instead, because they are the only path back to
it.

---

## What this classification cannot settle

Three things, stated so they are not silently assumed:

1. **The exam-row builders.** They look one-off but sit next to a live
   pipeline. The evaluation session should confirm.
2. ~~**`release_to_exam_rows.py`.**~~ **Settled:** it is a manual step, not a
   missing edge. It runs once when the release changes —
   `eml1_release.jsonl` → exam rows (14:44) → and then
   `preflight_exam_rows.py`, which `run_exam_cell.py` does call, produces
   `release535_playable.jsonl` (14:46). A reviewer using the shipped release
   never runs it, because the playable file is already there.
3. **Whether any historical script is the only record of how a released
   artifact was made.** Archiving is safe for this; deleting is not, which is
   why nothing here proposes deleting.
