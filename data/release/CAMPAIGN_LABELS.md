# Run labels

Each released row records the campaign that produced it. The labels are
run identifiers, in the order the runs happened; they are letters rather
than descriptive names because a run's name is not a property of the rows
it produced, and the original names carried the development sequence.

| Label | Benchmark | Generations | Groups |
|---|---|---|---|
| `run-a` | both | 5 | 20 |
| `run-b` | ProofNet | 10 | 10 |
| `run-c` | miniF2F | 10 | 5 |
| `run-d` | miniF2F | 10 | 10 |
| `run-e` | ProofNet | 10 | 10 |
| `ablation/*` | both | — | operator ablations |

A row's `campaign` field names one of these. The generation index within a
run is in `generation`, and the lineage is in `parents` and `ancestor_ids`.

## The rejudge files

Admission needs the same judgement from two independent passes, and both
passes ship. The judging happened per run, so the files are per run too; the
name of each is the name the batch had when it was written, and the table says
which run it covers. Every record inside carries its `campaign`, so the run is
readable from the data as well as from the filename.

| File (`_1` = first pass, `_2` = second) | Run | Records |
|---|---|---|
| `rejudged.json` / `rejudged_run2.json` | all of them | 1235 |
| `gen10_*`, `gen10b_*`, `gen10c_*` | `run-b` | 80, 46, 122 |
| `minif2f_h_*` | `run-c` | 118 |
| `minif2f_k_*` | `run-d` | 239 |
| `proofnet_p_*` | `run-e` | 238 |
| `new_*` | `ablation/mutation` | 63 |
| `pruned_*` | rows whose hypotheses were pruned | 14 |

`pruned_*` is an overlay, not a batch: a row that lost a dead hypothesis is no
longer the row that was judged, so it was judged again in its pruned form and
those verdicts replace the earlier ones outright. `export_release.py` applies
it after the others for that reason.

Two rows appear twice inside one pass with opposite verdicts, both in
`run-e`; both were rejected, and neither is in the released corpus. The loader
takes the later record, which is the re-judgement.

