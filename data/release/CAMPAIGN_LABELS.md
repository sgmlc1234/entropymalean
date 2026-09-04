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

## The two judge passes

Admission needs the same judgement from two independent passes, and both ship:
`rejudged_1.json` and `rejudged_2.json`, one record per row per pass, 1247
each. Every record carries the `campaign` it belongs to, so the run is
readable from the data.

The judging happened in batches, one per run, and the rows whose hypotheses
were pruned were judged again in their pruned form -- a pruned row is no
longer the row that was judged. Those re-judgements are the last word here:
the merge applied them after every batch, which is the order the exporter used
to enforce with a separate overlay. Seven of them changed a verdict.

Two rows in `run-e` carry two records in one pass with opposite verdicts. The
later one is the re-judgement and is the one kept; both rows were rejected,
and neither is in the released corpus. `export_release.py` prints a note when
it meets a row in that state.
