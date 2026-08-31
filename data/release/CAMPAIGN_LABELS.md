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
