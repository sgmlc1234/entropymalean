# benchmarks

One rule: **the top level of a benchmark directory holds only what you would
open.** Everything a pipeline stage wrote on the way there goes in `raw/`.

```
proofnet_verified/
  seeds_50_levels.csv     the seed set, 11 columns — the file to look at
  raw/                    exam rows, enriched rows, reports, wide analysis CSV
minif2f_v2/
  seeds_50_levels.csv     same 11 columns, same meaning
  raw/
external/                 benchmarks we do not run experiments on
```

Both seed sets are 50 rows in the same schema, and every row carries a proof
that compiles here — checked, not inherited from the file that claimed it.

They differ in one column, and the difference is real rather than incidental.
ProofNet rows reach `reproducible`: they ship ground-truth proofs, so the proof
could be replayed through an independent kernel and re-exported byte-identically
on a second platform. miniF2F withholds its proofs to keep the test set
uncontaminated, so the only proof that exists is the one Gen-0 wrote, and those
rows stop at `proof_checked`. Same ladder, different reachable ceiling; the
`certificate` column says which, so the two never blur when the sets are
concatenated.

`seeds_50_levels.csv` is the playable view: `id topic difficulty ready`
to pick a level, `goal lean_header lean_goal` to read it, `tools hint_outline
hint_first_step` when stuck, `solution` to check against. Regenerate it with
`scripts/exam_rows_to_game_csv.py`.

`raw/` is the record of truth and is where every script reads from and writes
to. The wide 40-column analysis CSV lives there too — it exists for ablation
tables, not for reading.

`external/` holds the original miniF2F, ProofNet, and PutnamBench dumps. Nothing
in `src/` or `scripts/` references them; PutnamBench in particular is out of the
EML-1 campaign entirely. They are kept only so a provenance question can be
answered without a re-download.
