# Preparing input for the generation pipeline

`proofnet_example_group.csv` is a working seed group — five real ProofNet-Verified
seeds, cut down to the columns that carry meaning. Run the pipeline against it
directly:

```bash
set -a; source .env; set +a

python3 scripts/generate/run_pool_generation.py \
  --input examples/seeds/proofnet_example_group.csv \
  --output /tmp/example.jsonl \
  --summary-output /tmp/example_summary.json \
  --pool-size 5 --survivor-count 1 --crossover-count 2 \
  --max-generations 2 --max-retries 1 --max-parallel 2
```

Two generations rather than ten, because this is for seeing the shape of the
output rather than for producing a corpus.

## What a seed group is

**Five rows.** The pool size is the group: each generation, slots pick parents
from the pool, and the loader refuses a file with fewer rows than `--pool-size`
rather than running a degenerate pool.

Seeds are what the corpus is bred *from*, so the group is also the unit of
topical composition. Two seeds of the same topic in one group can cross with
each other; two seeds in different groups never meet.

## The columns

Only three are required. The rest are read when present and left alone when not.

| Column | Required | What it is |
|---|---|---|
| `id` | **yes** | The identifier. Becomes the root of every descendant's id. |
| `statement` | **yes** | The problem in prose. May be a placeholder — see below. |
| `formal_statement` | **yes** | The Lean theorem statement, no proof. |
| `lean_header` | | Imports and `open`s the statement needs. |
| `lean_code` | | The complete file: header, statement, and a proof that compiles. |
| `difficulty_label` | | `easy` / `hard`. Read by the planner when choosing operators. |
| `problem_style` | | `theorem_proof` for these. Selects the family template. |
| `formal_status` | | `certified` once the row's own proof has been replayed here. |
| `generation` | | `0` for a seed. |
| `parent_ids`, `ancestor_ids` | | `[]` for a seed. Populated for generated rows. |

Anything else in the file is carried through into the row's metadata untouched,
so a campaign export with thirty columns loads exactly as this one does.

### `lean_code` decides what the group can do

A seed without a proof can be mutated but is a weak crossover parent: the
operators that need to see how a parent argues have nothing to read. All five
rows here carry one.

This is also where the two benchmarks differ, and the difference is not
incidental. ProofNet ships ground-truth proofs, so its seeds arrive with
`lean_code` and reach `reproducible`. miniF2F withholds its proofs to keep the
test set uncontaminated, so the only proof that exists is the one Generation 0
writes here, and those rows stop at `proof_checked`.

### `statement` may be a placeholder

These rows carry `Prove the theorem Dummit_Foote_exercise_3_1_22b.` — not prose,
just a pointer. The loader replaces it at read time with the real informal
statement from `data/benchmarks/*/`, and says so:

```
[seeds] informal statement recovered for 5/5 seed(s)
```

If your seeds are not from a benchmark under `data/benchmarks/`, put the real
prose in `statement`. Nothing recovers it for you, and the judge that decides
whether the prose describes the theorem will be reading whatever is there.

## Checking a group before you run it

Loading is cheap and catches the mistakes that otherwise surface an hour in:

```bash
python3 -c "
from pathlib import Path
from src.orchestration.pool_generation import load_seed_inputs
rows = load_seed_inputs(Path('examples/seeds/proofnet_example_group.csv'), pool_size=5)
print(f'{len(rows)} rows')
print('with proofs:', sum(1 for r in rows if str(r.metadata.get('lean_code') or '').strip()))
print('benchmark:', rows[0].metadata.get('benchmark'))
"
```

Expected here: `5 rows`, `with proofs: 5`, `benchmark: proofnet`.

`benchmark` is inferred from the **filename**, not from a column — a file named
`proofnet_*.csv` is read as ProofNet. Name yours accordingly, or the row will
carry no benchmark and will be pooled with the wrong arm at evaluation time.

## Where the real groups are

Under `data/certified/<campaign>/seeds/`. The ten ProofNet groups behind the
released corpus are in `data/certified/run-e/seeds/`. They were planned
by `scripts/analysis/plan_proofnet_groups.py` from measured per-seed admission
rate and written out by `scripts/analysis/write_seed_groups.py`.

The seed sets themselves — 50 rows per benchmark, 11 columns, every one carrying
a proof that compiles here — are in `data/benchmarks/`, documented in the README
beside them.
