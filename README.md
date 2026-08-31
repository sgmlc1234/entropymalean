# EntropyMaLean

A Lean-certified corpus of generated theorem-proving problems, and the pipeline
that made it. Every released row is a theorem with a proof that compiles, and
every row that did not make it is shipped too, with the reason.

**535 rows. All 535 reach `reproducible`** — the top rung of the certificate
ladder: an independent kernel accepted an exported proof term it did not
produce, and a second platform regenerated that export byte for byte.

| | |
|---|---|
| Released rows | 535 — 287 from ProofNet, 248 from miniF2F |
| Held back | 693, each with its reason on file |
| Certificate | `reproducible`, 535/535 |
| Toolchain | `leanprover/lean4:v4.30.0-rc2`, Mathlib `0fb2045` |

**[Browse the corpus and the results →](https://claude.ai/code/artifact/4c0b6395-74e7-4785-acbb-36868ad108a5)**
Every row with its Lean, its certificate, and the reasoning behind every
judgement it received — including the passes that disagreed.

One thing the hosted page cannot show: what each model did on each problem,
attempt by attempt — Lean's verdict on every try, and the proof that closed the
goal where one did. Those traces are 24 MB across 750 files, so they are served
beside the page rather than bundled into it. Serve this repository and they
appear under every row:

```bash
python3 -m http.server 8000     # then open site/workspace.html
```

**Reproducing any of this: [`REVIEWERS.md`](REVIEWERS.md).** It is ordered by
cost, and the first section needs nothing but Lean.

---

## What the pipeline does

A seed is an existing benchmark problem with a proof that compiles here. An
operator mutates one seed or crosses two. The child is kept only if it survives
every check that applies to it — and the checks are the point, not the
generator.

```
scripts/generate/       one seed group through N generations
scripts/faithfulness/   the checks; see below
scripts/release/        applying the gates, writing the corpus
scripts/evaluate/       the panel: 11 models, two arms
scripts/analysis/       tools that produced a number in the paper
scripts/archive/        superseded, kept because released rows came out of them
```

`scripts/INVENTORY.md` says what every script is for and why it sits where it
does.

## The checks

A generated theorem can be sound Lean and still be worthless. It can be true
because nothing satisfies its hypotheses; it can restate something the corpus
already has; it can follow from a parent outright; and it can prove something
other than the problem its prose describes. None of these is visible to a type
checker, and no two are the same kind of question — so each is put to the
faculty that can settle it.

| Directory | Question | Settled by |
|---|---|---|
| `faithfulness/lean/` | Vacuity, dead hypotheses, redundancy against a parent | Lean, as a compilation whose outcome is the verdict |
| `faithfulness/identity/` | Is this the same theorem as another row? | An alpha-normal hash, against every earlier run |
| `faithfulness/reader/` | Does the child demand new reasoning? Does the prose describe the goal? | A model, asked twice; admission needs both |
| `faithfulness/kernel/` | Does it hold for someone who is not us? | A kernel that did not write the proof, on a second platform |

The last one is the rung most released benchmarks skip. We measured what its
absence costs: **a third of ProofNet-Verified does not compile under our pin,
despite shipping as verified.** Their checks were not wrong; they were local,
and nothing in the artifact said what local meant.

## What is in here

```
data/release/eml_v1_release.jsonl     the corpus, one JSON object per line
data/release/eml_v1_rejected.jsonl    everything that did not make it, with reasons
data/release/CAMPAIGN_LABELS.md       what `run-a` … `run-e` mean
data/benchmarks/                      the two 50-row seed sets, and how they were chosen
examples/seeds/                       a five-seed group, the input a campaign takes
docs/eml_v1_release_report*.pdf        every row as a card, 1,177 pages
site/                                  the browsable page and its data
config/exam_cells.json                 the evaluation panel, one entry per cell
```

Two things are deliberately absent. The raw campaign output is 1.1 GB and is not
in version control; what survived it is the release, and what did not is the
rejected file. The comparator workspaces are one per row and regenerate in
minutes — `data/release/comparator/` holds five as examples, and
`scripts/faithfulness/kernel/prepare_comparator_batch.py` builds the rest.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
lake exe cache get && lake build     # Mathlib at the pinned revision
```

`.env.example` lists what needs a key and what does not. Verifying the corpus
needs none.

## One command, to see whether any of this is true

```bash
python3 - <<'PY'
import json, pathlib
row = json.loads(open('data/release/eml_v1_release.jsonl').readline())
pathlib.Path('/tmp/Check.lean').write_text(row['lean_code'])
print(row['problem_id'])
PY
lake env lean /tmp/Check.lean          # silence means it compiled
```
