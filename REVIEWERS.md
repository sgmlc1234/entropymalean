# Reproducing this work

Everything below runs from a clean checkout. The three things a reviewer is
most likely to want are ordered by cost, cheapest first:

| Want to | Cost | Section |
|---|---|---|
| Check that our reported proofs are real | minutes, no GPU, no API key | [1](#1-verify-the-release-without-running-anything) |
| Recompute the paper's tables from the shipped episodes | under a minute, Python only | [3.1](#31-start-here-the-papers-tables-from-the-shipped-episodes-no-model-no-lean) |
| Re-run the evaluation | hours to days, GPU or API keys | [3](#3-evaluation-pipeline) |
| Re-run problem generation | days, API keys | [4](#4-generation-pipeline) |

**Section 1 needs nothing but Lean.** It settles the claim that matters most —
that every released row is a theorem with a proof that compiles — without
reproducing anything. We suggest starting there.

---

## 0. Setup

### 0.1 Python

```bash
git clone <repository> && cd entropy-malean
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python ≥ 3.10.

### 0.2 Lean and Mathlib — do this before anything in §1–§3

Everything that touches Lean in this repository assumes a **built** Mathlib at
one pinned revision. The toolchain alone is not enough: with `lake` on PATH and
no build, every check fails with

```
error: unknown module prefix 'Mathlib'
```

which the certifier, the evaluation environment and several tests report as a
*failing proof*, not as a missing build. If you see that line, nothing below
it is a result.

The pins are not suggestions — every released row's certificate names them,
and a different Mathlib fails rows that are correct under ours:

```
toolchain          leanprover/lean4:v4.30.0-rc2      (lean-toolchain)
mathlib revision   0fb2045029635862ffb234635a111c80a55e2a87   (lake-manifest.json)
```

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh    # if you have no elan
lake exe cache get                                        # prebuilt Mathlib oleans, a few GB
lake build                                                # links the workspace; minutes, not hours
python3 scripts/check_setup.py                            # says whether the above worked
```

`check_setup.py` checks that `lake` resolves to the pinned toolchain, that
`lake-manifest.json` still names the certified Mathlib revision, and that
`Mathlib.olean` exists; each failed line prints its fix. Do not run
`lake update` — it moves the manifest off the certified revision. Run
`python3 scripts/check_setup.py --cells` before an evaluation cell to see
which credential that cell reads and whether `.env` provides it.

Every comparator workspace we ship records its own pins, Mathlib's revision
included — `data/release/comparator/<row>/lake-manifest.json`.

### 0.3 What needs an API key, and what does not

| Stage | Needs |
|---|---|
| Verify the release (§1) | nothing but Lean |
| Kernel replay (§2) | Linux, for `landrun` |
| Evaluation, local provers (§3) | a GPU that fits a 7–8B model at Q8 |
| Evaluation, hosted models (§3) | OpenRouter / Mistral / FriendliAI keys |
| Generation (§4) | an OpenRouter key |

Copy `.env.example` to `.env` if you are running §3 or §4. Sections 1 and 2 read
no keys.

---

## 1. Verify the release without running anything

The release is `data/release/eml1_release.jsonl` — 535 rows, one JSON object per
line. Each carries the natural-language statement, the Lean statement, the Lean
proof, the header it compiles under, its certificate, its parent lineage, and
the result of every check that was run on it.

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('data/release/eml1_release.jsonl') if l.strip()]
print(len(rows), 'rows')
print(sorted(rows[0]))
"
```

### 1.1 Recompile a row yourself

```bash
python3 - <<'PY'
import json, pathlib, subprocess
row = json.loads(open('data/release/eml1_release.jsonl').readline())
pathlib.Path('/tmp/Check.lean').write_text(row['lean_code'])
print(row['problem_id'])
PY
lake env lean /tmp/Check.lean          # silence means it compiled
```

`lean_code` is a complete file — it carries its own imports. Do not prepend
`lean_header` to it.

### 1.2 What each check means

Every row has a `checks` block. Each entry names the faculty that settled it,
whether it ran, and what it found — a check that could not run says so rather
than reporting a negative verdict.

```
statement_type_check   lean    the statement elaborates
proof_accepted         lean    the proof closes, no `sorry`
axiom_audit            lean    closure ⊆ {propext, Quot.sound, Classical.choice}
vacuity                lean    the hypotheses are jointly satisfiable
dead_hypotheses        lean    every hypothesis is load-bearing
redundancy             lean    the child does not already follow from a parent
corpus_dedup           hash    not a restatement of another row
comparator             lean    an independent kernel replayed the exported term
goal_roundtrip         model   the prose describes the goal Lean elaborated
```

`redundancy` applies to 518 of the 535: the other 17 are silent mutations, which
restate a parent on purpose and are gated on sameness instead.

### 1.3 The rejected rows are shipped too

`data/release/eml1_rejected.jsonl` holds the 691 candidates the validation
layer refused, each with the reason. Two more are in
`data/release/eml1_preflight_excluded.jsonl`: they are sound and certified,
but their statements do not parse under the evaluation preamble, so they are
excluded from the panel rather than from the corpus. A corpus that shows only its survivors cannot be
audited, and the failure modes are as informative as the successes.

```bash
python3 -c "
import json, collections
c = collections.Counter(json.loads(l)['admission']['why_not']
                        for l in open('data/release/eml1_rejected.jsonl') if l.strip())
for k, v in c.most_common(): print(f'{v:4d}  {k}')
"
```

### 1.4 Read it as a document

`docs/eml1_release_report.pdf` renders all 535 rows as cards — statement, Lean,
certificate, and every judge's reasoning including the passes that disagreed.
Split by benchmark: `docs/eml1_release_report_proofnet.pdf` (287 rows),
`docs/eml1_release_report_minif2f.pdf` (248 rows).

Rebuild from the release with:

```bash
python3 -m scripts.release.build_release_report --split-by-benchmark
cd docs && lualatex eml1_release_report.tex          # twice, for the contents
```

`lualatex`, not `pdflatex` — the cards set Lean's Unicode.

---

## 2. Kernel replay and reproducibility

Section 1 uses our Lean installation. This section is about not needing it.

### 2.1 Kernel replay (Linux only)

`comparator` rebuilds the statement from the trusted row alone, compares it to
the submitted proof through `lean4export` rather than by trusting the file,
re-audits the axioms, and replays the term through the kernel. Its sandbox is
Landlock via `landrun`, so it does not run on macOS.

```bash
bash scripts/faithfulness/kernel/comparator_setup.sh                 # builds comparator, landrun, lean4export
python3 scripts/faithfulness/kernel/prepare_comparator_batch.py      # one workspace per row
COMPARATOR_MATHLIB=$HOME/mathlib4 bash scripts/faithfulness/kernel/run_comparator_batch.sh
```

Workspaces are portable — prepare them anywhere, run them on a Linux host.

### 2.2 Two-platform reproducibility

`reproducible` is the top rung and the one released benchmarks most often skip.
It asks whether the result holds for whoever runs it, not just for us.

`lean4export` is deterministic across operating systems, so the exported term's
SHA-256 is a compact certificate of the whole environment: match our pins,
regenerate the export, compare one hash.

```bash
python3 scripts/faithfulness/kernel/check_export_reproducible.py \
  --rows data/release/reproducible_rows.jsonl \
  --output /tmp/repro.json
```

All 535 rows agree between `macos-aarch64` and `linux-x86_64`, under
`lean4:v4.30.0-rc2`, Mathlib `0fb2045`, and `lean4export` `12581a6b`. Exports
run 0.1–77 MB; every digest is distinct. Each row's certificate records
`platforms_verified` and `export_digest`.

Two details that cost us runs, in case you adapt this:

- **Export the same way on both platforms.** Elaborate `Solution.lean` under an
  explicit `LEAN_PATH` and export from that module. Running it inside a Lake
  project pulls a different module set into the environment and changes the
  term's dependency closure — 34 MB against 20 MB for the same theorem, which
  looks exactly like platform nondeterminism.
- **Pin the exporter.** A different `lean4export` revision can serialise the
  same term differently. That, too, reads as a platform disagreement.

---

## 3. Evaluation pipeline

The experiment is a difference between two arms: a control of 100 certified
seeds (50 per benchmark) and a treatment of the released corpus. It is reported
per benchmark and never pooled — the control is balanced 50/50, the treatment is
287/248, and the benchmark gap exceeds the effect being measured.

### 3.1 Start here: the paper's tables from the shipped episodes, no model, no Lean

Every episode behind the 26 reported cells ships in
`data/evaluation/exam_evidence/` — one gzipped file per model and arm, 24,765
episodes, with `MANIFEST.json` recording each file's SHA-256 and the Pass@3 it
yields. Three commands recompute the paper from those bytes, in under a minute,
with nothing installed beyond Python:

```bash
python3 scripts/evaluate/bundle_exam_evidence.py --from-bundle        # digests + Pass@3 vs MANIFEST
python3 scripts/evaluate/finalize_panel_numbers.py --latex drops      # Table 1: drops with bootstrap CIs
python3 scripts/evaluate/lineage_gap_from_exam.py --contrast goedel,bfs  # Table 5: lineage proof gap
```

The first checks the bundle against its own manifest and prints the control /
treatment grid. The second prints the rows of the paper's Table 1 verbatim (the
one cell scoring zero on both arms prints a degenerate `[0.0,0.0]`; the paper
substitutes a Wilson interval there and marks it †). The third prints the
lineage proof gap per model and benchmark with Wilson intervals, the rescue
counts, and the Goedel-minus-BFS contrast with its Newcombe interval. The
scripts read the raw cell directories when those exist and the bundle
otherwise, so they run identically here and in the tree the bundle was cut
from.

**Then verify the proofs.** Episode outcomes are sampled at temperature 1.0
and there is no seed argument, so Pass@3 will not reproduce exactly.
**Verification is deterministic.** Every episode we scored as solved stored
the Lean it closed with, and that can be re-elaborated against a built Mathlib
(§0.2):

```bash
python3 scripts/evaluate/verify_solved_episodes.py \
  data/evaluation/exam_evidence/<model>_<arm>.jsonl.gz
```

This settles whether our reported successes are real without reproducing a
single episode. Raw `episodes_*.jsonl` files are accepted as well.

### 3.2 Serving the local provers

```bash
./scripts/provers.sh up bfs        # llama.cpp on :8080, ctx 4096
./scripts/provers.sh up goedel     # llama.cpp or LM Studio on :8081, ctx 16384
./scripts/provers.sh status
```

The script expects the two GGUF checkpoints — BFS-Prover-V2-7B Q8_0 and
Goedel-Prover-V2-8B Q8_0 — under `$HOME/.cache/lm-studio/models` in LM
Studio's download layout; edit `MODELS` at the top of the script to point
elsewhere. `LM_STUDIO_BASE_URL` optionally names a remote LM Studio host that
can serve Goedel (not BFS, see below).

> **BFS must be served by llama.cpp.** Its search ranks the frontier by
> cumulative token log-probability, and LM Studio returns `logprobs: null` **on
> the completions endpoint** (`/v1/completions`). Nothing errors — the search
> silently degenerates into an unranked one, and the drop you measure is not the
> drop we measured.
>
> LM Studio *does* return logprobs on `/v1/chat/completions`, so that looks like
> a way around it. It is worse: the chat template wraps the prompt, and a
> completion-style prover then sees different input entirely. Use llama.cpp.

### 3.3 Running a cell

```bash
set -a; source .env; set +a

python3 scripts/evaluate/run_panel.py --plan                       # what remains
python3 scripts/evaluate/run_panel.py --model goedel --arm control
python3 scripts/evaluate/run_panel.py --model goedel --arm treatment
```

Per-prover differences live in `config/exam_cells.json` under `budgets`, so the
command is the same for every model. `run_panel.py` calls `run_exam_cell.py`,
which calls `run_seed_exam.py`; the budget-parity gate refuses a cell whose
budget differs from the control it will be compared against.

Resuming is the normal case. `--resume` keys on `(seed, attempt)`, so nothing
already measured is replayed.

> **Run the control to completion before starting the treatment.** The parity
> gate skips its check when the control is empty. Launching `--arm both` starts
> the treatment first and the gate never fires. We lost 546 Leanstral episodes
> that way.

The panel is 13 models in three groups:

```
lean_provers    bfs, goedel, pythagoras, leanstral
reasoning_slms  muse, qwen3_14b, nemotron_nano_9b, nemotron, qwen36, gptoss
frontier_llms   luna, muse_spark, gemini_flash
```

`groups` is the panel. `controls`, `treatments` and `budgets` are a registry and
keep entries for models that were withdrawn — reading those instead of `groups`
pulls in cells the paper does not report.

**A reduced path with no API keys:** serve BFS and Goedel locally and run those
two cells. They reproduce both headline results — the ProofNet drop and the
lineage proof gap — and need no hosted model. The remaining nine cells split
across OpenRouter, Mistral API, and a dedicated FriendliAI endpoint.

### 3.4 Turning episodes into the paper's numbers

```bash
python3 scripts/evaluate/finalize_panel_numbers.py --latex drops         # Table 2
python3 scripts/evaluate/lineage_gap_from_exam.py --contrast goedel,bfs  # Table 3
```

`analyze_exam_arms.py` is a different tool — paired Wilcoxon/McNemar between
arms — and does not produce the table numbers.

Both read the model order from `groups` in `config/exam_cells.json`, so a cell
added or withdrawn does not leave the table behind.

**Reading the tables.** All 26 cells ran — 13 models across two arms, 300
control episodes and 1,605 treatment episodes each, 24,765 in total, every one
carrying a Lean verdict. Nothing renders as `tbd`.

`data/evaluation/exam_evidence/MANIFEST.json` records, for every cell, the
SHA-256 of the shipped episodes and the Pass@3 recomputed from them, so the
table numbers can be checked against the logs rather than against our
transcription of them; `bundle_exam_evidence.py --from-bundle` (§3.1) does that
check.

Four models were withdrawn and are not in the tables: Grok-4.6 (control only),
Qwen3.8-27B, Nemotron-3-Ultra and Inkling. Their cells are not shipped. Nor
are two contaminated Luna cells whose prompts could read the ground-truth
proofs and which score 100/100 — they exist in our working tree as the record
of a defect, and `groups` in the config is what keeps every script from
reading them.

### 3.5 Where the results live

```
data/evaluation/exam_evidence/<model>_<arm>.jsonl.gz      episodes, one file per reported cell
data/evaluation/exam_evidence/MANIFEST.json               SHA-256 and Pass@3 per cell
data/evaluation/exam/seeds_all100.jsonl                   the 100 control rows as played
data/evaluation/exam/release537_playable.jsonl            the 535 released rows as played
config/exam_cells.json                                    panel (`groups`) and cell registry
```

A cell you run yourself lands under `data/evaluation/exam/<cell>/` as
`episodes_<model>_closed_book.jsonl` plus a `summary_*.json`; the config's
`controls` and `treatments` maps name those directories, and every scoring
script reads a directory when it exists and the shipped bundle when it does
not. Cell names in the registry are historical and not guessable — BFS's
treatment cell is `release309_bfs`, its control `control100_matched` — so read
the maps rather than inferring a name.

The control corpus is `seeds_all100.jsonl` (50 ProofNet-Verified + 50
miniF2F). Report per benchmark: the two differ in difficulty by more than any
effect being measured, and a pooled Pass@3 confounds the two.

### 3.6 Two ways to miscount

- **An empty generation is not a failed proof.** A `generator_empty` episode
  means the model returned nothing, which aggregates as "the model failed to
  prove it". Strip them with `drop_unmeasured_episodes.py` and replay with
  `--resume`.
- **Report per benchmark.** Pooling the arms mixes an effect with a difficulty
  difference that is larger than it.

### 3.7 Sizing

```
LEAN_REPL_POOL_SIZE   2–4 per cell. Each REPL holds a Mathlib environment
                      resident; 30 of them swapped a 48 GB machine.
episode_concurrency   bounded by provider rate limits, not memory — episodes
                      are coroutines. 4.5× over serial (82 s → 18.3 s/episode).
verification cost     0.12 s warm, 4.6 s cold per REPL — 0.2% of episode time.
volume                300 control episodes, 1,605 treatment episodes per model.
```

Lean is not the bottleneck. The prover is.

---

## 4. Generation pipeline

Generation runs one seed group at a time. A group is five seeds; each generation
opens a fixed number of slots; a slot picks parents and an operator, proposes
one candidate, and either lands it in the ledger or quarantines it with a
status.

### 4.1 Preparing input

A worked example is in `examples/seeds/` — five real seeds, cut to the columns
that carry meaning, with a README that says which three are required and what
the rest do. It runs as-is:

```bash
python3 scripts/generate/run_pool_generation.py \
  --input examples/seeds/proofnet_example_group.csv \
  --output /tmp/example.jsonl --summary-output /tmp/example_summary.json \
  --pool-size 5 --survivor-count 1 --crossover-count 2 \
  --max-generations 2 --max-retries 1 --max-parallel 2
```

Two things there are load-bearing and easy to miss: a group must have at least
`--pool-size` rows or the loader refuses it, and the benchmark is inferred from
the **filename**, so a file not named `proofnet_*` or `minif2f_*` produces rows
with no benchmark.

Environment variables are documented in `.env.example`, grouped by which stage
reads them. Note that `GENERATOR_MODEL` and `GENERATION_MODEL` are different
names read by different layers; setting only one is a quiet no-op.

### 4.2 One group, in full

```bash
set -a; source .env; set +a

GENERATION_PROVIDER=codex_cli LEAN_VERIFIER=repl \
python3 scripts/generate/run_pool_generation.py \
  --input examples/seeds/proofnet_example_group.csv \   # or your own certified seed group
  --output /tmp/p01.jsonl \
  --summary-output /tmp/p01_summary.json \
  --generation-model gpt-5.6-luna \
  --pool-size 5 --survivor-count 1 --crossover-count 2 \
  --max-generations 10 --max-retries 1 --max-parallel 2
```

### 4.3 A campaign

`scripts/generate/run_proofnet_p.sh` drives ten groups back to back and is the shortest
readable example of a full campaign. It skips any group that already has output,
so re-running after an interruption resumes rather than restarting.

```bash
bash scripts/generate/run_proofnet_p.sh
```

Two guards in the driver are worth knowing about, because both were added after
losing a run:

- **Quota exhaustion halts the queue.** An exhausted provider is not a slot
  failure and not retryable — every remaining call returns the same refusal. An
  early run wrote 207 "generation failures" that were all the provider saying it
  was out of credit.
- **A group that certifies zero rows halts the queue.** That is a broken run
  whatever the cause, and the next group will break the same way.

### 4.4 Seeds

The two seed sets are in `data/benchmarks/*/seeds_50_levels.csv`, 50 rows each,
11 columns, documented in `data/benchmarks/README.md`. Every seed carries a
proof that compiles here — checked, not inherited from the file that claimed it.

The two differ in reachable ceiling, and the `certificate` column says which:
ProofNet ships ground-truth proofs and reaches `reproducible`; miniF2F withholds
its proofs to keep the test set uncontaminated, so the only proof that exists is
the one Generation 0 wrote, and those rows stop at `proof_checked`.

Per-campaign seed groups are under `data/certified/<campaign>/seeds/`.

### 4.5 From certified rows to a release

A certified row is not a release candidate. The admission path:

```bash
# 1. two independent judge passes, recorded not applied
GENERATION_PROVIDER=codex_cli PROBLEM_JUDGE=1 PROBLEM_JUDGE_MODEL=gpt-5.6-luna \
python3 -m scripts.faithfulness.reader.rejudge_corpus --candidates <candidates>.json \
        --output data/release/rejudged.json --concurrency 3
#    then again, to a second output

# 2. prose-vs-goal round trip
python3 scripts/faithfulness/reader/run_goal_roundtrip_batch.py --only-missing

# 3. redundancy scan, prover on
python3 scripts/faithfulness/lean/scan_release_redundancy.py --only-missing

# 4. merge everything into the release
python3 scripts/release/export_release.py
```

Admission requires **`strong` from both judge passes and `keep` from both**. Run
twice over the same rows in the same order, the judge agreed with itself on
keep-or-reject 80% of the time — 984 of 1,233 rows in the accumulated verdict
files, and 77–82% across four separate measurements. That is the ceiling on what
a single verdict is worth, and the reason there are two. Figures quoted for one
campaign's subset differ slightly from this corpus-wide one; both are computed
by the same rule.

Reading only the quality and not the vote once admitted six rows the judge had
voted to reject.

Both passes ship, for all nine campaigns: `data/release/*_rejudged_1.json` and
`_2.json`, with `rejudged.json` / `rejudged_run2.json` as the first campaign's
unprefixed pair. They are the record of what the judge saw, not the route to the
release's numbers. Two of the nine campaigns re-judged rows an earlier campaign
had already seen, so an id appears in more than one file with different verdicts
and only the later one counts; a reader who merges them without that order gets
272/247/102 against the release's 276/249/102.

Nothing needs that merge. §1.3 counts `admission.why_not` in
`eml1_rejected.jsonl` — one file, no order to know, and the number the release
was actually built on. Joined that way, 626 of 627 judge-category rows reconcile
exactly. The one that does not is `Munkres_exercise_18_8a__mh__357e4f99`: two
different theorems from two campaigns hash to the same id, refused in one and
kept in the other, and a dict keyed on the id holds only one of them.

`export_release.py` reads only the campaigns in its `SOURCES` map, and
`rejudge_corpus.py` only the ones in `DEFAULT_INDEX`. A campaign missing from
either contributes nothing, silently: the judge is handed a child with no prose
and no parents and answers anyway. Eighty rows came back `ran: false` that way
and the summary line read them as eighty rejections.

---

## 5. The website

An HTML working page carries the same numbers as the paper, plus the full
release with every judge's reasoning, browsable.

```bash
python3 -m http.server 8791          # from the repository root
```

Then open `http://localhost:8791/site/workspace.html`. The page loads its
data and traces by relative path, so the server has to be rooted at the
repository, not inside `site/`.

```
#/results          corpus integrity, drops, lineage gap
#/release          all 535 rows, with the reasoning behind each judgement
#/certification    the ladder, and what each rung is worth
#/seeds            the seed sets and how they were chosen
```

Regenerate the release page from the current release with:

```bash
python3 -m scripts.release.publish_release_page
```

---

## 6. What we would check first, if we were reviewing this

0. **Run `python3 scripts/check_setup.py`** (§0.2). Ten seconds, and it says
   whether anything Lean-related below can be trusted on this machine.
1. **Recompute Table 1 and Table 5 from the bundle** (§3.1). Under a minute,
   Python only, and it settles whether the numbers in the paper are the numbers
   in the logs.
2. **Recompile a row** (§1.1). One command, and it settles whether the artifact
   is what it says it is.
3. **Run `verify_solved_episodes.py`** (§3.1). Deterministic, no GPU, and it
   settles whether the successes we report are real.
4. **Check a rejected row's reason** (§1.3). The gate is only meaningful if what
   it rejected was worth rejecting.
5. **Compare an export digest** (§2.2), if you have a second machine. That is
   the claim we are least able to make on your behalf.

Reproducing the full evaluation is the most expensive check and the least
conclusive one, because sampling is not seeded. We would do it last.
