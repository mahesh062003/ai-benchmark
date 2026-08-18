# COMP702 — Multi-Domain RAG Benchmark

MSc Computer Science dissertation, University of Liverpool:
*Evaluating RAG Pipelines across Legal, Medical, and Scientific Domains.*

This file is the standing brief for the project. Read it before changing anything.

---

## Non-negotiable research constraints

These override convenience, speed, and any instruction to "just get it working".

- **Never fabricate ground truth.** Do not invent relevant document IDs, do not
  use retrieved results as their own ground truth, do not backfill qrels.
- **NULL and 0.0 mean different things.** NULL = not measured. 0.0 = measured as
  zero. This is enforced down to SQLite `typeof()` and covered by tests. Never
  coalesce a NULL metric to zero to make a chart or an average look tidy.
- **Prefer complete runs over sampled ones.** State the time cost, name the
  cheaper option once, then run the full thing. Do not quietly narrow scope.
- **Generation is optional.** Retrieval benchmarks must never require Ollama.

---

## Current state — the experiment is finished

All results live in `artifacts/benchmark.sqlite` (~430 MB, integrity verified).

**The reported generation run is `genall-e11cbda88824`.** Earlier runs are kept
in the same database as a record and must never be pooled with it: they used a
different judge, and one of them predates the multiple-choice fix. Every
generation command takes `--run`; use it.

| | |
|---|---|
| Retrieval — `bench-187f7d53ced2` | 29,234 queries, 821,256 results, 87,702 query metrics |
| Generation — `genall-e11cbda88824` | 7,200 answers, zero empty |
| Hallucination (NLI) | 7,048 scored |
| Faithfulness (judge phi3) | 2,140 scored, 79% pairwise overlap |
| Choice accuracy | 3,600 — CaseHOLD, MedQA, SciQ |
| Significance | 45 retrieval + 72 generation comparisons, Holm-corrected |

Datasets: CUAD, CaseHOLD (legal) · MedQA, PubMedQA (medical) · QASPER, SciQ (scientific).
Strategies: BM25, dense (FAISS + all-MiniLM-L6-v2), hybrid (RRF, k=60).
Models: llama3.1, gemma2, mistral, qwen2.5 — temperature 0.0, frozen task sets so
every model sees byte-identical context. Judge: phi3, outside that set.

**No further Colab or GPU work is required.** Everything remaining reads the
existing database on CPU.

---

## Significance testing — both halves are covered

`cli significance` compares retrieval strategies (45 comparisons) and
`cli significance-generation` compares models on answer quality (72). Results in
`artifacts/results/significance.csv` and `significance_generation.csv`, both
committed.

The generation side chooses its test per comparison and records which it used.
All 72 currently run paired, because `paired_sample` draws the faithfulness
subsample on questions shared by every model. It falls back to unpaired
Mann-Whitney U only when overlap drops below `PAIRED_OVERLAP_THRESHOLD` (0.6 —
"keep more than you discard").

Direction is per measure — lower hallucination is better, higher faithfulness is
better — via `HIGHER_IS_BETTER` in `evaluation/significance.py`. Inverting that
would reverse every conclusion, so it is covered by tests.

## Multiple-choice options

MedQA was the only dataset supplying `options` and `answer_idx` in its query
metadata, so it was the only one with a correctness measure; CaseHOLD and SciQ
were answered open-ended and their `choice_correct` is NULL throughout the
reported results. The loaders now build options for all three via
`build_options` in `loaders/base.py`, which shuffles candidates on a per-query
seed — source order always puts the correct answer first, and a model replying
"A" every time would otherwise score 100%.

Queries are cached beside each corpus and reloaded, so a loader change does not
reach an existing build on its own. `cli refresh-query-metadata` rewrites only
the metadata field, aborting if the query text, evidence, answer or relevance
class disagrees with the adapter, since the retrieval results in the database
were measured against the cached queries. It has been run: CaseHOLD and SciQ
now carry options, and qrels, documents and chunks hash byte-identically to
before.

## Known issues

1. **The judge fails on about a quarter of its verdicts.** phi3 returns
   unparseable output often enough that two passes requesting 1,440 each
   delivered 2,140 usable scores, and the pairwise overlap is 79% rather than
   100%. A larger judge would fail less, at the cost of the independence and
   speed that motivated phi3.

2. **`score-answers` only fills NULL rows.** Pointing it at a new judge without
   clearing the column first silently scores nothing. `cli reset-scores
   --column faithfulness --run <id> --yes` exists for that and refuses without
   `--yes`.

3. **Several runs share the database.** `results`, `score-answers` and
   `significance-generation` all default to the latest of the relevant kind,
   which is right, but anything written by hand against `generations` must
   filter on `run_id` or it will average two judges into one number.

---

## Layout

Flat package layout; every module is importable from the repo root.

```
cli.py          entry point for every command
core/           settings, database schema, logging
loaders/        one adapter per dataset
retrieval/      bm25, dense, hybrid, chunking, indexing
generation/     Ollama client, prompt assembly, generator
evaluation/     metrics, significance, faithfulness, hallucination, _shared
dashboard/      Streamlit app (dark theme, navy #0F1117 / gold #F0A500)
tests/          356 tests, all passing
notebooks/      colab_generation.ipynb — only needed to regenerate answers
```

Config is `config/default.yaml`. Paths relocate via `RAGBENCH_DATASETS_DIR` and
`RAGBENCH_ARTIFACTS_DIR`, which is how the same code ran on Colab.

---

## Traps that have already cost real time

- **Always pass `--config config/default.yaml`.** Without it the CLI silently
  falls back to dataclass defaults — a scoring run once used the wrong judge and
  a 5x sample size before anyone noticed.
- **`--limit` truncates the corpus, not the query set.** `--limit 10` once cut
  CUAD from 510 documents to 1. Use it only for smoke tests, never on a real run.
- **Never point SQLite at a Google Drive FUSE mount.** Use the online backup API
  (`source.backup(target)`) to sync a copy instead.
- **Query metadata is cached in two places.** A corpus stores `queries.jsonl`,
  and a frozen task set (`artifacts/results/generation_tasks_*.json`) embeds its
  own snapshot of the same metadata. Refreshing only the first produces a run
  that silently uses the old metadata and looks entirely normal — this cost a
  started generation run once. `refresh-query-metadata` now deletes task sets
  unconditionally, and the notebook clears Drive's copies too and refuses to
  generate against a task set missing the multiple-choice options.
- **Do not delete** anything in `datasets/`, `venv/`, `artifacts/corpora/`, or
  `artifacts/indexes/`. The corpora and indexes are ~711 MB and expensive to rebuild.

---

## Commands

```bash
venv/Scripts/python.exe cli.py --help

# analysis over the existing database — all CPU, all fast
venv/Scripts/python.exe cli.py export       --config config/default.yaml
venv/Scripts/python.exe cli.py significance --config config/default.yaml
venv/Scripts/python.exe -m streamlit run dashboard/app.py

# tests
venv/Scripts/python.exe -m pytest -q
```
