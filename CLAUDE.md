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

All results live in `artifacts/benchmark.sqlite` (~360 MB, integrity verified).

| | |
|---|---|
| Retrieval | 29,234 queries, 821,256 retrieval results, 87,702 query metrics |
| Generation | 7,200 answers, zero empty (6 datasets x 3 strategies x 4 models x 100 queries) |
| Hallucination (NLI) | 7,033 scored |
| Faithfulness (LLM judge) | 1,679 scored — stratified, 72 cells |
| Significance | 45 Holm-corrected retrieval comparisons in `artifacts/results/significance.csv` |

Datasets: CUAD, CaseHOLD (legal) · MedQA, PubMedQA (medical) · QASPER, SciQ (scientific).
Strategies: BM25, dense (FAISS + all-MiniLM-L6-v2), hybrid (RRF, k=60).
Models: llama3.1, gemma2, mistral, qwen2.5 — temperature 0.0, frozen task sets so
every model sees byte-identical context.

**No further Colab or GPU work is required.** Everything remaining reads the
existing database on CPU.

---

## Significance testing — both halves are covered

`cli significance` compares retrieval strategies (45 comparisons) and
`cli significance-generation` compares models on answer quality (72). Results in
`artifacts/results/significance.csv` and `significance_generation.csv`, both
committed.

The generation side chooses its test per comparison and records which it used:
hallucination is scored for every answer, so models are paired on the same
(strategy, query) units; faithfulness is a subsample drawn independently per
cell with only ~20% overlap between models, so it uses unpaired Mann-Whitney U.
Direction is per measure — lower hallucination is better, higher faithfulness is
better — via `HIGHER_IS_BETTER` in `evaluation/significance.py`. Inverting that
would reverse every conclusion, so it is covered by tests.

## Known issues — open, in priority order

1. **CaseHOLD has no correctness measure.** `choice_acc` is NULL and
   `exact_match` is 0.000. Either add multiple-choice extraction or state it
   explicitly as a limitation — do not leave it looking like a score of zero.

2. **MedQA hallucination (~0.9) is partly an artifact** of answers citing
   textbook passages that do not lexically entail them. Needs framing in the
   write-up, not a code change.

3. **The faithfulness subsample is not shared across models**, which forces the
   weaker unpaired test and leaves only 3 of 36 comparisons significant against
   14 of 36 for hallucination. Drawing the subsample on a common set of
   questions would make the paired test available. Requires re-running scoring
   on a GPU, so it is a future-work item rather than a fix.

4. **The judge is one of the models under test.** mistral scores highest on the
   faithfulness metric it judges, but that advantage is significant in only 1 of
   18 comparisons, so the ordering is neither evidence of self-preference nor
   evidence against it. A judge outside the model set would settle it.

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
tests/          283+ tests, all passing
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
