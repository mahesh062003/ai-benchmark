# ragbench — retrieval strategies across specialist domains

An MSc COMP702 research framework for benchmarking **BM25**, **dense (FAISS)**
and **hybrid (Reciprocal Rank Fusion)** retrieval across legal, medical and
scientific question-answering datasets, with an optional local-LLM generation
stage.

**Research question:** *does the relative effectiveness of retrieval strategies
vary across specialist domains?*

The design priority is research validity over a complete results table. The
framework never fabricates retrieval ground truth: where a dataset does not
support retrieval evaluation, its metrics are reported as **NULL**, not zero,
and the dataset is kept for the evaluation it *does* support.

---

## Datasets and their ground truth

Determined by direct inspection of the shipped files, not from documentation.
Run `ragbench datasets` to see this table with full justifications.

| Dataset | Domain | Class | Corpus scope | Relevance comes from | Retrieval metrics |
|---|---|---|---|---|---|
| **CUAD** | legal | `GOLD` | per contract | Verified answer character spans | valid |
| **CaseHOLD** | legal | `DERIVED` | pooled holdings | The labelled correct holding | valid (constructed corpus) |
| **PubMedQA** | medical | `DERIVED` | pooled abstracts | The source abstract (`pubid`) | valid |
| **MedQA** | medical | `UNSUPPORTED` | textbooks | **nothing — none exists** | **NULL** |
| **QASPER** | scientific | `GOLD` | per paper | Annotated evidence paragraphs | valid |
| **SciQ** | scientific | `DERIVED` | pooled supports | The question's own support passage | valid |

- **GOLD** — annotators marked the evidence itself; it maps deterministically to chunks.
- **DERIVED** — the dataset records a real provenance link, and relevance follows from that link. Document-level, so it over-counts relative to GOLD.
- **UNSUPPORTED** — no relevance information exists at all.
- **HEURISTIC** — exists in the taxonomy and is **deliberately unused**; a test enforces this.

**Why MedQA has no retrieval metrics.** It ships 18 medical textbooks (89 MB) —
a genuine external corpus — but no annotation whatsoever linking a question to
a supporting passage. Computing Recall@k would require inventing relevance.
Instead MedQA is used for the generation stage, where multiple-choice accuracy
is measurable without any retrieval ground truth. See
[docs/METHODOLOGY.md §3.6](docs/METHODOLOGY.md) for the alternatives considered
and rejected.

---

## The distinction that governs the whole framework

| Value | Meaning |
|---|---|
| `0.0` | Ground truth existed and the retriever failed to find it. **A measurement.** |
| `NULL` | There was no ground truth to score against. **Not a measurement.** |

Aggregates take the mean over scoreable queries and skip NULLs rather than
coercing them to zero. Every result row also carries **coverage** — the share of
queries that actually had a judgement — because a mean over 28% of queries is a
different claim from a mean over 100%.

---

## Installation

Python 3.10+. The project uses the existing virtual environment in `venv/`.

```bash
cd "C:/AI BENCHMARK"
./venv/Scripts/python.exe -m pip install -e .
```

All runtime dependencies (faiss-cpu, sentence-transformers, torch, rank-bm25,
pandas, pyarrow, typer, rich, ollama) were already present; nothing new was
installed. `pip install -e .` only registers the package and the `ragbench`
command.

Verify:

```bash
./venv/Scripts/python.exe -m cli datasets
```

Every command below can also be run as `ragbench <command>` once the venv is
activated. This README uses the explicit interpreter path so the commands work
without activation.

### Expected layout

```
C:/AI BENCHMARK/
├── datasets/          # research data (never modified)
│   ├── Legal/{CUAD, LexGLUE/case_hold}
│   ├── Medical/{MedQA, PubMedQA}
│   └── Scientific/{qasper, sciq}
├── core/              # data model, settings, database, stable IDs
├── loaders/           # one module per dataset + registry
├── retrieval/         # bm25, dense, hybrid
├── evaluation/        # metrics, faithfulness, hallucination
├── generation/        # optional Ollama stage
├── benchmark/         # corpus, indexer, runner
├── dashboard/         # Streamlit app
├── tests/             # 288 tests
├── config/            # default.yaml
├── cli.py             # single entry point
├── docs/METHODOLOGY.md
└── artifacts/         # generated: corpora, indexes, results, SQLite
```

The two directories that differ between machines are relocatable without
editing any file, so the same checkout runs on Windows, Colab and Barkla:

```bash
export RAGBENCH_DATASETS_DIR=/content/drive/MyDrive/datasets
export RAGBENCH_ARTIFACTS_DIR=/content/drive/MyDrive/ragbench_artifacts
```

---

## Workflow

### 1. Check what is available

```bash
./venv/Scripts/python.exe -m cli datasets
./venv/Scripts/python.exe -m cli validate --all
```

`validate` reads every dataset and reports record counts and skips without
building anything.

### 2. Smoke run (verifies the whole pipeline)

`--limit` caps the number of queries per dataset, which is all a smoke run
needs. Always point `RAGBENCH_ARTIFACTS_DIR` at a scratch directory first:

```bash
# fast check -- one small dataset, all three retrieval methods, ~1 minute
RAGBENCH_ARTIFACTS_DIR=/tmp/ragbench-smoke \
  ./venv/Scripts/python.exe -m cli benchmark --dataset sciq --limit 10

# full check -- all six datasets
RAGBENCH_ARTIFACTS_DIR=/tmp/ragbench-smoke \
  ./venv/Scripts/python.exe -m cli benchmark --all --limit 10
```

> **Run smoke tests against a scratch artifacts directory, never the real one.**
> `--limit` also narrows the *corpus* to the documents those queries need, and
> the reduced corpus is written back over the full one — a `--limit 10` run
> against `artifacts/` rewrites CUAD from 510 contracts down to 1. The
> environment variable keeps the real corpora and indexes untouched.

The first `--all` run against an empty scratch directory is **not** quick: it
rebuilds every corpus and index from scratch, and embedding MedQA's ~67k
textbook chunks dominates the wall time. Later runs against the same scratch
directory reuse those indexes and finish in about a minute. Use the single
dataset form for a quick confidence check, and cap MedQA with `max_documents`
under `datasets.medqa` if you want a fast cold `--all` run.

A correct smoke run reports MedQA as `NULL` across all three methods, and every
other dataset with numeric scores.

### 3. Full experiment

```bash
# corpora: documents -> chunks -> qrels
./venv/Scripts/python.exe -m cli build-corpus --all --config config/default.yaml

# BM25 + FAISS indexes, persisted for reuse
./venv/Scripts/python.exe -m cli build-indexes --all --config config/default.yaml

# retrieval benchmark -> artifacts/benchmark.sqlite
./venv/Scripts/python.exe -m cli benchmark --all --config config/default.yaml
```

Indexing MedQA's full 89 MB textbook corpus dominates the runtime (roughly
100k chunks to embed). Reduce it with `max_documents` under `datasets.medqa`
in the config for faster iteration.

### 4. Inspect and export

```bash
./venv/Scripts/python.exe -m cli results
./venv/Scripts/python.exe -m cli inspect --dataset cuad --method hybrid --limit 3
./venv/Scripts/python.exe -m cli export --output artifacts/results/aggregates.csv
```

`inspect` retrieves live and prints the actual chunks with `RELEVANT` /
`unjudged` labels, so ground-truth mappings can be checked by eye.

### 5. Generation (optional — needs Ollama)

Generation is disabled by default and imported nowhere on the retrieval path,
so **retrieval benchmarks run on a machine with no Ollama installed** and the
dashboard still shows every retrieval result.

```bash
ollama serve
ollama pull llama3.1 && ollama pull gemma2 && ollama pull mistral && ollama pull qwen2.5

./venv/Scripts/python.exe -m cli models          # what the server can serve
./venv/Scripts/python.exe -m cli generate-all    # every model x method x dataset
```

`generate-all` answers the same questions with each configured model
(`generation.models`), 50 seeded-sample questions per dataset per retrieval
method by default. Three properties make the comparison meaningful and the run
survivable:

- **Identical inputs.** Retrieval runs once and is frozen to
  `artifacts/results/generation_tasks_*.json`; every model answers from
  byte-identical context.
- **Grouped by model.** Ollama keeps one model resident, so all questions for
  one model are completed before the next is loaded.
- **Resumable.** Every answer is committed as produced. Re-run with
  `--run <run_id>` and it skips what is already stored.

The full default sweep is 4 models × 6 datasets × 3 methods × 50 questions =
**3,600 generations**. Scale it with `--limit`, `--models` or `--methods`.

**Runtime is dominated by prompt length, not answer length.** Measured on a
4 GB GTX 1650, where an 8B Q4 model only half fits in VRAM and the rest runs on
CPU:

| Dataset | Context size | Seconds per answer |
|---|---|---|
| CaseHOLD | ~1.0k chars | ~34 |
| PubMedQA | ~2.1k chars | ~45 |
| SciQ / QASPER | ~2.6–3.1k chars | ~50–65 |
| **CUAD / MedQA** | **~8–9k chars** | **~170** |

That is roughly **23 h per model**, so the full four-model sweep is on the order
of **four days**. On a GPU that fits the whole model it is far faster. The two
effective levers are `generation.max_context_chunks` (prompt length) and
`generation.queries_per_dataset`.

**Two operational rules for long runs:**

```bash
# Resume an interrupted sweep -- nothing already answered is regenerated.
./venv/Scripts/python.exe -m cli generate-all --run <run_id>
```

- **Do not run `score-answers` while `generate-all` is running.** Both drive the
  same Ollama server with different models, and Ollama keeps only one resident:
  the two processes force a ~10–15 s model reload on nearly every call and each
  runs several times slower. Run them one after the other.
- Progress is committed continuously, so killing the run loses at most the
  answer in flight.

### 6. Answer grounding: faithfulness and hallucination (optional)

```bash
./venv/Scripts/python.exe -m cli score-answers
```

A second pass over stored answers, so a scorer can be re-run or re-thresholded
without regenerating anything.

- **RAGAS faithfulness** — the published procedure (decompose the answer into
  atomic statements, verify each against the retrieved context, score =
  supported / total) implemented directly against a local Ollama judge. One
  fixed judge (`scoring.judge_model`) rates every model so no model grades its
  own homework on a different footing; the judge is recorded per row. Costs two
  LLM calls per answer.
- **NLI hallucination** — a cross-encoder
  (`cross-encoder/nli-deberta-v3-base`) scores each answer sentence against
  each retrieved chunk. A sentence is supported when some chunk entails it;
  contradicted only when a chunk predicts contradiction as its winning class.
  Score = unsupported sentences / total, so **lower is better**. Cheap: it
  batches on the GPU.

Neither is ground truth — both are model-based estimates, and faithfulness
measures agreement with one judge. Anything unscoreable (empty answer, no
context, unparseable judge output) stays **NULL**, never 0.0.

### 7. Dashboard

```bash
./venv/Scripts/python.exe -m cli dashboard      # http://localhost:8501
```

Retrieval and generation tabs: aggregate tables, per-domain bar charts for
retrieval strategies and for LLM models, and an answer inspector. The
generation tab degrades to an explanatory panel when no answers exist, so the
dashboard is fully usable with retrieval results alone.

When more than one generation run exists the dashboard shows **one at a time**,
newest first, and never pools them — two runs over the same task set would
otherwise average together and double the apparent answer count.

### 8. Tests

```bash
./venv/Scripts/python.exe -m pytest tests/ -m "not slow"   # 167 tests, ~21s
./venv/Scripts/python.exe -m pytest tests/                 # 236 tests, ~60s
```

`slow` tests read the real datasets and skip cleanly if `datasets/` is absent.

---

## Architecture

Each stage of the pipeline is a top-level package, so the directory tree and
the data flow have the same shape.

```
core/
├── types.py          core data model + the ground-truth taxonomy
├── settings.py       dataclass config, YAML + environment overrides
├── ids.py            stable, readable, collision-safe identifiers
├── db.py             SQLite schema and experiment tracking
└── logging_setup.py  console and file logging

loaders/              one module per dataset + registry
├── base.py           the adapter contract every dataset implements
└── {cuad, casehold, medqa, pubmedqa, qasper, sciq}.py

benchmark/
├── chunking.py       chunking + the single rule that decides relevance
├── corpus.py         corpus build, qrels, persistence
├── indexer.py        shared sentence embedder (loaded once) + index building
└── runner.py         the benchmark loop

retrieval/            base, bm25, dense, hybrid (RRF)

evaluation/
├── metrics.py        Recall@K / MRR / nDCG@K with NULL semantics
├── faithfulness.py   RAGAS faithfulness against a local LLM judge
├── hallucination.py  NLI cross-encoder: unsupported-sentence rate
└── _shared.py        sentence splitting + JSON recovery used by both

generation/
├── generator.py      optional Ollama stage: prompt, client, answer measures
└── runner.py         multi-model sweep: frozen task set, resume, model probe

dashboard/app.py      Streamlit: retrieval + generation tables and charts
cli.py                Typer CLI — the single entry point
```

**The data flow, which is also the traceability chain:**

```
dataset file → SourceDocument → Chunk (with char offsets)
                     ↓                        ↓
              EvidenceSpan  ──overlap──→  Qrel (relevant chunk ids | unavailable + reason)
                                              ↓
query → retriever → ranked chunk ids → metrics (float | NULL) → SQLite run
```

Every dataset's ground truth — CUAD offsets, QASPER evidence text, PubMedQA
provenance — is normalised into character spans, so **relevance is decided in
exactly one function**, `chunking.relevant_chunk_ids`. Its signature admits no
retrieval results, and a test asserts this, making it structurally impossible
for retrieval output to influence its own ground truth.

---

## Configuration

Everything tunable lives in `config/*.yaml`; paths resolve relative to the
project root, so commands work from the repo root on any machine.

```yaml
chunk:      {strategy: fixed, size_tokens: 256, overlap_tokens: 64}
retrieval:  {top_k: 10, metric_ks: [1,5,10], methods: [bm25,dense,hybrid], rrf_k: 60}
embedding:  {model_name: sentence-transformers/all-MiniLM-L6-v2, device: null}
generation: {enabled: false, models: [llama3.1, gemma2, mistral, qwen2.5], queries_per_dataset: 50}
scoring:    {judge_model: llama3.1, nli_model: cross-encoder/nli-deberta-v3-base, entail_threshold: 0.5}
datasets:
  medqa: {split: test, max_documents: 2}    # cap the corpus for quick runs
```

Chunking is per-dataset: `fixed` windows for long text (CUAD, MedQA),
`passage` for datasets whose own units are the annotation units (QASPER
paragraphs, PubMedQA sections). A corpus build is keyed by its chunk
fingerprint (`fixed-256-64`), so changing chunking creates a separate build
rather than silently invalidating an index.

---

## Results database

`artifacts/benchmark.sqlite` — five tables, all metric columns nullable:

| Table | Contents |
|---|---|
| `runs` | full config JSON + environment snapshot per run |
| `retrieval_results` | every ranked chunk; `is_relevant` is 1/0/**NULL** |
| `query_metrics` | per-query scores, nullable |
| `aggregate_metrics` | per (dataset, method) means + coverage |
| `generations` | one row per (run, dataset, method, **model**, query): answer, latency, accuracy, faithfulness, hallucination — each nullable |

```sql
-- strategy comparison within each dataset
SELECT dataset, method, relevance_class, coverage, mrr,
       json_extract(metrics_json,'$.recall."10"') AS recall_10
FROM aggregate_metrics ORDER BY dataset, method;
```

---

## Verified smoke results

Query samples, not the full experiment — reported to show the pipeline works
end to end and produces plausible, non-degenerate numbers.
**Read down a column (strategies within a dataset), not across rows.**

| Dataset | Class | Scoreable | Coverage | BM25 R@10 | Dense R@10 | Hybrid R@10 |
|---|---|---|---|---|---|---|
| CUAD | GOLD | 231/820 | 0.28 | 0.6926 | 0.6839 | **0.7078** |
| CaseHOLD | DERIVED | 200/200 | 1.00 | 0.4250 | 0.5350 | **0.6050** |
| PubMedQA | DERIVED | 200/200 | 1.00 | 0.7050 | **0.8350** | 0.8000 |
| MedQA | UNSUPPORTED | 0/50 | 0.00 | **NULL** | **NULL** | **NULL** |
| QASPER | GOLD | 116/127 | 0.91 | 0.5080 | **0.5975** | 0.5919 |
| SciQ | DERIVED | 174/200 | 0.87 | 0.9828 | 0.9655 | **0.9885** |

The pattern is already the substance of the research question: hybrid wins on
legal, dense on medical and scientific evidence retrieval, and BM25 is
competitive only on SciQ — the dataset whose construction guarantees lexical
overlap. MedQA is NULL throughout, exactly as intended.

---

## Limitations

Summarised here; argued in full in [docs/METHODOLOGY.md §11](docs/METHODOLOGY.md).

1. **Cross-dataset scores are not commensurable** — corpus sizes, relevant
   counts per query and provenance all differ. Compare strategies *within* a
   dataset.
2. **DERIVED relevance is document-level** and over-counts relative to GOLD.
3. **CaseHOLD's corpus is constructed by this framework**, not shipped by the
   dataset; results are not open-domain legal retrieval.
4. **SciQ is easy by construction** (questions written from their supports) and
   structurally favours lexical matching.
5. **CUAD and QASPER are document-scoped**, not open-domain.
6. **MedQA contributes no retrieval metrics.**
7. **CUAD coverage is ~0.28** — most clause categories are absent from any
   given contract (`is_impossible`), and those queries are NULL, not zero.
8. **One embedding model** — conclusions about "dense retrieval" are
   conclusions about MiniLM-L6-v2 until the sweep is repeated.
9. **No significance testing** — differences are point estimates.
10. **Faithfulness and hallucination are model-based estimates, not ground
    truth.** Faithfulness measures agreement with one LLM judge, which is not a
    validated rater; hallucination is limited by the cross-encoder's 512-token
    pair truncation and by regex sentence splitting. Both count "the context is
    silent" as unsupported, which is deliberate but means the hallucination
    rate is an *ungroundedness* rate, not a count of false statements.
11. **PubMedQA and CUAD use single splits** because no official split ships in
    this distribution.

---

## Citation

Datasets: CUAD (Hendrycks et al., 2021); CaseHOLD/LexGLUE (Zheng et al., 2021;
Chalkidis et al., 2022); MedQA (Jin et al., 2020); PubMedQA (Jin et al., 2019);
QASPER (Dasigi et al., 2021); SciQ (Welbl et al., 2017).
Method: Reciprocal Rank Fusion (Cormack et al., 2009).
