# Methodology

This document records the methodological decisions behind the benchmark, the
evidence from the dataset audit that motivated each one, and the limitations
that follow. It is written to be defensible in a dissertation viva: where a
decision was a judgement call, the alternatives considered and the reason for
rejecting them are stated.

---

## 1. Research question

> Does the relative effectiveness of retrieval strategies vary across
> specialist domains?

Three strategies (BM25, dense FAISS, hybrid RRF) are compared over six datasets
spanning three domains (legal, medical, scientific). The comparison is
controlled: the same chunking, the same embedding model, the same top-k and the
same metric definitions are applied within each dataset, so the only thing that
varies within a dataset row is the retrieval strategy.

**Comparisons are valid down a column (strategies within one dataset), and only
qualitatively valid across rows (between datasets).** Different datasets have
different corpus sizes, different numbers of relevant items per query, and
different ground-truth provenance, so their absolute scores are not
commensurable. This is the single most important caveat in reading the results.

---

## 2. The ground-truth taxonomy

Every dataset is classified by the *provenance* of its relevance judgements.
The classification is declared in code (`ragbench/types.py::RelevanceClass`),
exposed by `ragbench datasets`, and stored on every row of the results table.

| Class | Meaning | Datasets |
|---|---|---|
| **GOLD** | Human annotators marked the evidence itself, at or below document level, and it maps deterministically onto retrieval units. | CUAD, QASPER |
| **DERIVED** | The dataset records an unambiguous provenance link (this question was written from this passage; this holding is the cited one). Relevance is derived from that recorded link — never from retrieval output. The link is real, but it is document-level, so some text marked relevant may not itself support the answer. | PubMedQA, SciQ, CaseHOLD |
| **HEURISTIC** | Relevance guessed by a rule such as answer-string containment. **Defined in the taxonomy but deliberately unused.** A test asserts no dataset carries this class. | *(none)* |
| **UNSUPPORTED** | No relevance information exists. Retrieval metrics are NULL. | MedQA |

The reason for separating GOLD from DERIVED rather than calling both "ground
truth": a DERIVED judgement systematically over-counts relevant text. A
PubMedQA question is genuinely answered by its source abstract, but marking
every section of that abstract relevant credits the METHODS section too. This
inflates recall relative to a GOLD dataset such as QASPER, where annotators
marked the specific supporting paragraph. Reporting both in one column without
that annotation would be misleading.

`HEURISTIC` is kept in the enum precisely so that the absence of heuristic
relevance is *visible and testable* rather than merely claimed.

---

## 3. Per-dataset decisions

### 3.1 CUAD (legal) — GOLD, document-scoped

**Audited:** 510 contracts, one paragraph each holding the full contract text
(mean 52,563 characters; max 338,211). 41 clause-category questions per
contract giving 20,910 records, of which 6,702 carry ≥1 answer span and 14,208
are `is_impossible`. **Every** shipped `answer_start` offset was verified to
reproduce its `text` exactly, so the offsets are trustworthy without fuzzy
matching.

- **Query:** the clause-category question.
- **Corpus:** chunks of *one* contract.
- **Relevant:** any chunk overlapping an annotated answer span.

**Why document-scoped.** The query text is a template — *"Highlight the parts
(if any) of this contract related to 'Governing Law'..."* — that is
byte-for-byte identical across all 510 contracts. Pooling the contracts would
make the query ill-posed rather than merely hard: no ranking over the pool
could be correct, because the query carries no information identifying which
contract is meant. In-contract clause location is also CUAD's actual task.

**`is_impossible` handling.** These 14,208 records assert the clause is *absent*
from that contract. There is no relevant chunk, so they are excluded from
metrics as NULL. Scoring them zero would penalise a retriever for failing to
find something the annotators recorded as not present. This is why CUAD's
coverage is ~0.28 — the figure is expected and is reported explicitly.

**Splits.** The shipped `CUAD_v1.json` contains no official train/test split
(the official split ships as separate files not present here). A single `all`
split is exposed and this is stated rather than a split being invented.
`CUADv1.json` was verified byte-identical to `CUAD_v1.json` (same MD5).

### 3.2 QASPER (scientific) — GOLD, document-scoped

**Audited:** train 888 papers / 2,593 questions; val 281 / 1,005; test 416 /
1,451. The decisive finding is that `evidence` entries are the **literal
paragraph text** copied from `full_text`, so they match by exact string lookup.

Measured evidence-matching rates over the shipped files:

| split | evidence items | exact | after `strip()` | FLOAT (figure/table) | unmatched |
|---|---|---|---|---|---|
| train | 4,209 | 3,534 | 119 | 386 | 170 |
| val | 2,808 | 2,349 | 72 | 253 | 134 |
| test | 5,744 | 4,855 | 176 | 459 | 254 |

- **Query:** the question. **Corpus:** paragraphs of its own paper.
- **Relevant:** chunks overlapping a matched evidence paragraph, unioned across
  the multiple annotators who answered each question (the standard treatment).

**The unmatched remainder is dropped and counted, never guessed at.**
`FLOAT SELECTED: ...` entries reference figures and tables that do not exist
among the body paragraphs; a small residue are bare section headings. A
question left with no mappable evidence — including every `unanswerable`
question — becomes NULL, not zero.

**Why document-scoped.** QASPER questions are deictic (*"which datasets did
they experiment with?"*) and only meaningful relative to their own paper. This
mirrors QASPER's own evidence-selection task.

### 3.3 PubMedQA (medical) — DERIVED, pooled

**Audited:** `pqa_labeled` has 1,000 expert-labelled records, no duplicate
`pubid`, each with a structured abstract of 1–9 sections (mean 3.36) carrying
labels (RESULTS 938, METHODS 634, BACKGROUND 385, ...), plus a `long_answer`
and a `final_decision` of yes (552) / no (338) / maybe (110).

- **Query:** the research question. **Corpus:** abstract sections, pooled.
- **Relevant:** the sections of the abstract the question was written from,
  identified by the record's own `pubid`.

**Why DERIVED and not GOLD.** No annotation marks *which section* answers the
question; only the source abstract is recorded. Projecting document-level
provenance onto every section over-counts. This is stated rather than hidden.

**Splits.** `pqa_labeled` ships as a single HuggingFace `train` split; the
official PQA-L folds are not present in this distribution. One `labeled` split
of all 1,000 records is exposed, and this is documented rather than a fold
being invented. `pqa_unlabeled` (61,249) and `pqa_artificial` (211,269) exist
and could expand the corpus but carry no expert labels; they are off by default.

### 3.4 SciQ (scientific) — DERIVED, pooled

**Audited:** train 11,679 rows (1,198 with empty support), validation 1,000
(113 empty), test 1,000 (116 empty). Non-empty supports average 464 characters
and are near-disjoint across splits (2 shared strings between train and test),
so pooling within a split introduces no cross-split leakage.

- **Query:** the exam question. **Corpus:** de-duplicated support passages.
- **Relevant:** the question's own support passage.

**Known and expected easiness.** Crowdworkers wrote each question *from* its
support, so question and support share heavy lexical overlap, and there is
exactly one relevant document in a pool of ~884 (test). SciQ is therefore an
unusually easy, closed-domain task that structurally favours lexical matching.
**High SciQ scores should be read as a pipeline sanity check, not as evidence
that a retriever generalises.** The smoke results bear this out: BM25 leads
dense on SciQ and trails it everywhere else.

Rows with empty support have no relevant document and are excluded as NULL
rather than dropped silently or scored zero.

### 3.5 CaseHOLD (legal) — DERIVED, pooled, with a constructed corpus

**Audited:** train 45,000 / validation 3,900 / test 3,600 rows. `context` is a
citing excerpt (mean 843 chars) with the cited holding masked as `<HOLDING>`
(present in 3,549 of 3,600 test rows). `endings` holds five candidate holdings;
`label` (0–4, near-uniform) indexes the correct one. In test, the 18,000
endings reduce to 13,297 unique strings; 3,596 of 3,600 golds are unique; 752
gold holdings also appear as some other row's distractor.

**A parsing hazard worth recording.** `endings` is serialised as a NumPy array
repr — quoted strings separated by newlines, with no commas. `ast.literal_eval`
*appears to succeed* on this but silently concatenates the five adjacent string
literals into one, which would corrupt every label lookup without raising an
error. The adapter parses quoted elements explicitly and asserts a count of
five; a regression test documents the trap.

**What is and is not dataset-provided.** CaseHOLD provides a genuine relevance
signal (the labelled correct holding, taken from the case actually cited). It
provides **no external retrieval corpus** — no case-law database, no document
ids, no evidence offsets; its native task is 5-way multiple choice. This
framework pools the de-duplicated candidate holdings of the split into a shared
corpus and ranks it with the citing context as the query.

So the *relevance* is dataset-provided while the *corpus* is constructed here —
which is exactly what DERIVED denotes. CaseHOLD results answer "can this
strategy locate the correct holding among ~13k real holdings" and **must not be
presented as open-domain legal retrieval**. Every candidate joins the corpus,
so the distractors act as genuine hard negatives.

### 3.6 MedQA (medical) — UNSUPPORTED

This is the most important negative finding of the audit.

**Audited:** `questions/{train,dev,test}.jsonl` hold 10,178 / 1,272 / 1,273
records with 5 options each; `answer_idx` and `answer` agree on every record;
the three splits share no question text, so the official split is clean and is
preserved. `textbooks/` holds 18 medical reference texts totalling 89.4 MB.

MedQA ships a genuine external retrieval corpus — the textbooks — which makes
it the only dataset here with a realistic open-domain medical corpus. What it
does **not** ship is any annotation linking a question to a supporting passage:
no evidence spans, no passage ids, no source references, not even a book-level
attribution.

**Therefore Recall@k, MRR and nDCG@k cannot be computed for MedQA without
fabricating relevance, and this framework reports them as NULL.**

Alternatives considered and rejected, recorded so the decision can be defended:

1. *Mark a chunk relevant because it contains the answer string.* The options
   are clinical sentences (*"Tell the attending that he cannot fail to disclose
   this mistake"*), so containment would almost never fire, and where it did it
   would measure string overlap rather than evidential support. This is the
   HEURISTIC class and is deliberately unused.
2. *Treat the question as its own corpus.* Trivially self-satisfying and
   meaningless.
3. *Record unavailable metrics as 0.0.* Would be read as "the retriever failed"
   when the truth is "there was nothing to score against", and would drag any
   cross-dataset average toward zero for a reason that has nothing to do with
   retrieval quality.

**What MedQA is used for instead:** the end-to-end generation stage. Retrieval
over the textbooks supplies context, the generator answers the multiple-choice
question, and multiple-choice accuracy is a legitimate downstream measure
requiring no retrieval ground truth. Retrieved passages are still persisted
with `is_relevant = NULL` so they can be inspected qualitatively.

---

## 4. Corpus construction

Source documents are stored once and referenced by many queries; SciQ supports
and CaseHOLD holdings are de-duplicated by content hash, so a passage shared by
two questions is one corpus document relevant to two queries rather than two
documents. Original dataset ids (`pubid`, `question_id`, contract title, row
index) are preserved in metadata, and the corpus representation is independent
of the QA representation.

**Corpus scope** is an explicit, per-dataset property:

- `document` — the query ranks only chunks of its own document (CUAD, QASPER).
- `pooled` — all documents of the split form one corpus (the other four).

Document scoping is implemented with a filter at search time (an `allowed` set
of chunk ids, and a FAISS `IDSelector` for dense), so one index serves all
queries rather than one index per document.

**Query subsampling never shrinks the corpus.** Setting `max_queries` reduces
the query sample but leaves every corpus document in place, so the distractor
pool is unchanged and a sampled run remains representative of the full task.

---

## 5. Chunking

Two strategies, chosen per dataset rather than applied uniformly:

- **`fixed`** — sliding window of 256 whitespace tokens with 64 overlap, for
  long continuous text (CUAD contracts, MedQA textbooks).
- **`passage`** — cut on the document's own declared segment boundaries
  (QASPER paragraphs, PubMedQA abstract sections), sub-splitting only a segment
  exceeding `max_tokens`. A document with no declared structure (a SciQ
  support, a CaseHOLD holding) is one segment.

**Why 256/64 rather than a value copied from a tutorial.** The default
embedding model (`all-MiniLM-L6-v2`) truncates input at 256 word-pieces, so a
larger chunk would be silently truncated on the dense side while remaining
fully visible to BM25 — an asymmetry that would confound the very comparison
being made. 64 tokens (25%) of overlap makes it unlikely that a clause spanning
a boundary is split away from its context in every chunk containing it. Both
are configurable and the fingerprint (`fixed-256-64`) is part of the corpus
build path, so changing them produces a separate build rather than silently
invalidating an index.

**Why `passage` exists at all.** Where the dataset's annotators judged a
paragraph, making a retrieval unit equal to that paragraph keeps the evidence
mapping one-to-one. Imposing a fixed window over QASPER would split annotated
paragraphs across chunks and blur what "relevant" means.

**Short documents are never dropped.** `min_chunk_chars` exists to discard
whitespace fragments of a sliding window; a short-but-real document (CaseHOLD
has 12-character holdings) still enters the corpus, because dropping it would
silently remove a document that is some query's gold answer and convert a
scoreable query into an unscoreable one.

---

## 6. Evidence-to-chunk mapping

All dataset ground truth is normalised into `EvidenceSpan(doc_id, start, end)`
character ranges, and chunk relevance is then a single rule in a single place
(`chunking.relevant_chunk_ids`):

> A chunk is relevant iff it shares at least one character with an evidence
> span from the same document.

- CUAD answer offsets are already character spans.
- QASPER evidence text is located in the assembled paper to get its span.
- PubMedQA / SciQ / CaseHOLD provenance covers the whole source document.

The function's signature admits no retrieval results at all — a test asserts
this — so it is structurally impossible for retrieval output to influence
relevance. If evidence exists but maps to no chunk, the qrel is recorded as
*unavailable* with a reason, so a chunking artefact is never scored as a
retrieval failure.

---

## 7. Retrieval strategies

**BM25** (`rank_bm25`, Okapi, k1=1.5, b=0.75). Tokenisation is lowercase
`\w+` with a length-1 filter, **identical for indexing and querying** — a
mismatch there is a classic silent cause of depressed BM25 scores. No stemming
or stopword removal: both interact differently with legal, medical and
scientific vocabulary and would confound the cross-domain comparison.

**Dense** (`sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP`).
Embeddings are L2-normalised so inner product is exactly cosine. The model is
loaded once per process and shared by every dataset and retriever.

*Why exact rather than approximate search:* ANN indexes (IVF, HNSW) trade
recall for speed, and that lost recall would appear in the results table as a
property of "dense retrieval" when it is really a property of the
approximation. Exact search removes the confound. The corpora make this
affordable — the largest, MedQA's textbooks, is ~10⁵ chunks, where a flat
384-dimensional index is well under a gigabyte. A project scaling to millions
of chunks would need to revisit this and report the recall cost.

**Hybrid** (Reciprocal Rank Fusion, k=60, candidate depth 100):

```
score(d) = Σ_r 1 / (k + rank_r(d))
```

*Why rank-based fusion:* BM25 scores are unbounded and corpus-dependent while
cosine similarities lie in [-1, 1]. Any score-level combination needs a
normalisation step that silently becomes a tuned hyperparameter and confounds
the strategy comparison. RRF uses positions only. Each base ranker is consulted
to depth 100, deeper than the reported top-10, so fusion can promote a document
neither ranker placed in its own head — which is the point of hybrid retrieval.
Fusion never mutates the objects its base retrievers return.

Ties break deterministically (on corpus position within a retriever, on chunk
id after fusion) so runs are reproducible.

---

## 8. Metrics

Binary relevance throughout; the datasets provide relevant/not-relevant with no
graded judgements.

- **Recall@k** = |top-k ∩ relevant| / |relevant|
- **MRR** = 1 / rank of the first relevant item (0.0 if none retrieved)
- **nDCG@k** = DCG/IDCG with gain 1, IDCG placing `min(|relevant|, k)` relevant
  items first — so a query with more relevant items than the cutoff can still
  reach 1.0.

**The NULL/zero invariant, restated because it governs everything:**

| Value | Meaning |
|---|---|
| `0.0` | Ground truth existed; the retriever did not find it. A measurement. |
| `NULL` | There was no ground truth to score against. Not a measurement. |

Aggregation takes the mean over scoreable queries and *skips* NULLs rather than
coercing them to zero. Every aggregate row also reports **coverage** — the share
of queries that carried a judgement — because a mean over 28% of queries (CUAD)
is a different claim from a mean over 100% (PubMedQA).

---

## 9. Experiment tracking and reproducibility

Every run records: run id, UTC timestamp, stage, dataset, split, methods,
embedding model, generation model, chunk strategy/size/overlap, top-k, RRF k
and depth, relevance class, corpus scope, corpus fingerprint, document/chunk/
query counts, the **full resolved configuration** as JSON, and an environment
snapshot (Python, platform, numpy, faiss, torch, sentence-transformers,
rank_bm25, pandas versions, CUDA availability and device).

Runs never overwrite one another: primary keys include `run_id` throughout.

---

## 10. Generation and its evaluation

Generation is optional, disabled by default, and imported nowhere on the
retrieval path, so retrieval benchmarks run on a machine with no Ollama.

**Implemented:** `choice_correct` (multiple-choice accuracy for MedQA, SciQ,
CaseHOLD) and `exact_match` (normalised equality, for PubMedQA's yes/no/maybe).
Both are real, checkable, and need no retrieval ground truth — which is what
makes MedQA evaluable despite having no relevance annotations.

A failed generation (server error) records the error and leaves the quality
columns NULL — a failure to answer is not scored as a wrong answer.

### 10.1 Comparing several models

`generate-all` answers the same questions with each configured Ollama model.
Three design choices make the comparison defensible:

**Identical inputs.** Retrieval executes once per (dataset, method, query) and
is frozen to a task set on disk; every model then answers from byte-identical
context. Re-running retrieval per model would let an index or a score tie shift
between models and contaminate the comparison.

**Seeded sampling, not a head slice.** The 50 questions per dataset are a
seeded random sample. Several datasets order queries by source document — CUAD
in particular groups ~41 clause queries per contract — so the first 50 rows
would draw from one or two documents and would not represent the dataset.

**Deterministic decoding.** `temperature: 0.0`, so differences between models
are model differences rather than sampling noise.

The `model` column is part of the `generations` primary key, so two models
answering one question are two rows, not one overwriting the other.

### 10.2 Faithfulness (RAGAS)

The published RAGAS faithfulness procedure, implemented directly against a
local Ollama judge: decompose the answer into atomic statements, verify each
against the retrieved context, and score = supported / total.

**One fixed judge rates every model.** A model judging its own output is not on
the same footing as its competitors, so a single judge (`scoring.judge_model`)
is used throughout and recorded per row in `faithfulness_judge` — a score is
only meaningful relative to its rater. Per-statement verdicts are kept in
`faithfulness_json` so any score can be audited back to the reasoning.

**It is judged on a stratified subsample.** Two judge calls per answer make
this the most expensive measure in the framework. `scoring.faithfulness_sample`
draws an evenly-allocated sample across every (dataset, method, model) cell, so
no comparison cell is thin or empty. Unsampled answers keep NULL, which already
means "not measured" everywhere else in the schema.

The shipped value is **1,440**. With six datasets, three retrieval strategies
and four models there are 72 cells, so this is 20 judged answers per cell. The
figure that constrains a claim is the count inside a cell, not the total: a
larger sample spread over the same 72 cells would raise confidence in the
overall faithfulness average while leaving any model-versus-model comparison
just as underpowered. NLI hallucination is cheap enough to run over every
answer, so that comparison is not subsampled at all.

**The sample is drawn on shared questions.** Filling each cell independently
would leave two models overlapping on only about a fifth of their judged
questions, which forces an unpaired comparison between models on a few dozen
answers each. `paired_sample` therefore chooses the *questions* first and takes
every model's answer to each, spending the same budget on a comparison that is
paired by construction. Only questions answered by all models are eligible.

**Expect attrition.** A judge that returns unreadable output leaves the row
NULL, and those failures do not fall on the same questions for every model, so
a sample drawn perfectly paired arrives with less than perfect overlap. In the
reported run phi3 failed on roughly 26% of what it was given: two passes
requesting 1,440 each delivered 2,140 usable scores at 79% pairwise overlap.
Each comparison is then paired on the questions both models were successfully
judged on, provided that majority survives
(`evaluation.significance.PAIRED_OVERLAP_THRESHOLD`).

An unreadable verdict counts as *unsupported*: this can understate faithfulness
but can never invent support. An answer that decomposes to no factual
statements ("the context does not answer this") is NULL, not zero — declining
to assert something is not a faithfulness failure.

### 10.3 Hallucination (NLI)

A supervised NLI cross-encoder scores each answer sentence against each
retrieved chunk as premise. A sentence is **supported** when at least one chunk
entails it above `entail_threshold`, and **contradicted** only when some chunk
predicts contradiction as its *winning class* above `contradict_threshold`.
That second condition matters: comparing the entailment and contradiction
maxima alone labels a flatly neutral pair (entailment 0.001, contradiction
0.003) as a contradiction and grossly overstates how often models assert
something the context denies.

    hallucination = unsupported sentences / total sentences

Higher is worse — the opposite direction to faithfulness. "Unsupported" spans
both contradiction and content the context is simply silent about, so this is
an **ungroundedness rate rather than a count of false statements**; the two
components are kept separately in `hallucination_json`.

Both measures return NULL rather than a number whenever they cannot produce a
defensible score — empty answer, no context retrieved, judge failure,
unparseable output. Neither is ground truth: faithfulness measures agreement
with one judge model, and the cross-encoder truncates a premise+hypothesis pair
at 512 tokens, so a sentence supported only by the tail of a long chunk can be
scored unsupported.

---

## 11. Known limitations

1. **Cross-dataset scores are not commensurable** (§1). Corpus sizes, relevant
   counts per query and provenance all differ.
2. **DERIVED relevance is document-level** and over-counts relevant text
   relative to GOLD.
3. **CaseHOLD's corpus is constructed here**, not shipped by the dataset.
4. **SciQ is easy by construction** and favours lexical matching.
5. **CUAD and QASPER are document-scoped**, so their numbers are not comparable
   to open-domain retrieval results.
6. **MedQA contributes no retrieval metrics at all.**
7. **CUAD coverage is ~0.28** because most clause categories are absent from
   any given contract.
8. **A single embedding model** was used; conclusions about "dense retrieval"
   are conclusions about MiniLM-L6-v2 unless the sweep is repeated.
9. **No statistical significance testing** is implemented; differences between
   strategies are reported as point estimates.
10. **Faithfulness and hallucination are model-based estimates, not ground
    truth.** Faithfulness measures agreement with one unvalidated LLM judge and
    is computed on a stratified subsample; hallucination is bounded by the
    cross-encoder's 512-token pair truncation and by regex sentence splitting,
    and counts "the context is silent" as unsupported.
11. **PubMedQA and CUAD use single splits** because no official split ships in
    this distribution.
