"""SciQ -- crowdsourced science exam questions with support passages (scientific).

Audited shape of ``datasets/Scientific/sciq/{train,validation,test}.csv``:
  columns question, distractor1-3, correct_answer, support.
  train 11,679 rows (1,198 with empty support), validation 1,000 (113 empty),
  test 1,000 (116 empty). Non-empty supports average 464 characters and are
  almost entirely distinct per question. Support text is near-disjoint across
  splits (2 shared strings between train and test), so pooling within a split
  introduces no cross-split leakage.

Ground truth: DERIVED. SciQ's crowdworkers wrote each question *from* a
retrieved support passage, and the dataset records that pairing. Relevance is
derived from that recorded pairing: a question's own support passage is
relevant. No relevance is inferred from retrieval output.

The honest caveat, which matters for interpreting results: because the question
was authored from the passage, question and support share heavy lexical
overlap, and each question has exactly one relevant document in a pool of ~900
(test). This makes SciQ an unusually easy, closed-domain retrieval task and
systematically favours lexical matching. SciQ scores near a ceiling are
expected and should be read as a sanity check on the pipeline rather than as
evidence that a retriever generalises.

Rows with an empty support have no relevant document. They are kept as queries
with no ground truth (excluded from metrics as NULL) rather than dropped
silently or scored zero.
"""

from __future__ import annotations

import csv
from typing import Dict, List

from core.settings import ChunkConfig
from core.ids import content_hash
from core.ids import doc_id as make_doc_id
from core.ids import query_id as make_query_id
from core.logging_setup import SkipLog, get_logger
from core.types import (
    CorpusScope,
    DatasetSpec,
    EvidenceSpan,
    Query,
    RelevanceClass,
    SourceDocument,
)
from .base import DatasetAdapter, LoadedSplit, build_options

log = get_logger("adapters.sciq")

_SPLITS = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


class SciQAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="sciq",
        domain="scientific",
        relevance_class=RelevanceClass.DERIVED,
        corpus_scope=CorpusScope.POOLED,
        splits=["train", "validation", "test"],
        query_unit="science exam question",
        corpus_unit="support passage",
        relevance_source=(
            "The support passage the crowdworker wrote the question from, as "
            "recorded in the dataset's 'support' column."
        ),
        limitations=(
            "Closed-domain and unusually easy: the question was authored from "
            "its support, giving high lexical overlap that favours BM25, and "
            "there is exactly one relevant passage per question in a pool of "
            "roughly 900 (test split). Treat high SciQ scores as a pipeline "
            "sanity check, not as evidence of generalisation. Rows with empty "
            "support (116 of 1,000 in test) have no ground truth and are "
            "excluded from metrics."
        ),
    )
    default_chunk = ChunkConfig(strategy="passage", max_tokens=512, overlap_tokens=64)

    def load(self, split: str = "test") -> LoadedSplit:
        if split not in _SPLITS:
            raise ValueError(f"sciq split must be one of {sorted(_SPLITS)}; got {split!r}")
        path = self._require(self.root / "Scientific" / "sciq" / _SPLITS[split])
        skips = SkipLog("sciq", split)

        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        ds_cfg = self.config.dataset(self.name)

        documents: List[SourceDocument] = []
        queries: List[Query] = []
        # De-duplicate identical supports so a passage shared by two questions
        # is one corpus document with two relevant queries, not two documents.
        by_content: Dict[str, str] = {}
        n_empty_support = 0

        for index, row in enumerate(rows):
            question = (row.get("question") or "").strip()
            if not question:
                skips.skip("empty_question")
                continue
            support = (row.get("support") or "").strip()

            evidence: List[EvidenceSpan] = []
            if support:
                key = content_hash(support)
                document_id = by_content.get(key)
                if document_id is None:
                    document_id = make_doc_id("sciq", split, f"support-{key}")
                    by_content[key] = document_id
                    documents.append(
                        SourceDocument(
                            doc_id=document_id,
                            dataset="sciq",
                            split=split,
                            text=support,
                            title=f"SciQ support {key}",
                            metadata={"content_hash": key, "n_chars": len(support)},
                        )
                    )
                evidence.append(
                    EvidenceSpan(doc_id=document_id, start=0, end=len(support))
                )
            else:
                n_empty_support += 1
                skips.skip("row_has_empty_support_no_ground_truth")

            correct = (row.get("correct_answer") or "").strip()
            distractors = [
                (row.get(f"distractor{i}") or "").strip() for i in (1, 2, 3)
            ]
            options, gold_key = build_options(
                correct, distractors, seed=f"sciq-{split}-{index}"
            )

            queries.append(
                Query(
                    query_id=make_query_id("sciq", split, f"{index:05d}"),
                    dataset="sciq",
                    split=split,
                    text=question,
                    scope_doc_id=None,
                    evidence=evidence,
                    relevance_class=RelevanceClass.DERIVED,
                    answer=correct or None,
                    metadata={
                        "row_index": index,
                        "correct_answer": correct,
                        "distractors": distractors,
                        # Consumed by the generation stage: `options` is rendered
                        # into the prompt and `answer_idx` is the key scored
                        # against. Both are absent when the row is malformed, so
                        # choice accuracy stays NULL rather than becoming 0.
                        **({"options": options, "answer_idx": gold_key} if options else {}),
                        "has_support": bool(support),
                        "answer_type": "multiple_choice",
                    },
                )
            )

        if ds_cfg.max_queries:
            queries = queries[: ds_cfg.max_queries]
            # Corpus intentionally left whole: the unqueried supports are the
            # distractors that make a sampled run representative.

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": sum(1 for q in queries if q.has_ground_truth),
            "rows_without_support": n_empty_support,
        }
        skips.report(log, kept=len(queries))
        log.info(
            "sciq/%s: %d unique support passages, %d questions, %d with derived "
            "relevance (%d rows have no support)",
            split, len(documents), len(queries), stats["queries_with_evidence"], n_empty_support,
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)
