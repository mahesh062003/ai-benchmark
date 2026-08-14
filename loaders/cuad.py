"""CUAD -- Contract Understanding Atticus Dataset (legal).

Audited shape of ``datasets/Legal/CUAD/CUAD_v1.json`` (SQuAD v2 style):
  510 contracts, each with exactly one paragraph holding the full contract text
  (mean 52k characters), and 41 clause-category questions per contract, giving
  20,910 QA records of which 6,702 carry at least one answer span and 14,208
  are marked ``is_impossible``. Answer spans were verified to be exact
  character offsets into the contract text.

Ground truth: GOLD. Answer offsets are human annotations of the evidence
itself, so they map onto chunks by interval overlap with no interpretation.

Corpus scope: DOCUMENT. The query text is a clause-category template
("Highlight the parts ... related to 'Governing Law' ...") that is byte-for-byte
identical across all 510 contracts. Pooling the contracts would make the query
ill-posed -- no ranking over the pool could be correct, because the query
carries no information about which contract is meant. Retrieval is therefore
in-contract clause location, which is CUAD's actual task.

``is_impossible`` records assert the clause is *absent* from that contract, so
there is no relevant chunk. They are excluded from retrieval metrics rather
than scored zero: a retriever cannot be penalised for failing to find something
the annotators recorded as not present.
"""

from __future__ import annotations

import json
from typing import List

from core.settings import ChunkConfig
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
from .base import DatasetAdapter, LoadedSplit

log = get_logger("adapters.cuad")


class CUADAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="cuad",
        domain="legal",
        relevance_class=RelevanceClass.GOLD,
        corpus_scope=CorpusScope.DOCUMENT,
        splits=["all"],
        query_unit="clause-category question about one contract",
        corpus_unit="fixed-size chunk of that contract",
        relevance_source=(
            "Human-annotated answer character spans (SQuAD-style answer_start + "
            "text), verified to match the contract text exactly."
        ),
        limitations=(
            "The shipped CUAD_v1.json contains no official train/test split, so a "
            "single 'all' split is used and this is stated rather than silently "
            "invented. Retrieval is document-scoped, so results are not "
            "comparable to open-domain retrieval numbers. Queries marked "
            "is_impossible (14,208 of 20,910) have no relevant chunk and are "
            "excluded from metrics as NULL."
        ),
    )
    default_chunk = ChunkConfig(strategy="fixed", size_tokens=256, overlap_tokens=64)

    def load(self, split: str = "all") -> LoadedSplit:
        if split != "all":
            raise ValueError(
                f"cuad exposes only the 'all' split (no official split ships with "
                f"CUAD_v1.json); got {split!r}"
            )
        path = self._require(self.root / "Legal" / "CUAD" / "CUAD_v1.json")
        skips = SkipLog("cuad", split)
        raw = json.loads(path.read_text(encoding="utf-8"))

        limit_docs = self.config.dataset(self.name).max_documents
        limit_queries = self.config.dataset(self.name).max_queries

        documents: List[SourceDocument] = []
        queries: List[Query] = []
        n_impossible = 0
        n_span_mismatch = 0

        entries = raw["data"]
        if limit_docs:
            entries = entries[:limit_docs]

        for entry in entries:
            title = entry["title"]
            paragraphs = entry.get("paragraphs", [])
            if not paragraphs:
                skips.skip("document_without_paragraph")
                continue
            # Audited: every CUAD entry has exactly one paragraph. Guard anyway.
            if len(paragraphs) > 1:
                skips.skip("document_with_multiple_paragraphs", len(paragraphs) - 1)
            paragraph = paragraphs[0]
            text = paragraph["context"]
            if not text.strip():
                skips.skip("empty_contract_text")
                continue

            document_id = make_doc_id("cuad", split, title)
            documents.append(
                SourceDocument(
                    doc_id=document_id,
                    dataset="cuad",
                    split=split,
                    text=text,
                    title=title,
                    metadata={"contract_title": title, "n_chars": len(text)},
                )
            )

            for qa in paragraph.get("qas", []):
                evidence: List[EvidenceSpan] = []
                for answer in qa.get("answers", []):
                    start = int(answer["answer_start"])
                    span_text = answer["text"]
                    end = start + len(span_text)
                    # Only trust the offset if it actually reproduces the text.
                    if text[start:end] != span_text:
                        n_span_mismatch += 1
                        skips.skip("answer_span_offset_mismatch")
                        continue
                    evidence.append(EvidenceSpan(doc_id=document_id, start=start, end=end))

                impossible = bool(qa.get("is_impossible"))
                if impossible:
                    n_impossible += 1

                queries.append(
                    Query(
                        query_id=make_query_id("cuad", split, qa["id"]),
                        dataset="cuad",
                        split=split,
                        text=qa["question"],
                        scope_doc_id=document_id,
                        evidence=evidence,
                        # GOLD remains the classification even when a particular
                        # record has no span; has_ground_truth then reports False
                        # and the query is excluded from metrics.
                        relevance_class=RelevanceClass.GOLD,
                        # Taken from the first *validated* span, so it cannot
                        # drift out of sync with the evidence list when an
                        # offset mismatch causes a span to be dropped.
                        answer=(
                            text[evidence[0].start : evidence[0].end] if evidence else None
                        ),
                        metadata={
                            "category": qa["id"].split("__")[-1],
                            "is_impossible": impossible,
                            "n_answer_spans": len(evidence),
                            "contract_title": title,
                        },
                    )
                )

        if limit_queries:
            queries = queries[:limit_queries]
            kept_docs = {q.scope_doc_id for q in queries}
            documents = [d for d in documents if d.doc_id in kept_docs]

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": sum(1 for q in queries if q.has_ground_truth),
            "queries_is_impossible": n_impossible,
            "answer_span_mismatches": n_span_mismatch,
        }
        skips.report(log, kept=len(queries))
        log.info(
            "cuad/%s: %d contracts, %d queries, %d with gold spans (%d is_impossible)",
            split, len(documents), len(queries),
            stats["queries_with_evidence"], n_impossible,
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)
