"""
Dataset-specific ground-truth (relevance) mapping for retrieval evaluation.

This module answers the question:
    "Given a benchmark Sample and the set of corpus Chunks, which chunk_ids
     are genuinely relevant to this query?"

Relevance definitions
---------------------
CUAD
    A chunk is relevant if it contains the gold answer text as a substring
    (case-insensitive).  Only answerable QAs have valid ground truth.

PubMedQA
    All chunks that belong to the same source document (same doc_id) are
    relevant — the context sentences were curated specifically for that question.

QASPER
    A chunk is relevant if its text contains any evidence passage as a
    substring, or if the chunk text is fully contained within an evidence
    passage.  (Evidence passages can be longer or shorter than a chunk.)

SciQ
    The single support passage for each question is the relevant document.
    Ground truth = all chunks produced from that question's CorpusDocument.

CaseHOLD
    NOT VALID — this is a classification task.  Returns None.

MedQA
    NOT VALID — no external retrieval corpus.  Returns None.

Return values
-------------
list[str]  — the relevant chunk_ids (may be empty list if no match is found
              despite has_retrieval_gt=True — this can happen with very short
              answers not preserved by the chunker).
None       — ground truth is not applicable for this dataset.
"""

import logging
from typing import Optional

from loaders.base import Sample
from preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


# ── Dataset dispatch ──────────────────────────────────────────────────────────

def get_relevant_chunk_ids(
    sample: Sample,
    corpus_chunks: list[Chunk],
) -> Optional[list[str]]:
    """
    Return relevant chunk_ids for sample, or None if not applicable.

    Parameters
    ----------
    sample : Sample
        The benchmark QA sample being evaluated.
    corpus_chunks : list[Chunk]
        All chunks in the dataset-specific corpus index.

    Returns
    -------
    list[str] | None
        Relevant chunk IDs, or None if retrieval metrics are not applicable.
    """
    if not sample["has_retrieval_gt"]:
        return None

    dataset = sample["dataset"]

    if dataset == "CUAD":
        return _cuad_relevant(sample, corpus_chunks)
    if dataset == "PubMedQA":
        return _pubmedqa_relevant(sample, corpus_chunks)
    if dataset == "QASPER":
        return _qasper_relevant(sample, corpus_chunks)
    if dataset == "SciQ":
        return _sciq_relevant(sample, corpus_chunks)

    # Datasets without valid retrieval ground truth
    return None


# ── Per-dataset implementations ───────────────────────────────────────────────

def _cuad_relevant(sample: Sample, chunks: list[Chunk]) -> list[str]:
    """
    CUAD: a chunk is relevant if it contains any gold answer span.

    The answer text is the exact answer span from the SQuAD-style annotation.
    Matching is case-insensitive to tolerate OCR/formatting differences.
    """
    answers: list[str] = sample["evidence"]  # same as sample["answer"] for CUAD
    if not answers:
        return []

    # Only search chunks from the same source document (same paragraph)
    source_doc_id = sample.get("source_doc_id")
    candidate_chunks = (
        [c for c in chunks if c["doc_id"] == source_doc_id]
        if source_doc_id
        else chunks
    )

    relevant: list[str] = []
    for chunk in candidate_chunks:
        chunk_lower = chunk["text"].lower()
        for ans in answers:
            ans_clean = ans.strip().lower()
            if ans_clean and ans_clean in chunk_lower:
                relevant.append(chunk["chunk_id"])
                break   # count this chunk once even if multiple answers match

    if not relevant:
        # Emit a debug warning — could indicate chunker is discarding the answer span
        logger.debug(
            "CUAD: no relevant chunks found for sample_id=%s (answers=%s)",
            sample["sample_id"], answers[:1],
        )

    return relevant


def _pubmedqa_relevant(sample: Sample, chunks: list[Chunk]) -> list[str]:
    """
    PubMedQA: all chunks from the same source document are relevant.

    The context sentences were curated from PubMed for this specific question,
    so every sentence/chunk is considered relevant background evidence.
    """
    source_doc_id = sample.get("source_doc_id")
    if not source_doc_id:
        return []
    return [c["chunk_id"] for c in chunks if c["doc_id"] == source_doc_id]


def _qasper_relevant(sample: Sample, chunks: list[Chunk]) -> list[str]:
    """
    QASPER: a chunk is relevant if it contains (or is contained in) any
    gold evidence passage.

    Evidence passages are verbatim text spans from the paper.
    A chunk may cover part of an evidence passage, or one chunk may contain
    a short evidence string.
    """
    evidence_passages: list[str] = sample["evidence"]
    if not evidence_passages:
        return []

    # Only examine chunks from the same paper
    source_doc_id = sample.get("source_doc_id")
    candidate_chunks = (
        [c for c in chunks if c["doc_id"] == source_doc_id]
        if source_doc_id
        else chunks
    )

    relevant: list[str] = []
    for chunk in candidate_chunks:
        chunk_lower = chunk["text"].strip().lower()
        for evidence in evidence_passages:
            ev_lower = evidence.strip().lower()
            if not ev_lower:
                continue
            # Either the evidence contains the chunk, or the chunk contains evidence
            if ev_lower in chunk_lower or chunk_lower in ev_lower:
                relevant.append(chunk["chunk_id"])
                break

    if not relevant:
        logger.debug(
            "QASPER: no evidence-overlapping chunks for sample_id=%s",
            sample["sample_id"],
        )

    return relevant


def _sciq_relevant(sample: Sample, chunks: list[Chunk]) -> list[str]:
    """
    SciQ: the support passage is the only relevant document.

    Ground truth = all chunks produced from that question's CorpusDocument
    (identified by source_doc_id).
    """
    source_doc_id = sample.get("source_doc_id")
    if not source_doc_id:
        return []
    return [c["chunk_id"] for c in chunks if c["doc_id"] == source_doc_id]


# ── Corpus index helper ────────────────────────────────────────────────────────

def build_chunk_index(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """
    Group chunks by doc_id for fast lookup.

    Returns a dict mapping doc_id → list of chunks with that doc_id.
    Used by the benchmark runner to avoid O(n) scans per sample.
    """
    from collections import defaultdict
    index: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        index[chunk["doc_id"]].append(chunk)
    return dict(index)
