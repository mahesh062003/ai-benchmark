"""Chunking, and the single rule that decides chunk relevance.

Every chunk records the character range of its parent document that it came
from. That is the whole trick: dataset evidence is normalised into character
spans (see ``core.types.EvidenceSpan``), so mapping evidence onto chunks is
an interval-overlap test rather than a fuzzy text search.

Two strategies are provided, chosen per dataset:

``fixed``    sliding window over whitespace tokens, with overlap. Used for long
             continuous text (CUAD contracts, MedQA textbooks).
``passage``  cut on the document's own declared ``segments`` -- QASPER
             paragraphs, PubMedQA abstract sections -- so a retrieval unit
             equals the unit the dataset's annotators judged. A document
             without declared segments (a SciQ support, a CaseHOLD holding) is
             one segment. A segment longer than ``max_tokens`` is sub-split by
             sliding window rather than truncated.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from core.settings import ChunkConfig
from core.ids import chunk_id as make_chunk_id
from core.types import Chunk, EvidenceSpan, SourceDocument

_TOKEN = re.compile(r"\S+")


def _token_spans(text: str) -> List[tuple[int, int]]:
    """Character spans of whitespace-delimited tokens."""
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _window_bounds(n_tokens: int, size: int, overlap: int) -> Iterable[tuple[int, int]]:
    """Token index windows ``[start, end)`` for a sliding window.

    Guards against the classic non-terminating case where overlap >= size.
    """
    if size <= 0:
        raise ValueError("chunk size_tokens must be positive")
    stride = size - overlap
    if stride <= 0:
        raise ValueError(
            f"overlap_tokens ({overlap}) must be smaller than size_tokens ({size})"
        )
    if n_tokens == 0:
        return
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        yield start, end
        if end == n_tokens:
            return
        start += stride


def _passage_bounds(
    document: SourceDocument,
    spans: List[tuple[int, int]],
    config: ChunkConfig,
) -> List[tuple[int, int]]:
    """Token windows that follow the document's declared segment boundaries."""
    limit = config.max_tokens
    sub_overlap = min(config.overlap_tokens, limit - 1)

    if not document.segments:
        segments = [(0, len(spans))]
    else:
        # Map each character segment onto the token indices it contains. A
        # token belongs to the segment its start character falls inside.
        segments = []
        for seg_start, seg_end in document.segments:
            first = next(
                (i for i, (s, _) in enumerate(spans) if s >= seg_start and s < seg_end), None
            )
            if first is None:
                continue
            last = first
            for i in range(first, len(spans)):
                if spans[i][0] >= seg_end:
                    break
                last = i
            segments.append((first, last + 1))
        if not segments:
            segments = [(0, len(spans))]

    bounds: List[tuple[int, int]] = []
    for t_start, t_end in segments:
        length = t_end - t_start
        if length <= limit:
            bounds.append((t_start, t_end))
            continue
        for w_start, w_end in _window_bounds(length, limit, sub_overlap):
            bounds.append((t_start + w_start, t_start + w_end))
    return bounds


def chunk_document(document: SourceDocument, config: ChunkConfig) -> List[Chunk]:
    """Split one source document into retrieval units."""
    text = document.text
    spans = _token_spans(text)
    if not spans:
        return []

    if config.strategy == "passage":
        bounds = _passage_bounds(document, spans, config)
    elif config.strategy == "fixed":
        bounds = list(_window_bounds(len(spans), config.size_tokens, config.overlap_tokens))
    else:
        raise ValueError(f"unknown chunk strategy: {config.strategy!r}")

    def build(char_start: int, char_end: int, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=make_chunk_id(document.doc_id, ordinal),
            doc_id=document.doc_id,
            dataset=document.dataset,
            split=document.split,
            text=text[char_start:char_end],
            ordinal=ordinal,
            char_start=char_start,
            char_end=char_end,
            metadata=dict(document.metadata),
        )

    chunks: List[Chunk] = []
    for t_start, t_end in bounds:
        char_start = spans[t_start][0]
        char_end = spans[t_end - 1][1]
        if len(text[char_start:char_end].strip()) < config.min_chunk_chars:
            continue
        chunks.append(build(char_start, char_end, len(chunks)))

    if not chunks:
        # min_chunk_chars exists to discard whitespace fragments of a sliding
        # window, not to delete documents. A short-but-real document (CaseHOLD
        # has holdings of 12 characters) must still enter the corpus: dropping
        # it would silently remove a document that is some query's gold answer,
        # turning a scoreable query into an unscoreable one.
        chunks.append(build(spans[0][0], spans[-1][1], 0))
    return chunks


def chunk_documents(
    documents: Sequence[SourceDocument], config: ChunkConfig
) -> List[Chunk]:
    out: List[Chunk] = []
    for doc in documents:
        out.extend(chunk_document(doc, config))
    return out


def relevant_chunk_ids(
    evidence: Sequence[EvidenceSpan],
    chunks: Sequence[Chunk],
    min_overlap_chars: int = 1,
) -> List[str]:
    """The one place in the codebase where a chunk is declared relevant.

    A chunk is relevant iff it shares at least ``min_overlap_chars`` characters
    with an evidence span from the same document. Nothing about retrieval
    output, answer strings, or model behaviour enters this decision, which is
    what keeps the ground truth non-fabricated.

    Returns ids in corpus order, de-duplicated. An empty result means the
    evidence did not land on any chunk; callers must treat that as *no ground
    truth* (NULL), not as a retrieval failure (zero).
    """
    if not evidence:
        return []
    by_doc: dict[str, List[EvidenceSpan]] = {}
    for span in evidence:
        by_doc.setdefault(span.doc_id, []).append(span)

    found: List[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for span in by_doc.get(chunk.doc_id, ()):
            if chunk.overlaps(span.start, span.end) >= min_overlap_chars:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    found.append(chunk.chunk_id)
                break
    return found
