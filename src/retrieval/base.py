"""Shared result type for all retrievers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """
    One retrieved chunk returned by any retriever.

    frozen=True prevents in-place mutation (the original RRF bug was caused
    by mutating these objects across retrievers).
    """
    chunk_id: str
    doc_id:   str       # parent document ID (links back to CorpusDocument)
    dataset:  str
    domain:   str
    text:     str
    score:    float
    rank:     int
