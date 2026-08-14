"""Retriever interface.

A retriever is built over an ordered list of chunks and answers queries with
ranked chunk ids. Two properties are required of every implementation:

  * results are addressed by stable chunk id, never by text or list position,
  * ``search`` has no side effects on the retriever or on any object it
    returns, so the same index can serve BM25, dense and hybrid runs without
    one mutating another's results.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence, Set

from core.types import Chunk, RetrievedChunk


class Retriever(abc.ABC):
    """Ranks corpus chunks against a query string."""

    name: str = "retriever"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.chunk_ids: List[str] = [c.chunk_id for c in self.chunks]
        self._position = {cid: i for i, cid in enumerate(self.chunk_ids)}
        if len(self._position) != len(self.chunk_ids):
            raise ValueError("duplicate chunk ids in corpus")

    def __len__(self) -> int:
        return len(self.chunks)

    @abc.abstractmethod
    def search(
        self, query: str, top_k: int, allowed: Optional[Set[str]] = None
    ) -> List[RetrievedChunk]:
        """Return up to ``top_k`` results, best first, ranks starting at 1.

        ``allowed`` restricts the search to a subset of chunk ids, which is how
        document-scoped datasets (CUAD, QASPER) confine a query to its own
        document without building one index per document.
        """

    def _rank(self, scored: List[tuple[int, float]], top_k: int) -> List[RetrievedChunk]:
        """Sort scored (index, score) pairs into a stable ranking.

        Ties break on corpus position so that a run is reproducible and no
        retriever gains an advantage from arbitrary ordering.
        """
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [
            RetrievedChunk(chunk_id=self.chunk_ids[index], rank=rank, score=float(score))
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]
