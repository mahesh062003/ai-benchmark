"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) combines rankings using positions only:

    score(d) = sum over rankers r of  1 / (k + rank_r(d))

with ranks 1-based and ``k`` a smoothing constant, 60 by convention and
configurable here. Documents missing from a ranker's list contribute nothing
from that ranker.

Rank-based fusion is the right choice for this comparison because BM25 scores
are unbounded and corpus-dependent while cosine similarities sit in [-1, 1];
any score-level combination would need a normalisation step that silently
becomes a tuned hyperparameter and confounds the strategy comparison.

Each base ranker is consulted to ``candidate_depth`` (default 100), deeper than
the reported top_k, so fusion can promote a document that neither ranker placed
in its own head -- which is the entire point of hybrid retrieval.

The fusion never mutates the RetrievedChunk objects returned by its base
retrievers; it reads their ranks and emits new objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from core.types import RetrievedChunk
from .base import Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    top_k: int,
    k: int = 60,
) -> List[RetrievedChunk]:
    """Fuse ranked lists. Ties break on chunk id for reproducibility."""
    if k <= 0:
        raise ValueError("rrf_k must be positive")

    fused: Dict[str, float] = {}
    for ranking in rankings:
        seen: Set[str] = set()
        for item in ranking:
            # Guard against a malformed ranker returning a duplicate: the first
            # (best) occurrence is the one that counts.
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (k + item.rank)

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        RetrievedChunk(chunk_id=cid, rank=rank, score=score)
        for rank, (cid, score) in enumerate(ordered[:top_k], start=1)
    ]


class HybridRetriever(Retriever):
    """BM25 + dense, fused with RRF."""

    name = "hybrid"

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        rrf_k: int = 60,
        candidate_depth: int = 100,
    ) -> None:
        if sparse.chunk_ids != dense.chunk_ids:
            raise ValueError(
                "hybrid retrieval requires both retrievers to be built over the "
                "same corpus in the same order"
            )
        super().__init__(sparse.chunks)
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def search(
        self, query: str, top_k: int, allowed: Optional[Set[str]] = None
    ) -> List[RetrievedChunk]:
        depth = max(self.candidate_depth, top_k)
        sparse_hits = self.sparse.search(query, depth, allowed=allowed)
        dense_hits = self.dense.search(query, depth, allowed=allowed)
        return reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k, k=self.rrf_k)
