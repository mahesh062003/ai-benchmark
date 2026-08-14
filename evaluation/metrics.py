"""Retrieval metrics: Recall@K, MRR, nDCG@K.

Every function here returns ``Optional[float]`` and returns ``None`` when the
query has no relevance judgement. That is not a defensive habit, it is the
central invariant of this project:

    0.0   the ground truth existed and the retriever did not find it
    None  there was no ground truth to score against

Collapsing the second into the first would understate every retriever on
datasets where evidence is merely absent, and would let MedQA -- which has no
relevance information at all -- appear in the results table as a uniform row of
zeros. Aggregation therefore skips None rather than treating it as zero, and
reports how many queries were skipped.

Binary relevance is used throughout: the datasets provide relevant / not
relevant, with no graded judgements, so nDCG uses gain 1 for relevant items and
its ideal ranking places all relevant items first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from core.types import RetrievedChunk


def _ranked_ids(retrieved: Sequence[RetrievedChunk | str]) -> List[str]:
    """Accept either RetrievedChunk objects or bare ids, in rank order."""
    return [r if isinstance(r, str) else r.chunk_id for r in retrieved]


def recall_at_k(
    retrieved: Sequence[RetrievedChunk | str],
    relevant: Set[str],
    k: int,
) -> Optional[float]:
    """Fraction of relevant items appearing in the top k.

    With multiple relevant items the denominator is the number of relevant
    items, so a query with 5 relevant chunks and a top-10 cutoff cannot exceed
    the 10-item ceiling artificially -- it is scored against what is reachable.
    """
    if not relevant:
        return None
    top = set(_ranked_ids(retrieved)[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(
    retrieved: Sequence[RetrievedChunk | str],
    relevant: Set[str],
    k: Optional[int] = None,
) -> Optional[float]:
    """1 / rank of the first relevant item, else 0.0 if none is retrieved.

    Zero here is a genuine measurement: ground truth existed and nothing
    relevant was returned within the cutoff.
    """
    if not relevant:
        return None
    ids = _ranked_ids(retrieved)
    if k is not None:
        ids = ids[:k]
    for rank, chunk_id in enumerate(ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[RetrievedChunk | str],
    relevant: Set[str],
    k: int,
) -> Optional[float]:
    """Normalised discounted cumulative gain with binary gains.

    DCG = sum over positions i (1-based) of rel_i / log2(i + 1).
    IDCG is the DCG of the best achievable ranking, which places
    min(len(relevant), k) relevant items first -- so a query with more relevant
    items than the cutoff can still reach 1.0.
    """
    if not relevant:
        return None
    ids = _ranked_ids(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(ids, start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return None
    return dcg / idcg


@dataclass
class QueryMetrics:
    """Per-query scores. ``None`` values mean unscoreable, never zero."""

    query_id: str
    scoreable: bool
    n_relevant: int
    recall: Dict[int, Optional[float]]
    ndcg: Dict[int, Optional[float]]
    mrr: Optional[float]
    reason: str = ""


def evaluate_query(
    query_id: str,
    retrieved: Sequence[RetrievedChunk | str],
    relevant: Set[str],
    ks: Sequence[int],
    available: bool = True,
    reason: str = "",
) -> QueryMetrics:
    """Score one query, or record why it cannot be scored."""
    if not available or not relevant:
        return QueryMetrics(
            query_id=query_id,
            scoreable=False,
            n_relevant=len(relevant),
            recall={k: None for k in ks},
            ndcg={k: None for k in ks},
            mrr=None,
            reason=reason or "no relevance judgement available",
        )
    return QueryMetrics(
        query_id=query_id,
        scoreable=True,
        n_relevant=len(relevant),
        recall={k: recall_at_k(retrieved, relevant, k) for k in ks},
        ndcg={k: ndcg_at_k(retrieved, relevant, k) for k in ks},
        mrr=reciprocal_rank(retrieved, relevant),
    )


@dataclass
class AggregateMetrics:
    """Mean scores over the scoreable queries of a run."""

    n_queries: int
    n_scoreable: int
    n_unscoreable: int
    recall: Dict[int, Optional[float]]
    ndcg: Dict[int, Optional[float]]
    mrr: Optional[float]

    @property
    def coverage(self) -> Optional[float]:
        """Share of queries that carried a relevance judgement.

        Reported alongside the metrics because a mean over 5% of queries is a
        different claim from a mean over 95% of them.
        """
        if self.n_queries == 0:
            return None
        return self.n_scoreable / self.n_queries

    def to_row(self) -> Dict[str, Optional[float]]:
        row: Dict[str, Optional[float]] = {"mrr": self.mrr}
        for k, v in sorted(self.recall.items()):
            row[f"recall@{k}"] = v
        for k, v in sorted(self.ndcg.items()):
            row[f"ndcg@{k}"] = v
        return row


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values; None if there are none.

    None values are *excluded*, never coerced to 0.0.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def aggregate(results: Sequence[QueryMetrics], ks: Sequence[int]) -> AggregateMetrics:
    scoreable = [r for r in results if r.scoreable]
    return AggregateMetrics(
        n_queries=len(results),
        n_scoreable=len(scoreable),
        n_unscoreable=len(results) - len(scoreable),
        recall={k: _mean(r.recall.get(k) for r in scoreable) for k in ks},
        ndcg={k: _mean(r.ndcg.get(k) for r in scoreable) for k in ks},
        mrr=_mean(r.mrr for r in scoreable),
    )
