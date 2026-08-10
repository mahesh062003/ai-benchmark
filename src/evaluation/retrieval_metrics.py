"""
Retrieval evaluation metrics.

All metrics operate on plain lists of strings (IDs), not on result objects,
so they are independent of the retrieval implementation.

Implementations
---------------
recall_at_k     : fraction of relevant items found in the top-k retrieved.
mean_reciprocal_rank : reciprocal of the rank of the first relevant item.
ndcg_at_k       : normalised Discounted Cumulative Gain at cutoff k.

All functions return 0.0 for degenerate inputs (empty retrieved / relevant sets).
They return None only when called with has_relevant_gt=False, but that guard
is applied at the caller level (benchmark runner), not here.
"""

import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids:  list[str],
    k:             int,
) -> float:
    """
    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|

    Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k  = set(retrieved_ids[:k])
    relset = set(relevant_ids)
    return len(top_k & relset) / len(relset)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids:  list[str],
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_item

    Returns 0.0 if no relevant item appears in retrieved_ids.
    """
    relset = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relset:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids:  list[str],
    k:             int,
) -> float:
    """
    nDCG@K with binary relevance.

    DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i + 1)
    IDCG@K = Σ_{i=1}^{min(|relevant|, K)}  1 / log2(i + 1)
    nDCG@K = DCG@K / IDCG@K

    Returns 0.0 if relevant_ids is empty or IDCG is 0.
    """
    relset = set(relevant_ids)

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relset
    )

    ideal_hits = min(len(relset), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0
