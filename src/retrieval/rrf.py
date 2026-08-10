"""
Reciprocal Rank Fusion (RRF).

Fuses ranked result lists from multiple retrievers into a single ranked list.

Formula (Cormack et al., 2009):
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  i

where k=60 is a smoothing constant that dampens the impact of very high ranks.

Implementation notes:
  - Deduplication key: chunk_id (stable document identifier).
    Using text content as the key (as in the original code) is fragile because
    it can fail on whitespace differences and is O(n²) for long texts.
  - Result objects are immutable (frozen dataclass), so fusion produces NEW
    RetrievalResult objects rather than mutating the originals.
"""

from collections import defaultdict

from retrieval.base import RetrievalResult


class ReciprocalRankFusion:
    """
    Fuse two or more ranked result lists with RRF.

    Parameters
    ----------
    k : int
        Smoothing constant (default 60, standard value from the original paper).
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, *result_lists: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Return a new ranked list fusing all input lists.

        Parameters
        ----------
        *result_lists : list[RetrievalResult]
            Two or more result lists to fuse (order does not affect the output).

        Returns
        -------
        list[RetrievalResult]
            De-duplicated results sorted by descending RRF score,
            with ranks reassigned starting from 1.
        """
        scores:  dict[str, float]           = defaultdict(float)
        lookup:  dict[str, RetrievalResult] = {}

        for results in result_lists:
            for result in results:
                key = result.chunk_id             # stable, unique identifier
                scores[key] += 1.0 / (self.k + result.rank)
                # Keep whichever result we see first (any retriever's version is fine)
                if key not in lookup:
                    lookup[key] = result

        # Build fused list with new score/rank values (new objects, originals unchanged)
        fused = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                doc_id=lookup[chunk_id].doc_id,
                dataset=lookup[chunk_id].dataset,
                domain=lookup[chunk_id].domain,
                text=lookup[chunk_id].text,
                score=scores[chunk_id],
                rank=new_rank,
            )
            for new_rank, chunk_id in enumerate(fused, start=1)
        ]
