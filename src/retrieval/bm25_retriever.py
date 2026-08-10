"""BM25 retriever — loads a pre-built BM25Okapi index from disk."""

import logging
import pickle
import re
from pathlib import Path

from retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    Search a BM25Okapi index serialised by BM25Indexer.

    Parameters
    ----------
    index_path : str | Path
        Path to the .pkl file produced by BM25Indexer.save().
    """

    def __init__(self, index_path: str | Path) -> None:
        index_path = Path(index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25   = data["bm25"]
        self.chunks = data["chunks"]
        logger.info("BM25 index loaded: %d chunks ← %s", len(self.chunks), index_path)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return up to top_k results ranked by BM25 score (descending)."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk.get("doc_id", chunk["chunk_id"]),
                dataset=chunk["dataset"],
                domain=chunk["domain"],
                text=chunk["text"],
                score=float(score),
                rank=rank,
            ))
        return results
