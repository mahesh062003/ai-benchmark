"""Hybrid retriever: BM25 + FAISS fused with Reciprocal Rank Fusion."""

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieval.base import RetrievalResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.rrf import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 and FAISS results with RRF.

    Parameters
    ----------
    bm25_path : str | Path
        Path to the BM25 .pkl index.
    faiss_index_path : str | Path
        Path to the FAISS .index file.
    metadata_path : str | Path
        Path to the FAISS metadata .pkl file.
    model : SentenceTransformer | None
        Shared embedding model forwarded to FAISSRetriever.
    model_name : str
        Hugging Face model ID used if model is None.
    rrf_k : int
        RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        bm25_path:        str | Path,
        faiss_index_path: str | Path,
        metadata_path:    str | Path,
        model:            SentenceTransformer | None = None,
        model_name:       str = "sentence-transformers/all-mpnet-base-v2",
        rrf_k:            int = 60,
    ) -> None:
        self.bm25  = BM25Retriever(bm25_path)
        self.faiss = FAISSRetriever(
            faiss_index_path,
            metadata_path,
            model=model,
            model_name=model_name,
        )
        self.rrf = ReciprocalRankFusion(k=rrf_k)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return top_k fused results."""
        bm25_results  = self.bm25.search(query,  top_k=top_k)
        faiss_results = self.faiss.search(query, top_k=top_k)
        fused         = self.rrf.fuse(bm25_results, faiss_results)
        return fused[:top_k]
