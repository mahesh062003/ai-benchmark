"""FAISS dense retriever — loads a pre-built FAISS index from disk."""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    Search a FAISS inner-product index serialised by FAISSIndexer.

    Parameters
    ----------
    index_path : str | Path
        Path to the .index file produced by FAISSIndexer.save().
    metadata_path : str | Path
        Path to the companion _metadata.pkl file.
    model : SentenceTransformer | None
        Shared pre-loaded embedding model.  If None, model_name is loaded.
    model_name : str
        Hugging Face model ID used when model is None.
    """

    def __init__(
        self,
        index_path:    str | Path,
        metadata_path: str | Path,
        model:         SentenceTransformer | None = None,
        model_name:    str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        index_path    = Path(index_path)
        metadata_path = Path(metadata_path)

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

        self._model_name = model_name
        self._model      = model

        logger.info(
            "FAISS index loaded: %d vectors ← %s",
            self.index.ntotal, index_path,
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return up to top_k results ranked by cosine similarity (descending)."""
        embedding: np.ndarray = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            embedding.astype(np.float32),
            top_k,
        )

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                # FAISS returns -1 for padding when fewer than top_k results exist
                continue
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
