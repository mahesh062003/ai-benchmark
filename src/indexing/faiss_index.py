"""
FAISS dense retrieval indexer.

Index type: IndexFlatIP (exact inner-product / cosine search after L2 normalisation).
Embedding model: sentence-transformers/all-mpnet-base-v2 (768 dimensions).

The model is loaded ONCE and should be passed in via the `model` parameter to
avoid reloading 768 MB of weights per dataset.  The build() entry point will
load the model internally only when no shared instance is provided.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """
    Build and persist a FAISS inner-product index over a list of Chunks.

    Parameters
    ----------
    model : SentenceTransformer | None
        A pre-loaded embedding model.  If None, the model named by `model_name`
        is loaded when build() is called.
    model_name : str
        Hugging Face model ID used when no shared model is provided.
    """

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self._model_name = model_name
        self._model      = model      # may remain None until build() is called
        self.index:  faiss.Index | None = None
        self.chunks: list[Chunk] | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunk list.")

        logger.info("Encoding %d chunks with %s…", len(chunks), self._model_name)
        texts = [c["text"] for c in chunks]
        embeddings: np.ndarray = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        logger.info("FAISS index built: %d vectors, dim=%d.", len(chunks), dim)

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None or self.chunks is None:
            raise RuntimeError("Call build() before save().")

        index_path    = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("FAISS index saved → %s", index_path)
        logger.info("FAISS metadata saved → %s", metadata_path)
