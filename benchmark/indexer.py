"""Sentence embedding, loaded once per process.

The model is expensive to construct, so ``get_embedder`` memoises one instance
per (model, device). Every dataset and every retriever in a run shares it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.settings import EmbeddingConfig
from core.logging_setup import get_logger

log = get_logger("embedding")

_CACHE: Dict[Tuple[str, Optional[str]], "Embedder"] = {}


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return "cpu"


class Embedder:
    """Thin wrapper over a SentenceTransformer.

    Embeddings are L2-normalised by default so that a FAISS inner-product index
    computes cosine similarity exactly.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = resolve_device(config.device)
        log.info("loading embedding model %s on %s", config.model_name, self.device)
        self.model = SentenceTransformer(config.model_name, device=self.device)
        # Renamed in sentence-transformers 5.x; the old name still works but
        # warns, so prefer the new one and fall back for older versions.
        get_dimension = getattr(
            self.model, "get_embedding_dimension", None
        ) or self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dimension())
        log.info("embedding model ready (dimension=%d)", self.dimension)

    def encode(
        self, texts: Sequence[str], batch_size: Optional[int] = None, progress: bool = False
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size or self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder(config: EmbeddingConfig) -> Embedder:
    """Return the shared embedder for this configuration, constructing it once."""
    key = (config.model_name, resolve_device(config.device))
    embedder = _CACHE.get(key)
    if embedder is None:
        embedder = Embedder(config)
        _CACHE[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Drop cached models. Intended for tests."""
    _CACHE.clear()
