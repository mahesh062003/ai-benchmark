"""
BM25 indexer using BM25Okapi (rank_bm25).

Tokenisation: lowercase, alphanumeric word boundary split (same as retriever).
Serialisation: pickle containing the BM25Okapi object and the chunk list.
"""

import logging
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Indexer:
    """Build and persist a BM25 index over a list of Chunks."""

    def __init__(self) -> None:
        self.index:     BM25Okapi | None = None
        self.chunks:    list[Chunk] | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        logger.info("Building BM25 index over %d chunks…", len(chunks))
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.index  = BM25Okapi(tokenized)
        self.chunks = chunks
        logger.info("BM25 index built.")

    def save(self, path: str | Path) -> None:
        if self.index is None or self.chunks is None:
            raise RuntimeError("Call build() before save().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.index, "chunks": self.chunks}, f)
        logger.info("BM25 index saved → %s", path)
