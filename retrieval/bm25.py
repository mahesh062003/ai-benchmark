"""BM25 sparse retrieval (Okapi BM25 via ``rank_bm25``).

Tokenisation is lowercase word-character n-grams with a length-1 filter. It is
deliberately simple and, importantly, identical for indexing and querying --
a mismatch between the two is a classic silent cause of depressed BM25 scores.
No stemming or stopword removal is applied: both would interact differently
with legal, medical and scientific vocabulary and would confound the
cross-domain comparison this project exists to make.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set

from rank_bm25 import BM25Okapi

from core.logging_setup import get_logger
from core.types import Chunk, RetrievedChunk
from .base import Retriever

log = get_logger("retrieval.bm25")

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        _index: Optional[BM25Okapi] = None,
    ) -> None:
        super().__init__(chunks)
        self.k1 = k1
        self.b = b
        if _index is not None:
            self.index = _index
        else:
            corpus = [tokenize(c.text) for c in self.chunks]
            # rank_bm25 divides by the average document length, which is
            # zero if every document tokenises to nothing. A sentinel
            # token keeps the index constructible; such a corpus retrieves
            # nothing meaningful either way, but must not crash the run.
            if corpus and not any(corpus):
                corpus = [tokens or ["__empty__"] for tokens in corpus]
            self.index = BM25Okapi(corpus, k1=k1, b=b)
            log.info("built BM25 index over %d chunks (k1=%.2f, b=%.2f)", len(corpus), k1, b)

    def search(
        self, query: str, top_k: int, allowed: Optional[Set[str]] = None
    ) -> List[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        if allowed is None:
            scored = list(enumerate(scores))
        else:
            scored = [
                (i, scores[i]) for i, cid in enumerate(self.chunk_ids) if cid in allowed
            ]
        return self._rank([(i, float(s)) for i, s in scored], top_k)

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {"index": self.index, "chunk_ids": self.chunk_ids, "k1": self.k1, "b": self.b},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.info("saved BM25 index -> %s", path)

    @classmethod
    def load(cls, path: Path, chunks: Sequence[Chunk]) -> "BM25Retriever":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        ids = [c.chunk_id for c in chunks]
        if payload["chunk_ids"] != ids:
            raise ValueError(
                f"BM25 index at {path} was built over a different corpus "
                f"({len(payload['chunk_ids'])} chunks vs {len(ids)}). Rebuild the index."
            )
        return cls(chunks, k1=payload["k1"], b=payload["b"], _index=payload["index"])
