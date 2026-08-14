"""Dense vector retrieval with FAISS.

Index choice: ``IndexFlatIP`` -- exhaustive inner-product search over
L2-normalised vectors, which is exactly cosine similarity.

The justification is methodological rather than one of scale. Approximate
indexes (IVF, HNSW) trade recall for speed, and that lost recall would appear
in the results table as a property of "dense retrieval" when it is really a
property of the approximation. Exact search removes that confound, so any
difference measured between BM25, dense and hybrid is attributable to the
retrieval strategy. The corpora make this affordable: the largest here is the
MedQA textbook set at roughly 10^5 chunks, where a flat 384-dimensional index
is well under a gigabyte and searches in milliseconds. A project scaling to
millions of chunks would need to revisit this and report the recall cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Set

import faiss
import numpy as np

from benchmark.indexer import Embedder
from core.logging_setup import get_logger
from core.types import Chunk, RetrievedChunk
from .base import Retriever

log = get_logger("retrieval.dense")


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        index: Optional[faiss.Index] = None,
        show_progress: bool = True,
    ) -> None:
        super().__init__(chunks)
        self.embedder = embedder
        if index is not None:
            self.index = index
        else:
            vectors = embedder.encode(
                [c.text for c in self.chunks], progress=show_progress
            )
            self.index = faiss.IndexFlatIP(embedder.dimension)
            if len(vectors):
                self.index.add(vectors)
            log.info(
                "built FAISS IndexFlatIP over %d chunks (dim=%d)",
                self.index.ntotal, embedder.dimension,
            )
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"FAISS index holds {self.index.ntotal} vectors but the corpus has "
                f"{len(self.chunks)} chunks"
            )

    def search(
        self, query: str, top_k: int, allowed: Optional[Set[str]] = None
    ) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        vector = self.embedder.encode_query(query).reshape(1, -1)

        if allowed is None:
            depth = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(vector, depth)
            pairs = [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i != -1
            ]
            return self._rank(pairs, top_k)

        # Restricted search (document-scoped datasets). An IDSelector keeps this
        # exact rather than over-fetching and hoping enough survive filtering.
        wanted = np.array(
            [i for i, cid in enumerate(self.chunk_ids) if cid in allowed], dtype=np.int64
        )
        if wanted.size == 0:
            return []
        selector = faiss.IDSelectorArray(wanted.size, faiss.swig_ptr(wanted))
        params = faiss.SearchParameters()
        params.sel = selector
        depth = min(top_k, int(wanted.size))
        scores, indices = self.index.search(vector, depth, params=params)
        pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
        return self._rank(pairs, top_k)

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.write_text(
            json.dumps(
                {
                    "chunk_ids": self.chunk_ids,
                    "model": self.embedder.config.model_name,
                    "dimension": self.embedder.dimension,
                    "normalized": self.embedder.config.normalize,
                    "index_type": "IndexFlatIP",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("saved FAISS index -> %s", path)

    @classmethod
    def load(
        cls, path: Path, chunks: Sequence[Chunk], embedder: Embedder
    ) -> "DenseRetriever":
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        ids = [c.chunk_id for c in chunks]
        if meta["chunk_ids"] != ids:
            raise ValueError(
                f"FAISS index at {path} was built over a different corpus. Rebuild it."
            )
        if meta["model"] != embedder.config.model_name:
            raise ValueError(
                f"FAISS index at {path} was built with embedding model "
                f"{meta['model']!r} but the current configuration uses "
                f"{embedder.config.model_name!r}. Rebuild the index or restore the model."
            )
        index = faiss.read_index(str(path))
        return cls(chunks, embedder, index=index)
