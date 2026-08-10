"""
Document chunker.

Converts CorpusDocuments into retrieval-ready Chunk dicts.

Each CorpusDocument is chunked exactly once.  Deduplication of documents
(e.g. CUAD paragraphs shared across multiple QA pairs, QASPER papers
referenced by multiple questions) is the responsibility of the dataset loaders
— by the time documents reach this module they should already be unique.

Chunk ID format: "{doc_id}__chunk{i:04d}"
  e.g. "CUAD__ContractTitle__para3__chunk0007"

This format guarantees:
  - Globally unique within a dataset corpus
  - Traceable back to the source document (doc_id)
  - Stable across runs (deterministic from content + position)
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.base import CorpusDocument

logger = logging.getLogger(__name__)


class Chunk(dict):
    """
    A retrieval unit produced from one CorpusDocument.

    Keys:
        chunk_id   — globally unique, stable ID
        doc_id     — parent CorpusDocument ID
        dataset    — dataset name
        domain     — domain label
        text       — chunk text
        chunk_idx  — position within the document (0-based)
        metadata   — inherited from the parent CorpusDocument
    """


def _make_chunk(
    doc: CorpusDocument,
    text: str,
    chunk_idx: int,
) -> Chunk:
    chunk: dict[str, Any] = {
        "chunk_id":  f"{doc['doc_id']}__chunk{chunk_idx:04d}",
        "doc_id":    doc["doc_id"],
        "dataset":   doc["dataset"],
        "domain":    doc["domain"],
        "text":      text,
        "chunk_idx": chunk_idx,
        "metadata":  dict(doc["metadata"]),
    }
    return chunk  # type: ignore[return-value]


class DocumentChunker:
    """
    Splits CorpusDocuments into fixed-size overlapping text chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk (passed to LangChain splitter).
    chunk_overlap : int
        Number of characters of overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: CorpusDocument) -> list[Chunk]:
        """Chunk a single CorpusDocument.  Returns [] for empty documents."""
        text = doc["text"].strip()
        if not text:
            return []

        pieces = self._splitter.split_text(text)
        return [_make_chunk(doc, piece, i) for i, piece in enumerate(pieces)]

    def chunk_documents(self, documents: list[CorpusDocument]) -> list[Chunk]:
        """
        Chunk all documents in the list.

        Logs a warning for any document that produces zero chunks.
        Returns the flat list of all chunks across all documents.
        """
        all_chunks: list[Chunk] = []
        empty_docs = 0

        for doc in documents:
            chunks = self.chunk_document(doc)
            if not chunks:
                empty_docs += 1
                logger.debug("Empty document skipped: %s", doc["doc_id"])
            all_chunks.extend(chunks)

        if empty_docs:
            logger.warning(
                "%d empty documents produced no chunks (dataset=%s)",
                empty_docs,
                documents[0]["dataset"] if documents else "unknown",
            )

        logger.info(
            "Chunked %d documents → %d chunks (chunk_size=%d, overlap=%d)",
            len(documents), len(all_chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return all_chunks


# ── Chunk serialisation helpers ───────────────────────────────────────────────

def save_chunks(chunks: list[Chunk], output_path: str | Path) -> None:
    """Write chunks to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks → %s", len(chunks), output_path)


def load_chunks(input_path: str | Path) -> list[Chunk]:
    """Load chunks from a JSON file produced by save_chunks()."""
    input_path = Path(input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d chunks ← %s", len(data), input_path)
    return data
