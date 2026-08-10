"""
Entry point 1: Build BM25 and FAISS indexes for all datasets.

Usage (from the project root):
    python src/build_indexes.py

What this does
--------------
For each dataset in the registry:
  1. Load the retrieval corpus (unique documents, not per-QA duplicates).
  2. Chunk documents into fixed-size overlapping text units.
  3. Save chunks as JSON for inspection.
  4. Build and persist a BM25Okapi index.
  5. Build and persist a FAISS IndexFlatIP index (cosine similarity).

The embedding model is loaded ONCE and shared across all datasets.

Outputs (relative to project root):
    results/chunks/{dataset}_chunks.json
    results/indexes/{dataset}_bm25.pkl
    results/indexes/{dataset}_faiss.index
    results/indexes/{dataset}_metadata.pkl

Run this before run_benchmark.py.
"""

import logging
import sys
from pathlib import Path

# Make src/ importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR, INDEX_DIR, EMBEDDING_MODEL, DATASET_PATHS,
)
from benchmark.registry import DATASETS
from preprocessing.chunker import DocumentChunker, save_chunks
from indexing.bm25_index import BM25Indexer
from indexing.faiss_index import FAISSIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_all() -> None:
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Load embedding model ONCE and share across all FAISS indexes
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    shared_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    print("\n" + "=" * 70)
    print("AI BENCHMARK — INDEX BUILDING")
    print("=" * 70)

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n{'=' * 70}\n  Dataset: {name}\n{'=' * 70}")

        # 1. Load corpus
        logger.info("[%s] Loading corpus documents…", name)
        documents = ds["load_corpus"]()

        if not documents:
            logger.warning("[%s] No corpus documents — skipping index build.", name)
            continue

        # 2. Chunk
        chunks = chunker.chunk_documents(documents)
        print(f"  Documents : {len(documents)}")
        print(f"  Chunks    : {len(chunks)}")

        # 3. Save chunks
        chunk_path = CHUNKS_DIR / f"{name.lower()}_chunks.json"
        save_chunks(chunks, chunk_path)

        # 4. BM25
        bm25 = BM25Indexer()
        bm25.build(chunks)
        bm25.save(INDEX_DIR / f"{name.lower()}_bm25.pkl")

        # 5. FAISS
        faiss_idx = FAISSIndexer(model=shared_model, model_name=EMBEDDING_MODEL)
        faiss_idx.build(chunks)
        faiss_idx.save(
            index_path    = INDEX_DIR / f"{name.lower()}_faiss.index",
            metadata_path = INDEX_DIR / f"{name.lower()}_metadata.pkl",
        )

        print(f"  ✓ {name} complete.")

    print("\n" + "=" * 70)
    print("ALL INDEXES BUILT SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    build_all()
