"""
Shared pytest fixtures and configuration.

All fixtures use small, self-contained synthetic data.
No real datasets, external models, or network access are required for any
unit test in this suite.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Tiny synthetic corpus ─────────────────────────────────────────────────────

SYNTHETIC_CHUNKS = [
    {
        "chunk_id": "DS__doc1__chunk0000",
        "doc_id":   "DS__doc1",
        "dataset":  "TestDS",
        "domain":   "Test",
        "text":     "The quick brown fox jumps over the lazy dog.",
        "chunk_idx": 0,
        "metadata": {"title": "Doc 1"},
    },
    {
        "chunk_id": "DS__doc1__chunk0001",
        "doc_id":   "DS__doc1",
        "dataset":  "TestDS",
        "domain":   "Test",
        "text":     "A quick fox is rarely lazy. Brown dogs sleep all day.",
        "chunk_idx": 1,
        "metadata": {"title": "Doc 1"},
    },
    {
        "chunk_id": "DS__doc2__chunk0000",
        "doc_id":   "DS__doc2",
        "dataset":  "TestDS",
        "domain":   "Test",
        "text":     "Machine learning models require large amounts of training data.",
        "chunk_idx": 0,
        "metadata": {"title": "Doc 2"},
    },
    {
        "chunk_id": "DS__doc3__chunk0000",
        "doc_id":   "DS__doc3",
        "dataset":  "TestDS",
        "domain":   "Test",
        "text":     "Natural language processing enables computers to understand text.",
        "chunk_idx": 0,
        "metadata": {"title": "Doc 3"},
    },
    {
        "chunk_id": "DS__doc4__chunk0000",
        "doc_id":   "DS__doc4",
        "dataset":  "TestDS",
        "domain":   "Test",
        "text":     "The contract was signed by both parties on the effective date.",
        "chunk_idx": 0,
        "metadata": {"title": "Doc 4"},
    },
]


@pytest.fixture
def synthetic_chunks():
    return list(SYNTHETIC_CHUNKS)


@pytest.fixture
def bm25_index_path(tmp_path, synthetic_chunks):
    """Build a tiny BM25 index and return its path."""
    from rank_bm25 import BM25Okapi
    import re

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    tokenized = [tokenize(c["text"]) for c in synthetic_chunks]
    bm25 = BM25Okapi(tokenized)

    path = tmp_path / "test_bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": synthetic_chunks}, f)
    return path


@pytest.fixture
def faiss_index_paths(tmp_path, synthetic_chunks):
    """Build a tiny FAISS index with mock embeddings and return (index_path, meta_path)."""
    import numpy as np
    faiss = pytest.importorskip("faiss")

    dim = 8  # very small for tests
    rng = np.random.default_rng(42)
    embeddings = rng.random((len(synthetic_chunks), dim)).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = tmp_path / "test_faiss.index"
    meta_path  = tmp_path / "test_metadata.pkl"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(synthetic_chunks, f)

    # Mock model that returns deterministic embeddings matching the index
    mock_model = MagicMock()
    def fake_encode(texts, **kwargs):
        arr = rng.random((len(texts), dim)).astype(np.float32)
        norms_enc = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= norms_enc
        return arr
    mock_model.encode.side_effect = fake_encode

    return index_path, meta_path, mock_model


@pytest.fixture
def tmp_db(tmp_path):
    """Return a BenchmarkDatabase backed by a temporary file."""
    from database.database import BenchmarkDatabase
    db = BenchmarkDatabase(db_path=tmp_path / "test_benchmark.db")
    yield db
    db.close()
