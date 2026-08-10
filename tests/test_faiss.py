"""Tests for FAISS indexer and retriever."""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from retrieval.faiss_retriever import FAISSRetriever
from retrieval.base import RetrievalResult


class TestFAISSRetriever:
    def test_returns_retrieval_results(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        results = r.search("test query", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(x, RetrievalResult) for x in results)

    def test_top_k_respected(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        results = r.search("test query", top_k=2)
        assert len(results) <= 2

    def test_ranks_sequential(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        results = r.search("fox", top_k=5)
        for i, res in enumerate(results, start=1):
            assert res.rank == i

    def test_scores_descending(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        results = r.search("natural language", top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_ids_are_strings(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        results = r.search("query", top_k=3)
        assert all(isinstance(res.chunk_id, str) for res in results)

    def test_uses_shared_model_not_loading_new_one(self, faiss_index_paths):
        idx_path, meta_path, mock_model = faiss_index_paths
        r = FAISSRetriever(idx_path, meta_path, model=mock_model)
        r.search("test", top_k=1)
        # encode should have been called on the mock, not on a freshly loaded model
        assert mock_model.encode.called


class TestFAISSIndexer:
    def test_build_and_save(self, tmp_path, synthetic_chunks):
        from unittest.mock import MagicMock
        from indexing.faiss_index import FAISSIndexer

        mock_model = MagicMock()
        dim = 8
        rng = np.random.default_rng(0)
        embeddings = rng.random((len(synthetic_chunks), dim)).astype(np.float32)
        mock_model.encode.return_value = embeddings

        idx = FAISSIndexer(model=mock_model)
        idx.build(synthetic_chunks)
        idx.save(
            index_path=tmp_path / "test.index",
            metadata_path=tmp_path / "test_meta.pkl",
        )
        assert (tmp_path / "test.index").exists()
        assert (tmp_path / "test_meta.pkl").exists()

    def test_empty_chunks_raises(self):
        from indexing.faiss_index import FAISSIndexer
        idx = FAISSIndexer()
        with pytest.raises(ValueError):
            idx.build([])
