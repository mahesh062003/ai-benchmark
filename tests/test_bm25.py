"""Tests for BM25 indexer and retriever."""

import pytest
from retrieval.bm25_retriever import BM25Retriever
from retrieval.base import RetrievalResult


class TestBM25Retriever:
    def test_returns_retrieval_results(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("fox dog", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(x, RetrievalResult) for x in results)

    def test_top_k_respected(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("the", top_k=2)
        assert len(results) <= 2

    def test_ranks_sequential(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("quick fox", top_k=5)
        for i, res in enumerate(results, start=1):
            assert res.rank == i

    def test_scores_descending(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("quick fox", top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_retrieved_first(self, bm25_index_path):
        # "contract parties" should rank the contract chunk highest
        r = BM25Retriever(bm25_index_path)
        results = r.search("contract signed parties", top_k=5)
        assert any("contract" in res.text.lower() for res in results[:2])

    def test_chunk_ids_are_strings(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("learning", top_k=3)
        assert all(isinstance(res.chunk_id, str) for res in results)

    def test_result_fields_populated(self, bm25_index_path):
        r = BM25Retriever(bm25_index_path)
        results = r.search("natural language", top_k=1)
        assert len(results) == 1
        res = results[0]
        assert res.chunk_id
        assert res.dataset == "TestDS"
        assert res.text


class TestBM25Indexer:
    def test_build_and_save(self, tmp_path, synthetic_chunks):
        from indexing.bm25_index import BM25Indexer
        indexer = BM25Indexer()
        indexer.build(synthetic_chunks)
        out = tmp_path / "bm25.pkl"
        indexer.save(out)
        assert out.exists()

    def test_saved_index_loadable(self, tmp_path, synthetic_chunks):
        from indexing.bm25_index import BM25Indexer
        indexer = BM25Indexer()
        indexer.build(synthetic_chunks)
        out = tmp_path / "bm25.pkl"
        indexer.save(out)

        retriever = BM25Retriever(out)
        results = retriever.search("fox", top_k=3)
        assert len(results) > 0

    def test_empty_chunks_raises(self):
        from indexing.bm25_index import BM25Indexer
        indexer = BM25Indexer()
        with pytest.raises(ValueError):
            indexer.build([])

    def test_save_before_build_raises(self, tmp_path):
        from indexing.bm25_index import BM25Indexer
        indexer = BM25Indexer()
        with pytest.raises(RuntimeError):
            indexer.save(tmp_path / "x.pkl")
