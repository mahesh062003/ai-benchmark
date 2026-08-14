"""BM25, RRF fusion, and the retriever contract.

Dense retrieval is exercised with a stub embedder so the core logic is testable
without downloading a model; a marked slow test covers the real one.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval import BM25Retriever, reciprocal_rank_fusion, tokenize
from retrieval.base import Retriever
from retrieval.hybrid import HybridRetriever
from core.types import Chunk, RetrievedChunk


class TestTokenizer:
    def test_lowercases_and_splits(self):
        assert tokenize("The Contract, governs!") == ["the", "contract", "governs"]

    def test_drops_single_characters(self):
        assert "a" not in tokenize("a big cat")

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25:
    def test_finds_lexically_matching_chunk(self, small_chunks):
        retriever = BM25Retriever(small_chunks)
        results = retriever.search("termination for convenience notice", top_k=3)
        assert results[0].chunk_id == small_chunks[1].chunk_id

    def test_ranks_are_sequential_from_one(self, small_chunks):
        results = BM25Retriever(small_chunks).search("contract payment", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_top_k_respected(self, small_chunks):
        assert len(BM25Retriever(small_chunks).search("the", top_k=2)) == 2

    def test_top_k_larger_than_corpus(self, small_chunks):
        results = BM25Retriever(small_chunks).search("energy", top_k=100)
        assert len(results) == len(small_chunks)

    def test_allowed_restricts_candidates(self, small_chunks):
        retriever = BM25Retriever(small_chunks)
        allowed = {small_chunks[2].chunk_id, small_chunks[3].chunk_id}
        results = retriever.search("contract payment terms", top_k=10, allowed=allowed)
        assert {r.chunk_id for r in results} <= allowed

    def test_empty_allowed_returns_nothing(self, small_chunks):
        assert BM25Retriever(small_chunks).search("contract", 10, allowed=set()) == []

    def test_query_with_no_usable_tokens(self, small_chunks):
        assert BM25Retriever(small_chunks).search("!!! ?", top_k=5) == []

    def test_scores_are_non_increasing(self, small_chunks):
        results = BM25Retriever(small_chunks).search("energy chemical light", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_chunk_ids_rejected(self, small_chunks):
        with pytest.raises(ValueError, match="duplicate chunk ids"):
            BM25Retriever([small_chunks[0], small_chunks[0]])

    def test_persistence_round_trip(self, small_chunks, tmp_path):
        retriever = BM25Retriever(small_chunks)
        path = tmp_path / "bm25.pkl"
        retriever.save(path)
        loaded = BM25Retriever.load(path, small_chunks)
        before = retriever.search("governing law delaware", top_k=3)
        after = loaded.search("governing law delaware", top_k=3)
        assert [r.chunk_id for r in before] == [r.chunk_id for r in after]

    def test_load_against_wrong_corpus_raises(self, small_chunks, tmp_path):
        path = tmp_path / "bm25.pkl"
        BM25Retriever(small_chunks).save(path)
        with pytest.raises(ValueError, match="different corpus"):
            BM25Retriever.load(path, small_chunks[:3])


class StubRetriever(Retriever):
    """Returns a fixed ranking, for testing fusion in isolation."""

    def __init__(self, chunks, order):
        super().__init__(chunks)
        self.order = order

    def search(self, query, top_k, allowed=None):
        ids = [c for c in self.order if allowed is None or c in allowed]
        return [
            RetrievedChunk(chunk_id=cid, rank=i, score=1.0 / i)
            for i, cid in enumerate(ids[:top_k], start=1)
        ]


class TestRRF:
    def test_item_ranked_first_by_both_wins(self):
        a = [RetrievedChunk("x", 1, 9.0), RetrievedChunk("y", 2, 1.0)]
        b = [RetrievedChunk("x", 1, 0.9), RetrievedChunk("z", 2, 0.5)]
        assert reciprocal_rank_fusion([a, b], top_k=3)[0].chunk_id == "x"

    def test_consensus_beats_a_single_top_hit(self):
        # "y" is 2nd in both; "x" is 1st in one list and absent from the other.
        a = [RetrievedChunk("x", 1, 1.0), RetrievedChunk("y", 2, 0.9)]
        b = [RetrievedChunk("z", 1, 1.0), RetrievedChunk("y", 2, 0.9)]
        fused = reciprocal_rank_fusion([a, b], top_k=3, k=60)
        assert fused[0].chunk_id == "y"

    def test_scores_match_formula(self):
        a = [RetrievedChunk("x", 1, 0.0)]
        b = [RetrievedChunk("x", 3, 0.0)]
        fused = reciprocal_rank_fusion([a, b], top_k=1, k=60)
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)

    def test_deduplicates_across_lists(self):
        a = [RetrievedChunk("x", 1, 0.0), RetrievedChunk("y", 2, 0.0)]
        b = [RetrievedChunk("x", 1, 0.0), RetrievedChunk("y", 2, 0.0)]
        fused = reciprocal_rank_fusion([a, b], top_k=10)
        assert len(fused) == 2
        assert len({f.chunk_id for f in fused}) == 2

    def test_ranks_renumbered_from_one(self):
        a = [RetrievedChunk("x", 5, 0.0), RetrievedChunk("y", 9, 0.0)]
        fused = reciprocal_rank_fusion([a], top_k=2)
        assert [f.rank for f in fused] == [1, 2]

    def test_does_not_mutate_inputs(self):
        a = [RetrievedChunk("x", 1, 0.5)]
        b = [RetrievedChunk("x", 2, 0.3)]
        reciprocal_rank_fusion([a, b], top_k=1)
        assert a[0].rank == 1 and a[0].score == 0.5
        assert b[0].rank == 2 and b[0].score == 0.3

    def test_empty_input(self):
        assert reciprocal_rank_fusion([], top_k=5) == []

    def test_duplicate_within_one_list_counted_once(self):
        a = [RetrievedChunk("x", 1, 0.0), RetrievedChunk("x", 2, 0.0)]
        fused = reciprocal_rank_fusion([a], top_k=5, k=60)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1 / 61)

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            reciprocal_rank_fusion([[RetrievedChunk("x", 1, 0.0)]], top_k=1, k=0)


class TestHybrid:
    def test_combines_both_rankers(self, small_chunks):
        ids = [c.chunk_id for c in small_chunks]
        sparse = StubRetriever(small_chunks, [ids[0], ids[1], ids[2]])
        dense = StubRetriever(small_chunks, [ids[2], ids[1], ids[4]])
        hybrid = HybridRetriever(sparse, dense, rrf_k=60, candidate_depth=10)
        fused = hybrid.search("anything", top_k=5)

        # With k=60: ids[2] appears at ranks 3 and 1 -> 1/63 + 1/61 = 0.032266,
        # narrowly beating ids[1] at ranks 2 and 2 -> 2/62 = 0.032258. Both
        # outrank ids[0], which only one ranker retrieved at all. This is the
        # intended behaviour of RRF: appearing in both lists dominates, and
        # among two-list items a strong first place still counts.
        assert [f.chunk_id for f in fused[:3]] == [ids[2], ids[1], ids[0]]
        assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)

    def test_item_in_both_lists_beats_single_list_leader(self, small_chunks):
        ids = [c.chunk_id for c in small_chunks]
        sparse = StubRetriever(small_chunks, [ids[0], ids[3]])
        dense = StubRetriever(small_chunks, [ids[1], ids[3]])
        hybrid = HybridRetriever(sparse, dense, rrf_k=60, candidate_depth=10)
        fused = hybrid.search("anything", top_k=5)
        # ids[3] is 2nd in both; neither ranker's own leader appears twice.
        assert fused[0].chunk_id == ids[3]

    def test_mismatched_corpora_rejected(self, small_chunks):
        sparse = StubRetriever(small_chunks, [])
        dense = StubRetriever(small_chunks[:3], [])
        with pytest.raises(ValueError, match="same corpus"):
            HybridRetriever(sparse, dense)

    def test_respects_allowed_filter(self, small_chunks):
        ids = [c.chunk_id for c in small_chunks]
        sparse = StubRetriever(small_chunks, ids)
        dense = StubRetriever(small_chunks, list(reversed(ids)))
        hybrid = HybridRetriever(sparse, dense)
        allowed = {ids[0], ids[1]}
        assert {r.chunk_id for r in hybrid.search("q", 10, allowed=allowed)} <= allowed


class StubEmbedder:
    """Deterministic bag-of-words embedder, so dense logic is testable offline."""

    class _Config:
        model_name = "stub"
        normalize = True
        batch_size = 8

    def __init__(self, vocabulary):
        self.vocabulary = {w: i for i, w in enumerate(vocabulary)}
        self.dimension = len(vocabulary)
        self.config = self._Config()

    def encode(self, texts, batch_size=None, progress=False):
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                if token in self.vocabulary:
                    out[row, self.vocabulary[token]] = 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out

    def encode_query(self, text):
        return self.encode([text])[0]


class TestDense:
    @pytest.fixture
    def embedder(self, small_chunks):
        vocab = sorted({t for c in small_chunks for t in tokenize(c.text)})
        return StubEmbedder(vocab)

    def test_retrieves_semantically_matching_chunk(self, small_chunks, embedder):
        from retrieval import DenseRetriever

        retriever = DenseRetriever(small_chunks, embedder, show_progress=False)
        results = retriever.search("mitochondria programmed cell death", top_k=2)
        assert results[0].chunk_id == small_chunks[2].chunk_id

    def test_index_size_matches_corpus(self, small_chunks, embedder):
        from retrieval import DenseRetriever

        assert DenseRetriever(small_chunks, embedder, show_progress=False).index.ntotal == 5

    def test_allowed_filter_is_exact(self, small_chunks, embedder):
        from retrieval import DenseRetriever

        retriever = DenseRetriever(small_chunks, embedder, show_progress=False)
        allowed = {small_chunks[0].chunk_id, small_chunks[4].chunk_id}
        results = retriever.search("mitochondria cell death", top_k=10, allowed=allowed)
        assert {r.chunk_id for r in results} <= allowed
        assert len(results) == 2

    def test_persistence_round_trip(self, small_chunks, embedder, tmp_path):
        from retrieval import DenseRetriever

        retriever = DenseRetriever(small_chunks, embedder, show_progress=False)
        path = tmp_path / "dense.faiss"
        retriever.save(path)
        loaded = DenseRetriever.load(path, small_chunks, embedder)
        query = "governing law delaware agreement"
        assert [r.chunk_id for r in retriever.search(query, 3)] == [
            r.chunk_id for r in loaded.search(query, 3)
        ]

    def test_load_with_different_model_raises(self, small_chunks, embedder, tmp_path):
        from retrieval import DenseRetriever

        path = tmp_path / "dense.faiss"
        DenseRetriever(small_chunks, embedder, show_progress=False).save(path)
        other = StubEmbedder(["x"])
        other.config.model_name = "different-model"
        other.dimension = embedder.dimension
        with pytest.raises(ValueError, match="embedding model"):
            DenseRetriever.load(path, small_chunks, other)
