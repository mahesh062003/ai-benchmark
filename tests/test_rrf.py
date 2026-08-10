"""Tests for Reciprocal Rank Fusion."""

import pytest
from retrieval.base import RetrievalResult
from retrieval.rrf import ReciprocalRankFusion


def _make_result(chunk_id: str, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        dataset="Test",
        domain="Test",
        text=f"text for {chunk_id}",
        score=score,
        rank=rank,
    )


class TestRRF:
    def setup_method(self):
        self.rrf = ReciprocalRankFusion(k=60)

    def test_single_list_passthrough(self):
        results = [_make_result("A", 1), _make_result("B", 2), _make_result("C", 3)]
        fused = self.rrf.fuse(results)
        assert [r.chunk_id for r in fused] == ["A", "B", "C"]

    def test_deduplication_by_chunk_id(self):
        list1 = [_make_result("A", 1), _make_result("B", 2)]
        list2 = [_make_result("B", 1), _make_result("C", 2)]
        fused = self.rrf.fuse(list1, list2)
        ids = [r.chunk_id for r in fused]
        # B appears in both → should appear exactly once
        assert ids.count("B") == 1
        assert set(ids) == {"A", "B", "C"}

    def test_b_ranked_higher_when_in_both_lists(self):
        # B appears at rank 1 in both lists → should outscore A (rank 1 in one)
        list1 = [_make_result("A", 1), _make_result("B", 2)]
        list2 = [_make_result("B", 1), _make_result("C", 2)]
        fused = self.rrf.fuse(list1, list2)
        b_score = next(r.score for r in fused if r.chunk_id == "B")
        a_score = next(r.score for r in fused if r.chunk_id == "A")
        assert b_score > a_score

    def test_ranks_reassigned_from_one(self):
        list1 = [_make_result("A", 1), _make_result("B", 2)]
        list2 = [_make_result("C", 1), _make_result("A", 2)]
        fused = self.rrf.fuse(list1, list2)
        for expected_rank, result in enumerate(fused, start=1):
            assert result.rank == expected_rank

    def test_original_objects_not_mutated(self):
        r1 = _make_result("A", 1, score=10.0)
        r2 = _make_result("A", 1, score=5.0)
        list1 = [r1]
        list2 = [r2]
        self.rrf.fuse(list1, list2)
        # Original objects must be unchanged (frozen dataclass guarantees this)
        assert r1.score == 10.0
        assert r2.score == 5.0
        assert r1.rank == 1
        assert r2.rank == 1

    def test_scores_are_positive(self):
        results = [_make_result("A", 1), _make_result("B", 2), _make_result("C", 3)]
        fused = self.rrf.fuse(results)
        assert all(r.score > 0 for r in fused)

    def test_empty_lists(self):
        fused = self.rrf.fuse([], [])
        assert fused == []

    def test_one_empty_list(self):
        results = [_make_result("A", 1), _make_result("B", 2)]
        fused = self.rrf.fuse(results, [])
        assert len(fused) == 2
