"""Tests for retrieval evaluation metrics."""

import pytest
from evaluation.retrieval_metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["A", "B", "C"], ["A", "B"], k=3) == 1.0

    def test_none_retrieved(self):
        assert recall_at_k(["X", "Y", "Z"], ["A", "B"], k=3) == 0.0

    def test_partial(self):
        assert recall_at_k(["A", "X", "B"], ["A", "B", "C"], k=3) == pytest.approx(2 / 3)

    def test_k_cutoff_respected(self):
        # B is at rank 4 — beyond k=3 cutoff
        assert recall_at_k(["A", "X", "Y", "B"], ["A", "B"], k=3) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k(["A", "B"], [], k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["A", "B"], k=5) == 0.0

    def test_single_relevant_found_first(self):
        assert recall_at_k(["A", "B", "C"], ["A"], k=5) == 1.0


class TestMeanReciprocalRank:
    def test_first_position(self):
        assert mean_reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["X", "A", "B"], ["A"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert mean_reciprocal_rank(["X", "Y", "A"], ["A"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mean_reciprocal_rank(["X", "Y", "Z"], ["A"]) == 0.0

    def test_multiple_relevant_first_hit_counts(self):
        assert mean_reciprocal_rank(["X", "B", "A"], ["A", "B"]) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], ["A"]) == 0.0

    def test_empty_relevant(self):
        assert mean_reciprocal_rank(["A", "B"], []) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Both relevant items at top-2 positions
        assert ndcg_at_k(["A", "B", "C"], ["A", "B"], k=5) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert ndcg_at_k(["X", "Y", "Z"], ["A", "B"], k=5) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["A", "B"], [], k=5) == 0.0

    def test_single_relevant_at_position_1(self):
        import math
        expected = (1 / math.log2(2)) / (1 / math.log2(2))   # DCG/IDCG = 1.0
        assert ndcg_at_k(["A", "B", "C"], ["A"], k=3) == pytest.approx(expected)

    def test_single_relevant_at_position_2(self):
        import math
        dcg  = 1 / math.log2(3)
        idcg = 1 / math.log2(2)
        assert ndcg_at_k(["X", "A", "Y"], ["A"], k=3) == pytest.approx(dcg / idcg)

    def test_k_cutoff(self):
        # Relevant item only at position 4, k=3 → not found
        assert ndcg_at_k(["X", "Y", "Z", "A"], ["A"], k=3) == 0.0
