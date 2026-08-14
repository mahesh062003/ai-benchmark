"""Retrieval metric tests, with emphasis on the NULL / zero distinction."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    QueryMetrics,
    aggregate,
    evaluate_query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def ids(*names: str) -> list[str]:
    return list(names)


class TestRecall:
    def test_single_relevant_found(self):
        assert recall_at_k(ids("a", "b", "c"), {"b"}, 3) == 1.0

    def test_single_relevant_outside_cutoff(self):
        assert recall_at_k(ids("a", "b", "c"), {"c"}, 2) == 0.0

    def test_multiple_relevant_partial(self):
        assert recall_at_k(ids("a", "b", "c", "d"), {"a", "d"}, 2) == 0.5

    def test_multiple_relevant_all(self):
        assert recall_at_k(ids("a", "b"), {"a", "b"}, 2) == 1.0

    def test_top_k_larger_than_result_list(self):
        assert recall_at_k(ids("a"), {"a"}, 100) == 1.0

    def test_no_ground_truth_returns_none_not_zero(self):
        assert recall_at_k(ids("a", "b"), set(), 10) is None

    def test_empty_retrieval_with_ground_truth_is_zero(self):
        # A genuine failure: the judgement existed and nothing was returned.
        assert recall_at_k([], {"a"}, 10) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(ids("a", "b"), {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(ids("a", "b", "c"), {"c"}) == pytest.approx(1 / 3)

    def test_uses_first_relevant_only(self):
        assert reciprocal_rank(ids("a", "b", "c"), {"b", "c"}) == 0.5

    def test_not_found_is_zero(self):
        assert reciprocal_rank(ids("a", "b"), {"z"}) == 0.0

    def test_no_ground_truth_is_none(self):
        assert reciprocal_rank(ids("a"), set()) is None

    def test_cutoff_excludes_late_hit(self):
        assert reciprocal_rank(ids("a", "b", "c"), {"c"}, k=2) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(ids("a", "b", "c"), {"a"}, 3) == 1.0

    def test_second_position(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        assert ndcg_at_k(ids("x", "a"), {"a"}, 2) == pytest.approx(1 / math.log2(3))

    def test_two_relevant_perfectly_ranked(self):
        assert ndcg_at_k(ids("a", "b", "c"), {"a", "b"}, 3) == pytest.approx(1.0)

    def test_two_relevant_reversed_is_less_than_one(self):
        value = ndcg_at_k(ids("x", "a", "b"), {"a", "b"}, 3)
        assert 0 < value < 1

    def test_ideal_capped_at_k_so_perfect_head_scores_one(self):
        # 5 relevant items, cutoff 2: retrieving 2 of them first is the best
        # achievable ranking at that cutoff and must score 1.0.
        relevant = {"a", "b", "c", "d", "e"}
        assert ndcg_at_k(ids("a", "b", "z"), relevant, 2) == pytest.approx(1.0)

    def test_nothing_relevant_retrieved_is_zero(self):
        assert ndcg_at_k(ids("x", "y"), {"a"}, 2) == 0.0

    def test_no_ground_truth_is_none(self):
        assert ndcg_at_k(ids("a"), set(), 5) is None


class TestEvaluateQuery:
    def test_unavailable_judgement_yields_all_none(self):
        result = evaluate_query(
            "q1", ids("a", "b"), set(), [1, 5, 10], available=False, reason="no evidence"
        )
        assert result.scoreable is False
        assert result.mrr is None
        assert all(v is None for v in result.recall.values())
        assert all(v is None for v in result.ndcg.values())
        assert result.reason == "no evidence"

    def test_available_but_missed_yields_zero_not_none(self):
        result = evaluate_query("q1", ids("x", "y"), {"a"}, [1, 5], available=True)
        assert result.scoreable is True
        assert result.mrr == 0.0
        assert result.recall[5] == 0.0
        assert result.ndcg[5] == 0.0

    def test_available_flag_true_but_empty_relevant_is_unscoreable(self):
        # Defensive: an empty relevant set can never produce a valid metric.
        result = evaluate_query("q1", ids("a"), set(), [1], available=True)
        assert result.scoreable is False
        assert result.recall[1] is None


class TestAggregate:
    def _metrics(self, query_id, scoreable, recall, mrr):
        return QueryMetrics(
            query_id=query_id, scoreable=scoreable, n_relevant=1 if scoreable else 0,
            recall={10: recall}, ndcg={10: recall}, mrr=mrr,
        )

    def test_unscoreable_queries_are_excluded_not_counted_as_zero(self):
        results = [
            self._metrics("a", True, 1.0, 1.0),
            self._metrics("b", False, None, None),
        ]
        agg = aggregate(results, [10])
        # Mean over the single scoreable query, not (1.0 + 0)/2.
        assert agg.recall[10] == 1.0
        assert agg.mrr == 1.0
        assert agg.n_queries == 2
        assert agg.n_scoreable == 1
        assert agg.n_unscoreable == 1
        assert agg.coverage == 0.5

    def test_zeros_do_count_toward_the_mean(self):
        results = [
            self._metrics("a", True, 1.0, 1.0),
            self._metrics("b", True, 0.0, 0.0),
        ]
        agg = aggregate(results, [10])
        assert agg.recall[10] == 0.5

    def test_all_unscoreable_gives_null_metrics(self):
        results = [self._metrics(f"q{i}", False, None, None) for i in range(3)]
        agg = aggregate(results, [10])
        assert agg.recall[10] is None
        assert agg.ndcg[10] is None
        assert agg.mrr is None
        assert agg.coverage == 0.0

    def test_empty_run(self):
        agg = aggregate([], [10])
        assert agg.n_queries == 0
        assert agg.coverage is None
        assert agg.mrr is None
