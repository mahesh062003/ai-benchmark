"""Paired significance testing between retrieval strategies.

The numbers here are chosen so the expected answer can be worked out by hand,
because a statistics module that is only tested against its own output cannot
catch a systematic error.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from cli import _latest_retrieval_run
from evaluation.significance import (
    Comparison,
    align,
    bootstrap_ci,
    compare_pair,
    holm_bonferroni,
    load_per_query,
    run_comparisons,
    wilcoxon_p,
)


@pytest.fixture
def database():
    """An in-memory query_metrics table mirroring the real schema."""
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE query_metrics (run_id TEXT, dataset TEXT, method TEXT,"
        " query_id TEXT, scoreable INTEGER, n_relevant INTEGER, mrr REAL,"
        " metrics_json TEXT, reason TEXT)"
    )
    connection.execute(
        "CREATE TABLE aggregate_metrics (run_id TEXT, dataset TEXT, method TEXT,"
        " relevance_class TEXT)"
    )
    return connection


def add(connection, dataset, method, query_id, recall, mrr, scoreable=1):
    connection.execute(
        "INSERT INTO query_metrics VALUES ('r1',?,?,?,?,1,?,?,'')",
        (
            dataset, method, query_id, scoreable, mrr,
            json.dumps({"recall": {"10": recall}, "ndcg": {"10": recall}}),
        ),
    )


class TestHolmBonferroni:
    def test_smallest_p_is_multiplied_by_the_family_size(self):
        # Three tests: the smallest p is scaled by 3, the next by 2, the last by 1.
        assert holm_bonferroni([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])

    def test_adjusted_values_never_decrease_with_rank(self):
        """A weaker result must never come out looking stronger than a better one."""
        adjusted = holm_bonferroni([0.01, 0.04, 0.045])
        ordered = sorted(zip([0.01, 0.04, 0.045], adjusted))
        assert all(b >= a for (_, a), (_, b) in zip(ordered, ordered[1:]))

    def test_capped_at_one(self):
        assert all(p <= 1.0 for p in holm_bonferroni([0.6, 0.7, 0.9]))

    def test_order_is_preserved(self):
        """Adjusted values must line up with the inputs, not with their sorted order."""
        assert holm_bonferroni([0.5, 0.001])[1] < holm_bonferroni([0.5, 0.001])[0]

    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_test_is_unchanged(self):
        assert holm_bonferroni([0.03]) == pytest.approx([0.03])


class TestAlignment:
    def test_only_queries_scored_by_both_are_compared(self):
        a, b = align({"q1": 1.0, "q2": 0.5, "q3": 0.0}, {"q1": 0.0, "q2": 0.5})
        assert list(a) == [1.0, 0.5]
        assert list(b) == [0.0, 0.5]

    def test_no_overlap_gives_empty_arrays(self):
        a, b = align({"q1": 1.0}, {"q2": 1.0})
        assert a.size == 0 and b.size == 0

    def test_pairing_is_by_query_id_not_position(self):
        """Misaligned insertion order must not silently pair the wrong scores."""
        a, b = align({"q2": 0.2, "q1": 0.9}, {"q1": 0.9, "q2": 0.2})
        assert list(a) == list(b)


class TestWilcoxon:
    def test_identical_scores_report_no_evidence(self):
        values = np.array([0.5, 0.5, 1.0])
        assert wilcoxon_p(values, values.copy()) == 1.0

    def test_empty_input_reports_no_evidence(self):
        assert wilcoxon_p(np.array([]), np.array([])) == 1.0

    def test_consistent_difference_is_detected(self):
        a = np.array([0.9] * 20)
        b = np.array([0.1] * 20)
        assert wilcoxon_p(a, b) < 0.01

    def test_symmetric_noise_is_not_detected(self):
        a = np.array([0.6, 0.4] * 15)
        b = np.array([0.4, 0.6] * 15)
        assert wilcoxon_p(a, b) > 0.05


class TestBootstrap:
    def test_interval_brackets_the_mean_difference(self):
        differences = np.array([0.1] * 50 + [0.2] * 50)
        low, high = bootstrap_ci(differences, resamples=2000)
        assert low <= differences.mean() <= high

    def test_constant_difference_gives_a_degenerate_interval(self):
        low, high = bootstrap_ci(np.array([0.25] * 40), resamples=500)
        assert low == pytest.approx(0.25) and high == pytest.approx(0.25)

    def test_deterministic_for_a_fixed_seed(self):
        d = np.array([0.1, -0.2, 0.3, 0.05, -0.15])
        assert bootstrap_ci(d, resamples=1000, seed=7) == bootstrap_ci(d, resamples=1000, seed=7)

    def test_interval_spans_zero_when_there_is_no_effect(self):
        low, high = bootstrap_ci(np.array([0.2, -0.2] * 40), resamples=2000)
        assert low < 0 < high

    def test_empty_differences_give_nan(self):
        low, high = bootstrap_ci(np.array([]))
        assert np.isnan(low) and np.isnan(high)


class TestLoadPerQuery:
    def test_unscoreable_rows_are_excluded_not_zeroed(self, database):
        """NULL must shrink the sample, never enter the comparison as 0.0."""
        add(database, "d", "bm25", "q1", 1.0, 1.0)
        add(database, "d", "bm25", "q2", 0.0, 0.0, scoreable=0)
        scores = load_per_query(database, "r1", "d", "bm25", "recall")
        assert scores == {"q1": 1.0}

    def test_mrr_is_read_from_its_column(self, database):
        add(database, "d", "bm25", "q1", 0.25, 0.75)
        assert load_per_query(database, "r1", "d", "bm25", "mrr") == {"q1": 0.75}

    def test_missing_k_yields_no_score(self, database):
        add(database, "d", "bm25", "q1", 1.0, 1.0)
        assert load_per_query(database, "r1", "d", "bm25", "recall", k=999) == {}


class TestComparePair:
    def test_detects_a_real_difference_and_names_the_winner(self, database):
        for i in range(30):
            add(database, "d", "bm25", f"q{i}", 0.2, 0.2)
            add(database, "d", "hybrid", f"q{i}", 0.8, 0.8)
        result = compare_pair(database, "r1", "d", "recall", "bm25", "hybrid", resamples=500)
        result.p_corrected = result.p_value
        assert result.n_pairs == 30
        assert result.mean_diff == pytest.approx(-0.6)
        assert result.significant and result.winner == "hybrid"

    def test_identical_strategies_are_not_significant(self, database):
        for i in range(30):
            add(database, "d", "bm25", f"q{i}", 0.5, 0.5)
            add(database, "d", "dense", f"q{i}", 0.5, 0.5)
        result = compare_pair(database, "r1", "d", "recall", "bm25", "dense", resamples=500)
        result.p_corrected = result.p_value
        assert result.n_differing == 0
        assert not result.significant and result.winner is None

    def test_returns_none_when_nothing_is_scoreable(self, database):
        add(database, "d", "bm25", "q1", 0.0, 0.0, scoreable=0)
        add(database, "d", "dense", "q1", 0.0, 0.0, scoreable=0)
        assert compare_pair(database, "r1", "d", "recall", "bm25", "dense") is None


class TestRunComparisons:
    def test_produces_one_row_per_pair_and_corrects_within_the_family(self, database):
        for i in range(25):
            add(database, "d", "bm25", f"q{i}", 0.2, 0.2)
            add(database, "d", "dense", f"q{i}", 0.5, 0.5)
            add(database, "d", "hybrid", f"q{i}", 0.9, 0.9)
        results = run_comparisons(
            database, "r1", ["d"], ["bm25", "dense", "hybrid"], [("recall", 10)],
            resamples=400,
        )
        assert len(results) == 3
        assert all(r.p_corrected >= r.p_value for r in results)

    def test_unsupported_dataset_contributes_nothing(self, database):
        add(database, "medqa", "bm25", "q1", 0.0, 0.0, scoreable=0)
        add(database, "medqa", "dense", "q1", 0.0, 0.0, scoreable=0)
        assert run_comparisons(
            database, "r1", ["medqa"], ["bm25", "dense"], [("recall", 10)], resamples=100
        ) == []


class TestDefaultRunSelection:
    """Which run `significance` tests when the user does not pass --run.

    Once a database has reached the generation stage its newest run is a
    generation run, which writes a `runs` row but no query_metrics. Selecting
    it would report "nothing to test" on a database full of perfectly testable
    retrieval results, so the default has to skip past it.
    """

    @pytest.fixture
    def with_runs(self, database):
        database.execute("CREATE TABLE runs (run_id TEXT, created_at TEXT, stage TEXT)")
        return database

    @staticmethod
    def _add_run(connection, run_id, created_at, stage):
        connection.execute("INSERT INTO runs VALUES (?,?,?)", (run_id, created_at, stage))

    def test_skips_a_newer_run_that_has_no_retrieval_metrics(self, with_runs):
        self._add_run(with_runs, "r1", "2026-08-14T00:00:00+00:00", "retrieval-batch")
        self._add_run(with_runs, "genall-1", "2026-08-15T00:00:00+00:00", "generation-multi-model")
        add(with_runs, "d", "bm25", "q1", 0.5, 0.5)  # belongs to r1

        assert _latest_retrieval_run(with_runs) == "r1"

    def test_prefers_the_most_recent_run_that_does_have_metrics(self, with_runs):
        self._add_run(with_runs, "old", "2026-08-01T00:00:00+00:00", "retrieval-batch")
        self._add_run(with_runs, "r1", "2026-08-14T00:00:00+00:00", "retrieval-batch")
        with_runs.execute(
            "INSERT INTO query_metrics VALUES ('old','d','bm25','q1',1,1,0.5,'{}','')"
        )
        add(with_runs, "d", "bm25", "q1", 0.5, 0.5)  # belongs to r1, the newer run

        assert _latest_retrieval_run(with_runs) == "r1"

    def test_returns_none_when_no_run_has_metrics(self, with_runs):
        self._add_run(with_runs, "genall-1", "2026-08-15T00:00:00+00:00", "generation-multi-model")

        assert _latest_retrieval_run(with_runs) is None
