"""SQLite persistence: run separation, NULL semantics, traceability."""

from __future__ import annotations

import json

import pytest

from core.db import Database, environment_info
from evaluation.metrics import AggregateMetrics, QueryMetrics
from core.types import RetrievedChunk


@pytest.fixture
def database(tmp_path) -> Database:
    db = Database(tmp_path / "test.sqlite")
    yield db
    db.close()


def make_run(database, config, run_id="run-1", dataset="cuad"):
    return database.create_run(
        run_id, stage="retrieval", config=config, dataset=dataset, split="all",
        methods=["bm25"], relevance_class="GOLD", corpus_scope="document",
        corpus_fingerprint="fixed-256-64", n_documents=3, n_chunks=100, n_queries=10,
    )


class TestConcurrency:
    def test_journal_mode_is_wal(self, database):
        """A long sweep is backed up while it writes.

        Copying this database to Google Drive holds a read lock for minutes.
        Under the default rollback journal that blocks the writer and the
        commit eventually fails with "database is locked", ending a run that
        may be hours in. WAL lets the reader and the writer proceed together.
        """
        mode = database.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_a_reader_does_not_block_a_writer(self, database, config):
        """The exact shape of the failure: backup reads while generation writes."""
        import sqlite3

        reader = sqlite3.connect(f"file:{database.path}?mode=ro", uri=True)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM runs").fetchone()
            make_run(database, config)          # must not raise "database is locked"
        finally:
            reader.close()

        assert len(database.runs()) == 1


class TestRuns:
    def test_created_run_is_retrievable(self, database, config):
        make_run(database, config)
        rows = database.runs()
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-1"
        assert rows[0]["dataset"] == "cuad"
        assert rows[0]["relevance_class"] == "GOLD"

    def test_run_records_full_config_and_environment(self, database, config):
        make_run(database, config)
        row = database.runs()[0]
        stored = json.loads(row["config_json"])
        assert stored["embedding"]["model_name"] == config.embedding.model_name
        assert stored["retrieval"]["rrf_k"] == config.retrieval.rrf_k
        environment = json.loads(row["environment_json"])
        assert "python" in environment and "platform" in environment

    def test_reproducibility_fields_present(self, database, config):
        make_run(database, config)
        row = database.runs()[0]
        for field in ("created_at", "embedding_model", "chunk_strategy", "chunk_size",
                      "chunk_overlap", "top_k", "rrf_k", "corpus_fingerprint"):
            assert row[field] is not None, field

    def test_generation_model_null_when_disabled(self, database, config):
        make_run(database, config)
        assert database.runs()[0]["generation_model"] is None

    def test_two_runs_are_separate(self, database, config):
        make_run(database, config, "run-1")
        make_run(database, config, "run-2")
        assert {r["run_id"] for r in database.runs()} == {"run-1", "run-2"}


class TestRetrievalResults:
    def test_results_persisted_with_relevance_flags(self, database, config):
        make_run(database, config)
        retrieved = [RetrievedChunk("c1", 1, 0.9), RetrievedChunk("c2", 2, 0.5)]
        database.save_retrieval_results("run-1", "cuad", "bm25", "q1", retrieved, {"c2"})
        rows = database.results_for_query("run-1", "bm25", "q1")
        assert [r["chunk_id"] for r in rows] == ["c1", "c2"]
        assert [r["is_relevant"] for r in rows] == [0, 1]

    def test_unjudgeable_results_store_null_relevance(self, database, config):
        make_run(database, config, dataset="medqa")
        retrieved = [RetrievedChunk("c1", 1, 0.9)]
        # relevant=None means "no judgement", which must be NULL, not 0.
        database.save_retrieval_results("run-1", "medqa", "bm25", "q1", retrieved, None)
        rows = database.results_for_query("run-1", "bm25", "q1")
        assert rows[0]["is_relevant"] is None

    def test_two_runs_do_not_overwrite_each_other(self, database, config):
        make_run(database, config, "run-1")
        make_run(database, config, "run-2")
        database.save_retrieval_results(
            "run-1", "cuad", "bm25", "q1", [RetrievedChunk("a", 1, 1.0)], set()
        )
        database.save_retrieval_results(
            "run-2", "cuad", "bm25", "q1", [RetrievedChunk("b", 1, 1.0)], set()
        )
        assert database.results_for_query("run-1", "bm25", "q1")[0]["chunk_id"] == "a"
        assert database.results_for_query("run-2", "bm25", "q1")[0]["chunk_id"] == "b"

    def test_methods_stored_separately(self, database, config):
        make_run(database, config)
        for method in ("bm25", "dense"):
            database.save_retrieval_results(
                "run-1", "cuad", method, "q1", [RetrievedChunk(method, 1, 1.0)], set()
            )
        assert database.results_for_query("run-1", "bm25", "q1")[0]["chunk_id"] == "bm25"
        assert database.results_for_query("run-1", "dense", "q1")[0]["chunk_id"] == "dense"


class TestMetricPersistence:
    def test_null_metrics_stored_as_sql_null(self, database, config):
        make_run(database, config, dataset="medqa")
        unscoreable = QueryMetrics(
            "q1", scoreable=False, n_relevant=0,
            recall={10: None}, ndcg={10: None}, mrr=None, reason="no ground truth",
        )
        database.save_query_metrics("run-1", "medqa", "bm25", [unscoreable])
        row = database.connection.execute(
            "SELECT * FROM query_metrics WHERE run_id='run-1'"
        ).fetchone()
        assert row["mrr"] is None
        assert row["scoreable"] == 0
        assert json.loads(row["metrics_json"])["recall"]["10"] is None

    def test_zero_metrics_stored_as_zero_not_null(self, database, config):
        make_run(database, config)
        missed = QueryMetrics(
            "q1", scoreable=True, n_relevant=1,
            recall={10: 0.0}, ndcg={10: 0.0}, mrr=0.0,
        )
        database.save_query_metrics("run-1", "cuad", "bm25", [missed])
        row = database.connection.execute(
            "SELECT * FROM query_metrics WHERE run_id='run-1'"
        ).fetchone()
        assert row["mrr"] == 0.0
        assert row["mrr"] is not None
        assert json.loads(row["metrics_json"])["recall"]["10"] == 0.0

    def test_aggregate_null_round_trip(self, database, config):
        make_run(database, config, dataset="medqa")
        agg = AggregateMetrics(
            n_queries=5, n_scoreable=0, n_unscoreable=5,
            recall={10: None}, ndcg={10: None}, mrr=None,
        )
        database.save_aggregate("run-1", "medqa", "medical", "bm25", "UNSUPPORTED", agg)
        row = database.aggregates("run-1")[0]
        assert row["mrr"] is None
        assert row["coverage"] == 0.0
        assert row["n_unscoreable"] == 5
        assert json.loads(row["metrics_json"])["ndcg"]["10"] is None

    def test_aggregate_with_values(self, database, config):
        make_run(database, config)
        agg = AggregateMetrics(
            n_queries=10, n_scoreable=8, n_unscoreable=2,
            recall={10: 0.75}, ndcg={10: 0.6}, mrr=0.5,
        )
        database.save_aggregate("run-1", "cuad", "legal", "bm25", "GOLD", agg)
        row = database.aggregates("run-1")[0]
        assert row["mrr"] == 0.5
        assert row["coverage"] == pytest.approx(0.8)


class TestGenerations:
    def test_unscored_metrics_are_null(self, database, config):
        make_run(database, config, dataset="medqa")
        database.save_generation("run-1", "medqa", "hybrid", "llama3.1", {
            "query_id": "q1", "prompt": "p", "answer": "B", "reference_answer": "B",
            "context_chunk_ids": ["c1"], "latency_ms": 12.0,
            "exact_match": 1.0, "choice_correct": 1.0,
            "faithfulness": None, "hallucination": None, "error": None,
        })
        row = database.connection.execute("SELECT * FROM generations").fetchone()
        assert row["choice_correct"] == 1.0
        assert row["faithfulness"] is None
        assert row["hallucination"] is None
        assert json.loads(row["context_chunk_ids"]) == ["c1"]

    def test_models_answering_the_same_question_are_kept_apart(self, database, config):
        # The whole point of the multi-model stage: two models answering one
        # question must be two rows, not one overwriting the other.
        make_run(database, config, dataset="medqa")
        for model, answer in (("llama3.1", "B"), ("mistral", "C")):
            database.save_generation("run-1", "medqa", "hybrid", model, {
                "query_id": "q1", "prompt": "p", "answer": answer,
                "reference_answer": "B", "context_chunk_ids": ["c1"],
                "latency_ms": 1.0, "exact_match": None,
                "choice_correct": 1.0 if answer == "B" else 0.0,
            })
        rows = database.connection.execute(
            "SELECT model, answer FROM generations ORDER BY model"
        ).fetchall()
        assert [(r["model"], r["answer"]) for r in rows] == [
            ("llama3.1", "B"), ("mistral", "C"),
        ]

    def test_completed_generations_supports_resume(self, database, config):
        make_run(database, config, dataset="medqa")
        database.save_generation("run-1", "medqa", "hybrid", "llama3.1", {
            "query_id": "q1", "prompt": "p", "answer": "B", "reference_answer": "B",
            "context_chunk_ids": [], "latency_ms": 1.0,
            "exact_match": None, "choice_correct": None,
        })
        # A row whose generation failed has no answer and must be retried.
        database.save_generation("run-1", "medqa", "hybrid", "llama3.1", {
            "query_id": "q2", "prompt": "p", "answer": None, "reference_answer": "B",
            "context_chunk_ids": [], "latency_ms": None,
            "exact_match": None, "choice_correct": None, "error": "boom",
        })
        done = database.completed_generations("run-1")
        assert ("medqa", "hybrid", "llama3.1", "q1") in done
        assert ("medqa", "hybrid", "llama3.1", "q2") not in done

    def test_scores_are_written_with_their_rater(self, database, config):
        make_run(database, config, dataset="medqa")
        database.save_generation("run-1", "medqa", "hybrid", "llama3.1", {
            "query_id": "q1", "prompt": "p", "answer": "B", "reference_answer": "B",
            "context_chunk_ids": [], "latency_ms": 1.0,
            "exact_match": None, "choice_correct": None,
        })
        database.update_generation_score(
            "run-1", "medqa", "hybrid", "llama3.1", "q1", "faithfulness", 0.75,
            rater="llama3.1", detail='{"n_statements": 4}',
        )
        row = database.connection.execute("SELECT * FROM generations").fetchone()
        assert row["faithfulness"] == 0.75
        assert row["faithfulness_judge"] == "llama3.1"
        assert row["hallucination"] is None


def test_environment_info_reports_versions():
    info = environment_info()
    assert info["python"]
    assert "numpy" in info and "faiss" in info
    assert "cuda_available" in info
