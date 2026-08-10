"""Tests for BenchmarkDatabase."""

import pytest


class TestBenchmarkDatabase:
    def test_start_run_returns_uuid_string(self, tmp_db):
        run_id = tmp_db.start_run({"method": "BM25", "top_k": 5})
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID4 format

    def test_two_runs_have_different_ids(self, tmp_path):
        from database.database import BenchmarkDatabase
        db = BenchmarkDatabase(db_path=tmp_path / "db.sqlite")
        id1 = db.start_run({"x": 1})
        id2 = db.start_run({"x": 2})
        assert id1 != id2
        db.close()

    def test_insert_result_with_ground_truth(self, tmp_db):
        tmp_db.start_run({"top_k": 5})
        tmp_db.insert_result(
            dataset="CUAD", domain="Legal", split="all",
            retrieval_method="BM25", sample_id="cuad_1",
            question="What is the term?",
            generated_answer=None,
            has_retrieval_gt=True,
            recall=0.8, mrr=1.0, ndcg=0.9,
            response_time_s=0.05,
        )
        tmp_db.commit_batch()
        rows = tmp_db.cursor.execute("SELECT * FROM results").fetchall()
        assert len(rows) == 1

    def test_null_metrics_when_no_ground_truth(self, tmp_db):
        tmp_db.start_run({})
        tmp_db.insert_result(
            dataset="MedQA", domain="Medical", split="dev",
            retrieval_method="FAISS", sample_id="medqa_1",
            question="Which drug?",
            generated_answer=None,
            has_retrieval_gt=False,
            recall=None, mrr=None, ndcg=None,
            response_time_s=0.1,
        )
        tmp_db.commit_batch()
        row = tmp_db.cursor.execute(
            "SELECT recall, mrr, ndcg, has_retrieval_gt FROM results"
        ).fetchone()
        recall, mrr, ndcg, has_gt = row
        assert recall is None
        assert mrr    is None
        assert ndcg   is None
        assert has_gt == 0

    def test_faithfulness_not_zero_but_null(self, tmp_db):
        """faithfulness/hallucination must be NULL, not 0.0."""
        tmp_db.start_run({})
        tmp_db.insert_result(
            dataset="CUAD", domain="Legal", split="all",
            retrieval_method="Hybrid", sample_id="cuad_2",
            question="parties?", generated_answer="the parties",
            has_retrieval_gt=True,
            recall=1.0, mrr=1.0, ndcg=1.0,
            response_time_s=0.2,
        )
        tmp_db.commit_batch()
        row = tmp_db.cursor.execute(
            "SELECT faithfulness, hallucination FROM results"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_summary_excludes_no_ground_truth_rows(self, tmp_db):
        tmp_db.start_run({})
        # One row with GT
        tmp_db.insert_result(
            dataset="CUAD", domain="Legal", split="all",
            retrieval_method="BM25", sample_id="cuad_1",
            question="q1", generated_answer=None,
            has_retrieval_gt=True,
            recall=0.5, mrr=0.5, ndcg=0.5,
            response_time_s=0.1,
        )
        # One row without GT (MedQA)
        tmp_db.insert_result(
            dataset="MedQA", domain="Medical", split="dev",
            retrieval_method="BM25", sample_id="medqa_1",
            question="q2", generated_answer=None,
            has_retrieval_gt=False,
            recall=None, mrr=None, ndcg=None,
            response_time_s=0.1,
        )
        tmp_db.commit_batch()
        summary = tmp_db.summary()
        # Only CUAD should appear in summary (has_retrieval_gt=1)
        datasets_in_summary = [row["dataset"] for row in summary]
        assert "CUAD" in datasets_in_summary
        assert "MedQA" not in datasets_in_summary

    def test_run_stored_in_runs_table(self, tmp_db):
        run_id = tmp_db.start_run({"embedding": "all-mpnet"})
        row = tmp_db.cursor.execute(
            "SELECT run_id, config FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        import json
        config = json.loads(row[1])
        assert config["embedding"] == "all-mpnet"
