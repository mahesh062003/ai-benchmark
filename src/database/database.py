"""
SQLite persistence for benchmark results.

Schema
------
Two tables:

runs
    One row per benchmark run.  A run covers all datasets × retrieval methods
    executed in one invocation of run_benchmark.py.
    Stores configuration so any past run can be exactly reproduced.

results
    One row per (run, dataset, retrieval_method, sample).
    Metric columns use NULL — not 0.0 — when a metric is not applicable,
    so that SQL aggregations (AVG, COUNT) exclude them correctly.

Design principles
-----------------
- Every run receives a UUID run_id so runs are never confused.
- NULL is used for genuinely missing metrics; 0.0 is a measured value.
- has_retrieval_gt distinguishes "no ground truth" from "ground truth = miss".
- Results are committed in batches (one commit per dataset × method) to reduce
  SQLite I/O while preserving recoverability.
- The original benchmark.db is NEVER overwritten; a new file is used by default.
"""

import json
import logging
import sqlite3
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT    PRIMARY KEY,
    started_at    TEXT    NOT NULL,        -- ISO 8601 timestamp
    config        TEXT    NOT NULL         -- JSON blob of run configuration
);

CREATE TABLE IF NOT EXISTS results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT    NOT NULL REFERENCES runs(run_id),

    -- Experiment identifiers
    dataset            TEXT    NOT NULL,
    domain             TEXT    NOT NULL,
    split              TEXT    NOT NULL,
    retrieval_method   TEXT    NOT NULL,

    -- Sample
    sample_id          TEXT    NOT NULL,
    question           TEXT    NOT NULL,
    generated_answer   TEXT,

    -- Retrieval evaluation (NULL when has_retrieval_gt = 0)
    has_retrieval_gt   INTEGER NOT NULL DEFAULT 0,
    recall             REAL,              -- NULL if not applicable
    mrr                REAL,              -- NULL if not applicable
    ndcg               REAL,             -- NULL if not applicable

    -- Generation evaluation (NULL = not implemented / not run)
    faithfulness       REAL,
    hallucination      REAL,

    -- Performance
    response_time_s    REAL    NOT NULL,

    -- Configuration snapshot (redundant with runs.config but convenient)
    embedding_model    TEXT,
    generation_model   TEXT,
    chunk_size         INTEGER,
    chunk_overlap      INTEGER,
    top_k              INTEGER
);

CREATE INDEX IF NOT EXISTS idx_results_run_id
    ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_dataset_method
    ON results(dataset, retrieval_method);
"""


class BenchmarkDatabase:
    """
    Manage the benchmark SQLite database.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite file.  Defaults to results/benchmark_refactored.db
        to avoid overwriting the original database.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from config import DATABASE_PATH
            # Use a distinct file so the original benchmark.db is preserved
            db_path = DATABASE_PATH.parent / "benchmark_refactored.db"

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn   = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        self._create_schema()
        self._current_run_id: str | None = None
        logger.info("Database opened: %s", db_path)

    def _create_schema(self) -> None:
        self.conn.executescript(_DDL)
        self.conn.commit()

    # ── Run management ─────────────────────────────────────────────────────────

    def start_run(self, config: dict) -> str:
        """
        Register a new benchmark run and return its run_id (UUID4).

        Parameters
        ----------
        config : dict
            Full configuration dict that will be serialised to JSON.
        """
        from datetime import datetime, timezone
        run_id     = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()

        self.cursor.execute(
            "INSERT INTO runs (run_id, started_at, config) VALUES (?, ?, ?)",
            (run_id, started_at, json.dumps(config)),
        )
        self.conn.commit()
        self._current_run_id = run_id
        logger.info("Run started: run_id=%s  started_at=%s", run_id, started_at)
        return run_id

    @property
    def current_run_id(self) -> str:
        if self._current_run_id is None:
            raise RuntimeError("Call start_run() before inserting results.")
        return self._current_run_id

    # ── Result insertion ───────────────────────────────────────────────────────

    def insert_result(
        self,
        *,
        dataset:          str,
        domain:           str,
        split:            str,
        retrieval_method: str,
        sample_id:        str,
        question:         str,
        generated_answer: str | None,
        has_retrieval_gt: bool,
        recall:           float | None,
        mrr:              float | None,
        ndcg:             float | None,
        response_time_s:  float,
        embedding_model:  str  | None = None,
        generation_model: str  | None = None,
        chunk_size:       int  | None = None,
        chunk_overlap:    int  | None = None,
        top_k:            int  | None = None,
    ) -> None:
        """Insert one result row.  Commit is deferred — call commit_batch() after a batch."""
        self.cursor.execute(
            """
            INSERT INTO results (
                run_id, dataset, domain, split, retrieval_method,
                sample_id, question, generated_answer,
                has_retrieval_gt, recall, mrr, ndcg,
                faithfulness, hallucination,
                response_time_s,
                embedding_model, generation_model,
                chunk_size, chunk_overlap, top_k
            ) VALUES (
                ?,?,?,?,?, ?,?,?, ?,?,?,?, ?,?, ?, ?,?, ?,?,?
            )
            """,
            (
                self.current_run_id, dataset, domain, split, retrieval_method,
                sample_id, question, generated_answer,
                int(has_retrieval_gt), recall, mrr, ndcg,
                None, None,   # faithfulness / hallucination: not implemented
                response_time_s,
                embedding_model, generation_model,
                chunk_size, chunk_overlap, top_k,
            ),
        )

    def commit_batch(self) -> None:
        """Commit the current transaction (call after inserting a dataset×method batch)."""
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
        logger.info("Database closed.")

    # ── Query helpers (for inspection / reporting) ─────────────────────────────

    def summary(self) -> list[dict]:
        """Return per-dataset × method aggregate metrics for the current run."""
        if self._current_run_id is None:
            return []
        rows = self.cursor.execute(
            """
            SELECT dataset, retrieval_method,
                   COUNT(*) AS n,
                   AVG(recall) AS avg_recall,
                   AVG(mrr)    AS avg_mrr,
                   AVG(ndcg)   AS avg_ndcg,
                   AVG(response_time_s) AS avg_time_s
            FROM results
            WHERE run_id = ?
              AND has_retrieval_gt = 1
            GROUP BY dataset, retrieval_method
            ORDER BY dataset, retrieval_method
            """,
            (self._current_run_id,),
        ).fetchall()
        cols = ["dataset", "retrieval_method", "n",
                "avg_recall", "avg_mrr", "avg_ndcg", "avg_time_s"]
        return [dict(zip(cols, row)) for row in rows]
