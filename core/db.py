"""SQLite results store and experiment tracking.

Design goals, in order: every number traceable to the run that produced it,
NULL representable everywhere a metric can be unavailable, and runs never
silently overwriting each other.

Schema (five tables):

  runs              one row per benchmark invocation, with the full resolved
                    configuration and environment captured as JSON
  retrieval_results one row per (run, query, rank) -- the actual ranked chunks,
                    so results can be re-inspected without re-running
  query_metrics     one row per (run, query) with nullable metric columns
  aggregate_metrics one row per (run, dataset, method) with nullable means
  generations       one row per (run, dataset, method, model, query) generated
                    answer, nullable quality columns. ``model`` is part of the
                    key so several LLMs can answer the same question in one run
                    and stay separable for comparison.

Metric columns are REAL and nullable on purpose: an unavailable metric is
stored as SQL NULL, never as 0.0.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .settings import Config
from .logging_setup import get_logger
from evaluation.metrics import AggregateMetrics, QueryMetrics
from .types import RetrievedChunk

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    stage             TEXT NOT NULL,
    dataset           TEXT,
    split             TEXT,
    methods           TEXT,
    embedding_model   TEXT,
    generation_model  TEXT,
    chunk_strategy    TEXT,
    chunk_size        INTEGER,
    chunk_overlap     INTEGER,
    top_k             INTEGER,
    rrf_k             INTEGER,
    rrf_depth         INTEGER,
    relevance_class   TEXT,
    corpus_scope      TEXT,
    corpus_fingerprint TEXT,
    n_documents       INTEGER,
    n_chunks          INTEGER,
    n_queries         INTEGER,
    config_json       TEXT NOT NULL,
    environment_json  TEXT NOT NULL,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    run_id     TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    method     TEXT NOT NULL,
    query_id   TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    chunk_id   TEXT NOT NULL,
    score      REAL,
    is_relevant INTEGER,     -- 1/0 when judged, NULL when unjudgeable
    PRIMARY KEY (run_id, method, query_id, rank),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS query_metrics (
    run_id       TEXT NOT NULL,
    dataset      TEXT NOT NULL,
    method       TEXT NOT NULL,
    query_id     TEXT NOT NULL,
    scoreable    INTEGER NOT NULL,
    n_relevant   INTEGER NOT NULL,
    mrr          REAL,
    metrics_json TEXT NOT NULL,
    reason       TEXT,
    PRIMARY KEY (run_id, method, query_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS aggregate_metrics (
    run_id          TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    domain          TEXT,
    method          TEXT NOT NULL,
    relevance_class TEXT NOT NULL,
    n_queries       INTEGER NOT NULL,
    n_scoreable     INTEGER NOT NULL,
    n_unscoreable   INTEGER NOT NULL,
    coverage        REAL,
    mrr             REAL,
    metrics_json    TEXT NOT NULL,
    PRIMARY KEY (run_id, dataset, method),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS generations (
    run_id           TEXT NOT NULL,
    dataset          TEXT NOT NULL,
    method           TEXT NOT NULL,
    model            TEXT NOT NULL,   -- the Ollama model that produced `answer`
    query_id         TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    answer           TEXT,
    reference_answer TEXT,
    context_chunk_ids TEXT,
    latency_ms       REAL,
    exact_match      REAL,   -- NULL where the dataset has no checkable answer
    choice_correct   REAL,   -- NULL where the dataset is not multiple choice
    faithfulness     REAL,   -- RAGAS faithfulness; NULL until scored
    faithfulness_judge TEXT, -- judge model, so a score is traceable to its rater
    faithfulness_json  TEXT, -- per-statement verdicts behind the score
    hallucination    REAL,   -- NLI contradiction/unsupported rate; NULL until scored
    hallucination_model TEXT,-- the NLI cross-encoder used
    hallucination_json  TEXT,-- per-sentence NLI labels behind the score
    error            TEXT,
    -- `dataset` belongs in the key: without it, two datasets that happen to
    -- share a query_id silently overwrite each other. Ids are namespaced by
    -- dataset in practice, but the schema must not depend on that convention.
    PRIMARY KEY (run_id, dataset, method, model, query_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run   ON retrieval_results(run_id, dataset, method);
CREATE INDEX IF NOT EXISTS idx_qmetrics_run  ON query_metrics(run_id, dataset, method);
CREATE INDEX IF NOT EXISTS idx_agg_dataset   ON aggregate_metrics(dataset, method);
CREATE INDEX IF NOT EXISTS idx_gen_model     ON generations(model, dataset, method);
"""

# Columns added to `generations` after the first release. Older databases are
# upgraded in place by add_missing_columns(); `model` is handled separately
# because it also belongs to the primary key.
BUSY_TIMEOUT_SECONDS = 30.0

GENERATION_PRIMARY_KEY = ("run_id", "dataset", "method", "model", "query_id")

GENERATION_COLUMNS = {
    "model": "TEXT",
    "faithfulness_judge": "TEXT",
    "faithfulness_json": "TEXT",
    "hallucination_model": "TEXT",
    "hallucination_json": "TEXT",
}


# Module name -> the distribution that provides it, for version lookup.
TRACKED_PACKAGES = {
    "numpy": "numpy",
    "faiss": "faiss-cpu",
    "torch": "torch",
    "sentence_transformers": "sentence-transformers",
    "rank_bm25": "rank-bm25",
    "pandas": "pandas",
}


def environment_info() -> Dict[str, Any]:
    """Version information recorded with every run, for reproducibility.

    Versions come from installed distribution metadata rather than by importing
    each package. Importing them merely to read ``__version__`` pulls megabytes
    of native extensions into any process that writes a run row -- including
    the test suite, where importing pandas (and through it pyarrow) alongside
    faiss and torch reliably crashed the interpreter with an access violation
    on Windows. Metadata lookup records the same fact without loading a thing.
    """
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module, distribution in TRACKED_PACKAGES.items():
        try:
            info[module] = version(distribution)
        except PackageNotFoundError:
            info[module] = None

    # CUDA state is a runtime property, not metadata, so this one genuinely
    # needs the import. torch is a core dependency that every real run loads
    # anyway; pandas -- the package that made this function dangerous -- is not
    # imported here at all.
    try:
        import torch

        available = bool(torch.cuda.is_available())
        info["cuda_available"] = available
        info["cuda_device"] = torch.cuda.get_device_name(0) if available else None
    except Exception:
        info["cuda_available"] = None
        info["cuda_device"] = None
    return info


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        # Long generation sweeps hold this database for hours, so wait for a
        # busy writer instead of failing the moment one is mid-commit.
        self.connection.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
        # Readers must not block writers. Backing this database up to Google
        # Drive holds a read lock for as long as the copy takes, and copying
        # several hundred megabytes over a FUSE mount takes minutes -- longer
        # than any busy timeout worth setting. Under the default rollback
        # journal that stalls the sweep and eventually fails a commit with
        # "database is locked", which ends the run rather than one answer. In
        # WAL a reader and a writer proceed concurrently, so the periodic
        # backup and the sweep stop contending.
        #
        # WAL is unsafe on a network filesystem. This is correct only because
        # the live database is always on local disk; only the copy goes to
        # Drive.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate_generations()
        try:
            self.connection.executescript(SCHEMA)
            self.connection.commit()
        except sqlite3.OperationalError as exc:
            # Only reachable when another process holds the write lock, which
            # means the tables it is writing to already exist. Opening
            # read-only alongside a running sweep is a normal thing to do.
            log.warning("database busy, skipped schema check (%s)", exc)
            self.connection.rollback()

    def _migrate_generations(self) -> None:
        """Bring an older ``generations`` table up to the current schema.

        Two historical shapes are repaired. The original table had no ``model``
        column, so it could not hold two models' answers to one question; it
        also keyed on (run_id, method, query_id), omitting ``dataset``, so two
        datasets sharing a query_id would overwrite each other. Rebuilding is
        the only way to change a SQLite primary key, so rows are copied into a
        correctly-keyed table. Every column present in both shapes is carried
        across; a missing ``model`` is filled from the run's recorded
        ``generation_model``. Retrieval tables are never touched.
        """
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='generations'"
        ).fetchone()
        if existing is None:
            return
        info = list(self.connection.execute("PRAGMA table_info(generations)"))
        columns = {row["name"] for row in info}
        primary_key = tuple(
            row["name"] for row in sorted(
                (r for r in info if r["pk"]), key=lambda r: r["pk"]
            )
        )
        if "model" in columns and primary_key == GENERATION_PRIMARY_KEY:
            for name, decl in GENERATION_COLUMNS.items():
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE generations ADD COLUMN {name} {decl}"
                    )
            self.connection.commit()
            return

        n_rows = self.connection.execute("SELECT count(*) FROM generations").fetchone()[0]
        log.info(
            "migrating generations table (%d rows): key %s -> %s",
            n_rows, primary_key or "(none)", GENERATION_PRIMARY_KEY,
        )
        conn = self.connection
        # A rebuild renames and drops a table another process may be writing to
        # (a multi-hour `generate-all` holds the database for days). Take the
        # write lock up front: if a writer is active, leave the old table alone
        # and carry on read-only rather than pulling it out from under them.
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            log.warning(
                "generations table needs migrating but the database is busy (%s); "
                "leaving it unchanged. Re-open it when no run is writing.", exc,
            )
            return

        conn.execute("ALTER TABLE generations RENAME TO generations_legacy")
        conn.executescript(SCHEMA)

        target = [
            row["name"] for row in conn.execute("PRAGMA table_info(generations)")
        ]
        carried = [name for name in target if name in columns and name != "model"]
        selected = [f"g.{name}" for name in carried]
        if "model" in columns:
            carried.append("model")
            selected.append("g.model")
        else:
            # A legacy row's model is whatever its run declared; '' marks a row
            # whose producing model was never recorded, rather than guessing.
            carried.append("model")
            selected.append("COALESCE(r.generation_model, '')")

        conn.execute(
            f"INSERT OR REPLACE INTO generations ({', '.join(carried)}) "
            f"SELECT {', '.join(selected)} "
            "FROM generations_legacy g LEFT JOIN runs r ON r.run_id = g.run_id"
        )
        migrated = conn.execute("SELECT count(*) FROM generations").fetchone()[0]
        conn.execute("DROP TABLE generations_legacy")
        conn.commit()
        if migrated != n_rows:
            log.warning(
                "generations migration kept %d of %d rows; the surplus were "
                "duplicates under the old, narrower key",
                migrated, n_rows,
            )
        log.info("generations table migrated (%d rows)", migrated)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    # -- writes -------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        stage: str,
        config: Config,
        *,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        methods: Sequence[str] = (),
        relevance_class: Optional[str] = None,
        corpus_scope: Optional[str] = None,
        corpus_fingerprint: Optional[str] = None,
        n_documents: Optional[int] = None,
        n_chunks: Optional[int] = None,
        n_queries: Optional[int] = None,
        notes: str = "",
    ) -> str:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, stage, dataset, split, methods,
                    embedding_model, generation_model, chunk_strategy, chunk_size,
                    chunk_overlap, top_k, rrf_k, rrf_depth, relevance_class,
                    corpus_scope, corpus_fingerprint, n_documents, n_chunks,
                    n_queries, config_json, environment_json, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    stage, dataset, split, ",".join(methods),
                    config.embedding.model_name,
                    config.generation.model if config.generation.enabled else None,
                    config.chunk.strategy, config.chunk.size_tokens,
                    config.chunk.overlap_tokens, config.retrieval.top_k,
                    config.retrieval.rrf_k, config.retrieval.rrf_candidate_depth,
                    relevance_class, corpus_scope, corpus_fingerprint,
                    n_documents, n_chunks, n_queries,
                    json.dumps(config.to_dict(), default=str),
                    json.dumps(environment_info()),
                    notes,
                ),
            )
        log.info("created run %s (stage=%s dataset=%s)", run_id, stage, dataset)
        return run_id

    def save_retrieval_results(
        self,
        run_id: str,
        dataset: str,
        method: str,
        query_id: str,
        retrieved: Sequence[RetrievedChunk],
        relevant: Optional[set] = None,
    ) -> None:
        """Persist a ranked list. ``relevant=None`` marks judgements unavailable."""
        rows = [
            (
                run_id, dataset, method, query_id, item.rank, item.chunk_id, item.score,
                None if relevant is None else int(item.chunk_id in relevant),
            )
            for item in retrieved
        ]
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO retrieval_results "
                "(run_id, dataset, method, query_id, rank, chunk_id, score, is_relevant) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )

    def save_query_metrics(
        self, run_id: str, dataset: str, method: str, results: Iterable[QueryMetrics]
    ) -> None:
        rows = [
            (
                run_id, dataset, method, r.query_id, int(r.scoreable), r.n_relevant, r.mrr,
                json.dumps(
                    {
                        "recall": {str(k): v for k, v in r.recall.items()},
                        "ndcg": {str(k): v for k, v in r.ndcg.items()},
                    }
                ),
                r.reason,
            )
            for r in results
        ]
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO query_metrics "
                "(run_id, dataset, method, query_id, scoreable, n_relevant, mrr, "
                "metrics_json, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def save_aggregate(
        self,
        run_id: str,
        dataset: str,
        domain: str,
        method: str,
        relevance_class: str,
        aggregate: AggregateMetrics,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO aggregate_metrics "
                "(run_id, dataset, domain, method, relevance_class, n_queries, "
                "n_scoreable, n_unscoreable, coverage, mrr, metrics_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, dataset, domain, method, relevance_class,
                    aggregate.n_queries, aggregate.n_scoreable, aggregate.n_unscoreable,
                    aggregate.coverage, aggregate.mrr,
                    json.dumps(
                        {
                            "recall": {str(k): v for k, v in aggregate.recall.items()},
                            "ndcg": {str(k): v for k, v in aggregate.ndcg.items()},
                        }
                    ),
                ),
            )

    def save_generation(
        self, run_id: str, dataset: str, method: str, model: str, record: Dict[str, Any]
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO generations "
                "(run_id, dataset, method, model, query_id, prompt, answer, "
                "reference_answer, context_chunk_ids, latency_ms, exact_match, "
                "choice_correct, faithfulness, faithfulness_judge, faithfulness_json, "
                "hallucination, hallucination_model, hallucination_json, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, dataset, method, model, record["query_id"], record["prompt"],
                    record.get("answer"), record.get("reference_answer"),
                    json.dumps(record.get("context_chunk_ids", [])),
                    record.get("latency_ms"), record.get("exact_match"),
                    record.get("choice_correct"), record.get("faithfulness"),
                    record.get("faithfulness_judge"), record.get("faithfulness_json"),
                    record.get("hallucination"), record.get("hallucination_model"),
                    record.get("hallucination_json"), record.get("error"),
                ),
            )

    def completed_generations(self, run_id: str) -> set:
        """(dataset, method, model, query_id) keys already generated for a run.

        This is what makes a multi-hour generation run resumable: an interrupted
        run is restarted with the same run_id and skips what it already has.
        """
        return {
            (r["dataset"], r["method"], r["model"], r["query_id"])
            for r in self.connection.execute(
                "SELECT dataset, method, model, query_id FROM generations "
                "WHERE run_id = ? AND answer IS NOT NULL",
                (run_id,),
            )
        }

    def generations_needing_scores(
        self, column: str, run_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[sqlite3.Row]:
        """Answered rows whose ``column`` score is still NULL."""
        if column not in ("faithfulness", "hallucination"):
            raise ValueError(f"not a scoreable column: {column}")
        sql = (
            "SELECT run_id, dataset, method, model, query_id, prompt, answer, "
            "context_chunk_ids FROM generations "
            f"WHERE answer IS NOT NULL AND answer != '' AND {column} IS NULL"
        )
        params: List[Any] = []
        if run_id:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY dataset, method, model, query_id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.connection.execute(sql, params).fetchall()

    def update_generation_score(
        self,
        run_id: str,
        dataset: str,
        method: str,
        model: str,
        query_id: str,
        column: str,
        value: Optional[float],
        rater: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        if column not in ("faithfulness", "hallucination"):
            raise ValueError(f"not a scoreable column: {column}")
        rater_column = (
            "faithfulness_judge" if column == "faithfulness" else "hallucination_model"
        )
        detail_column = f"{column}_json"
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE generations SET {column} = ?, {rater_column} = ?, "
                f"{detail_column} = ? WHERE run_id = ? AND dataset = ? AND method = ? "
                "AND model = ? AND query_id = ?",
                (value, rater, detail, run_id, dataset, method, model, query_id),
            )

    # -- reads --------------------------------------------------------------

    def runs(self, limit: int = 50) -> List[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def aggregates(self, run_id: Optional[str] = None) -> List[sqlite3.Row]:
        if run_id:
            return self.connection.execute(
                "SELECT * FROM aggregate_metrics WHERE run_id = ? ORDER BY dataset, method",
                (run_id,),
            ).fetchall()
        return self.connection.execute(
            "SELECT a.*, r.created_at FROM aggregate_metrics a "
            "JOIN runs r ON r.run_id = a.run_id ORDER BY r.created_at DESC, dataset, method"
        ).fetchall()

    def generation_aggregates(self, run_id: Optional[str] = None) -> List[sqlite3.Row]:
        """Per (dataset, method, model) generation means.

        Averages ignore NULLs, so a dataset that is not multiple choice
        contributes no accuracy rather than a zero, matching how the retrieval
        side treats unscoreable queries.
        """
        sql = """
            SELECT dataset, method, model,
                   count(*)                          AS n_rows,
                   sum(answer IS NOT NULL)           AS n_answered,
                   sum(error IS NOT NULL)            AS n_errors,
                   avg(choice_correct)               AS choice_correct,
                   count(choice_correct)             AS n_choice,
                   avg(exact_match)                  AS exact_match,
                   count(exact_match)                AS n_exact,
                   avg(faithfulness)                 AS faithfulness,
                   count(faithfulness)               AS n_faithfulness,
                   avg(hallucination)                AS hallucination,
                   count(hallucination)              AS n_hallucination,
                   avg(latency_ms)                   AS latency_ms
            FROM generations
        """
        params: List[Any] = []
        if run_id:
            sql += " WHERE run_id = ?"
            params.append(run_id)
        sql += " GROUP BY dataset, method, model ORDER BY dataset, method, model"
        return self.connection.execute(sql, params).fetchall()

    def results_for_query(self, run_id: str, method: str, query_id: str) -> List[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM retrieval_results WHERE run_id=? AND method=? AND query_id=? "
            "ORDER BY rank",
            (run_id, method, query_id),
        ).fetchall()


def open_database(config: Config) -> Database:
    return Database(config.paths.database_path)
