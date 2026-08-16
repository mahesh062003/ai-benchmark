"""End-to-end tests: corpus -> index -> retrieval -> metrics -> database.

The synthetic pipeline test uses a fake adapter so it runs offline in
milliseconds. Real-data tests are marked slow.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from loaders import ADAPTERS, get_adapter
from loaders.base import DatasetAdapter, LoadedSplit
from benchmark.runner import run_retrieval_benchmark
from core.settings import ChunkConfig, DatasetConfig
from benchmark.corpus import (
    QueryMismatch,
    build_corpus,
    load_corpus,
    refresh_query_metadata,
    save_corpus,
)
from core.db import Database
from core.logging_setup import SkipLog
from core.types import (
    CorpusScope,
    DatasetSpec,
    EvidenceSpan,
    Query,
    RelevanceClass,
    SourceDocument,
)


# --------------------------------------------------------------------------
# a tiny synthetic dataset, registered only for the duration of a test
# --------------------------------------------------------------------------

class FakeAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="fake", domain="scientific", relevance_class=RelevanceClass.GOLD,
        corpus_scope=CorpusScope.POOLED, splits=["test"],
        query_unit="question", corpus_unit="passage",
        relevance_source="synthetic fixture with known answers",
        limitations="synthetic fixture used only for pipeline tests",
    )
    default_chunk = ChunkConfig(strategy="passage", max_tokens=64, min_chunk_chars=1)

    TEXTS = [
        "photosynthesis converts light energy into chemical energy in chloroplasts",
        "mitochondria are the primary site of cellular respiration and atp synthesis",
        "the nucleus stores genetic material as chromatin within a nuclear envelope",
        "ribosomes translate messenger rna into polypeptide chains during synthesis",
    ]
    QUESTIONS = [
        ("what converts light energy?", 0),
        ("where does cellular respiration occur?", 1),
        ("what stores genetic material?", 2),
    ]

    def load(self, split: str = "test") -> LoadedSplit:
        documents = [
            SourceDocument(f"fake/test/d{i}", "fake", "test", text, title=f"doc {i}")
            for i, text in enumerate(self.TEXTS)
        ]
        queries = []
        for i, (text, target) in enumerate(self.QUESTIONS):
            queries.append(Query(
                query_id=f"fake/test/q/{i}", dataset="fake", split="test", text=text,
                evidence=[EvidenceSpan(f"fake/test/d{target}", 0, len(self.TEXTS[target]))],
                relevance_class=RelevanceClass.GOLD,
            ))
        # One query with no evidence: must become NULL, not zero.
        queries.append(Query(
            query_id="fake/test/q/nogt", dataset="fake", split="test",
            text="a question with no annotation", evidence=[],
            relevance_class=RelevanceClass.GOLD,
        ))
        return LoadedSplit(documents, queries, SkipLog("fake", "test"), {"documents": 4})


@pytest.fixture
def fake_dataset():
    ADAPTERS["fake"] = FakeAdapter
    yield "fake"
    ADAPTERS.pop("fake", None)


@pytest.fixture
def stub_embedder(monkeypatch):
    """Replace the real embedding model with a deterministic bag-of-words stub."""
    import numpy as np

    from benchmark import runner as benchmark_module
    from retrieval.bm25 import tokenize

    vocabulary = sorted({
        t for text in FakeAdapter.TEXTS + [q for q, _ in FakeAdapter.QUESTIONS]
        for t in tokenize(text)
    })
    index = {w: i for i, w in enumerate(vocabulary)}

    class Stub:
        class config:
            model_name = "stub-embedder"
            normalize = True
            batch_size = 8

        dimension = len(vocabulary)

        def encode(self, texts, batch_size=None, progress=False):
            out = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
            for row, text in enumerate(texts):
                for token in tokenize(text):
                    if token in index:
                        out[row, index[token]] = 1.0
                norm = np.linalg.norm(out[row])
                if norm:
                    out[row] /= norm
            return out

        def encode_query(self, text):
            return self.encode([text])[0]

    monkeypatch.setattr(benchmark_module, "get_embedder", lambda _config: Stub())
    return Stub()


class TestPipeline:
    def test_corpus_build_produces_expected_qrels(self, config, fake_dataset):
        build = build_corpus(config, "fake", "test")
        assert len(build.chunks) == 4
        scoreable = [q for q in build.qrels.values() if q.available]
        assert len(scoreable) == 3
        unavailable = build.qrels["fake/test/q/nogt"]
        assert unavailable.available is False
        assert unavailable.relevant_chunk_ids == []
        assert unavailable.reason

    def test_refresh_updates_metadata_and_leaves_ground_truth_alone(
        self, config, fake_dataset
    ):
        """A loader that gains metadata must reach an already-built corpus.

        Queries are cached on disk and reloaded, so without this a loader
        improvement is invisible to the next generation run.
        """
        build = build_corpus(config, "fake", "test")
        directory = save_corpus(config, build)
        qrels_before = (directory / "qrels.jsonl").read_bytes()
        chunks_before = (directory / "chunks.jsonl").read_bytes()

        class WithOptions(FakeAdapter):
            def load(self, split: str = "test") -> LoadedSplit:
                loaded = super().load(split)
                for query in loaded.queries:
                    query.metadata["options"] = {"A": "first", "B": "second"}
                return loaded

        ADAPTERS["fake"] = WithOptions
        stats = refresh_query_metadata(config, "fake", "test", build.fingerprint)

        assert stats["updated"] == stats["queries"]
        assert (directory / "qrels.jsonl").read_bytes() == qrels_before
        assert (directory / "chunks.jsonl").read_bytes() == chunks_before
        restored = load_corpus(config, "fake", "test", build.fingerprint)
        assert all(q.metadata["options"] == {"A": "first", "B": "second"}
                   for q in restored.queries)

    def test_refresh_discards_frozen_task_sets(self, config, fake_dataset):
        """A task set embeds its own copy of query metadata.

        Leaving one in place after refreshing the corpus feeds the next
        generation run the metadata that was just replaced, and the run looks
        entirely normal while doing it.
        """
        build = build_corpus(config, "fake", "test")
        save_corpus(config, build)
        results = config.paths.results_dir
        results.mkdir(parents=True, exist_ok=True)
        stale = results / "generation_tasks_abc123.json"
        stale.write_text('{"tasks": []}', encoding="utf-8")

        stats = refresh_query_metadata(config, "fake", "test", build.fingerprint)

        assert stats["task_sets_removed"] == 1
        assert not stale.exists()

    def test_task_sets_go_even_when_the_metadata_was_already_current(
        self, config, fake_dataset
    ):
        """The corpus can be up to date while the task set is stale from elsewhere."""
        build = build_corpus(config, "fake", "test")
        save_corpus(config, build)
        results = config.paths.results_dir
        results.mkdir(parents=True, exist_ok=True)
        stale = results / "generation_tasks_def456.json"
        stale.write_text('{"tasks": []}', encoding="utf-8")

        stats = refresh_query_metadata(config, "fake", "test", build.fingerprint)

        assert stats["updated"] == 0, "nothing to change in the corpus itself"
        assert stats["task_sets_removed"] == 1, "the task set must still go"
        assert not stale.exists()

    def test_refresh_refuses_when_the_evidence_itself_changed(self, config, fake_dataset):
        """Rewriting there would invalidate results already in the database."""
        build = build_corpus(config, "fake", "test")
        directory = save_corpus(config, build)
        before = (directory / "queries.jsonl").read_bytes()

        class MovedEvidence(FakeAdapter):
            def load(self, split: str = "test") -> LoadedSplit:
                loaded = super().load(split)
                loaded.queries[0] = dataclasses.replace(
                    loaded.queries[0], evidence=[EvidenceSpan("fake/test/d3", 0, 5)]
                )
                return loaded

        ADAPTERS["fake"] = MovedEvidence
        with pytest.raises(QueryMismatch, match="more than metadata"):
            refresh_query_metadata(config, "fake", "test", build.fingerprint)

        assert (directory / "queries.jsonl").read_bytes() == before, \
            "nothing may be written when the check fails"

    def test_corpus_round_trip(self, config, fake_dataset):
        build = build_corpus(config, "fake", "test")
        save_corpus(config, build)
        restored = load_corpus(config, "fake", "test", build.fingerprint)
        assert [c.chunk_id for c in restored.chunks] == [c.chunk_id for c in build.chunks]
        assert restored.qrels.keys() == build.qrels.keys()
        assert restored.scope is build.scope
        assert restored.chunk_config.fingerprint() == build.fingerprint

    def test_missing_corpus_raises_actionable_error(self, config, fake_dataset):
        with pytest.raises(FileNotFoundError, match="build-corpus"):
            load_corpus(config, "fake", "test", "nonexistent-fingerprint")

    def test_full_benchmark_all_methods(self, config, fake_dataset, stub_embedder, tmp_path):
        build = build_corpus(config, "fake", "test")
        database = Database(tmp_path / "bench.sqlite")
        outcomes = run_retrieval_benchmark(
            config, build, ["bm25", "dense", "hybrid"], database=database,
        )
        assert set(outcomes) == {"bm25", "dense", "hybrid"}
        for method, outcome in outcomes.items():
            # 3 scoreable + 1 unscoreable
            assert outcome.aggregate.n_queries == 4
            assert outcome.aggregate.n_scoreable == 3
            assert outcome.aggregate.n_unscoreable == 1
            # The fixture is easy; every method should find the answer.
            assert outcome.aggregate.recall[10] == 1.0, method
            assert outcome.aggregate.mrr == 1.0, method
        database.close()

    def test_benchmark_persists_traceable_results(
        self, config, fake_dataset, stub_embedder, tmp_path
    ):
        build = build_corpus(config, "fake", "test")
        database = Database(tmp_path / "bench.sqlite")
        run_retrieval_benchmark(config, build, ["bm25"], database=database)

        run_id = database.runs()[0]["run_id"]
        # dataset -> query -> retrieved chunk -> relevance -> metric
        rows = database.results_for_query(run_id, "bm25", "fake/test/q/0")
        assert rows and rows[0]["chunk_id"].startswith("fake/test/d")
        assert rows[0]["is_relevant"] in (0, 1)

        metrics = database.connection.execute(
            "SELECT * FROM query_metrics WHERE run_id=? AND query_id=?",
            (run_id, "fake/test/q/nogt"),
        ).fetchone()
        assert metrics["scoreable"] == 0
        assert metrics["mrr"] is None
        database.close()

    def test_unscoreable_query_still_retrieves_but_scores_null(
        self, config, fake_dataset, stub_embedder, tmp_path
    ):
        build = build_corpus(config, "fake", "test")
        database = Database(tmp_path / "bench.sqlite")
        run_retrieval_benchmark(config, build, ["bm25"], database=database)
        run_id = database.runs()[0]["run_id"]
        rows = database.results_for_query(run_id, "bm25", "fake/test/q/nogt")
        assert len(rows) > 0                       # results were produced
        assert all(r["is_relevant"] is None for r in rows)  # but not judged
        database.close()

    def test_indexes_persist_and_reload(self, config, fake_dataset, stub_embedder):
        from benchmark.runner import bm25_path, build_indexes, faiss_path

        build = build_corpus(config, "fake", "test")
        build_indexes(config, build, ["bm25", "dense"])
        assert bm25_path(config, build).exists()
        assert faiss_path(config, build).exists()
        # Second call must load rather than rebuild, and still work.
        again = build_indexes(config, build, ["bm25", "dense"])
        assert again["bm25"].search("light energy", 2)

    def test_different_chunk_config_yields_separate_build(self, config, fake_dataset):
        build = build_corpus(config, "fake", "test")
        save_corpus(config, build)
        config.datasets["fake"] = DatasetConfig(
            chunk=ChunkConfig(strategy="fixed", size_tokens=4, overlap_tokens=1)
        )
        other = build_corpus(config, "fake", "test")
        assert other.fingerprint != build.fingerprint
        assert len(other.chunks) > len(build.chunks)


@pytest.mark.slow
class TestRealDataSmoke:
    def test_small_cuad_benchmark(self, config, real_datasets_available, tmp_path):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        config.datasets["cuad"] = DatasetConfig(max_documents=2)
        build = build_corpus(config, "cuad", "all")
        database = Database(tmp_path / "smoke.sqlite")
        outcomes = run_retrieval_benchmark(config, build, ["bm25"], database=database)
        aggregate = outcomes["bm25"].aggregate
        assert aggregate.n_scoreable > 0
        assert aggregate.n_unscoreable > 0   # is_impossible queries
        assert aggregate.recall[10] is not None
        assert 0.0 <= aggregate.recall[10] <= 1.0
        database.close()

    def test_medqa_benchmark_yields_null_metrics(
        self, config, real_datasets_available, tmp_path
    ):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        config.datasets["medqa"] = DatasetConfig(max_documents=1, max_queries=5)
        build = build_corpus(config, "medqa", "test")
        database = Database(tmp_path / "smoke.sqlite")
        outcomes = run_retrieval_benchmark(config, build, ["bm25"], database=database)
        aggregate = outcomes["bm25"].aggregate
        # The whole point: retrieval ran, but nothing is scored.
        assert aggregate.n_scoreable == 0
        assert aggregate.recall[10] is None
        assert aggregate.mrr is None
        assert aggregate.coverage == 0.0

        row = database.aggregates()[0]
        assert row["mrr"] is None
        assert json.loads(row["metrics_json"])["recall"]["10"] is None
        database.close()
