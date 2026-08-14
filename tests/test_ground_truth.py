"""Ground-truth construction, including the guarantee that none is fabricated.

The central property under test: a chunk is marked relevant only because a
dataset evidence annotation overlaps it. Retrieval output must never influence
relevance.
"""

from __future__ import annotations

import pytest

from loaders import ADAPTERS, all_dataset_names, retrieval_capable_datasets
from loaders.casehold import parse_endings
from benchmark.chunking import relevant_chunk_ids
from benchmark.corpus import Qrel, build_corpus
from core.types import CorpusScope, EvidenceSpan, Query, RelevanceClass


class TestTaxonomy:
    def test_every_dataset_declares_a_class(self):
        for name, cls in ADAPTERS.items():
            assert isinstance(cls.spec.relevance_class, RelevanceClass), name

    def test_medqa_is_the_only_unsupported_dataset(self):
        unsupported = [
            n for n, c in ADAPTERS.items()
            if c.spec.relevance_class is RelevanceClass.UNSUPPORTED
        ]
        assert unsupported == ["medqa"]

    def test_no_dataset_uses_heuristic_relevance(self):
        # The class exists in the taxonomy but must remain unused: a heuristic
        # proxy would be exactly the fabricated ground truth this project bans.
        heuristic = [
            n for n, c in ADAPTERS.items()
            if c.spec.relevance_class is RelevanceClass.HEURISTIC
        ]
        assert heuristic == []

    def test_retrieval_capable_excludes_medqa(self):
        assert "medqa" not in retrieval_capable_datasets()
        assert len(retrieval_capable_datasets()) == 5

    def test_all_six_datasets_registered(self):
        assert all_dataset_names() == [
            "casehold", "cuad", "medqa", "pubmedqa", "qasper", "sciq"
        ]

    def test_every_spec_documents_limitations(self):
        for name, cls in ADAPTERS.items():
            assert len(cls.spec.limitations) > 40, name
            assert len(cls.spec.relevance_source) > 10, name

    def test_domains_cover_three_specialisms(self):
        domains = {c.spec.domain for c in ADAPTERS.values()}
        assert domains == {"legal", "medical", "scientific"}


class TestQueryGroundTruthFlag:
    def test_gold_query_with_evidence_is_scoreable(self, query_with_evidence):
        assert query_with_evidence.has_ground_truth is True

    def test_gold_query_without_evidence_is_not_scoreable(self):
        query = Query("q", "ds", "s", "text", relevance_class=RelevanceClass.GOLD)
        assert query.has_ground_truth is False

    def test_unsupported_query_is_never_scoreable(self):
        query = Query(
            "q", "ds", "s", "text",
            evidence=[EvidenceSpan("d", 0, 10)],
            relevance_class=RelevanceClass.UNSUPPORTED,
        )
        # Even if evidence were somehow attached, the class forbids scoring.
        assert query.has_ground_truth is False


class TestQrel:
    def test_round_trip(self):
        qrel = Qrel("q1", ["c1", "c2"], RelevanceClass.GOLD, True)
        assert Qrel.from_json(qrel.to_json()) == qrel

    def test_unavailable_carries_reason(self):
        qrel = Qrel("q1", [], RelevanceClass.DERIVED, False, "no support passage")
        restored = Qrel.from_json(qrel.to_json())
        assert restored.available is False
        assert restored.reason == "no support passage"


class TestNoFabrication:
    def test_relevance_ignores_retrieval_output(self, small_chunks):
        """The relevance function's signature admits no retrieval results."""
        import inspect

        parameters = set(inspect.signature(relevant_chunk_ids).parameters)
        assert parameters == {"evidence", "chunks", "min_overlap_chars"}

    def test_no_evidence_never_yields_relevant_chunks(self, small_chunks):
        assert relevant_chunk_ids([], small_chunks) == []

    def test_relevance_requires_matching_document(self, small_chunks):
        # A span on an unrelated document must not mark anything relevant even
        # though the offsets would overlap numerically.
        span = EvidenceSpan("toy/test/does-not-exist", 0, 1000)
        assert relevant_chunk_ids([span], small_chunks) == []


class TestCaseHOLDParsing:
    """The NumPy-repr parser, which a literal_eval implementation gets wrong."""

    RAW = (
        "['holding one here'\n 'holding two here'\n 'holding three here'\n"
        " 'holding four here'\n 'holding five here']"
    )

    def test_parses_five_elements(self):
        assert len(parse_endings(self.RAW)) == 5

    def test_preserves_content(self):
        assert parse_endings(self.RAW)[2] == "holding three here"

    def test_literal_eval_would_have_been_wrong(self):
        import ast

        # Documents the trap: adjacent string literals concatenate silently.
        collapsed = ast.literal_eval(self.RAW)
        assert len(collapsed) == 1
        assert len(parse_endings(self.RAW)) == 5

    def test_handles_double_quoted_elements(self):
        raw = "['it\\'s here' \"contains 'quotes'\"]"
        assert len(parse_endings(raw)) >= 1

    def test_malformed_input_returns_empty(self):
        assert parse_endings("") == []


@pytest.mark.slow
class TestRealDatasetGroundTruth:
    """Verifies the ground truth actually produced from the shipped files."""

    def _build(self, config, dataset, split, **limits):
        from core.settings import DatasetConfig

        config.datasets[dataset] = DatasetConfig(**limits)
        return build_corpus(config, dataset, split)

    def test_cuad_spans_land_inside_their_chunks(self, config, real_datasets_available):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        build = self._build(config, "cuad", "all", max_documents=2)
        chunk_by_id = {c.chunk_id: c for c in build.chunks}
        checked = 0
        for query in build.queries:
            qrel = build.qrels[query.query_id]
            if not qrel.available:
                continue
            for chunk_id in qrel.relevant_chunk_ids:
                chunk = chunk_by_id[chunk_id]
                assert any(
                    chunk.overlaps(span.start, span.end) > 0
                    for span in query.evidence
                    if span.doc_id == chunk.doc_id
                )
                checked += 1
        assert checked > 0

    def test_medqa_produces_no_relevance_at_all(self, config, real_datasets_available):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        build = self._build(config, "medqa", "test", max_documents=1, max_queries=5)
        assert all(not q.available for q in build.qrels.values())
        assert all(q.relevant_chunk_ids == [] for q in build.qrels.values())
        assert all(
            q.relevance_class is RelevanceClass.UNSUPPORTED for q in build.qrels.values()
        )

    def test_sciq_rows_without_support_are_unavailable(self, config, real_datasets_available):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        build = self._build(config, "sciq", "test")
        unavailable = [q for q in build.qrels.values() if not q.available]
        assert len(unavailable) > 0
        for qrel in unavailable:
            assert qrel.relevant_chunk_ids == []
            assert qrel.reason

    def test_qasper_evidence_maps_to_paragraph_chunks(self, config, real_datasets_available):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        build = self._build(config, "qasper", "val", max_documents=5)
        scoreable = [q for q in build.qrels.values() if q.available]
        assert len(scoreable) > 0
        chunk_ids = {c.chunk_id for c in build.chunks}
        for qrel in scoreable:
            assert set(qrel.relevant_chunk_ids) <= chunk_ids

    def test_document_scoped_relevance_stays_in_its_document(
        self, config, real_datasets_available
    ):
        if not real_datasets_available:
            pytest.skip("datasets/ not present")
        build = self._build(config, "cuad", "all", max_documents=3)
        assert build.scope is CorpusScope.DOCUMENT
        chunk_doc = {c.chunk_id: c.doc_id for c in build.chunks}
        for query in build.queries:
            qrel = build.qrels[query.query_id]
            for chunk_id in qrel.relevant_chunk_ids:
                assert chunk_doc[chunk_id] == query.scope_doc_id
