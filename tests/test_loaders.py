"""Dataset adapters against the real shipped files.

Marked slow: these read the actual dataset directory and skip if it is absent.
Every adapter and every split it declares is exercised. Counts are asserted
against the figures recorded during the dataset audit, so a silent change in
the data or in parsing is caught.
"""

from __future__ import annotations

import pytest

from loaders import ADAPTERS, get_adapter
from loaders.base import build_options
from core.settings import DatasetConfig
from core.types import RelevanceClass

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def require_datasets(real_datasets_available):
    if not real_datasets_available:
        pytest.skip("datasets/ not present")


def load(config, name, split, **limits):
    if limits:
        config.datasets[name] = DatasetConfig(**limits)
    return get_adapter(name, config).load(split)


class TestCUAD:
    def test_loads_all_contracts(self, config):
        loaded = load(config, "cuad", "all")
        assert len(loaded.documents) == 510
        assert len(loaded.queries) == 20910

    def test_audited_evidence_counts(self, config):
        loaded = load(config, "cuad", "all")
        assert loaded.stats["queries_with_evidence"] == 6702
        assert loaded.stats["queries_is_impossible"] == 14208
        # Every shipped answer offset reproduced its text exactly.
        assert loaded.stats["answer_span_mismatches"] == 0

    def test_spans_reproduce_document_text(self, config):
        loaded = load(config, "cuad", "all", max_documents=5)
        by_id = {d.doc_id: d for d in loaded.documents}
        for query in loaded.queries:
            for span in query.evidence:
                assert 0 <= span.start < span.end <= len(by_id[span.doc_id].text)

    def test_impossible_queries_have_no_evidence(self, config):
        loaded = load(config, "cuad", "all", max_documents=5)
        for query in loaded.queries:
            if query.metadata["is_impossible"]:
                assert query.evidence == []
                assert query.has_ground_truth is False

    def test_queries_are_document_scoped(self, config):
        loaded = load(config, "cuad", "all", max_documents=3)
        doc_ids = {d.doc_id for d in loaded.documents}
        assert all(q.scope_doc_id in doc_ids for q in loaded.queries)

    def test_rejects_unknown_split(self, config):
        with pytest.raises(ValueError, match="only the 'all' split"):
            load(config, "cuad", "train")


class TestCaseHOLD:
    @pytest.mark.parametrize("split,expected", [
        ("test", 3600), ("validation", 3900), ("train", 45000),
    ])
    def test_row_counts_match_audit(self, config, split, expected):
        loaded = load(config, "casehold", split)
        assert len(loaded.queries) == expected

    def test_every_query_has_exactly_one_gold_holding(self, config):
        loaded = load(config, "casehold", "test", max_documents=200)
        assert all(len(q.evidence) == 1 for q in loaded.queries)

    def test_all_five_candidates_enter_the_corpus(self, config):
        loaded = load(config, "casehold", "test", max_documents=50)
        doc_ids = {d.doc_id for d in loaded.documents}
        for query in loaded.queries:
            assert set(query.metadata["candidate_doc_ids"]) <= doc_ids
            assert len(query.metadata["candidate_doc_ids"]) == 5

    def test_gold_is_the_labelled_candidate(self, config):
        loaded = load(config, "casehold", "test", max_documents=50)
        for query in loaded.queries:
            expected = query.metadata["candidate_doc_ids"][query.metadata["label"]]
            assert query.evidence[0].doc_id == expected

    def test_identical_holdings_deduplicated(self, config):
        loaded = load(config, "casehold", "test", max_documents=200)
        texts = [d.text for d in loaded.documents]
        assert len(texts) == len(set(texts))
        assert len(loaded.documents) < 200 * 5


class TestMedQA:
    @pytest.mark.parametrize("split,expected", [
        ("test", 1273), ("dev", 1272), ("train", 10178),
    ])
    def test_question_counts_match_audit(self, config, split, expected):
        loaded = load(config, "medqa", split, max_documents=1)
        assert len(loaded.queries) == expected

    def test_no_query_ever_carries_evidence(self, config):
        loaded = load(config, "medqa", "test", max_documents=1)
        assert all(q.evidence == [] for q in loaded.queries)
        assert all(q.has_ground_truth is False for q in loaded.queries)
        assert loaded.stats["queries_with_evidence"] == 0

    def test_relevance_class_is_unsupported(self, config):
        loaded = load(config, "medqa", "test", max_documents=1, max_queries=5)
        assert all(
            q.relevance_class is RelevanceClass.UNSUPPORTED for q in loaded.queries
        )

    def test_textbooks_form_the_corpus(self, config):
        loaded = load(config, "medqa", "test", max_documents=3, max_queries=5)
        assert len(loaded.documents) == 3
        # Textbooks are split-independent and labelled as such.
        assert all(d.split == "textbooks" for d in loaded.documents)

    def test_options_preserved_for_generation(self, config):
        loaded = load(config, "medqa", "test", max_documents=1, max_queries=10)
        for query in loaded.queries:
            assert len(query.metadata["options"]) == 5
            assert query.metadata["answer_idx"] in query.metadata["options"]
            assert query.metadata["options"][query.metadata["answer_idx"]] == query.answer


class TestPubMedQA:
    def test_loads_one_thousand_labelled_records(self, config):
        loaded = load(config, "pubmedqa", "labeled")
        assert len(loaded.documents) == 1000
        assert len(loaded.queries) == 1000

    def test_every_question_maps_to_its_own_abstract(self, config):
        loaded = load(config, "pubmedqa", "labeled", max_documents=50)
        by_pubid = {d.metadata["pubid"]: d.doc_id for d in loaded.documents}
        for query in loaded.queries:
            assert query.evidence[0].doc_id == by_pubid[query.metadata["pubid"]]

    def test_sections_recorded_as_segments(self, config):
        loaded = load(config, "pubmedqa", "labeled", max_documents=20)
        for document in loaded.documents:
            assert len(document.segments) == document.metadata["n_sections"]
            for start, end in document.segments:
                assert document.text[start:end].strip()

    def test_decision_values_are_expected(self, config):
        loaded = load(config, "pubmedqa", "labeled", max_documents=100)
        assert {q.answer for q in loaded.queries} <= {"yes", "no", "maybe"}

    def test_rejects_unknown_split(self, config):
        with pytest.raises(ValueError, match="only the 'labeled' split"):
            load(config, "pubmedqa", "test")


class TestQASPER:
    @pytest.mark.parametrize("split,papers,questions", [
        ("val", 281, 1005), ("test", 416, 1451), ("train", 888, 2593),
    ])
    def test_counts_match_audit(self, config, split, papers, questions):
        loaded = load(config, "qasper", split)
        assert len(loaded.documents) == papers
        assert len(loaded.queries) == questions

    def test_evidence_text_is_recoverable_from_document(self, config):
        loaded = load(config, "qasper", "val", max_documents=10)
        by_id = {d.doc_id: d for d in loaded.documents}
        checked = 0
        for query in loaded.queries:
            for span in query.evidence:
                extract = by_id[span.doc_id].text[span.start : span.end]
                assert extract.strip()
                checked += 1
        assert checked > 0

    def test_figure_evidence_is_dropped_and_counted(self, config):
        loaded = load(config, "qasper", "val")
        assert loaded.stats["evidence_figure_or_table_dropped"] == 253
        assert loaded.stats["evidence_unmatched_dropped"] == 134

    def test_paragraphs_recorded_as_segments(self, config):
        loaded = load(config, "qasper", "val", max_documents=5)
        for document in loaded.documents:
            assert len(document.segments) > 1

    def test_unanswerable_questions_lack_evidence(self, config):
        loaded = load(config, "qasper", "val", max_documents=60)
        unanswerable = [q for q in loaded.queries if not q.metadata["answerable"]]
        assert unanswerable
        assert all(q.evidence == [] for q in unanswerable)


class TestSciQ:
    @pytest.mark.parametrize("split,rows,empty", [
        ("test", 1000, 116), ("validation", 1000, 113), ("train", 11679, 1198),
    ])
    def test_counts_and_missing_support_match_audit(self, config, split, rows, empty):
        loaded = load(config, "sciq", split)
        assert len(loaded.queries) == rows
        assert loaded.stats["rows_without_support"] == empty

    def test_rows_without_support_kept_but_unscoreable(self, config):
        loaded = load(config, "sciq", "test")
        without = [q for q in loaded.queries if not q.metadata["has_support"]]
        assert len(without) == 116
        assert all(q.evidence == [] for q in without)
        assert all(q.has_ground_truth is False for q in without)

    def test_supports_deduplicated(self, config):
        loaded = load(config, "sciq", "test")
        texts = [d.text for d in loaded.documents]
        assert len(texts) == len(set(texts))

    def test_evidence_covers_whole_support(self, config):
        loaded = load(config, "sciq", "test")
        by_id = {d.doc_id: d for d in loaded.documents}
        for query in loaded.queries[:200]:
            for span in query.evidence:
                assert (span.start, span.end) == (0, len(by_id[span.doc_id].text))

    def test_options_are_supplied_so_choice_accuracy_can_be_scored(self, config):
        loaded = load(config, "sciq", "test")
        for query in loaded.queries[:200]:
            options = query.metadata["options"]
            gold = query.metadata["answer_idx"]
            assert 2 <= len(options) <= 4
            assert options[gold] == query.metadata["correct_answer"]

    def test_the_correct_answer_is_not_always_the_first_option(self, config):
        """Source order puts it first every time; an unshuffled set scores a
        model that always answers "A" at 100%."""
        loaded = load(config, "sciq", "test")
        gold_keys = {q.metadata["answer_idx"] for q in loaded.queries[:200]}
        assert len(gold_keys) > 1

    def test_option_order_is_identical_across_loads(self, config):
        """Frozen task sets require every run to build the same prompt."""
        first = load(config, "sciq", "test").queries[:50]
        second = load(config, "sciq", "test").queries[:50]
        assert [q.metadata["options"] for q in first] == [
            q.metadata["options"] for q in second
        ]


class TestBuildOptions:
    """The shared multiple-choice helper, away from any dataset on disk."""

    def test_an_answer_with_no_distractors_is_not_a_choice_question(self):
        assert build_options("water", [], seed="s") == ({}, None)

    def test_a_missing_answer_leaves_the_question_unscoreable(self):
        options, gold = build_options("", ["a", "b", "c"], seed="s")
        assert (options, gold) == ({}, None), "no answer must not score as wrong"

    def test_blank_distractors_are_dropped_rather_than_labelled(self):
        options, gold = build_options("water", ["ice", "", "  "], seed="s")
        assert sorted(options.values()) == ["ice", "water"]
        assert options[gold] == "water"

    def test_the_gold_key_always_points_at_the_correct_answer(self):
        for i in range(50):
            options, gold = build_options("right", ["w1", "w2", "w3"], seed=f"s{i}")
            assert options[gold] == "right"

    def test_the_same_seed_gives_the_same_order(self):
        assert build_options("a", ["b", "c"], seed="x") == build_options(
            "a", ["b", "c"], seed="x"
        )


class TestCaseHOLDOptions:
    def test_every_candidate_holding_is_offered(self, config):
        loaded = load(config, "casehold", "test", max_queries=100)
        for query in loaded.queries[:100]:
            options = query.metadata["options"]
            assert len(options) == 5
            assert options[query.metadata["answer_idx"]] == query.answer

    def test_the_gold_key_tracks_the_dataset_label(self, config):
        loaded = load(config, "casehold", "test", max_queries=100)
        for query in loaded.queries[:100]:
            assert query.metadata["answer_idx"] == "ABCDE"[query.metadata["label"]]


class TestAllAdapters:
    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_default_split_loads(self, config, name):
        adapter = get_adapter(name, config)
        config.datasets[name] = DatasetConfig(max_documents=2, max_queries=10)
        loaded = adapter.load(adapter.default_split())
        assert len(loaded.queries) > 0

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_ids_are_unique(self, config, name):
        adapter = get_adapter(name, config)
        config.datasets[name] = DatasetConfig(max_documents=3, max_queries=50)
        loaded = adapter.load(adapter.default_split())
        assert len({d.doc_id for d in loaded.documents}) == len(loaded.documents)
        assert len({q.query_id for q in loaded.queries}) == len(loaded.queries)

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_evidence_references_existing_documents(self, config, name):
        adapter = get_adapter(name, config)
        config.datasets[name] = DatasetConfig(max_documents=3, max_queries=50)
        loaded = adapter.load(adapter.default_split())
        doc_ids = {d.doc_id for d in loaded.documents}
        for query in loaded.queries:
            for span in query.evidence:
                assert span.doc_id in doc_ids

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_missing_data_raises_clear_error(self, config, name, tmp_path):
        config.paths.datasets = str(tmp_path / "nowhere")
        adapter = get_adapter(name, config)
        with pytest.raises(FileNotFoundError, match="missing"):
            adapter.load(adapter.default_split())
