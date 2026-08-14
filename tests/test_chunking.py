"""Chunking and evidence-to-chunk mapping."""

from __future__ import annotations

import pytest

from benchmark.chunking import chunk_document, chunk_documents, relevant_chunk_ids
from core.settings import ChunkConfig
from core.types import EvidenceSpan, SourceDocument


class TestFixedChunking:
    def test_offsets_reproduce_source_text(self, document, fixed_chunk_config):
        for chunk in chunk_document(document, fixed_chunk_config):
            assert document.text[chunk.char_start : chunk.char_end] == chunk.text

    def test_overlap_present(self, document, fixed_chunk_config):
        chunks = chunk_document(document, fixed_chunk_config)
        assert len(chunks) > 1
        # size 10, overlap 5 -> consecutive windows share characters
        assert chunks[1].char_start < chunks[0].char_end

    def test_ids_are_stable_and_ordered(self, document, fixed_chunk_config):
        first = chunk_document(document, fixed_chunk_config)
        second = chunk_document(document, fixed_chunk_config)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.ordinal for c in first] == list(range(len(first)))
        assert first[0].chunk_id.startswith(document.doc_id)

    def test_full_coverage_of_document(self, document, fixed_chunk_config):
        chunks = chunk_document(document, fixed_chunk_config)
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(document.text)

    def test_overlap_not_smaller_than_size_raises(self, document):
        bad = ChunkConfig(strategy="fixed", size_tokens=10, overlap_tokens=10)
        with pytest.raises(ValueError, match="must be smaller"):
            chunk_document(document, bad)

    def test_unknown_strategy_raises(self, document):
        with pytest.raises(ValueError, match="unknown chunk strategy"):
            chunk_document(document, ChunkConfig(strategy="nonsense"))

    def test_empty_document_yields_no_chunks(self):
        empty = SourceDocument("d", "ds", "s", "   \n  ")
        assert chunk_document(empty, ChunkConfig()) == []

    def test_short_document_still_produces_one_chunk(self):
        # min_chunk_chars must not delete a document from the corpus.
        doc = SourceDocument("d", "ds", "s", "tiny")
        chunks = chunk_document(doc, ChunkConfig(strategy="passage", min_chunk_chars=100))
        assert len(chunks) == 1
        assert chunks[0].text == "tiny"


class TestPassageChunking:
    def test_document_without_segments_is_one_chunk(self):
        doc = SourceDocument("d", "ds", "s", "one two three four five")
        chunks = chunk_document(doc, ChunkConfig(strategy="passage", max_tokens=100))
        assert len(chunks) == 1

    def test_segments_become_separate_chunks(self, segmented_document):
        chunks = chunk_document(segmented_document, ChunkConfig(strategy="passage", max_tokens=100))
        assert len(chunks) == 3
        assert chunks[0].text == "first section text here"
        assert chunks[2].text == "third and final section"

    def test_segment_offsets_reproduce_source(self, segmented_document):
        text = segmented_document.text
        for chunk in chunk_document(segmented_document, ChunkConfig(strategy="passage")):
            assert text[chunk.char_start : chunk.char_end] == chunk.text

    def test_oversized_segment_is_subsplit(self):
        body = " ".join(f"w{i}" for i in range(50))
        doc = SourceDocument("d", "ds", "s", body, segments=[(0, len(body))])
        chunks = chunk_document(doc, ChunkConfig(strategy="passage", max_tokens=10, overlap_tokens=2))
        assert len(chunks) > 1
        assert all(len(c.text.split()) <= 10 for c in chunks)


class TestRelevanceMapping:
    def _corpus(self):
        text = " ".join(f"word{i:02d}" for i in range(40))
        doc = SourceDocument("ds/s/d1", "ds", "s", text)
        config = ChunkConfig(strategy="fixed", size_tokens=10, overlap_tokens=0, min_chunk_chars=1)
        return doc, chunk_document(doc, config)

    def test_evidence_maps_to_containing_chunk(self):
        doc, chunks = self._corpus()
        span = EvidenceSpan(doc.doc_id, chunks[2].char_start + 2, chunks[2].char_start + 8)
        assert relevant_chunk_ids([span], chunks) == [chunks[2].chunk_id]

    def test_evidence_spanning_two_chunks_marks_both(self):
        doc, chunks = self._corpus()
        span = EvidenceSpan(doc.doc_id, chunks[0].char_end - 3, chunks[1].char_start + 5)
        found = relevant_chunk_ids([span], chunks)
        assert chunks[0].chunk_id in found and chunks[1].chunk_id in found

    def test_multiple_spans_deduplicated(self):
        doc, chunks = self._corpus()
        spans = [
            EvidenceSpan(doc.doc_id, chunks[1].char_start, chunks[1].char_start + 4),
            EvidenceSpan(doc.doc_id, chunks[1].char_start + 5, chunks[1].char_start + 9),
        ]
        assert relevant_chunk_ids(spans, chunks) == [chunks[1].chunk_id]

    def test_evidence_from_another_document_is_ignored(self):
        _, chunks = self._corpus()
        assert relevant_chunk_ids([EvidenceSpan("ds/s/other", 0, 10)], chunks) == []

    def test_no_evidence_gives_empty_list(self):
        _, chunks = self._corpus()
        assert relevant_chunk_ids([], chunks) == []

    def test_results_are_in_corpus_order(self):
        doc, chunks = self._corpus()
        spans = [
            EvidenceSpan(doc.doc_id, chunks[3].char_start, chunks[3].char_start + 2),
            EvidenceSpan(doc.doc_id, chunks[0].char_start, chunks[0].char_start + 2),
        ]
        assert relevant_chunk_ids(spans, chunks) == [chunks[0].chunk_id, chunks[3].chunk_id]

    def test_zero_width_span_matches_nothing(self):
        doc, chunks = self._corpus()
        span = EvidenceSpan(doc.doc_id, 5, 5)
        assert relevant_chunk_ids([span], chunks) == []

    def test_invalid_span_rejected(self):
        with pytest.raises(ValueError):
            EvidenceSpan("d", 10, 5)


def test_chunk_documents_concatenates(document, fixed_chunk_config):
    other = SourceDocument("ds/split/doc2", "ds", "split", "alpha beta gamma delta")
    chunks = chunk_documents([document, other], fixed_chunk_config)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"ds/split/doc1", "ds/split/doc2"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
