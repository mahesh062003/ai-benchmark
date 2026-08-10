"""Tests for DocumentChunker."""

import pytest
from loaders.base import CorpusDocument
from preprocessing.chunker import DocumentChunker, save_chunks, load_chunks


def _make_doc(doc_id: str, text: str, dataset: str = "TestDS") -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        dataset=dataset,
        domain="Test",
        text=text,
        metadata={"title": "test"},
    )


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    def test_single_short_doc_one_chunk(self):
        doc = _make_doc("DS__doc1", "Short text.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_chunk_ids_are_unique(self):
        doc = _make_doc("DS__doc1", "word " * 200)
        chunks = self.chunker.chunk_document(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self):
        doc = _make_doc("DS__doc1", "a " * 300)
        chunks = self.chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"DS__doc1__chunk{i:04d}"

    def test_doc_id_preserved(self):
        doc = _make_doc("CUAD__Contract__para0", "hello world " * 50)
        chunks = self.chunker.chunk_document(doc)
        assert all(c["doc_id"] == "CUAD__Contract__para0" for c in chunks)

    def test_dataset_and_domain_preserved(self):
        doc = _make_doc("DS__doc1", "hello world " * 50, dataset="CUAD")
        chunks = self.chunker.chunk_document(doc)
        assert all(c["dataset"] == "CUAD" for c in chunks)

    def test_empty_doc_returns_no_chunks(self):
        doc = _make_doc("DS__empty", "")
        chunks = self.chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_documents_flat(self):
        docs = [
            _make_doc("DS__doc1", "alpha " * 200),
            _make_doc("DS__doc2", "beta  " * 200),
        ]
        chunks = self.chunker.chunk_documents(docs)
        doc1_chunks = [c for c in chunks if c["doc_id"] == "DS__doc1"]
        doc2_chunks = [c for c in chunks if c["doc_id"] == "DS__doc2"]
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0

    def test_no_duplicate_chunk_ids_across_docs(self):
        docs = [_make_doc(f"DS__doc{i}", "word " * 200) for i in range(5)]
        chunks = self.chunker.chunk_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_long_doc_produces_multiple_chunks(self):
        doc = _make_doc("DS__long", "word " * 500)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 1


class TestChunkSerialisation:
    def test_save_and_reload(self, tmp_path, synthetic_chunks):
        path = tmp_path / "test_chunks.json"
        save_chunks(synthetic_chunks, path)
        reloaded = load_chunks(path)
        assert len(reloaded) == len(synthetic_chunks)
        assert reloaded[0]["chunk_id"] == synthetic_chunks[0]["chunk_id"]

    def test_all_fields_preserved(self, tmp_path, synthetic_chunks):
        path = tmp_path / "test_chunks.json"
        save_chunks(synthetic_chunks, path)
        reloaded = load_chunks(path)
        for orig, reloaded_chunk in zip(synthetic_chunks, reloaded):
            assert reloaded_chunk["chunk_id"] == orig["chunk_id"]
            assert reloaded_chunk["text"]     == orig["text"]
            assert reloaded_chunk["doc_id"]   == orig["doc_id"]
