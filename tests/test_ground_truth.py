"""
Tests for dataset-specific ground-truth relevance mapping.

These tests verify that get_relevant_chunk_ids() returns the correct chunk IDs
for each dataset strategy without requiring real datasets or large models.
"""

import pytest
from evaluation.ground_truth import get_relevant_chunk_ids


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample(
    dataset,
    has_retrieval_gt=True,
    evidence=None,
    source_doc_id=None,
    answer=None,
):
    return {
        "sample_id":      f"{dataset}__test1",
        "dataset":        dataset,
        "domain":         "Test",
        "split":          "test",
        "question":       "What is the answer?",
        "answer":         answer or "",
        "options":        None,
        "source_doc_id":  source_doc_id,
        "evidence":       evidence or [],
        "has_retrieval_gt": has_retrieval_gt,
        "metadata":       {},
    }


def _chunk(chunk_id, doc_id, text):
    return {
        "chunk_id":  chunk_id,
        "doc_id":    doc_id,
        "dataset":   "TestDS",
        "domain":    "Test",
        "text":      text,
        "chunk_idx": 0,
        "metadata":  {},
    }


# ── Datasets without ground truth ─────────────────────────────────────────────

class TestNoGroundTruth:
    def test_casehold_returns_none(self):
        sample = _sample("CaseHOLD", has_retrieval_gt=False)
        assert get_relevant_chunk_ids(sample, []) is None

    def test_medqa_returns_none(self):
        sample = _sample("MedQA", has_retrieval_gt=False)
        assert get_relevant_chunk_ids(sample, []) is None

    def test_has_retrieval_gt_false_always_returns_none(self):
        sample = _sample("CUAD", has_retrieval_gt=False, evidence=["contract"])
        assert get_relevant_chunk_ids(sample, []) is None


# ── CUAD ─────────────────────────────────────────────────────────────────────

class TestCUADGroundTruth:
    def test_chunk_containing_answer_is_relevant(self):
        sample = _sample(
            "CUAD",
            evidence=["DISTRIBUTOR AGREEMENT"],
            source_doc_id="CUAD__Contract__para0",
        )
        chunks = [
            _chunk("CUAD__Contract__para0__chunk0000", "CUAD__Contract__para0",
                   "THIS IS A DISTRIBUTOR AGREEMENT made between parties."),
            _chunk("CUAD__Contract__para0__chunk0001", "CUAD__Contract__para0",
                   "The payment terms are net 30 days."),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert "CUAD__Contract__para0__chunk0000" in relevant
        assert "CUAD__Contract__para0__chunk0001" not in relevant

    def test_case_insensitive_match(self):
        sample = _sample(
            "CUAD",
            evidence=["distributor agreement"],
            source_doc_id="CUAD__Contract__para0",
        )
        chunks = [
            _chunk("CUAD__Contract__para0__chunk0000", "CUAD__Contract__para0",
                   "DISTRIBUTOR AGREEMENT — This is a binding contract."),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert "CUAD__Contract__para0__chunk0000" in relevant

    def test_empty_answer_returns_empty(self):
        sample = _sample("CUAD", evidence=[], source_doc_id="CUAD__Contract__para0")
        chunks = [_chunk("CUAD__Contract__para0__chunk0000", "CUAD__Contract__para0", "text")]
        assert get_relevant_chunk_ids(sample, chunks) == []

    def test_only_same_source_doc_chunks_searched(self):
        sample = _sample(
            "CUAD",
            evidence=["AGREEMENT"],
            source_doc_id="CUAD__ContractA__para0",
        )
        chunks = [
            _chunk("CUAD__ContractA__para0__chunk0000", "CUAD__ContractA__para0",
                   "This AGREEMENT is between parties."),
            _chunk("CUAD__ContractB__para0__chunk0000", "CUAD__ContractB__para0",
                   "Another AGREEMENT in a different contract."),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert "CUAD__ContractA__para0__chunk0000" in relevant
        assert "CUAD__ContractB__para0__chunk0000" not in relevant


# ── PubMedQA ─────────────────────────────────────────────────────────────────

class TestPubMedQAGroundTruth:
    def test_all_chunks_from_same_doc_relevant(self):
        sample = _sample("PubMedQA", source_doc_id="PubMedQA__12345")
        chunks = [
            _chunk("PubMedQA__12345__chunk0000", "PubMedQA__12345", "sentence 1"),
            _chunk("PubMedQA__12345__chunk0001", "PubMedQA__12345", "sentence 2"),
            _chunk("PubMedQA__99999__chunk0000", "PubMedQA__99999", "different paper"),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert set(relevant) == {"PubMedQA__12345__chunk0000", "PubMedQA__12345__chunk0001"}

    def test_different_doc_excluded(self):
        sample = _sample("PubMedQA", source_doc_id="PubMedQA__12345")
        chunks = [_chunk("PubMedQA__99999__chunk0000", "PubMedQA__99999", "other")]
        assert get_relevant_chunk_ids(sample, chunks) == []


# ── QASPER ────────────────────────────────────────────────────────────────────

class TestQASPERGroundTruth:
    def test_chunk_containing_evidence_is_relevant(self):
        evidence_text = "We collected 220 human-human dialogs."
        sample = _sample(
            "QASPER",
            evidence=[evidence_text],
            source_doc_id="QASPER__paper1",
        )
        chunks = [
            _chunk("QASPER__paper1__chunk0000", "QASPER__paper1",
                   "We collected 220 human-human dialogs. The data was annotated."),
            _chunk("QASPER__paper1__chunk0001", "QASPER__paper1",
                   "Experiments were run on a GPU cluster."),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert "QASPER__paper1__chunk0000" in relevant
        assert "QASPER__paper1__chunk0001" not in relevant

    def test_evidence_longer_than_chunk_still_matches(self):
        evidence = "This is the evidence passage. It is quite long and spans text."
        sample = _sample("QASPER", evidence=[evidence], source_doc_id="QASPER__p1")
        chunk_text = "This is the evidence passage."  # shorter than evidence
        chunks = [_chunk("QASPER__p1__chunk0000", "QASPER__p1", chunk_text)]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert "QASPER__p1__chunk0000" in relevant

    def test_empty_evidence_returns_empty(self):
        sample = _sample("QASPER", evidence=[], source_doc_id="QASPER__p1")
        chunks = [_chunk("QASPER__p1__chunk0000", "QASPER__p1", "some text")]
        assert get_relevant_chunk_ids(sample, chunks) == []


# ── SciQ ─────────────────────────────────────────────────────────────────────

class TestSciQGroundTruth:
    def test_all_chunks_from_source_doc_relevant(self):
        sample = _sample("SciQ", source_doc_id="SciQ__test__0")
        chunks = [
            _chunk("SciQ__test__0__chunk0000", "SciQ__test__0", "support text 1"),
            _chunk("SciQ__test__0__chunk0001", "SciQ__test__0", "support text 2"),
            _chunk("SciQ__test__1__chunk0000", "SciQ__test__1", "different question"),
        ]
        relevant = get_relevant_chunk_ids(sample, chunks)
        assert set(relevant) == {"SciQ__test__0__chunk0000", "SciQ__test__0__chunk0001"}

    def test_no_source_doc_returns_empty(self):
        sample = _sample("SciQ", source_doc_id=None)
        chunks = [_chunk("SciQ__test__0__chunk0000", "SciQ__test__0", "text")]
        assert get_relevant_chunk_ids(sample, chunks) == []
