"""
Tests for dataset loaders.

These tests use real dataset files when available, or skip gracefully when files
are not present (CI environments without large datasets).
"""

import pytest
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Dataset file not found: {path}")


# ── CUAD ─────────────────────────────────────────────────────────────────────

class TestCUADLoader:
    CUAD_PATH = DATASETS_DIR / "Legal" / "CUAD" / "CUADv1.json"

    def test_load_corpus_returns_corpus_documents(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_corpus
        docs = load_corpus(self.CUAD_PATH)
        assert len(docs) > 0
        assert docs[0]["dataset"] == "CUAD"
        assert docs[0]["domain"]  == "Legal"
        assert docs[0]["text"]
        assert docs[0]["doc_id"].startswith("CUAD__")

    def test_corpus_doc_ids_are_unique(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_corpus
        docs = load_corpus(self.CUAD_PATH)
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate doc_ids detected"

    def test_load_samples_has_question_field(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_samples
        samples = load_samples(self.CUAD_PATH)
        assert len(samples) > 0
        for s in samples[:20]:
            assert "question" in s
            assert s["dataset"] == "CUAD"

    def test_answerable_samples_have_retrieval_gt(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_samples
        samples = load_samples(self.CUAD_PATH)
        answerable = [s for s in samples if not s["metadata"]["is_impossible"]]
        assert all(s["has_retrieval_gt"] for s in answerable)

    def test_impossible_samples_no_retrieval_gt(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_samples
        samples = load_samples(self.CUAD_PATH)
        impossible = [s for s in samples if s["metadata"]["is_impossible"]]
        assert all(not s["has_retrieval_gt"] for s in impossible)

    def test_source_doc_id_links_sample_to_corpus(self):
        _skip_if_missing(self.CUAD_PATH)
        from loaders.cuad_loader import load_corpus, load_samples
        corpus_ids = {d["doc_id"] for d in load_corpus(self.CUAD_PATH)}
        samples = load_samples(self.CUAD_PATH)
        for s in samples[:50]:
            assert s["source_doc_id"] in corpus_ids


# ── CaseHOLD ─────────────────────────────────────────────────────────────────

class TestCaseHOLDLoader:
    TEST_PATH  = DATASETS_DIR / "Legal" / "LexGLUE" / "case_hold" / "case_hold_test.csv"
    TRAIN_PATH = DATASETS_DIR / "Legal" / "LexGLUE" / "case_hold" / "case_hold_train.csv"

    def test_samples_have_no_retrieval_gt(self):
        _skip_if_missing(self.TEST_PATH)
        from loaders.casehold_loader import load_samples
        samples = load_samples(self.TEST_PATH)
        assert all(not s["has_retrieval_gt"] for s in samples)

    def test_corpus_from_train_split(self):
        _skip_if_missing(self.TRAIN_PATH)
        from loaders.casehold_loader import load_corpus
        docs = load_corpus(self.TRAIN_PATH)
        assert len(docs) > 0
        assert all(d["dataset"] == "CaseHOLD" for d in docs)

    def test_sample_question_is_context(self):
        _skip_if_missing(self.TEST_PATH)
        from loaders.casehold_loader import load_samples
        samples = load_samples(self.TEST_PATH)
        # question should be the context text
        assert all(len(s["question"]) > 10 for s in samples[:10])


# ── MedQA ────────────────────────────────────────────────────────────────────

class TestMedQALoader:
    DEV_PATH = DATASETS_DIR / "Medical" / "MedQA" / "questions" / "dev.jsonl"

    def test_samples_have_no_retrieval_gt(self):
        _skip_if_missing(self.DEV_PATH)
        from loaders.medqa_loader import load_samples
        samples = load_samples(self.DEV_PATH)
        assert all(not s["has_retrieval_gt"] for s in samples)

    def test_sample_has_question_and_options(self):
        _skip_if_missing(self.DEV_PATH)
        from loaders.medqa_loader import load_samples
        samples = load_samples(self.DEV_PATH)
        assert len(samples) > 0
        s = samples[0]
        assert s["question"]
        assert s["options"]

    def test_corpus_is_empty(self):
        _skip_if_missing(self.DEV_PATH)
        from loaders.medqa_loader import load_corpus
        docs = load_corpus(self.DEV_PATH)
        assert docs == []


# ── PubMedQA ─────────────────────────────────────────────────────────────────

class TestPubMedQALoader:
    DATA_PATH = DATASETS_DIR / "Medical" / "PubMedQA" / "pqa_labeled" / "train_labeled.csv"

    def test_samples_have_retrieval_gt(self):
        _skip_if_missing(self.DATA_PATH)
        from loaders.pubmedqa_loader import load_samples
        samples = load_samples(self.DATA_PATH)
        assert all(s["has_retrieval_gt"] for s in samples)

    def test_source_doc_id_matches_corpus(self):
        _skip_if_missing(self.DATA_PATH)
        from loaders.pubmedqa_loader import load_corpus, load_samples
        docs    = load_corpus(self.DATA_PATH)
        samples = load_samples(self.DATA_PATH)
        corpus_ids = {d["doc_id"] for d in docs}
        for s in samples[:20]:
            assert s["source_doc_id"] in corpus_ids

    def test_corpus_doc_ids_unique(self):
        _skip_if_missing(self.DATA_PATH)
        from loaders.pubmedqa_loader import load_corpus
        docs = load_corpus(self.DATA_PATH)
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids))


# ── QASPER ───────────────────────────────────────────────────────────────────

class TestQASPERLoader:
    TRAIN_PATH = DATASETS_DIR / "Scientific" / "qasper" / "train.parquet"
    TEST_PATH  = DATASETS_DIR / "Scientific" / "qasper" / "test.parquet"

    def test_corpus_includes_both_splits(self):
        _skip_if_missing(self.TRAIN_PATH)
        _skip_if_missing(self.TEST_PATH)
        from loaders.qasper_loader import load_corpus
        docs = load_corpus(train_path=self.TRAIN_PATH, test_path=self.TEST_PATH)
        assert len(docs) > 0
        assert all(d["dataset"] == "QASPER" for d in docs)

    def test_no_duplicate_doc_ids(self):
        _skip_if_missing(self.TRAIN_PATH)
        _skip_if_missing(self.TEST_PATH)
        from loaders.qasper_loader import load_corpus
        docs = load_corpus(train_path=self.TRAIN_PATH, test_path=self.TEST_PATH)
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids))

    def test_test_samples_have_source_doc_in_corpus(self):
        _skip_if_missing(self.TRAIN_PATH)
        _skip_if_missing(self.TEST_PATH)
        from loaders.qasper_loader import load_corpus, load_samples
        docs    = load_corpus(train_path=self.TRAIN_PATH, test_path=self.TEST_PATH)
        samples = load_samples(self.TEST_PATH)
        corpus_ids = {d["doc_id"] for d in docs}
        for s in samples[:20]:
            assert s["source_doc_id"] in corpus_ids


# ── SciQ ─────────────────────────────────────────────────────────────────────

class TestSciQLoader:
    TEST_PATH = DATASETS_DIR / "Scientific" / "SciQ" / "test.csv"

    def test_samples_have_retrieval_gt_when_support_exists(self):
        _skip_if_missing(self.TEST_PATH)
        from loaders.sciq_loader import load_samples
        samples = load_samples(self.TEST_PATH)
        with_support = [s for s in samples if s["evidence"]]
        assert all(s["has_retrieval_gt"] for s in with_support)

    def test_source_doc_id_in_corpus(self):
        _skip_if_missing(self.TEST_PATH)
        from loaders.sciq_loader import load_corpus, load_samples
        docs    = load_corpus(self.TEST_PATH)
        samples = load_samples(self.TEST_PATH)
        corpus_ids = {d["doc_id"] for d in docs}
        for s in samples[:20]:
            if s["source_doc_id"]:
                assert s["source_doc_id"] in corpus_ids

    def test_corpus_ids_unique(self):
        _skip_if_missing(self.TEST_PATH)
        from loaders.sciq_loader import load_corpus
        docs = load_corpus(self.TEST_PATH)
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids))
