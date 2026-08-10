"""
Dataset registry for the benchmark.

Each entry describes one dataset: how to load its corpus documents and
its benchmark samples, which domain it belongs to, and configuration notes.

Adding a new dataset: add one entry to DATASETS following the same pattern.
"""

from config import DATASET_PATHS

from loaders import (
    cuad_loader,
    casehold_loader,
    medqa_loader,
    pubmedqa_loader,
    qasper_loader,
    sciq_loader,
)


def _load_cuad_corpus():
    return cuad_loader.load_corpus(DATASET_PATHS["CUAD"]["corpus"])

def _load_cuad_samples():
    return cuad_loader.load_samples(DATASET_PATHS["CUAD"]["queries"], split="all")


def _load_casehold_corpus():
    return casehold_loader.load_corpus(DATASET_PATHS["CaseHOLD"]["corpus"])

def _load_casehold_samples():
    return casehold_loader.load_samples(DATASET_PATHS["CaseHOLD"]["queries"], split="test")


def _load_medqa_corpus():
    return medqa_loader.load_corpus(DATASET_PATHS["MedQA"]["queries"])

def _load_medqa_samples():
    return medqa_loader.load_samples(DATASET_PATHS["MedQA"]["queries"], split="dev")


def _load_pubmedqa_corpus():
    return pubmedqa_loader.load_corpus(DATASET_PATHS["PubMedQA"]["corpus"])

def _load_pubmedqa_samples():
    return pubmedqa_loader.load_samples(DATASET_PATHS["PubMedQA"]["queries"], split="train_labeled")


def _load_qasper_corpus():
    return qasper_loader.load_corpus(
        train_path=DATASET_PATHS["QASPER"]["corpus_train"],
        test_path=DATASET_PATHS["QASPER"]["corpus_test"],
    )

def _load_qasper_samples():
    return qasper_loader.load_samples(DATASET_PATHS["QASPER"]["queries"], split="test")


def _load_sciq_corpus():
    return sciq_loader.load_corpus(DATASET_PATHS["SciQ"]["corpus"])

def _load_sciq_samples():
    return sciq_loader.load_samples(DATASET_PATHS["SciQ"]["queries"], split="test")


# ── Registry ──────────────────────────────────────────────────────────────────

DATASETS: list[dict] = [
    {
        "name":            "CUAD",
        "domain":          "Legal",
        "load_corpus":     _load_cuad_corpus,
        "load_samples":    _load_cuad_samples,
        "retrieval_valid": True,
        "notes":           "Answerable QAs only.  Ground truth = answer span in chunk.",
    },
    {
        "name":            "CaseHOLD",
        "domain":          "Legal",
        "load_corpus":     _load_casehold_corpus,
        "load_samples":    _load_casehold_samples,
        "retrieval_valid": False,
        "notes":           "Classification task.  Retrieval metrics NOT applicable.",
    },
    {
        "name":            "MedQA",
        "domain":          "Medical",
        "load_corpus":     _load_medqa_corpus,
        "load_samples":    _load_medqa_samples,
        "retrieval_valid": False,
        "notes":           "No external corpus.  Retrieval metrics NOT applicable.",
    },
    {
        "name":            "PubMedQA",
        "domain":          "Medical",
        "load_corpus":     _load_pubmedqa_corpus,
        "load_samples":    _load_pubmedqa_samples,
        "retrieval_valid": True,
        "notes":           "Ground truth = all chunks from same abstract (pubid).",
    },
    {
        "name":            "QASPER",
        "domain":          "Scientific",
        "load_corpus":     _load_qasper_corpus,
        "load_samples":    _load_qasper_samples,
        "retrieval_valid": True,
        "notes":           "Corpus = train+test papers.  GT = evidence passage overlap.",
    },
    {
        "name":            "SciQ",
        "domain":          "Scientific",
        "load_corpus":     _load_sciq_corpus,
        "load_samples":    _load_sciq_samples,
        "retrieval_valid": True,
        "notes":           "GT = support passage chunk for that question.",
    },
]
