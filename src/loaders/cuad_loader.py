"""
CUAD (Contract Understanding Atticus Dataset) loader.

Dataset format: SQuAD-style JSON.
Each contract contains one or more paragraphs (contexts).
Each paragraph contains one or more QA pairs (questions + answer spans).

Corpus strategy:
  CorpusDocument = each unique paragraph from every contract.
  doc_id = "{contract_title}__para{para_idx}" (stable across runs).

Sample / ground-truth strategy:
  Each QA pair is one Sample.
  source_doc_id links the sample back to its paragraph CorpusDocument.
  Ground truth = corpus chunks whose text contains the answer text
                 (implemented in evaluation/ground_truth.py).
  is_impossible=True samples have no answer → has_retrieval_gt = False.

Retrieval validity: YES for answerable QAs.
"""

import json
import logging
from pathlib import Path
from typing import Any

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def load_corpus(file_path: str | Path) -> list[CorpusDocument]:
    """
    Return unique paragraph-level CorpusDocuments.
    Each distinct paragraph context appears exactly once.
    """
    file_path = Path(file_path)
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    documents: list[CorpusDocument] = []

    for contract in data["data"]:
        title: str = contract["title"]

        for para_idx, paragraph in enumerate(contract["paragraphs"]):
            context: str = paragraph["context"].strip()
            if not context:
                continue

            doc_id = f"CUAD__{title}__para{para_idx}"

            documents.append(CorpusDocument(
                doc_id=doc_id,
                dataset="CUAD",
                domain="Legal",
                text=context,
                metadata={
                    "title":   title,
                    "para_idx": para_idx,
                },
            ))

    logger.info("CUAD corpus: %d paragraph documents", len(documents))
    return documents


def load_samples(file_path: str | Path, split: str = "all") -> list[Sample]:
    """
    Return one Sample per QA pair.
    source_doc_id matches the doc_id produced by load_corpus().
    """
    file_path = Path(file_path)
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    samples: list[Sample] = []
    skipped = 0

    for contract in data["data"]:
        title: str = contract["title"]

        for para_idx, paragraph in enumerate(contract["paragraphs"]):
            doc_id = f"CUAD__{title}__para{para_idx}"

            for qa in paragraph["qas"]:
                question: str = qa["question"].strip()
                if not question:
                    skipped += 1
                    continue

                is_impossible: bool = qa.get("is_impossible", False)
                answers: list[str] = [
                    a["text"].strip() for a in qa.get("answers", [])
                    if a["text"].strip()
                ]

                samples.append(Sample(
                    sample_id=qa["id"],
                    dataset="CUAD",
                    domain="Legal",
                    split=split,
                    question=question,
                    answer=answers if answers else None,
                    options=None,
                    source_doc_id=doc_id,
                    evidence=answers,          # answer span = primary evidence
                    has_retrieval_gt=(not is_impossible and bool(answers)),
                    metadata={
                        "title":        title,
                        "para_idx":     para_idx,
                        "is_impossible": is_impossible,
                    },
                ))

    if skipped:
        logger.warning("CUAD: skipped %d QAs with empty question", skipped)
    logger.info("CUAD samples: %d (answerable: %d)",
                len(samples), sum(s["has_retrieval_gt"] for s in samples))
    return samples


def load_cuad(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    file_path = Path(file_path)
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for contract in data["data"]:
        title = contract["title"]
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = [a["text"] for a in qa["answers"]]
                records.append({
                    "id":           qa["id"],
                    "dataset":      "CUAD",
                    "domain":       "Legal",
                    "title":        title,
                    "question":     qa["question"],
                    "context":      context,
                    "answers":      answers,
                    "is_impossible": qa["is_impossible"],
                })
    return records
