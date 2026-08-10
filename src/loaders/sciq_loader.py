"""
SciQ loader (science multiple-choice QA).

Dataset format: CSV with columns: question, correct_answer, distractor1-3, support.
The 'support' field is a short paragraph directly supporting the correct answer.

Corpus strategy:
  CorpusDocument = one document per question row (the support passage).
  doc_id = "SciQ__{split}__{row_idx}".

Sample / ground-truth strategy:
  The support passage for question X is the relevant document for query X.
  Ground truth = the single corpus chunk whose source_doc_id matches the sample.

  has_retrieval_gt = True when a non-empty support passage exists.

  This is a CLOSED-DOMAIN retrieval task: the corpus consists of support
  passages from all questions in the test split, and the retriever must
  identify which passage answers a given question.

Split note:
  Original code used train.csv (incorrect for evaluation).
  Refactored: test.csv for both corpus and queries (1 000 samples).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def load_corpus(file_path: str | Path) -> list[CorpusDocument]:
    """Return one CorpusDocument per SciQ row (the support passage)."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    split_name = file_path.stem   # 'train', 'test', 'validation'
    documents: list[CorpusDocument] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        support: str = str(row.get("support", "")).strip()
        if not support:
            skipped += 1
            continue

        doc_id = f"SciQ__{split_name}__{row_idx}"
        documents.append(CorpusDocument(
            doc_id=doc_id,
            dataset="SciQ",
            domain="Scientific",
            text=support,
            metadata={"split": split_name, "row_idx": int(row_idx)},
        ))

    if skipped:
        logger.warning("SciQ corpus: skipped %d rows with empty support", skipped)
    logger.info("SciQ corpus: %d documents (split=%s)", len(documents), split_name)
    return documents


def load_samples(file_path: str | Path, split: str = "test") -> list[Sample]:
    """Return one Sample per SciQ row."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    split_name = file_path.stem
    samples: list[Sample] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        question: str = str(row.get("question", "")).strip()
        if not question:
            skipped += 1
            continue

        support: str = str(row.get("support", "")).strip()
        doc_id = f"SciQ__{split_name}__{row_idx}" if support else None

        options = {
            "A": str(row.get("correct_answer", "")).strip(),
            "B": str(row.get("distractor1",   "")).strip(),
            "C": str(row.get("distractor2",   "")).strip(),
            "D": str(row.get("distractor3",   "")).strip(),
        }

        samples.append(Sample(
            sample_id=f"SciQ__{split_name}__{row_idx}",
            dataset="SciQ",
            domain="Scientific",
            split=split_name,
            question=question,
            answer=options["A"],   # correct_answer is always option A
            options=options,
            source_doc_id=doc_id,
            evidence=[support] if support else [],
            has_retrieval_gt=bool(support),
            metadata={
                "split":   split_name,
                "row_idx": int(row_idx),
            },
        ))

    if skipped:
        logger.warning("SciQ: skipped %d rows with empty question", skipped)
    logger.info("SciQ samples: %d (split=%s)", len(samples), split_name)
    return samples


def load_sciq(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    dataset: list[dict] = []

    for idx, row in df.iterrows():
        options = {
            "A": row["correct_answer"],
            "B": row["distractor1"],
            "C": row["distractor2"],
            "D": row["distractor3"],
        }
        dataset.append({
            "id":         idx,
            "dataset":    "SciQ",
            "domain":     "Scientific",
            "question":   row["question"],
            "context":    row["support"],
            "options":    options,
            "answer":     row["correct_answer"],
            "answer_idx": "A",
        })
    return dataset
