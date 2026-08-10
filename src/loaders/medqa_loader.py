"""
MedQA loader (USMLE-style medical MCQ).

Dataset format: JSONL, one JSON object per line.
Fields: question, options (dict A-E), answer, answer_idx, meta_info.

Corpus strategy:
  MedQA has NO external retrieval corpus. The "context" in the original code
  was the question + options text — i.e., the query itself.
  This is NOT a valid retrieval corpus for standard RAG evaluation.

  has_retrieval_gt = False for ALL MedQA samples.
  Recall@K, MRR, nDCG are NOT valid for this dataset.

  MedQA is included for:
    - LLM generation quality evaluation (does the model select the correct option?)
    - Response time measurement
    - Qualitative comparison of retrieval methods (which context helps more)

Split used: dev.jsonl (correct for evaluation; train is for model training).
"""

import json
import logging
from pathlib import Path
from typing import Any

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def load_corpus(file_path: str | Path) -> list[CorpusDocument]:
    """
    MedQA has no valid external retrieval corpus.
    Returns an empty list; retrieval metrics are not applicable.
    """
    logger.warning(
        "MedQA: no external retrieval corpus. "
        "Retrieval metrics are NOT valid for this dataset."
    )
    return []


def load_samples(file_path: str | Path, split: str = "dev") -> list[Sample]:
    """Return one Sample per MedQA question."""
    file_path = Path(file_path)
    samples: list[Sample] = []

    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            question: str = row["question"].strip()
            options: dict[str, str] = row.get("options", {})

            samples.append(Sample(
                sample_id=f"MedQA__{split}__{idx}",
                dataset="MedQA",
                domain="Medical",
                split=split,
                question=question,
                answer=row.get("answer", ""),
                options=options,
                source_doc_id=None,
                evidence=[],
                has_retrieval_gt=False,   # NOT a valid retrieval task
                metadata={
                    "answer_idx": row.get("answer_idx", ""),
                    "meta_info":  row.get("meta_info", ""),
                    "row_idx":    idx,
                },
            ))

    logger.info(
        "MedQA samples: %d (split=%s, retrieval_gt=UNAVAILABLE)",
        len(samples), split,
    )
    return samples


def load_medqa(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    file_path = Path(file_path)
    records: list[dict] = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            records.append({
                "id":         idx,
                "dataset":    "MedQA",
                "domain":     "Medical",
                "question":   row["question"],
                "options":    row["options"],
                "answer":     row["answer"],
                "answer_idx": row["answer_idx"],
                "meta_info":  row.get("meta_info", ""),
            })
    return records
