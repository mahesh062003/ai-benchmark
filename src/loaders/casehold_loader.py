"""
CaseHOLD loader (LexGLUE benchmark).

Dataset format: CSV with columns: context, endings, label.
Task: multiple-choice legal holding selection (classification, not open QA).

Corpus strategy:
  CorpusDocument = each unique citation context.
  doc_id = "CaseHOLD__train__{row_idx}" (from training split for corpus).
  This builds a corpus of legal citation passages.

Sample / ground-truth strategy:
  This is a CLASSIFICATION task, not a standard retrieval task.
  There is no external "relevant document" to retrieve — the task is to select
  the correct holding from 5 given options.

  has_retrieval_gt = False for ALL CaseHOLD samples.
  Recall@K, MRR, nDCG are NOT valid for this dataset.

  The context itself serves as both the query and the document for any
  RAG-style generation experiment.

Note on dataset split:
  The ORIGINAL code used the TRAIN split for both corpus and queries.
  This refactored code uses the TEST split for evaluation queries (correct).
  The TRAIN split is used only to build the retrieval corpus.
"""

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def _parse_endings(raw: str) -> list[str]:
    """Parse the raw 'endings' string into a list of choice texts."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(c).strip() for c in parsed]
    except Exception:
        pass
    # Fallback: extract single-quoted strings
    import re
    choices = re.findall(r"'([^']*)'", str(raw))
    return choices


def load_corpus(file_path: str | Path) -> list[CorpusDocument]:
    """
    Build a retrieval corpus from CaseHOLD training contexts.
    Each unique citation context becomes one CorpusDocument.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, engine="python")

    split_name = file_path.stem.replace("case_hold_", "")  # 'train' / 'test'
    documents: list[CorpusDocument] = []

    for row_idx, row in df.iterrows():
        context: str = str(row["context"]).strip()
        if not context:
            continue

        doc_id = f"CaseHOLD__{split_name}__{row_idx}"
        documents.append(CorpusDocument(
            doc_id=doc_id,
            dataset="CaseHOLD",
            domain="Legal",
            text=context,
            metadata={"split": split_name, "row_idx": int(row_idx)},
        ))

    logger.info("CaseHOLD corpus: %d documents (split=%s)", len(documents), split_name)
    return documents


def load_samples(file_path: str | Path, split: str = "test") -> list[Sample]:
    """
    Return one Sample per CaseHOLD row from the given split file.

    There is NO valid retrieval ground truth for CaseHOLD.
    The context is stored as both 'question' (for search) and the document text.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, engine="python")

    split_name = file_path.stem.replace("case_hold_", "")
    samples: list[Sample] = []

    for row_idx, row in df.iterrows():
        context: str = str(row["context"]).strip()
        choices: list[str] = _parse_endings(row["endings"])

        try:
            label = int(row["label"])
            answer = choices[label] if 0 <= label < len(choices) else None
        except (ValueError, IndexError):
            label = -1
            answer = None

        # No meaningful short "question" exists; use context as the query.
        # This is documented as a limitation.
        samples.append(Sample(
            sample_id=f"CaseHOLD__{split_name}__{row_idx}",
            dataset="CaseHOLD",
            domain="Legal",
            split=split_name,
            question=context,         # context IS the query for retrieval
            answer=answer,
            options={str(i): c for i, c in enumerate(choices)},
            source_doc_id=None,       # classification — no specific source doc
            evidence=[],
            has_retrieval_gt=False,   # NOT a valid retrieval task
            metadata={
                "split":   split_name,
                "row_idx": int(row_idx),
                "label":   label,
                "choices": choices,
            },
        ))

    logger.info(
        "CaseHOLD samples: %d (split=%s, retrieval_gt=UNAVAILABLE)",
        len(samples), split_name,
    )
    return samples


def load_casehold(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    import re
    file_path = Path(file_path)
    df = pd.read_csv(file_path, engine="python")

    records: list[dict] = []
    for idx, row in df.iterrows():
        choices = re.findall(r"'([^']*)'", row["endings"])
        records.append({
            "id":      idx,
            "dataset": "CaseHOLD",
            "domain":  "Legal",
            "context": row["context"],
            "choices": choices,
            "answer":  choices[int(row["label"])] if choices else "",
            "label":   int(row["label"]),
        })
    return records
