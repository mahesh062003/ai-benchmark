"""
QASPER loader (NLP paper QA).

Dataset format: Parquet.  Each row is one paper with its full text and a set
of QA pairs (including evidence passages).

Corpus strategy:
  CorpusDocument = one document per paper (paper_id).
  Build corpus from BOTH train and test splits to ensure all paper texts
  needed by test QA pairs are present in the corpus.
  doc_id = "QASPER__{paper_id}".

Sample / ground-truth strategy:
  Each QA pair with non-empty evidence is one Sample.
  Ground truth = corpus chunks whose text overlaps with any evidence passage.

  QASPER evidence passages are exact text spans extracted from the paper.
  Matching is done via substring containment in evaluation/ground_truth.py.

  has_retrieval_gt = True when evidence is non-empty and question is answerable.
  Unanswerable questions have has_retrieval_gt = False.

Split note:
  Original code used train.parquet for both corpus and queries (incorrect).
  Refactored: corpus = train + test papers; queries = test QA pairs.
  Train/test paper sets are disjoint (verified: 0 overlap in paper IDs).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def _to_list(val: Any) -> list:
    """Safely convert a value that may be a numpy array, list, or None to a plain list."""
    if val is None:
        return []
    if hasattr(val, "tolist"):
        return val.tolist()
    return list(val)


def _build_paper_text(full_text: Any) -> str:
    """Concatenate section names and paragraph texts into one document string."""
    sections   = _to_list(full_text.get("section_name", None))
    paragraphs = _to_list(full_text.get("paragraphs", None))

    parts: list[str] = []
    for section, para_list in zip(sections, paragraphs):
        if section:
            parts.append(str(section).strip())
        for para in _to_list(para_list):
            if para:
                parts.append(str(para).strip())

    return "\n\n".join(p for p in parts if p)


def _extract_answer(annotation: dict) -> str:
    """Pick the best human-readable answer from one annotation dict."""
    free_form = annotation.get("free_form_answer", "")
    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip()

    spans = annotation.get("extractive_spans", [])
    if hasattr(spans, "tolist"):
        spans = spans.tolist()
    if spans:
        return " | ".join(str(s) for s in spans)

    yes_no = annotation.get("yes_no", None)
    if yes_no is True:
        return "Yes"
    if yes_no is False:
        return "No"

    return ""


def _extract_evidence(annotation: dict) -> list[str]:
    """Return list of evidence passage strings from one annotation dict."""
    evidence = annotation.get("evidence", [])
    if hasattr(evidence, "tolist"):
        evidence = evidence.tolist()
    return [str(e).strip() for e in (evidence or []) if str(e).strip()]


def load_corpus(
    train_path: str | Path | None = None,
    test_path:  str | Path | None = None,
) -> list[CorpusDocument]:
    """
    Build the QASPER corpus from all available papers (train + test splits).
    At least one of train_path / test_path must be provided.
    """
    documents: list[CorpusDocument] = []
    seen_ids: set[str] = set()

    for path in filter(None, [train_path, test_path]):
        path = Path(path)
        if not path.exists():
            logger.warning("QASPER corpus: path not found — %s", path)
            continue

        df = pd.read_parquet(path)
        split_label = "train" if "train" in path.name else "test"

        for _, row in df.iterrows():
            paper_id: str = str(row["id"])
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            text = _build_paper_text(row["full_text"])
            if not text:
                continue

            doc_id = f"QASPER__{paper_id}"
            documents.append(CorpusDocument(
                doc_id=doc_id,
                dataset="QASPER",
                domain="Scientific",
                text=text,
                metadata={
                    "paper_id": paper_id,
                    "title":    str(row.get("title", "")),
                    "split":    split_label,
                },
            ))

    logger.info("QASPER corpus: %d papers", len(documents))
    return documents


def load_samples(file_path: str | Path, split: str = "test") -> list[Sample]:
    """Return one Sample per answerable QA pair in the given file."""
    file_path = Path(file_path)
    df = pd.read_parquet(file_path)

    samples: list[Sample] = []
    skipped_unanswerable = 0
    skipped_no_question  = 0

    for _, row in df.iterrows():
        paper_id: str = str(row["id"])
        doc_id = f"QASPER__{paper_id}"

        qas = row["qas"]
        questions    = _to_list(qas.get("question",    None))
        question_ids = _to_list(qas.get("question_id", None))
        answers_list = _to_list(qas.get("answers",     None))

        for qid, question, answer_group in zip(question_ids, questions, answers_list):
            question = str(question).strip()
            if not question:
                skipped_no_question += 1
                continue

            annotations = answer_group.get("answer", [])
            if hasattr(annotations, "tolist"):
                annotations = annotations.tolist()
            annotations = list(annotations or [])

            if not annotations:
                skipped_unanswerable += 1
                continue

            first = annotations[0]
            if first.get("unanswerable", False):
                skipped_unanswerable += 1
                continue

            evidence = _extract_evidence(first)
            answer   = _extract_answer(first)

            samples.append(Sample(
                sample_id=f"QASPER__{paper_id}__{qid}",
                dataset="QASPER",
                domain="Scientific",
                split=split,
                question=question,
                answer=answer,
                options=None,
                source_doc_id=doc_id,
                evidence=evidence,
                has_retrieval_gt=bool(evidence),
                metadata={
                    "paper_id":    paper_id,
                    "question_id": str(qid),
                    "title":       str(row.get("title", "")),
                },
            ))

    logger.info(
        "QASPER samples: %d (split=%s, skipped unanswerable=%d, no-question=%d)",
        len(samples), split, skipped_unanswerable, skipped_no_question,
    )
    return samples


def load_qasper(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    file_path = Path(file_path)
    df = pd.read_parquet(file_path)
    dataset: list[dict] = []

    for _, row in df.iterrows():
        context = _build_paper_text(row["full_text"])
        qas = row["qas"]
        questions    = list(qas["question"])
        question_ids = list(qas["question_id"])
        answers      = list(qas["answers"])

        for qid, question, answer_group in zip(question_ids, questions, answers):
            annotations = answer_group["answer"]
            if hasattr(annotations, "tolist"):
                annotations = annotations.tolist()
            annotations = list(annotations)
            if not annotations:
                continue

            first    = annotations[0]
            evidence = _extract_evidence(first)

            dataset.append({
                "id":       qid,
                "dataset":  "QASPER",
                "domain":   "Scientific",
                "paper_id": row["id"],
                "title":    row["title"],
                "abstract": row["abstract"],
                "context":  context,
                "question": question,
                "answer":   _extract_answer(first),
                "evidence": evidence,
            })

    return dataset
