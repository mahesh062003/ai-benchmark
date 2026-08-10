"""
PubMedQA loader (biomedical research QA).

Dataset format: CSV with columns: pubid, question, context, long_answer,
                final_decision.
The 'context' column is a stringified Python dict with a 'contexts' key
containing a list of abstract sentences assembled for that question.

Corpus strategy:
  CorpusDocument = one document per pubid (the concatenated abstract sentences).
  doc_id = "PubMedQA__{pubid}".

Sample / ground-truth strategy:
  Each QA pair has a set of context sentences that are ALL relevant
  (they were curated from PubMed specifically for that question).
  Ground truth = all corpus chunks whose source doc_id matches
                 "PubMedQA__{pubid}".

  This is a CONSERVATIVE definition of relevance: all context sentences are
  relevant for the corresponding question.  It does NOT evaluate whether the
  retriever finds the single most informative sentence.

  has_retrieval_gt = True.
  Recall@K, MRR, nDCG are valid within this corpus construction.

Limitation: corpus and query sets overlap (same split).  No separate train/test
document split exists in the labeled PubMedQA release.  This limitation is
documented and does not involve training any model weights.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from loaders.base import CorpusDocument, Sample

logger = logging.getLogger(__name__)


def _parse_context_sentences(raw: str) -> list[str]:
    """
    Extract the list of context sentences from the raw stringified dict.

    The stored value looks like:
        {'contexts': ['sent1', 'sent2', ...], 'labels': [...], ...}
    """
    # Try literal_eval first (handles proper Python reprs)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            contexts = parsed.get("contexts", [])
            if isinstance(contexts, list):
                return [str(c).strip() for c in contexts if str(c).strip()]
    except Exception:
        pass

    # Fallback: regex extraction
    match = re.search(
        r"'contexts':\s*\[(.*?)\]\s*,\s*'labels'",
        str(raw),
        flags=re.DOTALL,
    )
    if match:
        return [s.strip() for s in re.findall(r"'([^']*)'", match.group(1)) if s.strip()]

    logger.warning("PubMedQA: could not parse context field")
    return []


def load_corpus(file_path: str | Path) -> list[CorpusDocument]:
    """Return one CorpusDocument per unique pubid (the abstract context)."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    documents: list[CorpusDocument] = []
    for _, row in df.iterrows():
        sentences = _parse_context_sentences(str(row["context"]))
        if not sentences:
            continue

        text = "\n".join(sentences)
        doc_id = f"PubMedQA__{row['pubid']}"

        documents.append(CorpusDocument(
            doc_id=doc_id,
            dataset="PubMedQA",
            domain="Medical",
            text=text,
            metadata={"pubid": int(row["pubid"])},
        ))

    logger.info("PubMedQA corpus: %d documents", len(documents))
    return documents


def load_samples(file_path: str | Path, split: str = "train_labeled") -> list[Sample]:
    """Return one Sample per PubMedQA question."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    samples: list[Sample] = []
    skipped = 0

    for _, row in df.iterrows():
        question: str = str(row["question"]).strip()
        if not question:
            skipped += 1
            continue

        sentences = _parse_context_sentences(str(row["context"]))
        pubid = int(row["pubid"])
        doc_id = f"PubMedQA__{pubid}"

        samples.append(Sample(
            sample_id=f"PubMedQA__{pubid}",
            dataset="PubMedQA",
            domain="Medical",
            split=split,
            question=question,
            answer=str(row.get("final_decision", "")).strip(),
            options={"yes": "yes", "no": "no", "maybe": "maybe"},
            source_doc_id=doc_id,
            evidence=sentences,      # all context sentences are relevant
            has_retrieval_gt=True,
            metadata={
                "pubid":         pubid,
                "final_decision": str(row.get("final_decision", "")).strip(),
                "long_answer":   str(row.get("long_answer", "")).strip(),
            },
        ))

    if skipped:
        logger.warning("PubMedQA: skipped %d rows with empty question", skipped)
    logger.info("PubMedQA samples: %d", len(samples))
    return samples


def load_pubmedqa(file_path: str | Path) -> list[dict[str, Any]]:
    """Legacy convenience wrapper — returns flat dicts compatible with old code."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    records: list[dict] = []
    for _, row in df.iterrows():
        contexts = _parse_context_sentences(str(row["context"]))
        records.append({
            "id":             row["pubid"],
            "dataset":        "PubMedQA",
            "domain":         "Medical",
            "question":       row["question"],
            "context":        contexts,
            "long_answer":    row["long_answer"],
            "final_decision": row["final_decision"],
        })
    return records
