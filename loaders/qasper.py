"""QASPER -- question answering over NLP research papers (scientific).

Audited shape of ``datasets/Scientific/qasper/{train,val,test}.parquet``:
  train 888 papers / 2,593 questions, val 281 / 1,005, test 416 / 1,451.
  ``full_text`` holds parallel arrays ``section_name`` and ``paragraphs``.
  ``qas`` holds parallel arrays of questions and, per question, a list of
  annotator answers each carrying an ``evidence`` array.

The decisive audit finding: ``evidence`` entries are the *literal paragraph
text* copied from ``full_text``, so they match paragraphs by exact string
lookup. Measured match rates over the shipped files:

  split  evidence items  exact   after strip()  FLOAT (figure/table)  unmatched
  train  4,209           3,534   119            386                  170
  val    2,808           2,349    72            253                  134
  test   5,744           4,855   176            459                  254

Ground truth: GOLD. Evidence is a human annotation of the supporting paragraph.

Handling of the non-matching remainder, none of which is fabricated:
  * ``FLOAT SELECTED: ...`` entries reference figures and tables, which do not
    exist in ``full_text`` paragraphs. They are counted and dropped.
  * A small residue of entries are bare section headings. They are counted and
    dropped rather than guessed at.
  * A question whose evidence yields no paragraph (including every
    ``unanswerable`` question) ends up with empty evidence and is excluded from
    retrieval metrics as NULL.

Corpus scope: DOCUMENT. QASPER questions are asked about a named paper and are
frequently deictic ("which datasets did they experiment with?"), so they are
only answerable relative to their own paper. This mirrors QASPER's own evidence
selection task.

Multiple annotators answer the same question; evidence is unioned across them,
which is the standard QASPER treatment and yields multiple relevant chunks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from core.settings import ChunkConfig
from core.ids import doc_id as make_doc_id
from core.ids import query_id as make_query_id
from core.logging_setup import SkipLog, get_logger
from core.types import (
    CorpusScope,
    DatasetSpec,
    EvidenceSpan,
    Query,
    RelevanceClass,
    SourceDocument,
)
from .base import DatasetAdapter, LoadedSplit

log = get_logger("adapters.qasper")

_SPLIT_FILES = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}


class QasperAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="qasper",
        domain="scientific",
        relevance_class=RelevanceClass.GOLD,
        corpus_scope=CorpusScope.DOCUMENT,
        splits=["train", "val", "test"],
        query_unit="question about one paper",
        corpus_unit="paragraph of that paper (passage chunking)",
        relevance_source=(
            "Annotator-selected evidence paragraphs, matched back to full_text "
            "paragraphs by exact string equality."
        ),
        limitations=(
            "Evidence referring to figures/tables (FLOAT SELECTED) cannot be "
            "mapped to body paragraphs and is dropped, as is a small residue of "
            "section-heading-only evidence; affected questions become NULL "
            "rather than zero. Unanswerable questions have no evidence by "
            "construction and are excluded. Retrieval is scoped to the paper."
        ),
    )
    # Paragraphs are QASPER's own annotation unit; preserving their boundaries
    # keeps the evidence mapping one-to-one.
    default_chunk = ChunkConfig(strategy="passage", max_tokens=512, overlap_tokens=64)

    def load(self, split: str = "test") -> LoadedSplit:
        if split not in _SPLIT_FILES:
            raise ValueError(f"qasper split must be one of {sorted(_SPLIT_FILES)}; got {split!r}")
        path = self._require(self.root / "Scientific" / "qasper" / _SPLIT_FILES[split])
        skips = SkipLog("qasper", split)
        frame = pd.read_parquet(path)

        ds_cfg = self.config.dataset(self.name)
        if ds_cfg.max_documents:
            frame = frame.head(ds_cfg.max_documents)

        documents: List[SourceDocument] = []
        queries: List[Query] = []
        n_float = 0
        n_unmatched = 0
        n_unanswerable = 0

        for _, row in frame.iterrows():
            paper_id = str(row["id"])
            document_id = make_doc_id("qasper", split, paper_id)

            text, offsets, segments = self._assemble(row)
            if not text.strip():
                skips.skip("paper_without_body_text")
                continue

            documents.append(
                SourceDocument(
                    doc_id=document_id,
                    dataset="qasper",
                    split=split,
                    text=text,
                    title=str(row["title"]),
                    # Paragraph boundaries, so a chunk == an annotatable paragraph.
                    segments=segments,
                    metadata={
                        "paper_id": paper_id,
                        "title": str(row["title"]),
                        "n_paragraphs": len(offsets),
                    },
                )
            )

            qas = row["qas"]
            for i, question in enumerate(qas["question"]):
                spans: List[EvidenceSpan] = []
                seen: set[Tuple[int, int]] = set()
                answerable = False
                for annotation in qas["answers"][i]["answer"]:
                    if annotation["unanswerable"]:
                        continue
                    answerable = True
                    for evidence_text in annotation["evidence"]:
                        if evidence_text.startswith("FLOAT SELECTED"):
                            n_float += 1
                            skips.skip("evidence_is_figure_or_table")
                            continue
                        located = offsets.get(evidence_text) or offsets.get(evidence_text.strip())
                        if located is None:
                            n_unmatched += 1
                            skips.skip("evidence_not_found_in_full_text")
                            continue
                        if located in seen:
                            continue
                        seen.add(located)
                        spans.append(
                            EvidenceSpan(doc_id=document_id, start=located[0], end=located[1])
                        )
                if not answerable:
                    n_unanswerable += 1

                queries.append(
                    Query(
                        query_id=make_query_id("qasper", split, str(qas["question_id"][i])),
                        dataset="qasper",
                        split=split,
                        text=str(question),
                        scope_doc_id=document_id,
                        evidence=spans,
                        relevance_class=RelevanceClass.GOLD,
                        answer=self._answer_text(qas["answers"][i]["answer"]),
                        metadata={
                            "paper_id": paper_id,
                            "question_id": str(qas["question_id"][i]),
                            "answerable": answerable,
                            "n_evidence_paragraphs": len(spans),
                        },
                    )
                )

        if ds_cfg.max_queries:
            queries = queries[: ds_cfg.max_queries]
            kept = {q.scope_doc_id for q in queries}
            documents = [d for d in documents if d.doc_id in kept]

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": sum(1 for q in queries if q.has_ground_truth),
            "evidence_figure_or_table_dropped": n_float,
            "evidence_unmatched_dropped": n_unmatched,
            "queries_unanswerable": n_unanswerable,
        }
        skips.report(log, kept=len(queries))
        log.info(
            "qasper/%s: %d papers, %d questions, %d with gold evidence "
            "(dropped %d float + %d unmatched evidence items, %d unanswerable)",
            split, len(documents), len(queries), stats["queries_with_evidence"],
            n_float, n_unmatched, n_unanswerable,
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)

    @staticmethod
    def _assemble(row) -> Tuple[str, Dict[str, Tuple[int, int]], List[Tuple[int, int]]]:
        """Concatenate the paper's paragraphs, recording each one's char span.

        Returns the joined text, a lookup from paragraph text to its character
        range (exactly the form QASPER's evidence annotations take), and the
        list of paragraph ranges used as chunk boundaries.

        Where a paragraph repeats verbatim the first occurrence wins in the
        lookup; that is a deliberate, documented choice and affects only which
        of two identical texts is credited.
        """
        full_text = row["full_text"]
        parts: List[str] = []
        offsets: Dict[str, Tuple[int, int]] = {}
        segments: List[Tuple[int, int]] = []
        cursor = 0
        separator = "\n\n"

        def add(paragraph: str) -> None:
            nonlocal cursor
            start = cursor
            end = start + len(paragraph)
            offsets.setdefault(paragraph, (start, end))
            stripped = paragraph.strip()
            if stripped != paragraph:
                offsets.setdefault(stripped, (start, end))
            segments.append((start, end))
            parts.append(paragraph)
            cursor = end + len(separator)

        abstract = str(row["abstract"]) if row["abstract"] is not None else ""
        if abstract.strip():
            add(abstract)

        for paragraphs in full_text["paragraphs"]:
            for paragraph in paragraphs:
                paragraph = str(paragraph)
                if paragraph.strip():
                    add(paragraph)
        return separator.join(parts), offsets, segments

    @staticmethod
    def _answer_text(annotations) -> str | None:
        """First non-empty human answer, preferring free-form over extractive.

        Array-valued fields are length-checked rather than truth-tested, since
        the parquet delivers them as NumPy arrays whose truthiness raises.
        """
        for annotation in annotations:
            if annotation["unanswerable"]:
                continue
            free_form = str(annotation["free_form_answer"] or "").strip()
            if free_form:
                return free_form
            spans = annotation["extractive_spans"]
            if spans is not None and len(spans) > 0:
                return "; ".join(str(s) for s in spans)
            yes_no = annotation["yes_no"]
            if yes_no is not None:
                return "yes" if yes_no else "no"
        return None
