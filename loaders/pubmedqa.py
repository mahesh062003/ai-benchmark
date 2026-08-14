"""PubMedQA -- biomedical yes/no/maybe QA over PubMed abstracts (medical).

Audited shape of ``datasets/Medical/PubMedQA/pqa_labeled/train-00000-of-00001.parquet``:
  1,000 expert-labelled records, no duplicate ``pubid``. Each record holds a
  question, a ``context`` dict whose ``contexts`` array carries the structured
  abstract sections (1-9, mean 3.36) with parallel ``labels``
  (BACKGROUND/METHODS/RESULTS/...), a ``long_answer`` conclusion, and a
  ``final_decision`` of yes (552) / no (338) / maybe (110).

Ground truth: DERIVED. The dataset carries no annotation marking which
*section* answers the question, but it does record which abstract the question
was written from -- the record's own ``pubid``. Relevance is derived from that
recorded provenance link: the sections of the source abstract are relevant, all
other abstracts' sections are not. That link is a real dataset fact, not an
inference from retrieval output.

The honest caveat, which is why this is DERIVED and not GOLD: relevance is
document-level projected onto every section, so a METHODS section counted as
relevant may not itself support the answer. This inflates Recall@k relative to
true section-level evidence and must not be compared directly against QASPER's
paragraph-level GOLD numbers.

Corpus scope: POOLED. Unlike CUAD/QASPER the question is self-contained
("Do mitochondria play a role in remodelling lace plant leaves...?"), so
ranking it against all abstracts in the split is a well-posed open-domain task.

Splits: the shipped parquet exposes ``pqa_labeled`` as a single HuggingFace
"train" split. The official PQA-L test fold files are not present here, so this
adapter exposes one ``labeled`` split covering all 1,000 records and says so,
rather than inventing a split. ``pqa_unlabeled`` (61,249) and ``pqa_artificial``
(211,269) are available as corpus-expansion options, off by default.
"""

from __future__ import annotations

from typing import List

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

log = get_logger("adapters.pubmedqa")


class PubMedQAAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="pubmedqa",
        domain="medical",
        relevance_class=RelevanceClass.DERIVED,
        corpus_scope=CorpusScope.POOLED,
        splits=["labeled"],
        query_unit="research question",
        corpus_unit="section of a PubMed abstract",
        relevance_source=(
            "Provenance: the abstract (pubid) the question was written from. All "
            "sections of that abstract are marked relevant."
        ),
        limitations=(
            "Relevance is document-level, not section-level: sections such as "
            "METHODS are marked relevant even where they do not support the "
            "answer, which inflates recall relative to true evidence "
            "annotation. Only pqa_labeled ships with expert labels and it "
            "carries no official train/test split in this distribution, so a "
            "single 'labeled' split of 1,000 records is used. The corpus is "
            "closed-domain over 1,000 abstracts unless expanded."
        ),
    )
    # Abstract sections are already passage-sized units.
    default_chunk = ChunkConfig(strategy="passage", max_tokens=512, overlap_tokens=64)

    def load(self, split: str = "labeled") -> LoadedSplit:
        if split != "labeled":
            raise ValueError(
                "pubmedqa exposes only the 'labeled' split (pqa_labeled ships as a "
                f"single HuggingFace train split with no official folds); got {split!r}"
            )
        path = self._require(
            self.root / "Medical" / "PubMedQA" / "pqa_labeled" / "train-00000-of-00001.parquet"
        )
        skips = SkipLog("pubmedqa", split)
        frame = pd.read_parquet(path)

        ds_cfg = self.config.dataset(self.name)
        if ds_cfg.max_documents:
            frame = frame.head(ds_cfg.max_documents)

        documents: List[SourceDocument] = []
        queries: List[Query] = []

        for _, row in frame.iterrows():
            pubid = str(row["pubid"])
            context = row["context"]

            def field(key: str) -> list:
                """Read an array field defensively.

                The parquet stores these as NumPy arrays, whose truthiness
                raises, so ``or []`` cannot be used here.
                """
                value = context.get(key) if context is not None else None
                return [] if value is None else list(value)

            sections = [str(s) for s in field("contexts")]
            labels = [str(x) for x in field("labels")]
            if not any(s.strip() for s in sections):
                skips.skip("abstract_without_sections")
                continue

            # One document per abstract, with the structured sections recorded
            # as segments so passage chunking reproduces the dataset's own
            # section boundaries instead of an arbitrary window.
            separator = "\n\n"
            segments: List[tuple] = []
            cursor = 0
            for section in sections:
                segments.append((cursor, cursor + len(section)))
                cursor += len(section) + len(separator)
            text = separator.join(sections)

            document_id = make_doc_id("pubmedqa", split, pubid)
            documents.append(
                SourceDocument(
                    doc_id=document_id,
                    dataset="pubmedqa",
                    split=split,
                    text=text,
                    title=f"PubMed abstract {pubid}",
                    segments=segments,
                    metadata={
                        "pubid": pubid,
                        "section_labels": labels,
                        "n_sections": len(sections),
                        "meshes": [str(m) for m in field("meshes")],
                    },
                )
            )

            question = str(row["question"]).strip()
            if not question:
                skips.skip("empty_question")
                continue

            queries.append(
                Query(
                    query_id=make_query_id("pubmedqa", split, pubid),
                    dataset="pubmedqa",
                    split=split,
                    text=question,
                    scope_doc_id=None,  # pooled corpus
                    # Whole-document provenance span.
                    evidence=[EvidenceSpan(doc_id=document_id, start=0, end=len(text))],
                    relevance_class=RelevanceClass.DERIVED,
                    answer=str(row["final_decision"]),
                    metadata={
                        "pubid": pubid,
                        "final_decision": str(row["final_decision"]),
                        "long_answer": str(row["long_answer"]),
                        "answer_type": "yes_no_maybe",
                    },
                )
            )

        if ds_cfg.max_queries:
            queries = queries[: ds_cfg.max_queries]
            # Documents are deliberately NOT filtered: the non-queried abstracts
            # remain in the corpus as distractors, which is what makes a reduced
            # query set a valid sample of the full retrieval task.

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": sum(1 for q in queries if q.has_ground_truth),
        }
        skips.report(log, kept=len(queries))
        log.info(
            "pubmedqa/%s: %d abstracts, %d questions (pooled corpus, %d with derived relevance)",
            split, len(documents), len(queries), stats["queries_with_evidence"],
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)
