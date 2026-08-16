"""CaseHOLD -- identifying the holding of a cited case (legal).

Audited shape of ``datasets/Legal/LexGLUE/case_hold/case_hold_{split}.csv``:
  columns context, endings, label. train 45,000 rows, validation 3,900,
  test 3,600. ``context`` is a citing excerpt (mean 843 chars) in which the
  cited holding has been masked with the literal token ``<HOLDING>`` (present
  in 3,549 of 3,600 test rows). ``endings`` is five candidate holding
  statements and ``label`` (0-4, near-uniform) indexes the correct one.

A parsing hazard worth recording: ``endings`` is serialised as a NumPy array
repr -- quoted strings separated by newlines with no commas. ``ast.literal_eval``
appears to succeed on it but silently concatenates the five adjacent string
literals into one, which would corrupt every label lookup without raising. This
adapter parses the quoted elements explicitly and asserts a count of five.

Ground truth: DERIVED, and the construction deserves scrutiny.

  What the dataset provides: one correct holding per citing context, taken from
  the case actually cited at that point. That relevance signal is genuine.

  What it does NOT provide: any external corpus. There is no case-law database,
  no document ids, no evidence offsets. CaseHOLD's native task is 5-way
  multiple choice, not retrieval.

  What this framework does: pools the deduplicated candidate holdings of the
  split into a shared corpus (18,000 endings -> 13,297 unique in test) and
  ranks it with the citing context as the query. The gold holding is the one
  the dataset labels. The *relevance* is therefore dataset-provided, while the
  *corpus* is constructed here, which is precisely what DERIVED denotes.

Because the corpus is our construction, CaseHOLD numbers answer "can this
retrieval strategy locate the correct legal holding among 13k real holdings"
and must not be presented as evidence about open-domain legal retrieval.

Corpus scope: POOLED. Note 752 test gold holdings also occur as another row's
distractor; deduplication by content means such a holding is a single corpus
document that is relevant to one query and not to others, which is handled
correctly by per-query relevance.
"""

from __future__ import annotations

import csv
import re
import sys
from typing import Dict, List

from core.settings import ChunkConfig
from core.ids import content_hash
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
from .base import DatasetAdapter, LoadedSplit, OPTION_KEYS

log = get_logger("adapters.casehold")

_SPLITS = {
    "train": "case_hold_train.csv",
    "validation": "case_hold_validation.csv",
    "test": "case_hold_test.csv",
}

# Matches one single- or double-quoted element of a NumPy array repr.
_ELEMENT = re.compile(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", re.S)


def parse_endings(raw: str) -> List[str]:
    """Parse the NumPy-array-repr ``endings`` field into its elements.

    Deliberately does not use ``ast.literal_eval``: Python's implicit
    adjacent-string-literal concatenation makes that silently return a single
    joined string for this format.
    """
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out: List[str] = []
    for match in _ELEMENT.finditer(text):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        out.append(value)
    return out


class CaseHOLDAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="casehold",
        domain="legal",
        relevance_class=RelevanceClass.DERIVED,
        corpus_scope=CorpusScope.POOLED,
        splits=["train", "validation", "test"],
        query_unit="citing context with the holding masked as <HOLDING>",
        corpus_unit="candidate holding statement",
        relevance_source=(
            "The dataset's own label identifying which of the five candidate "
            "holdings is the one actually cited."
        ),
        limitations=(
            "CaseHOLD ships no external retrieval corpus; its native task is "
            "5-way multiple choice. The corpus of pooled candidate holdings is "
            "constructed by this framework, so results characterise ranking "
            "within that constructed pool and are not open-domain legal "
            "retrieval. Exactly one holding is relevant per query, and holdings "
            "are short (mean 155 chars) relative to the citing context."
        ),
    )
    default_chunk = ChunkConfig(strategy="passage", max_tokens=512, overlap_tokens=64)

    def load(self, split: str = "test") -> LoadedSplit:
        if split not in _SPLITS:
            raise ValueError(f"casehold split must be one of {sorted(_SPLITS)}; got {split!r}")
        path = self._require(self.root / "Legal" / "LexGLUE" / "case_hold" / _SPLITS[split])
        skips = SkipLog("casehold", split)

        # LexGLUE rows carry large embedded text fields.
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        ds_cfg = self.config.dataset(self.name)
        if ds_cfg.max_documents:
            rows = rows[: ds_cfg.max_documents]

        documents: List[SourceDocument] = []
        queries: List[Query] = []
        by_content: Dict[str, str] = {}

        def intern(holding: str) -> str:
            """Register a holding as a corpus document, de-duplicating by text."""
            key = content_hash(holding)
            existing = by_content.get(key)
            if existing is not None:
                return existing
            document_id = make_doc_id("casehold", split, f"holding-{key}")
            by_content[key] = document_id
            documents.append(
                SourceDocument(
                    doc_id=document_id,
                    dataset="casehold",
                    split=split,
                    text=holding,
                    title=f"Holding {key}",
                    metadata={"content_hash": key, "n_chars": len(holding)},
                )
            )
            return document_id

        for index, row in enumerate(rows):
            context = (row.get("context") or "").strip()
            endings = parse_endings(row.get("endings") or "")
            if len(endings) != 5:
                # Never guess at a malformed row: dropping it is reported.
                skips.skip(f"endings_parse_expected_5_got_{len(endings)}")
                continue
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError):
                skips.skip("missing_or_malformed_label")
                continue
            if not 0 <= label < 5:
                skips.skip("label_out_of_range")
                continue
            if not context:
                skips.skip("empty_context")
                continue

            # Every candidate joins the corpus, so distractors act as genuine
            # hard negatives; only the labelled one is relevant to this query.
            document_ids = [intern(e) for e in endings]
            gold_document_id = document_ids[label]
            gold_text = endings[label]

            queries.append(
                Query(
                    query_id=make_query_id("casehold", split, f"{index:05d}"),
                    dataset="casehold",
                    split=split,
                    text=context,
                    scope_doc_id=None,
                    evidence=[
                        EvidenceSpan(doc_id=gold_document_id, start=0, end=len(gold_text))
                    ],
                    relevance_class=RelevanceClass.DERIVED,
                    answer=gold_text,
                    metadata={
                        "row_index": index,
                        "label": label,
                        "candidate_doc_ids": document_ids,
                        # The five candidate holdings, labelled for the
                        # generation stage. Source order is kept rather than
                        # shuffled: CaseHOLD already varies which position
                        # holds the correct answer, so re-ordering would only
                        # break correspondence with `label` and
                        # `candidate_doc_ids` for no gain.
                        "options": {
                            OPTION_KEYS[i]: ending for i, ending in enumerate(endings)
                        },
                        "answer_idx": OPTION_KEYS[label],
                        "has_holding_mask": "<HOLDING>" in context,
                        "answer_type": "multiple_choice",
                    },
                )
            )

        if ds_cfg.max_queries:
            queries = queries[: ds_cfg.max_queries]
            # Corpus kept whole so the pool of hard negatives is unchanged.

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": sum(1 for q in queries if q.has_ground_truth),
        }
        skips.report(log, kept=len(queries))
        log.info(
            "casehold/%s: %d unique holdings pooled as corpus, %d citing contexts",
            split, len(documents), len(queries),
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)
