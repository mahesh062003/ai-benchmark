"""Core data model for the RAG benchmarking framework.

The model is deliberately small. Five concepts carry the whole pipeline:

    SourceDocument  a de-duplicated unit of source text (a contract, a paper,
                    an abstract, a support passage, a textbook)
    Chunk           a retrieval unit carved out of a SourceDocument, carrying
                    the character offsets it came from
    Query           a question asked against a corpus
    EvidenceSpan    a character range of a SourceDocument that the *dataset*
                    asserts is evidence for a Query
    Qrels           the mapping Query -> relevant Chunk ids, plus the provenance
                    of that judgement

The key unification: every form of dataset ground truth (CUAD answer offsets,
QASPER evidence paragraphs, PubMedQA abstract provenance, ...) is normalised
into ``EvidenceSpan`` objects. Chunk relevance is then a single, testable rule:
a chunk is relevant iff its character range overlaps an evidence span. There is
exactly one place in the codebase where relevance is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RelevanceClass(str, Enum):
    """How trustworthy a dataset's retrieval ground truth is.

    The distinction matters more than any individual metric value: a number
    produced under HEURISTIC is not comparable to one produced under GOLD, and
    reporting them in the same column without annotation would be misleading.
    """

    GOLD = "GOLD"
    """Human annotators marked the evidence itself, at or below document level,
    and it maps deterministically onto retrieval units. CUAD, QASPER."""

    DERIVED = "DERIVED"
    """The dataset records an unambiguous provenance link (this question was
    written from this passage / this holding is the cited one). Relevance is
    derived from that recorded link -- never from retrieval output. The link is
    real, but it is document-level, so some relevant-marked text may not
    actually support the answer. PubMedQA, SciQ, CaseHOLD."""

    HEURISTIC = "HEURISTIC"
    """Relevance guessed by a rule such as answer-string containment. Defined
    here for completeness of the taxonomy; NO dataset in this framework uses
    it. If a future dataset is added under this class its metrics must be
    reported separately and never pooled with GOLD/DERIVED results."""

    UNSUPPORTED = "UNSUPPORTED"
    """No relevance information exists in the dataset. Retrieval metrics are
    reported as NULL, never as zero. MedQA."""


class CorpusScope(str, Enum):
    """What a query is allowed to retrieve from."""

    DOCUMENT = "document"
    """Closed-domain within one source document: the query only ranks chunks of
    its own document (CUAD clause extraction, QASPER evidence selection). Used
    where the query text carries no signal identifying which document it
    belongs to, so a pooled corpus would be ill-posed rather than merely hard."""

    POOLED = "pooled"
    """All documents of the dataset split form one shared corpus."""


@dataclass(frozen=True)
class SourceDocument:
    """A unit of source text, stored once regardless of how many queries use it.

    ``segments`` records the document's own structural boundaries as character
    ranges -- QASPER paragraphs, PubMedQA abstract sections. When present, the
    ``passage`` chunking strategy cuts on these boundaries instead of imposing
    an arbitrary window, which is what keeps a retrieval unit equal to the unit
    the dataset's annotators actually judged. An empty list means the document
    has no declared structure and is treated as one segment.
    """

    doc_id: str
    dataset: str
    split: str
    text: str
    title: str = ""
    segments: List[tuple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit.

    ``char_start``/``char_end`` index into the parent SourceDocument's ``text``
    and are what make evidence-to-chunk mapping exact rather than approximate.
    """

    chunk_id: str
    doc_id: str
    dataset: str
    split: str
    text: str
    ordinal: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overlaps(self, start: int, end: int) -> int:
        """Number of characters shared with ``[start, end)``."""
        return max(0, min(self.char_end, end) - max(self.char_start, start))


@dataclass(frozen=True)
class EvidenceSpan:
    """A character range of a source document that the dataset asserts is evidence.

    ``[start, end)``. For provenance-level ground truth (PubMedQA, SciQ,
    CaseHOLD) the span covers the whole document.
    """

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"evidence span end {self.end} precedes start {self.start}")


@dataclass(frozen=True)
class Query:
    """A question posed to the benchmark.

    ``scope_doc_id`` is set only for CorpusScope.DOCUMENT datasets and names
    the single document the query may retrieve from.
    """

    query_id: str
    dataset: str
    split: str
    text: str
    scope_doc_id: Optional[str] = None
    evidence: List[EvidenceSpan] = field(default_factory=list)
    relevance_class: RelevanceClass = RelevanceClass.UNSUPPORTED
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_ground_truth(self) -> bool:
        """True iff this query can legitimately be scored with retrieval metrics.

        A query in a GOLD/DERIVED dataset can still lack ground truth
        individually (an unanswerable CUAD category, a SciQ item with an empty
        support). Such queries are excluded from metrics, not scored zero.
        """
        return self.relevance_class in (
            RelevanceClass.GOLD,
            RelevanceClass.DERIVED,
            RelevanceClass.HEURISTIC,
        ) and bool(self.evidence)


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked result. ``rank`` is 1-based."""

    chunk_id: str
    rank: int
    score: float


@dataclass
class DatasetSpec:
    """Static, audited description of how one dataset is benchmarked.

    Declared per adapter so that the methodology is visible in code review and
    can be exported into the results database and the dissertation write-up.
    """

    name: str
    domain: str
    relevance_class: RelevanceClass
    corpus_scope: CorpusScope
    splits: List[str]
    query_unit: str
    corpus_unit: str
    relevance_source: str
    limitations: str
    supports_generation: bool = True

    @property
    def supports_retrieval_metrics(self) -> bool:
        return self.relevance_class is not RelevanceClass.UNSUPPORTED
