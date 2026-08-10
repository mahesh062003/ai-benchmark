"""
Shared data types for the benchmark.

Every loader returns a list of Sample dicts and a list of CorpusDocument dicts.
All downstream modules depend only on these structures.

Sample — one benchmark query/QA pair.
CorpusDocument — one document to be chunked and indexed.

Using TypedDict rather than dataclasses keeps the code readable and easy to
serialise to JSON without extra machinery.
"""

from typing import Any, TypedDict


class CorpusDocument(TypedDict):
    """One document unit to be chunked and indexed."""
    doc_id:  str            # stable, unique across the dataset corpus
    dataset: str
    domain:  str
    text:    str            # the full text that will be chunked
    metadata: dict[str, Any]


class Sample(TypedDict):
    """One benchmark evaluation query."""
    sample_id:    str       # stable unique ID for this QA pair
    dataset:      str
    domain:       str
    split:        str       # 'train' | 'dev' | 'test' | 'all'
    question:     str       # the query sent to the retriever
    answer:       Any       # gold answer (str, list, or None)
    options:      dict[str, str] | None   # MCQ option letter → text, or None
    source_doc_id: str | None             # which CorpusDocument this QA is about
    evidence:     list[str]               # gold evidence passages (may be empty)
    has_retrieval_gt: bool                # whether valid retrieval ground truth exists
    metadata:     dict[str, Any]
