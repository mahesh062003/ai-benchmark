"""Shared fixtures.

Unit tests use tiny synthetic fixtures and never require the real datasets or
an embedding model. Tests that do need them are marked ``slow`` and skip
cleanly when the data or model is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import ChunkConfig, Config, PathConfig
from core.types import (
    Chunk,
    CorpusScope,
    EvidenceSpan,
    Query,
    RelevanceClass,
    SourceDocument,
)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real datasets or models")


@pytest.fixture
def real_datasets_available() -> bool:
    return DATASETS_DIR.exists()


@pytest.fixture
def config(tmp_path) -> Config:
    """A config pointing all artefact paths at a temporary directory."""
    cfg = Config()
    cfg.paths = PathConfig(
        datasets=str(DATASETS_DIR),
        artifacts=str(tmp_path),
        corpora=str(tmp_path / "corpora"),
        indexes=str(tmp_path / "indexes"),
        results=str(tmp_path / "results"),
        database=str(tmp_path / "test.sqlite"),
    )
    return cfg


@pytest.fixture
def document() -> SourceDocument:
    # 40 words, so a size-10/overlap-5 window yields several chunks.
    words = [f"word{i:02d}" for i in range(40)]
    return SourceDocument(
        doc_id="ds/split/doc1",
        dataset="ds",
        split="split",
        text=" ".join(words),
        title="fixture document",
    )


@pytest.fixture
def segmented_document() -> SourceDocument:
    parts = ["first section text here", "second section is a bit longer than the first",
             "third and final section"]
    separator = "\n\n"
    segments = []
    cursor = 0
    for part in parts:
        segments.append((cursor, cursor + len(part)))
        cursor += len(part) + len(separator)
    return SourceDocument(
        doc_id="ds/split/seg1",
        dataset="ds",
        split="split",
        text=separator.join(parts),
        segments=segments,
    )


@pytest.fixture
def small_chunks() -> list[Chunk]:
    """A five-chunk toy corpus with distinguishable vocabulary."""
    texts = [
        "the contract governs payment terms and invoicing schedules",
        "termination for convenience requires ninety days written notice",
        "mitochondria regulate programmed cell death in plant leaves",
        "photosynthesis converts light energy into chemical energy in chloroplasts",
        "the governing law of this agreement is the state of delaware",
    ]
    return [
        Chunk(
            chunk_id=f"toy/test/doc{i}#c00000",
            doc_id=f"toy/test/doc{i}",
            dataset="toy",
            split="test",
            text=text,
            ordinal=0,
            char_start=0,
            char_end=len(text),
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def fixed_chunk_config() -> ChunkConfig:
    return ChunkConfig(strategy="fixed", size_tokens=10, overlap_tokens=5, min_chunk_chars=1)


@pytest.fixture
def query_with_evidence() -> Query:
    return Query(
        query_id="ds/split/q/1",
        dataset="ds",
        split="split",
        text="what are the payment terms?",
        scope_doc_id="ds/split/doc1",
        evidence=[EvidenceSpan("ds/split/doc1", 0, 20)],
        relevance_class=RelevanceClass.GOLD,
    )
