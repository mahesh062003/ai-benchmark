"""Corpus construction and persistence.

A *corpus build* is the frozen product of (dataset, split, chunking config):
de-duplicated source documents, their chunks, the queries, and the qrels
(query -> relevant chunk ids). Builds are written to
``artifacts/corpora/<dataset>/<split>/<chunk-fingerprint>/`` so that changing
the chunking configuration produces a separate build rather than silently
invalidating an index.

Qrels carry an explicit availability status. This is the mechanism that keeps
"the retriever found nothing" (score 0) distinct from "there was nothing to
score" (NULL) all the way from the dataset to the results table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loaders import get_adapter
from .chunking import chunk_documents, relevant_chunk_ids
from core.settings import ChunkConfig, Config
from core.ids import content_hash
from core.logging_setup import get_logger
from core.types import (
    Chunk,
    CorpusScope,
    EvidenceSpan,
    Query,
    RelevanceClass,
    SourceDocument,
)

log = get_logger("corpus")

MANIFEST = "manifest.json"


@dataclass
class Qrel:
    """Relevance judgement for one query.

    ``available`` False means no legitimate ground truth exists for this query;
    metrics must then be recorded as NULL. ``reason`` explains why, so an
    unavailable judgement can always be traced back to a dataset fact.
    """

    query_id: str
    relevant_chunk_ids: List[str]
    relevance_class: RelevanceClass
    available: bool
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "query_id": self.query_id,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "relevance_class": self.relevance_class.value,
            "available": self.available,
            "reason": self.reason,
        }

    @staticmethod
    def from_json(raw: dict) -> "Qrel":
        return Qrel(
            query_id=raw["query_id"],
            relevant_chunk_ids=list(raw["relevant_chunk_ids"]),
            relevance_class=RelevanceClass(raw["relevance_class"]),
            available=bool(raw["available"]),
            reason=raw.get("reason", ""),
        )


@dataclass
class CorpusBuild:
    dataset: str
    split: str
    scope: CorpusScope
    chunk_config: ChunkConfig
    documents: List[SourceDocument]
    chunks: List[Chunk]
    queries: List[Query]
    qrels: Dict[str, Qrel]
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return self.chunk_config.fingerprint()

    def chunks_for_query(self, query: Query) -> List[Chunk]:
        """The chunks a query is allowed to retrieve from."""
        if self.scope is CorpusScope.DOCUMENT and query.scope_doc_id:
            return [c for c in self.chunks if c.doc_id == query.scope_doc_id]
        return self.chunks

    def scoreable_queries(self) -> List[Query]:
        return [q for q in self.queries if self.qrels[q.query_id].available]


def build_directory(config: Config, dataset: str, split: str, fingerprint: str) -> Path:
    return config.paths.corpora_dir / dataset / split / fingerprint


def build_corpus(config: Config, dataset: str, split: Optional[str] = None) -> CorpusBuild:
    """Load a dataset split, chunk it, and compute qrels."""
    adapter = get_adapter(dataset, config)
    split = split or adapter.default_split()
    loaded = adapter.load(split)
    chunk_config = adapter.chunk_config()
    scope = adapter.spec.corpus_scope

    chunks = chunk_documents(loaded.documents, chunk_config)
    log.info(
        "%s/%s: %d documents -> %d chunks (%s)",
        dataset, split, len(loaded.documents), len(chunks), chunk_config.fingerprint(),
    )

    by_document: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    qrels: Dict[str, Qrel] = {}
    n_available = 0
    n_no_evidence = 0
    n_evidence_unmapped = 0

    for query in loaded.queries:
        if adapter.spec.relevance_class is RelevanceClass.UNSUPPORTED:
            qrels[query.query_id] = Qrel(
                query.query_id, [], RelevanceClass.UNSUPPORTED, False,
                "dataset provides no retrieval ground truth",
            )
            continue
        if not query.evidence:
            n_no_evidence += 1
            qrels[query.query_id] = Qrel(
                query.query_id, [], query.relevance_class, False,
                "record carries no evidence annotation",
            )
            continue

        candidates: List[Chunk] = []
        for span in query.evidence:
            candidates.extend(by_document.get(span.doc_id, ()))
        relevant = relevant_chunk_ids(query.evidence, candidates)

        if not relevant:
            # Evidence existed but landed on no chunk. Reporting this as an
            # unavailable judgement rather than an empty one prevents a
            # chunking artefact from being scored as retrieval failure.
            n_evidence_unmapped += 1
            qrels[query.query_id] = Qrel(
                query.query_id, [], query.relevance_class, False,
                "evidence spans did not map onto any chunk",
            )
            continue

        n_available += 1
        qrels[query.query_id] = Qrel(
            query.query_id, relevant, query.relevance_class, True,
        )

    stats = {
        **loaded.stats,
        "chunks": len(chunks),
        "queries_scoreable": n_available,
        "queries_without_evidence": n_no_evidence,
        "queries_evidence_unmapped": n_evidence_unmapped,
    }
    if n_evidence_unmapped:
        log.warning(
            "%s/%s: %d queries had evidence that mapped to no chunk; recorded as "
            "unavailable (NULL), not as zero",
            dataset, split, n_evidence_unmapped,
        )
    log.info(
        "%s/%s: %d/%d queries scoreable for retrieval metrics (%s)",
        dataset, split, n_available, len(loaded.queries),
        adapter.spec.relevance_class.value,
    )

    return CorpusBuild(
        dataset=dataset, split=split, scope=scope, chunk_config=chunk_config,
        documents=loaded.documents, chunks=chunks, queries=loaded.queries,
        qrels=qrels, stats=stats,
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_corpus(config: Config, build: CorpusBuild) -> Path:
    directory = build_directory(config, build.dataset, build.split, build.fingerprint)
    directory.mkdir(parents=True, exist_ok=True)

    _write_jsonl(directory / "documents.jsonl", (
        {
            "doc_id": d.doc_id, "dataset": d.dataset, "split": d.split,
            "title": d.title, "text": d.text,
            "segments": [list(s) for s in d.segments], "metadata": d.metadata,
        }
        for d in build.documents
    ))
    _write_jsonl(directory / "chunks.jsonl", (
        {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "dataset": c.dataset,
            "split": c.split, "text": c.text, "ordinal": c.ordinal,
            "char_start": c.char_start, "char_end": c.char_end,
        }
        for c in build.chunks
    ))
    _write_jsonl(directory / "queries.jsonl", (
        {
            "query_id": q.query_id, "dataset": q.dataset, "split": q.split,
            "text": q.text, "scope_doc_id": q.scope_doc_id,
            "evidence": [[e.doc_id, e.start, e.end] for e in q.evidence],
            "relevance_class": q.relevance_class.value,
            "answer": q.answer, "metadata": q.metadata,
        }
        for q in build.queries
    ))
    _write_jsonl(directory / "qrels.jsonl", (q.to_json() for q in build.qrels.values()))

    manifest = {
        "dataset": build.dataset,
        "split": build.split,
        "scope": build.scope.value,
        "chunk_config": {
            "strategy": build.chunk_config.strategy,
            "size_tokens": build.chunk_config.size_tokens,
            "overlap_tokens": build.chunk_config.overlap_tokens,
            "max_tokens": build.chunk_config.max_tokens,
            "min_chunk_chars": build.chunk_config.min_chunk_chars,
        },
        "fingerprint": build.fingerprint,
        "stats": build.stats,
        "corpus_hash": content_hash("".join(c.chunk_id for c in build.chunks)),
    }
    (directory / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("saved corpus build -> %s", directory)
    return directory


def load_corpus(config: Config, dataset: str, split: str, fingerprint: str) -> CorpusBuild:
    directory = build_directory(config, dataset, split, fingerprint)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no corpus build at {directory}. Run: ragbench build-corpus --dataset {dataset}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    documents = [
        SourceDocument(
            doc_id=r["doc_id"], dataset=r["dataset"], split=r["split"], text=r["text"],
            title=r["title"], segments=[tuple(s) for s in r.get("segments", [])],
            metadata=r.get("metadata", {}),
        )
        for r in _read_jsonl(directory / "documents.jsonl")
    ]
    chunks = [
        Chunk(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], dataset=r["dataset"],
            split=r["split"], text=r["text"], ordinal=r["ordinal"],
            char_start=r["char_start"], char_end=r["char_end"],
        )
        for r in _read_jsonl(directory / "chunks.jsonl")
    ]
    queries = [
        Query(
            query_id=r["query_id"], dataset=r["dataset"], split=r["split"], text=r["text"],
            scope_doc_id=r["scope_doc_id"],
            evidence=[EvidenceSpan(d, s, e) for d, s, e in r["evidence"]],
            relevance_class=RelevanceClass(r["relevance_class"]),
            answer=r.get("answer"), metadata=r.get("metadata", {}),
        )
        for r in _read_jsonl(directory / "queries.jsonl")
    ]
    qrels = {r["query_id"]: Qrel.from_json(r) for r in _read_jsonl(directory / "qrels.jsonl")}

    chunk_config = ChunkConfig(**manifest["chunk_config"])
    return CorpusBuild(
        dataset=manifest["dataset"], split=manifest["split"],
        scope=CorpusScope(manifest["scope"]), chunk_config=chunk_config,
        documents=documents, chunks=chunks, queries=queries, qrels=qrels,
        stats=manifest.get("stats", {}),
    )
