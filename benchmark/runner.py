"""Index construction and the retrieval benchmark loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loaders import get_adapter
from core.settings import Config
from .corpus import CorpusBuild, Qrel
from core.db import Database
from .indexer import get_embedder
from core.ids import run_id as make_run_id
from core.logging_setup import get_logger
from evaluation.metrics import AggregateMetrics, QueryMetrics, aggregate, evaluate_query
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from core.types import CorpusScope, RelevanceClass

log = get_logger("benchmark")


def index_directory(config: Config, build: CorpusBuild) -> Path:
    return config.paths.indexes_dir / build.dataset / build.split / build.fingerprint


def bm25_path(config: Config, build: CorpusBuild) -> Path:
    return index_directory(config, build) / "bm25.pkl"


def faiss_path(config: Config, build: CorpusBuild) -> Path:
    model = config.embedding.model_name.replace("/", "__")
    return index_directory(config, build) / f"dense__{model}.faiss"


def build_indexes(
    config: Config,
    build: CorpusBuild,
    methods: Sequence[str],
    rebuild: bool = False,
) -> Dict[str, Retriever]:
    """Construct (or load) the retrievers needed for ``methods``.

    ``hybrid`` implies both base retrievers; they are built once and shared
    rather than rebuilt for the fusion.
    """
    needed = set(methods)
    if "hybrid" in needed:
        needed |= {"bm25", "dense"}

    retrievers: Dict[str, Retriever] = {}

    if "bm25" in needed:
        path = bm25_path(config, build)
        if path.exists() and not rebuild:
            try:
                retrievers["bm25"] = BM25Retriever.load(path, build.chunks)
                log.info("loaded BM25 index from %s", path)
            except ValueError as exc:
                log.warning("%s -- rebuilding", exc)
        if "bm25" not in retrievers:
            started = time.time()
            retriever = BM25Retriever(
                build.chunks, k1=config.retrieval.bm25_k1, b=config.retrieval.bm25_b
            )
            retriever.save(path)
            retrievers["bm25"] = retriever
            log.info("BM25 index built in %.1fs", time.time() - started)

    if "dense" in needed:
        embedder = get_embedder(config.embedding)
        path = faiss_path(config, build)
        if path.exists() and path.with_suffix(path.suffix + ".meta.json").exists() and not rebuild:
            try:
                retrievers["dense"] = DenseRetriever.load(path, build.chunks, embedder)
                log.info("loaded FAISS index from %s", path)
            except ValueError as exc:
                log.warning("%s -- rebuilding", exc)
        if "dense" not in retrievers:
            started = time.time()
            retriever = DenseRetriever(build.chunks, embedder)
            retriever.save(path)
            retrievers["dense"] = retriever
            log.info("FAISS index built in %.1fs", time.time() - started)

    if "hybrid" in methods:
        retrievers["hybrid"] = HybridRetriever(
            retrievers["bm25"], retrievers["dense"],
            rrf_k=config.retrieval.rrf_k,
            candidate_depth=config.retrieval.rrf_candidate_depth,
        )

    return {m: retrievers[m] for m in methods}


@dataclass
class MethodOutcome:
    method: str
    aggregate: AggregateMetrics
    per_query: List[QueryMetrics]
    elapsed_seconds: float


def run_retrieval_benchmark(
    config: Config,
    build: CorpusBuild,
    methods: Sequence[str],
    database: Optional[Database] = None,
    run_id: Optional[str] = None,
    rebuild_indexes: bool = False,
    store_results: bool = True,
) -> Dict[str, MethodOutcome]:
    """Run every method over a corpus build and evaluate it.

    Queries whose qrel is unavailable are still *executed* (their retrieved
    chunks are persisted for inspection) but scored as NULL. That is what lets
    MedQA produce inspectable retrieval output while contributing no fabricated
    metric.
    """
    spec = get_adapter(build.dataset, config).spec
    ks = sorted(set(config.retrieval.metric_ks) | {config.retrieval.top_k})
    top_k = max(config.retrieval.top_k, max(ks))

    retrievers = build_indexes(config, build, methods, rebuild=rebuild_indexes)

    if database is not None and run_id is None:
        run_id = make_run_id("retr")
        database.create_run(
            run_id, stage="retrieval", config=config,
            dataset=build.dataset, split=build.split, methods=methods,
            relevance_class=spec.relevance_class.value,
            corpus_scope=build.scope.value,
            corpus_fingerprint=build.fingerprint,
            n_documents=len(build.documents), n_chunks=len(build.chunks),
            n_queries=len(build.queries),
        )

    # Precompute the allowed chunk set per document for document-scoped runs.
    per_document: Dict[str, set] = {}
    if build.scope is CorpusScope.DOCUMENT:
        for chunk in build.chunks:
            per_document.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

    outcomes: Dict[str, MethodOutcome] = {}
    for method in methods:
        retriever = retrievers[method]
        started = time.time()
        per_query: List[QueryMetrics] = []

        for query in build.queries:
            qrel: Qrel = build.qrels[query.query_id]
            allowed = (
                per_document.get(query.scope_doc_id, set())
                if build.scope is CorpusScope.DOCUMENT and query.scope_doc_id
                else None
            )
            if allowed is not None and not allowed:
                per_query.append(
                    evaluate_query(
                        query.query_id, [], set(), ks, available=False,
                        reason="query's document produced no chunks",
                    )
                )
                continue

            retrieved = retriever.search(query.text, top_k, allowed=allowed)
            relevant = set(qrel.relevant_chunk_ids)

            per_query.append(
                evaluate_query(
                    query.query_id, retrieved, relevant, ks,
                    available=qrel.available, reason=qrel.reason,
                )
            )
            if database is not None and store_results:
                database.save_retrieval_results(
                    run_id, build.dataset, method, query.query_id, retrieved,
                    relevant=relevant if qrel.available else None,
                )

        elapsed = time.time() - started
        agg = aggregate(per_query, ks)
        outcomes[method] = MethodOutcome(method, agg, per_query, elapsed)

        if database is not None:
            database.save_query_metrics(run_id, build.dataset, method, per_query)
            database.save_aggregate(
                run_id, build.dataset, spec.domain, method,
                spec.relevance_class.value, agg,
            )

        if spec.relevance_class is RelevanceClass.UNSUPPORTED:
            log.info(
                "%s/%s %s: %d queries retrieved in %.1fs; metrics NULL "
                "(dataset has no retrieval ground truth)",
                build.dataset, build.split, method, len(build.queries), elapsed,
            )
        else:
            log.info(
                "%s/%s %s: %d/%d scoreable | recall@%d=%s mrr=%s ndcg@%d=%s (%.1fs)",
                build.dataset, build.split, method, agg.n_scoreable, agg.n_queries,
                config.retrieval.top_k, _fmt(agg.recall.get(config.retrieval.top_k)),
                _fmt(agg.mrr), config.retrieval.top_k,
                _fmt(agg.ndcg.get(config.retrieval.top_k)), elapsed,
            )

    return outcomes


def _fmt(value: Optional[float]) -> str:
    return "NULL" if value is None else f"{value:.4f}"
