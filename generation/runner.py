"""Multi-model generation orchestration.

The generation stage answers the same questions with several local Ollama
models so the models can be compared on identical inputs. Three properties
matter and shape the design:

Identical inputs across models
    Retrieval is executed **once** per (dataset, method, query) and frozen into
    a *task set* on disk. Every model then answers from byte-identical context.
    If retrieval were re-run per model, nothing would stop an index or a tie
    break from shifting between models and contaminating the comparison.

Grouped by model
    Ollama holds one model resident at a time; interleaving models costs a
    10-15s reload on every question. The runner therefore completes all
    questions for one model before loading the next.

Resumable
    A full sweep takes hours. Each answer is committed as it is produced, and
    restarting with the same ``run_id`` skips the rows already present, so an
    interrupted run resumes instead of starting over.

Query sampling is a seeded random sample rather than the first N rows: several
datasets (notably CUAD) order queries by source document, so a head slice would
draw all its questions from one or two contracts.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from benchmark.runner import build_indexes
from core.settings import Config
from benchmark.corpus import CorpusBuild
from core.db import Database
from core.ids import content_hash
from core.logging_setup import get_logger
from core.types import Chunk, Query

log = get_logger("generation-run")


@dataclass
class GenerationTask:
    """One question, with the retrieved context frozen for every model."""

    dataset: str
    domain: str
    method: str
    query_id: str
    query_text: str
    reference_answer: Optional[str]
    metadata: Dict = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)
    chunk_texts: List[str] = field(default_factory=list)

    def as_query(self) -> Query:
        return Query(
            query_id=self.query_id,
            dataset=self.dataset,
            split="",
            text=self.query_text,
            answer=self.reference_answer,
            metadata=self.metadata,
        )

    def as_chunks(self) -> List[Chunk]:
        return [
            Chunk(
                chunk_id=cid, doc_id="", dataset=self.dataset, split="",
                text=text, ordinal=i, char_start=0, char_end=len(text),
            )
            for i, (cid, text) in enumerate(zip(self.chunk_ids, self.chunk_texts))
        ]

    def to_json(self) -> dict:
        return {
            "dataset": self.dataset, "domain": self.domain, "method": self.method,
            "query_id": self.query_id, "query_text": self.query_text,
            "reference_answer": self.reference_answer, "metadata": self.metadata,
            "chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts,
        }

    @staticmethod
    def from_json(raw: dict) -> "GenerationTask":
        return GenerationTask(**raw)


def task_set_path(config: Config, datasets: Sequence[str], methods: Sequence[str], limit: int) -> Path:
    key = content_hash(
        "|".join(sorted(datasets)) + "#" + "|".join(sorted(methods))
        + f"#n={limit}#k={config.retrieval.top_k}#seed={config.seed}"
        + f"#chunks={config.generation.max_context_chunks}"
    )[:12]
    return config.paths.results_dir / f"generation_tasks_{key}.json"


def sample_queries(build: CorpusBuild, limit: int, seed: int) -> List[Query]:
    """A deterministic, seeded sample of ``limit`` queries from a build."""
    queries = list(build.queries)
    if limit >= len(queries):
        return queries
    rng = random.Random(f"{seed}:{build.dataset}:{build.split}")
    picked = rng.sample(range(len(queries)), limit)
    return [queries[i] for i in sorted(picked)]


def build_task_set(
    config: Config,
    datasets: Sequence[str],
    methods: Sequence[str],
    limit: int,
    corpus_loader,
    rebuild: bool = False,
) -> List[GenerationTask]:
    """Retrieve context for every (dataset, method, sampled query), once.

    ``corpus_loader(dataset) -> CorpusBuild`` is injected so this module does
    not decide whether a corpus is loaded from disk or rebuilt.
    """
    path = task_set_path(config, datasets, methods, limit)
    if path.exists() and not rebuild:
        raw = json.loads(path.read_text(encoding="utf-8"))
        tasks = [GenerationTask.from_json(r) for r in raw["tasks"]]
        log.info("loaded %d frozen generation tasks from %s", len(tasks), path)
        return tasks

    from loaders import get_adapter

    tasks: List[GenerationTask] = []
    for dataset in datasets:
        adapter = get_adapter(dataset, config)
        spec = adapter.spec
        if not spec.supports_generation:
            log.info("skipping %s: adapter declares supports_generation=False", dataset)
            continue
        build = corpus_loader(dataset)
        retrievers = build_indexes(config, build, list(methods))
        chunk_by_id = {c.chunk_id: c for c in build.chunks}

        per_document: Dict[str, set] = {}
        if build.scope.value == "document":
            for chunk in build.chunks:
                per_document.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

        selected = sample_queries(build, limit, config.seed)
        n_context = config.generation.max_context_chunks

        for method in methods:
            retriever = retrievers[method]
            for query in selected:
                allowed = (
                    per_document.get(query.scope_doc_id)
                    if build.scope.value == "document" and query.scope_doc_id
                    else None
                )
                if allowed is not None and not allowed:
                    retrieved = []
                else:
                    retrieved = retriever.search(
                        query.text, config.retrieval.top_k, allowed=allowed
                    )
                top = retrieved[:n_context]
                tasks.append(
                    GenerationTask(
                        dataset=dataset, domain=spec.domain, method=method,
                        query_id=query.query_id, query_text=query.text,
                        reference_answer=query.answer, metadata=dict(query.metadata),
                        chunk_ids=[r.chunk_id for r in top],
                        chunk_texts=[chunk_by_id[r.chunk_id].text for r in top],
                    )
                )
        log.info("%s: %d tasks (%d queries x %d methods)",
                 dataset, len(selected) * len(methods), len(selected), len(methods))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "datasets": list(datasets), "methods": list(methods), "limit": limit,
                "seed": config.seed, "top_k": config.retrieval.top_k,
                "max_context_chunks": config.generation.max_context_chunks,
                "tasks": [t.to_json() for t in tasks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("froze %d generation tasks -> %s", len(tasks), path)
    return tasks


def run_generation(
    config: Config,
    tasks: Sequence[GenerationTask],
    models: Sequence[str],
    database: Database,
    run_id: str,
    resume: bool = True,
    progress=None,
) -> Dict[str, int]:
    """Answer every task with every model, committing each row as it lands.

    Returns the number of answers produced per model this invocation (rows
    skipped by ``resume`` are not counted).
    """
    from .generator import OllamaClient, generate_for_query

    done = database.completed_generations(run_id) if resume else set()
    if done:
        log.info("resuming run %s: %d answers already stored", run_id, len(done))

    produced: Dict[str, int] = {m: 0 for m in models}
    for model in models:
        client = OllamaClient(replace(config.generation, model=model, enabled=True))
        if not client.available():
            log.error("skipping model %r: not available on the Ollama server", model)
            continue

        pending = [t for t in tasks if (t.dataset, t.method, model, t.query_id) not in done]
        log.info("model %s: %d tasks pending (%d already done)",
                 model, len(pending), len(tasks) - len(pending))
        started = time.time()

        for index, task in enumerate(pending, start=1):
            evaluation = generate_for_query(
                client, task.as_query(), task.as_chunks(), config
            )
            database.save_generation(
                run_id, task.dataset, task.method, model, evaluation.to_record()
            )
            produced[model] += 1
            if progress is not None:
                progress(model, index, len(pending), task)
            if index % 25 == 0:
                rate = (time.time() - started) / index
                log.info(
                    "%s: %d/%d (%.1fs/answer, ~%.0f min left)",
                    model, index, len(pending), rate,
                    rate * (len(pending) - index) / 60.0,
                )
    return produced


def stratified_sample(rows: Sequence, size: int, seed: int, key=None) -> List:
    """Draw ``size`` rows spread evenly over every (dataset, method, model) cell.

    Faithfulness is judged on a subsample, and a naive head or uniform random
    draw would leave some cells thin or empty -- exactly the cells a
    model-versus-strategy comparison needs. Allocating a near-equal quota per
    cell keeps every combination represented; cells with fewer rows than their
    quota surrender the remainder to the others. Deterministic given ``seed``.
    """
    if size is None or size >= len(rows):
        return list(rows)
    if size <= 0:
        return []

    if key is None:
        def key(row):
            return (row["dataset"], row["method"], row["model"])

    cells: Dict[tuple, List] = {}
    for row in rows:
        cells.setdefault(key(row), []).append(row)

    rng = random.Random(seed)
    for members in cells.values():
        rng.shuffle(members)

    picked: List = []
    remaining = size
    # Repeatedly hand out one row per non-exhausted cell, so a small cell never
    # blocks a large one from contributing and the spread stays even.
    order = sorted(cells)
    cursor = {name: 0 for name in order}
    while remaining > 0:
        progressed = False
        for name in order:
            if remaining == 0:
                break
            members = cells[name]
            index = cursor[name]
            if index < len(members):
                picked.append(members[index])
                cursor[name] = index + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return picked


def load_chunk_texts(config: Config, dataset: str, needed: set) -> Dict[str, str]:
    """Map chunk_id -> text for ``needed`` ids, streamed from the saved corpus.

    The scoring pass needs the context text that a stored answer was given, but
    ``generations`` records only chunk ids. Reading them back from the corpus on
    disk keeps the answer table small and means a score is always computed
    against the same chunk text the generator saw. Streaming rather than
    loading the file avoids holding CUAD's full chunk set in memory to recover
    a few hundred strings.
    """
    from loaders import get_adapter
    from benchmark.corpus import build_directory

    if not needed:
        return {}
    adapter = get_adapter(dataset, config)
    directory = build_directory(
        config, dataset, adapter.default_split(), adapter.chunk_config().fingerprint()
    )
    path = directory / "chunks.jsonl"
    if not path.exists():
        log.warning("no saved corpus for %s at %s; context unavailable", dataset, path)
        return {}

    found: Dict[str, str] = {}
    remaining = set(needed)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id")
            if chunk_id in remaining:
                found[chunk_id] = record.get("text", "")
                remaining.discard(chunk_id)
    if remaining:
        log.warning("%s: %d chunk ids not found in the saved corpus", dataset, len(remaining))
    return found


def available_models(config: Config, wanted: Iterable[str]) -> Dict[str, bool]:
    """Which of ``wanted`` the Ollama server can actually serve.

    Never raises: a machine with no Ollama simply reports every model as
    unavailable, which is what keeps the generation stage optional.
    """
    from .generator import OllamaClient

    status: Dict[str, bool] = {}
    for model in wanted:
        probe = replace(config.generation, model=model, enabled=True)
        try:
            status[model] = OllamaClient(probe).available()
        except Exception:
            status[model] = False
    return status
