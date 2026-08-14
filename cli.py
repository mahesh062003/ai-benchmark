"""Command-line interface.

Workflow:

    ragbench datasets                 # what is available and how it is judged
    ragbench validate                 # can every dataset be read?
    ragbench build-corpus --all       # documents -> chunks -> qrels
    ragbench build-indexes --all      # BM25 + FAISS, persisted
    ragbench benchmark --all          # retrieval + metrics -> SQLite
    ragbench results                  # aggregate table
    ragbench inspect --dataset cuad   # look at actual retrieved chunks
    ragbench export --output x.csv    # results to CSV

Optional generation stage (needs a local Ollama server; retrieval never does):

    ragbench models                   # which configured models are servable
    ragbench generate --dataset medqa # one model, one method
    ragbench generate-all             # every model x method x dataset
    ragbench score-answers            # faithfulness + hallucination
    ragbench dashboard                # Streamlit view of everything above
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from loaders import ADAPTERS, all_dataset_names, get_adapter
from benchmark.runner import build_indexes, run_retrieval_benchmark
from core.settings import Config, load_config
from benchmark.corpus import build_corpus, load_corpus, save_corpus
from core.db import open_database
from core.ids import run_id as make_run_id
from core.logging_setup import get_logger, setup_logging
from core.types import RelevanceClass

app = typer.Typer(add_completion=False, help="RAG retrieval benchmark across specialist domains.")
console = Console()
log = get_logger("cli")

ConfigOption = typer.Option(None, "--config", "-c", help="Path to a YAML config file.")


def _load(config_path: Optional[str], verbose: bool = False) -> Config:
    config = load_config(config_path)
    setup_logging("DEBUG" if verbose else config.log_level)
    return config


def _targets(dataset: Optional[str], use_all: bool) -> List[str]:
    if use_all:
        return all_dataset_names()
    if dataset:
        names = [d.strip() for d in dataset.split(",") if d.strip()]
        for name in names:
            if name not in ADAPTERS:
                raise typer.BadParameter(
                    f"unknown dataset {name!r}; available: {', '.join(all_dataset_names())}"
                )
        return names
    raise typer.BadParameter("specify --dataset NAME or --all")


def _fmt(value) -> str:
    if value is None:
        return "[dim]NULL[/dim]"
    return f"{value:.4f}"


@app.command("datasets")
def list_datasets(config: Optional[str] = ConfigOption) -> None:
    """Show each dataset's ground-truth strategy and whether metrics are valid."""
    cfg = _load(config)
    table = Table(title="Datasets and retrieval ground truth", show_lines=True)
    for column in ("dataset", "domain", "class", "scope", "splits", "retrieval metrics"):
        table.add_column(column)
    for name in all_dataset_names():
        spec = ADAPTERS[name].spec
        supported = (
            "[green]valid[/green]" if spec.supports_retrieval_metrics
            else "[red]NULL (no ground truth)[/red]"
        )
        colour = {
            RelevanceClass.GOLD: "green",
            RelevanceClass.DERIVED: "yellow",
            RelevanceClass.HEURISTIC: "magenta",
            RelevanceClass.UNSUPPORTED: "red",
        }[spec.relevance_class]
        table.add_row(
            name, spec.domain, f"[{colour}]{spec.relevance_class.value}[/{colour}]",
            spec.corpus_scope.value, ", ".join(spec.splits), supported,
        )
    console.print(table)
    console.print("\n[bold]Relevance source and limitations[/bold]")
    for name in all_dataset_names():
        spec = ADAPTERS[name].spec
        console.print(f"\n[bold]{name}[/bold] ({spec.relevance_class.value})")
        console.print(f"  query:      {spec.query_unit}")
        console.print(f"  unit:       {spec.corpus_unit}")
        console.print(f"  relevance:  {spec.relevance_source}")
        console.print(f"  [dim]limits:     {spec.limitations}[/dim]")


@app.command("validate")
def validate(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    all_datasets: bool = typer.Option(False, "--all"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Read every dataset and report record counts without building anything."""
    cfg = _load(config, verbose)
    table = Table(title="Dataset validation")
    for column in ("dataset", "split", "documents", "queries", "with ground truth", "skipped"):
        table.add_column(column, justify="right")
    failures = 0
    for name in _targets(dataset, all_datasets):
        adapter = get_adapter(name, cfg)
        target_split = split or adapter.default_split()
        try:
            loaded = adapter.load(target_split)
        except Exception as exc:
            failures += 1
            console.print(f"[red]FAILED[/red] {name}/{target_split}: {exc}")
            continue
        table.add_row(
            name, target_split, str(len(loaded.documents)), str(len(loaded.queries)),
            str(loaded.stats.get("queries_with_evidence", 0)), str(loaded.skips.total),
        )
    console.print(table)
    if failures:
        raise typer.Exit(code=1)


@app.command("build-corpus")
def build_corpus_command(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    all_datasets: bool = typer.Option(False, "--all"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build and persist corpora: documents, chunks and qrels."""
    cfg = _load(config, verbose)
    for name in _targets(dataset, all_datasets):
        build = build_corpus(cfg, name, split)
        directory = save_corpus(cfg, build)
        console.print(
            f"[green]built[/green] {name}/{build.split} -> {len(build.chunks)} chunks, "
            f"{build.stats.get('queries_scoreable', 0)}/{len(build.queries)} scoreable  "
            f"[dim]{directory}[/dim]"
        )


@app.command("build-indexes")
def build_indexes_command(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    all_datasets: bool = typer.Option(False, "--all"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    methods: str = typer.Option("bm25,dense", "--methods", "-m"),
    rebuild: bool = typer.Option(False, "--rebuild"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build and persist BM25 and/or FAISS indexes for existing corpora."""
    cfg = _load(config, verbose)
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    for name in _targets(dataset, all_datasets):
        adapter = get_adapter(name, cfg)
        target_split = split or adapter.default_split()
        build = _load_or_build(cfg, name, target_split)
        build_indexes(cfg, build, method_list, rebuild=rebuild)
        console.print(f"[green]indexed[/green] {name}/{target_split}: {', '.join(method_list)}")


def _load_or_build(cfg: Config, name: str, split: str):
    adapter = get_adapter(name, cfg)
    fingerprint = adapter.chunk_config().fingerprint()
    try:
        return load_corpus(cfg, name, split, fingerprint)
    except FileNotFoundError:
        log.info("no saved corpus for %s/%s; building it now", name, split)
        build = build_corpus(cfg, name, split)
        save_corpus(cfg, build)
        return build


@app.command("benchmark")
def benchmark_command(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    all_datasets: bool = typer.Option(False, "--all"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    methods: Optional[str] = typer.Option(None, "--methods", "-m"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap queries (smoke runs)."),
    rebuild: bool = typer.Option(False, "--rebuild"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the retrieval benchmark and write results to SQLite."""
    cfg = _load(config, verbose)
    method_list = (
        [m.strip() for m in methods.split(",") if m.strip()]
        if methods else list(cfg.retrieval.methods)
    )
    names = _targets(dataset, all_datasets)

    if limit is not None:
        from core.settings import DatasetConfig

        for name in names:
            existing = cfg.datasets.get(name, DatasetConfig())
            existing.max_queries = limit
            cfg.datasets[name] = existing

    with open_database(cfg) as database:
        run = make_run_id("bench")
        database.create_run(
            run, stage="retrieval-batch", config=cfg,
            dataset=",".join(names), split=split or "", methods=method_list,
            notes=f"limit={limit}" if limit else "",
        )
        for name in names:
            adapter = get_adapter(name, cfg)
            target_split = split or adapter.default_split()
            build = build_corpus(cfg, name, target_split)
            save_corpus(cfg, build)
            run_retrieval_benchmark(
                cfg, build, method_list, database=database, run_id=run,
                rebuild_indexes=rebuild,
            )
        console.print(f"\n[green]run complete[/green]: {run}")
        _print_results(database, run)


@app.command("generate")
def generate_command(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    method: str = typer.Option("hybrid", "--method", "-m"),
    limit: int = typer.Option(20, "--limit"),
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the optional RAG generation stage against a local Ollama model."""
    cfg = _load(config, verbose)
    cfg.generation.enabled = True
    if model:
        cfg.generation.model = model

    from generation.generator import OllamaClient, generate_for_query

    client = OllamaClient(cfg.generation)
    if not client.available():
        console.print(
            f"[red]Ollama is not available[/red] at {cfg.generation.host} with model "
            f"{cfg.generation.model}.\nStart it with 'ollama serve' and "
            f"'ollama pull {cfg.generation.model}'. Retrieval benchmarks do not need it."
        )
        raise typer.Exit(code=1)

    adapter = get_adapter(dataset, cfg)
    target_split = split or adapter.default_split()
    build = _load_or_build(cfg, dataset, target_split)
    retrievers = build_indexes(cfg, build, [method])
    retriever = retrievers[method]
    chunk_by_id = {c.chunk_id: c for c in build.chunks}

    per_document = {}
    if build.scope.value == "document":
        for chunk in build.chunks:
            per_document.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

    with open_database(cfg) as database:
        run = make_run_id("gen")
        database.create_run(
            run, stage="generation", config=cfg, dataset=dataset, split=target_split,
            methods=[method], relevance_class=adapter.spec.relevance_class.value,
            corpus_scope=build.scope.value, corpus_fingerprint=build.fingerprint,
            n_documents=len(build.documents), n_chunks=len(build.chunks),
            n_queries=min(limit, len(build.queries)),
        )
        scored: List[float] = []
        for query in build.queries[:limit]:
            allowed = (
                per_document.get(query.scope_doc_id)
                if build.scope.value == "document" and query.scope_doc_id else None
            )
            retrieved = retriever.search(query.text, cfg.retrieval.top_k, allowed=allowed)
            chunks = [chunk_by_id[r.chunk_id] for r in retrieved]
            evaluation = generate_for_query(client, query, chunks, cfg)
            database.save_generation(
                run, dataset, method, cfg.generation.model, evaluation.to_record()
            )
            if evaluation.choice_correct is not None:
                scored.append(evaluation.choice_correct)

        console.print(f"[green]generation run complete[/green]: {run}")
        if scored:
            console.print(
                f"multiple-choice accuracy: {sum(scored)/len(scored):.4f} "
                f"over {len(scored)} answered queries"
            )
        else:
            console.print(
                "[yellow]no multiple-choice accuracy[/yellow] (dataset has no option set); "
                "run 'ragbench score-answers' for faithfulness and hallucination"
            )


@app.command("models")
def models_command(
    config: Optional[str] = ConfigOption,
) -> None:
    """Report which configured Ollama models the server can currently serve."""
    cfg = _load(config)
    from generation.runner import available_models

    status = available_models(cfg, cfg.generation.models)
    table = Table(title=f"Ollama models at {cfg.generation.host}")
    table.add_column("model")
    table.add_column("available")
    for model, ok in status.items():
        table.add_row(model, "[green]yes[/green]" if ok else "[red]no[/red]")
    console.print(table)
    if not any(status.values()):
        console.print(
            "[yellow]No models reachable.[/yellow] Generation is optional -- "
            "retrieval results remain available via 'ragbench results' and the dashboard."
        )


@app.command("generate-all")
def generate_all_command(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    all_datasets: bool = typer.Option(True, "--all/--no-all"),
    methods: Optional[str] = typer.Option(None, "--methods", "-m"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated Ollama models."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Queries per dataset."),
    run: Optional[str] = typer.Option(None, "--run", help="Resume an existing generation run."),
    rebuild_tasks: bool = typer.Option(False, "--rebuild-tasks"),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Answer the same questions with several Ollama models, for comparison.

    Retrieval is frozen once so every model sees identical context, models are
    run one at a time to avoid Ollama reload thrash, and each answer is
    committed as it is produced so an interrupted sweep can be resumed with
    --run.
    """
    cfg = _load(config, verbose)
    cfg.generation.enabled = True
    names = _targets(dataset, all_datasets and not dataset)
    method_list = (
        [m.strip() for m in methods.split(",") if m.strip()]
        if methods else list(cfg.retrieval.methods)
    )
    model_list = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models else list(cfg.generation.models)
    )
    n_queries = limit or cfg.generation.queries_per_dataset

    from generation.runner import available_models, build_task_set, run_generation

    status = available_models(cfg, model_list)
    usable = [m for m, ok in status.items() if ok]
    for model, ok in status.items():
        console.print(f"  {model:12s} {'[green]available[/green]' if ok else '[red]unavailable[/red]'}")
    if not usable:
        console.print(
            f"\n[red]No requested model is available[/red] at {cfg.generation.host}.\n"
            "Start the server with 'ollama serve' and pull a model. "
            "Retrieval results are unaffected and remain available."
        )
        raise typer.Exit(code=1)

    tasks = build_task_set(
        cfg, names, method_list, n_queries,
        corpus_loader=lambda name: _load_or_build(cfg, name, get_adapter(name, cfg).default_split()),
        rebuild=rebuild_tasks,
    )
    console.print(
        f"\n[bold]{len(tasks)}[/bold] frozen tasks x [bold]{len(usable)}[/bold] models "
        f"= [bold]{len(tasks) * len(usable)}[/bold] generations"
    )

    with open_database(cfg) as database:
        run_id = run or make_run_id("genall")
        if run is None:
            database.create_run(
                run_id, stage="generation-multi-model", config=cfg,
                dataset=",".join(names), split="", methods=method_list,
                n_queries=n_queries,
                notes=f"models={','.join(usable)}; {n_queries} queries/dataset/method",
            )
        else:
            console.print(f"[cyan]resuming[/cyan] run {run_id}")

        produced = run_generation(cfg, tasks, usable, database, run_id, resume=True)

    console.print(f"\n[green]generation complete[/green]: {run_id}")
    for model, count in produced.items():
        console.print(f"  {model:12s} {count} answers this session")
    console.print("\nNext: [bold]ragbench score-answers[/bold] for faithfulness and hallucination.")


@app.command("score-answers")
def score_answers_command(
    run: Optional[str] = typer.Option(None, "--run"),
    faithfulness: bool = typer.Option(True, "--faithfulness/--no-faithfulness"),
    hallucination: bool = typer.Option(True, "--hallucination/--no-hallucination"),
    judge: Optional[str] = typer.Option(None, "--judge", help="Ollama judge model."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Score at most N rows per measure."),
    sample: Optional[int] = typer.Option(
        None, "--sample",
        help="Judge faithfulness on a stratified sample of N answers "
             "(default: scoring.faithfulness_sample).",
    ),
    config: Optional[str] = ConfigOption,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score stored answers for RAGAS faithfulness and NLI hallucination.

    A second pass over answers already in the database, so scorers can be
    re-run or re-thresholded without regenerating anything. Rows that cannot be
    scored keep NULL rather than being given a zero.

    Hallucination covers every answer; faithfulness costs two judge calls each,
    so it runs over a stratified sample spanning every (dataset, method, model)
    cell.
    """
    cfg = _load(config, verbose)
    if judge:
        cfg.scoring.judge_model = judge
    if sample is not None:
        cfg.scoring.faithfulness_sample = sample

    from evaluation.faithfulness import FaithfulnessScorer
    from evaluation.hallucination import HallucinationScorer
    from generation.runner import load_chunk_texts

    with open_database(cfg) as database:
        if hallucination:
            _score_hallucination(cfg, database, run, limit, load_chunk_texts, HallucinationScorer)
        if faithfulness:
            _score_faithfulness(cfg, database, run, limit, load_chunk_texts, FaithfulnessScorer)


def _context_for(cfg, database, rows, load_chunk_texts) -> dict:
    """chunk_id -> text for every chunk referenced by ``rows``."""
    wanted: dict = {}
    for row in rows:
        for chunk_id in json.loads(row["context_chunk_ids"] or "[]"):
            wanted.setdefault(row["dataset"], set()).add(chunk_id)
    texts: dict = {}
    for dataset, ids in wanted.items():
        texts.update(load_chunk_texts(cfg, dataset, ids))
    return texts


def _row_context(row, texts) -> List[str]:
    ids = json.loads(row["context_chunk_ids"] or "[]")
    return [texts[i] for i in ids if i in texts]


def _score_hallucination(cfg, database, run, limit, load_chunk_texts, scorer_class) -> None:
    rows = database.generations_needing_scores("hallucination", run, limit)
    if not rows:
        console.print("[dim]hallucination: nothing to score[/dim]")
        return
    console.print(f"[bold]NLI hallucination[/bold]: {len(rows)} answers ({cfg.scoring.nli_model})")
    scorer = scorer_class(
        model_name=cfg.scoring.nli_model,
        entail_threshold=cfg.scoring.entail_threshold,
        contradict_threshold=cfg.scoring.contradict_threshold,
        batch_size=cfg.scoring.nli_batch_size,
        max_length=cfg.scoring.nli_max_length,
    )
    texts = _context_for(cfg, database, rows, load_chunk_texts)
    scored = 0
    null = 0
    with typer.progressbar(range(0, len(rows), 32), label="  scoring") as batches:
        for start in batches:
            batch = rows[start : start + 32]
            results = scorer.score_batch(
                [(r["answer"], _row_context(r, texts)) for r in batch]
            )
            for row, result in zip(batch, results):
                database.update_generation_score(
                    row["run_id"], row["dataset"], row["method"], row["model"],
                    row["query_id"], "hallucination", result.score,
                    rater=result.model, detail=result.to_json(),
                )
                if result.score is None:
                    null += 1
                else:
                    scored += 1
    console.print(f"  scored {scored}, NULL {null}")


def _score_faithfulness(cfg, database, run, limit, load_chunk_texts, scorer_class) -> None:
    rows = database.generations_needing_scores("faithfulness", run, limit)
    if not rows:
        console.print("[dim]faithfulness: nothing to score[/dim]")
        return

    from generation.runner import stratified_sample

    total = len(rows)
    quota = cfg.scoring.faithfulness_sample
    if quota:
        rows = stratified_sample(rows, quota, cfg.seed)
        cells = len({(r["dataset"], r["method"], r["model"]) for r in rows})
        console.print(
            f"[cyan]sampling[/cyan] {len(rows)} of {total} unscored answers "
            f"across {cells} (dataset, method, model) cells; the rest keep NULL"
        )

    scorer = scorer_class(cfg.generation, judge_model=cfg.scoring.judge_model)
    if not scorer.available():
        console.print(
            f"[yellow]skipping faithfulness[/yellow]: judge model "
            f"{cfg.scoring.judge_model!r} is not available at {cfg.generation.host}."
        )
        return
    console.print(
        f"[bold]RAGAS faithfulness[/bold]: {len(rows)} answers "
        f"(judge {cfg.scoring.judge_model}, 2 LLM calls each)"
    )
    texts = _context_for(cfg, database, rows, load_chunk_texts)
    scored = 0
    null = 0
    with typer.progressbar(rows, label="  judging") as judging:
        for row in judging:
            result = scorer.score(
                _question_from_prompt(row["prompt"]), row["answer"], _row_context(row, texts)
            )
            database.update_generation_score(
                row["run_id"], row["dataset"], row["method"], row["model"],
                row["query_id"], "faithfulness", result.score,
                rater=result.judge, detail=result.to_json(),
            )
            if result.score is None:
                null += 1
            else:
                scored += 1
    console.print(f"  scored {scored}, NULL {null}")


def _question_from_prompt(prompt: str) -> str:
    """Recover the question line from a stored RAG prompt."""
    for line in (prompt or "").splitlines():
        if line.startswith("Question:"):
            return line[len("Question:") :].strip()
    return ""


@app.command("dashboard")
def dashboard_command(
    port: int = typer.Option(8501, "--port"),
    config: Optional[str] = ConfigOption,
) -> None:
    """Launch the Streamlit results dashboard."""
    import subprocess
    import sys

    from core.settings import PROJECT_ROOT

    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    if not app_path.exists():
        console.print(f"[red]dashboard not found[/red] at {app_path}")
        raise typer.Exit(code=1)
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    if config:
        command += ["--", "--config", config]
    console.print(f"[green]starting dashboard[/green] on http://localhost:{port}")
    raise typer.Exit(code=subprocess.call(command))


@app.command("results")
def results_command(
    run: Optional[str] = typer.Option(None, "--run"),
    config: Optional[str] = ConfigOption,
) -> None:
    """Show aggregate metrics for retrieval and, when present, generation."""
    cfg = _load(config)
    with open_database(cfg) as database:
        _print_results(database, run)
        _print_generation_results(database, None)


def _print_results(database, run: Optional[str]) -> None:
    rows = database.aggregates(run)
    if not rows:
        console.print("[yellow]no results yet[/yellow]. Run: ragbench benchmark --all")
        return
    table = Table(title=f"Aggregate retrieval metrics{f' (run {run})' if run else ''}")
    for column in ("dataset", "domain", "class", "method", "scoreable", "coverage",
                   "recall@10", "MRR", "nDCG@10"):
        table.add_column(column)
    for row in rows:
        metrics = json.loads(row["metrics_json"])
        recall = metrics["recall"].get("10")
        ndcg = metrics["ndcg"].get("10")
        table.add_row(
            row["dataset"], row["domain"] or "-", row["relevance_class"], row["method"],
            f"{row['n_scoreable']}/{row['n_queries']}",
            _fmt(row["coverage"]), _fmt(recall), _fmt(row["mrr"]), _fmt(ndcg),
        )
    console.print(table)
    console.print(
        "[dim]NULL = no valid ground truth for those queries; distinct from 0.0, "
        "which means ground truth existed and nothing relevant was retrieved.[/dim]"
    )


def _print_generation_results(database, run: Optional[str]) -> None:
    rows = database.generation_aggregates(run)
    if not rows:
        return
    table = Table(title="Generation results by model")
    for column in ("dataset", "method", "model", "answered", "errors",
                   "choice acc", "exact match", "faithfulness", "hallucination", "latency"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["dataset"], row["method"], row["model"],
            str(row["n_answered"]), str(row["n_errors"] or 0),
            _fmt_n(row["choice_correct"], row["n_choice"]),
            _fmt_n(row["exact_match"], row["n_exact"]),
            _fmt_n(row["faithfulness"], row["n_faithfulness"]),
            _fmt_n(row["hallucination"], row["n_hallucination"]),
            "-" if row["latency_ms"] is None else f"{row['latency_ms']/1000:.1f}s",
        )
    console.print(table)
    console.print(
        "[dim]Cell shows mean (n scored). NULL = the measure does not apply to "
        "that dataset or has not been run; hallucination is a rate, lower is better.[/dim]"
    )


def _fmt_n(value, count) -> str:
    if value is None or not count:
        return "[dim]NULL[/dim]"
    return f"{value:.3f} [dim]({count})[/dim]"


@app.command("inspect")
def inspect_command(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    split: Optional[str] = typer.Option(None, "--split", "-s"),
    method: str = typer.Option("bm25", "--method", "-m"),
    limit: int = typer.Option(3, "--limit"),
    top: int = typer.Option(5, "--top"),
    config: Optional[str] = ConfigOption,
) -> None:
    """Retrieve live and print the actual chunks with their relevance labels."""
    cfg = _load(config)
    adapter = get_adapter(dataset, cfg)
    target_split = split or adapter.default_split()
    build = _load_or_build(cfg, dataset, target_split)
    retriever = build_indexes(cfg, build, [method])[method]
    chunk_by_id = {c.chunk_id: c for c in build.chunks}

    per_document = {}
    if build.scope.value == "document":
        for chunk in build.chunks:
            per_document.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

    shown = 0
    for query in build.queries:
        if shown >= limit:
            break
        qrel = build.qrels[query.query_id]
        shown += 1
        console.print(f"\n[bold]{query.query_id}[/bold]  ({method})")
        console.print(f"  query: {query.text[:220]}")
        if qrel.available:
            console.print(f"  [green]relevant chunks[/green]: {len(qrel.relevant_chunk_ids)}")
        else:
            console.print(f"  [red]no ground truth[/red]: {qrel.reason} -> metrics NULL")
        allowed = (
            per_document.get(query.scope_doc_id)
            if build.scope.value == "document" and query.scope_doc_id else None
        )
        for item in retriever.search(query.text, top, allowed=allowed):
            chunk = chunk_by_id[item.chunk_id]
            if qrel.available:
                mark = "[green]RELEVANT[/green]" if item.chunk_id in qrel.relevant_chunk_ids else "[dim]-[/dim]"
            else:
                mark = "[yellow]unjudged[/yellow]"
            console.print(
                f"    {item.rank}. {mark} score={item.score:.4f} {chunk.chunk_id}"
            )
            console.print(f"       [dim]{chunk.text[:150].strip()}[/dim]")


@app.command("significance")
def significance_command(
    run: Optional[str] = typer.Option(None, "--run", help="Run to test. Defaults to the latest."),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    metrics: str = typer.Option("recall,mrr,ndcg", "--metrics", help="Comma-separated."),
    k: int = typer.Option(10, "--k", help="Cut-off for recall and nDCG."),
    resamples: int = typer.Option(10000, "--resamples", help="Bootstrap resamples."),
    alpha: float = typer.Option(0.05, "--alpha"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write results to CSV."),
    config: Optional[str] = ConfigOption,
) -> None:
    """Test whether the gaps between retrieval strategies are statistically real.

    Pairwise Wilcoxon signed-rank tests on per-query scores, with bootstrap
    confidence intervals for the effect size and a Holm-Bonferroni correction
    across the three strategy pairs. Datasets with no retrieval ground truth
    are skipped rather than compared against zeros.
    """
    import csv

    from evaluation.significance import run_comparisons

    cfg = _load(config)
    requested = [m.strip().lower() for m in metrics.split(",") if m.strip()]
    unknown = [m for m in requested if m not in ("recall", "mrr", "ndcg")]
    if unknown:
        console.print(f"[red]unknown metric(s): {', '.join(unknown)}[/red]")
        raise typer.Exit(code=1)
    metric_specs = [(m, k) for m in requested]

    with open_database(cfg) as database:
        connection = database.connection
        run_id = run or (
            connection.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
            or [None]
        )[0]
        if run_id is None:
            console.print("[yellow]no runs in the database[/yellow]")
            raise typer.Exit(code=1)

        # Only datasets with scoreable queries can be tested at all.
        scoreable = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT dataset FROM query_metrics"
                " WHERE run_id = ? AND scoreable = 1 ORDER BY dataset",
                (run_id,),
            )
        ]
        skipped = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT dataset FROM aggregate_metrics WHERE run_id = ?"
                " AND relevance_class = 'UNSUPPORTED' ORDER BY dataset",
                (run_id,),
            )
        ]
        datasets = [dataset] if dataset else scoreable
        methods = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT method FROM query_metrics WHERE run_id = ? ORDER BY method",
                (run_id,),
            )
        ]

        comparisons = run_comparisons(
            connection, run_id, datasets, methods, metric_specs,
            resamples=resamples, alpha=alpha,
        )

    if not comparisons:
        console.print("[yellow]nothing to test[/yellow]")
        raise typer.Exit(code=1)

    # Means are omitted here: they are already in `results`, and dropping them
    # keeps the table readable in a standard terminal. The CSV carries them.
    table = Table(title=f"Pairwise significance (run {run_id}, Holm-corrected, alpha={alpha})")
    for column, justify in (
        ("dataset", "left"), ("metric", "left"), ("A vs B", "left"), ("n", "right"),
        ("diff", "right"), (f"{int((1 - alpha) * 100)}% CI", "right"),
        ("p (Holm)", "right"), ("verdict", "left"),
    ):
        table.add_column(column, justify=justify, no_wrap=True)
    previous = None
    for c in comparisons:
        if c.significant:
            verdict = f"[green]{c.winner}[/green]"
        else:
            verdict = "[dim]n.s.[/dim]"
        group = (c.dataset, c.metric)
        table.add_row(
            c.dataset if group != previous else "",
            c.metric if group != previous else "",
            f"{c.method_a}–{c.method_b}", str(c.n_pairs),
            f"{c.mean_diff:+.4f}",
            f"[{c.ci_low:+.3f},{c.ci_high:+.3f}]",
            f"{c.p_corrected:.1e}", verdict,
        )
        previous = group
    console.print(table)
    console.print(
        "[dim]A confidence interval spanning zero means the observed gap is "
        "indistinguishable from sampling noise, whatever the point estimates say.[/dim]"
    )
    if skipped:
        console.print(
            f"[dim]skipped (no retrieval ground truth): {', '.join(skipped)}[/dim]"
        )

    if output is not None:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = cfg.paths.results_dir / "significance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run_id", "dataset", "metric", "method_a", "method_b", "n_pairs",
            "n_differing", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
            "p_value", "p_holm", "significant", "winner",
        ])
        for c in comparisons:
            writer.writerow([
                run_id, c.dataset, c.metric, c.method_a, c.method_b, c.n_pairs,
                c.n_differing, f"{c.mean_a:.6f}", f"{c.mean_b:.6f}",
                f"{c.mean_diff:.6f}", f"{c.ci_low:.6f}", f"{c.ci_high:.6f}",
                f"{c.p_value:.6g}", f"{c.p_corrected:.6g}",
                int(c.significant), c.winner or "",
            ])
    console.print(f"[dim]written -> {path}[/dim]")


@app.command("export")
def export_command(
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination CSV. Defaults to aggregates.csv in the results directory.",
    ),
    run: Optional[str] = typer.Option(None, "--run"),
    config: Optional[str] = ConfigOption,
) -> None:
    """Export aggregate metrics to CSV (empty cells for NULL)."""
    import csv

    cfg = _load(config)
    with open_database(cfg) as database:
        rows = database.aggregates(run)
    if not rows:
        console.print("[yellow]nothing to export[/yellow]")
        raise typer.Exit(code=1)

    # Default through the configured results directory so that
    # RAGBENCH_ARTIFACTS_DIR relocates the export alongside the database it
    # was read from, rather than writing back into the repository.
    if output is None:
        path = cfg.paths.results_dir / "aggregates.csv"
    else:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    ks = set()
    parsed = []
    for row in rows:
        metrics = json.loads(row["metrics_json"])
        ks |= set(metrics["recall"]) | set(metrics["ndcg"])
        parsed.append((row, metrics))
    ordered_ks = sorted(ks, key=int)

    header = (
        ["run_id", "dataset", "domain", "method", "relevance_class", "n_queries",
         "n_scoreable", "n_unscoreable", "coverage", "mrr"]
        + [f"recall@{k}" for k in ordered_ks] + [f"ndcg@{k}" for k in ordered_ks]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, metrics in parsed:
            writer.writerow(
                [row["run_id"], row["dataset"], row["domain"], row["method"],
                 row["relevance_class"], row["n_queries"], row["n_scoreable"],
                 row["n_unscoreable"],
                 "" if row["coverage"] is None else row["coverage"],
                 "" if row["mrr"] is None else row["mrr"]]
                + ["" if metrics["recall"].get(k) is None else metrics["recall"][k] for k in ordered_ks]
                + ["" if metrics["ndcg"].get(k) is None else metrics["ndcg"][k] for k in ordered_ks]
            )
    console.print(f"[green]exported[/green] {len(parsed)} rows -> {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
