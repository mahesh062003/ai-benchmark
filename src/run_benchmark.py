"""
Entry point 2: Run the RAG retrieval (and optionally generation) benchmark.

Usage (from the project root):
    python src/run_benchmark.py [--generate] [--limit N] [--methods BM25 FAISS Hybrid]

Arguments
---------
--generate      Also run LLM generation (requires Ollama to be running).
--limit N       Cap the number of samples per dataset (default: 500).
--methods       Subset of retrieval methods to run (default: BM25 FAISS Hybrid).

What this does
--------------
For each dataset × retrieval method:
  1. Load the pre-built index (built by build_indexes.py).
  2. Load benchmark samples (QA pairs from the evaluation split).
  3. For each sample:
     a. Retrieve top-K chunks.
     b. Compute ground-truth relevant chunk IDs (dataset-specific).
     c. Compute Recall@K, MRR, nDCG@K  (only when ground truth is available).
     d. Optionally generate an answer with the local Ollama LLM.
     e. Store the result in the benchmark database.
  4. Print a summary table.

Scientific validity
-------------------
  - Recall@K / MRR / nDCG are written as NULL — not 0.0 — when no valid
    ground truth exists for a sample (has_retrieval_gt = False).
  - CaseHOLD and MedQA receive NULL for all retrieval metrics.
  - See evaluation/ground_truth.py for per-dataset relevance definitions.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SAMPLE_LIMIT, TOP_K, EMBEDDING_MODEL, LLM_MODEL,
    INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)
from benchmark.registry import DATASETS
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid_retriever import HybridRetriever
from evaluation.ground_truth import get_relevant_chunk_ids
from evaluation.retrieval_metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k
from database.database import BenchmarkDatabase
from preprocessing.chunker import load_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG benchmark.")
    parser.add_argument(
        "--generate", action="store_true",
        help="Run LLM generation (requires Ollama running locally).",
    )
    parser.add_argument(
        "--limit", type=int, default=SAMPLE_LIMIT,
        help=f"Max samples per dataset (default: {SAMPLE_LIMIT}).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["BM25", "FAISS", "Hybrid"],
        choices=["BM25", "FAISS", "Hybrid"],
        help="Retrieval methods to run.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    methods:       list[str],
    sample_limit:  int,
    do_generation: bool,
) -> None:

    # ── Load embedding model once ──────────────────────────────────────────
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    shared_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # ── Optionally load generator ──────────────────────────────────────────
    generator = None
    if do_generation:
        from generation.generator import LLMGenerator
        generator = LLMGenerator(model=LLM_MODEL)

    # ── Open database and register this run ───────────────────────────────
    db = BenchmarkDatabase()
    run_config = {
        "methods":         methods,
        "sample_limit":    sample_limit,
        "do_generation":   do_generation,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": LLM_MODEL if do_generation else None,
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "top_k":           TOP_K,
    }
    run_id = db.start_run(run_config)
    print(f"\nRun ID: {run_id}")

    print("\n" + "=" * 70)
    print("AI BENCHMARK — RETRIEVAL EVALUATION")
    print("=" * 70)

    # ── Main loop ─────────────────────────────────────────────────────────
    for ds in DATASETS:
        name   = ds["name"]
        domain = ds["domain"]

        print(f"\n{'=' * 70}")
        print(f"  Dataset : {name}  |  Domain : {domain}")
        print(f"  Retrieval valid : {ds['retrieval_valid']}")
        print(f"{'=' * 70}")

        # ── Load benchmark samples ─────────────────────────────────────────
        samples = ds["load_samples"]()
        samples = samples[:sample_limit]

        if not samples:
            logger.warning("[%s] No samples — skipping.", name)
            continue

        # ── Load corpus chunks (for ground truth lookup) ───────────────────
        chunk_path = Path(f"results/chunks/{name.lower()}_chunks.json")
        if chunk_path.exists():
            corpus_chunks = load_chunks(chunk_path)
        else:
            logger.warning(
                "[%s] Chunk file not found (%s). Ground truth unavailable.",
                name, chunk_path,
            )
            corpus_chunks = []

        # ── Loop over retrieval methods ────────────────────────────────────
        for method in methods:
            print(f"\n  → {method}…")

            bm25_path  = INDEX_DIR / f"{name.lower()}_bm25.pkl"
            faiss_path = INDEX_DIR / f"{name.lower()}_faiss.index"
            meta_path  = INDEX_DIR / f"{name.lower()}_metadata.pkl"

            # Instantiate retriever
            if method == "BM25":
                if not bm25_path.exists():
                    logger.error("[%s] BM25 index not found: %s", name, bm25_path)
                    continue
                retriever = BM25Retriever(bm25_path)

            elif method == "FAISS":
                if not faiss_path.exists() or not meta_path.exists():
                    logger.error("[%s] FAISS index not found.", name)
                    continue
                retriever = FAISSRetriever(faiss_path, meta_path, model=shared_model)

            else:  # Hybrid
                if not all(p.exists() for p in [bm25_path, faiss_path, meta_path]):
                    logger.error("[%s] One or more indexes missing for Hybrid.", name)
                    continue
                retriever = HybridRetriever(
                    bm25_path, faiss_path, meta_path, model=shared_model,
                )

            # ── Sample loop ────────────────────────────────────────────────
            skipped_no_gt = 0

            for i, sample in enumerate(samples):
                question = sample["question"]
                if not question or not question.strip():
                    continue

                t0 = time.perf_counter()
                retrieved = retriever.search(question, top_k=TOP_K)
                answer    = ""
                if generator and retrieved:
                    answer = generator.generate(question, retrieved)
                elapsed = time.perf_counter() - t0

                retrieved_ids = [r.chunk_id for r in retrieved]

                # Ground truth
                if ds["retrieval_valid"] and corpus_chunks:
                    relevant_ids = get_relevant_chunk_ids(sample, corpus_chunks)
                else:
                    relevant_ids = None

                has_gt = relevant_ids is not None

                if has_gt:
                    recall = recall_at_k(retrieved_ids, relevant_ids, TOP_K)
                    mrr    = mean_reciprocal_rank(retrieved_ids, relevant_ids)
                    ndcg   = ndcg_at_k(retrieved_ids, relevant_ids, TOP_K)
                else:
                    recall = mrr = ndcg = None
                    skipped_no_gt += 1

                db.insert_result(
                    dataset=name,
                    domain=domain,
                    split=sample["split"],
                    retrieval_method=method,
                    sample_id=sample["sample_id"],
                    question=question,
                    generated_answer=answer if answer else None,
                    has_retrieval_gt=has_gt,
                    recall=recall,
                    mrr=mrr,
                    ndcg=ndcg,
                    response_time_s=elapsed,
                    embedding_model=EMBEDDING_MODEL,
                    generation_model=LLM_MODEL if do_generation else None,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    top_k=TOP_K,
                )

                if (i + 1) % 50 == 0:
                    print(f"     {i+1}/{len(samples)} processed…")

            db.commit_batch()

            if skipped_no_gt:
                print(f"     {skipped_no_gt} samples had no valid retrieval ground truth "
                      f"(metrics stored as NULL).")
            print(f"  ✓ {method} on {name} complete.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (samples with valid ground truth only)")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Method':<8} {'N':>5} {'Recall@K':>10} {'MRR':>8} {'nDCG@K':>8} {'Time(s)':>9}")
    print("-" * 70)
    for row in db.summary():
        n         = row["n"] or 0
        avg_recall = row["avg_recall"]
        avg_mrr    = row["avg_mrr"]
        avg_ndcg   = row["avg_ndcg"]
        avg_time   = row["avg_time_s"]
        r_str = f"{avg_recall:.4f}" if avg_recall is not None else "  N/A  "
        m_str = f"{avg_mrr:.4f}"    if avg_mrr    is not None else "  N/A  "
        n_str = f"{avg_ndcg:.4f}"   if avg_ndcg   is not None else "  N/A  "
        t_str = f"{avg_time:.3f}"   if avg_time   is not None else "  N/A  "
        print(f"{row['dataset']:<12} {row['retrieval_method']:<8} {n:>5} "
              f"{r_str:>10} {m_str:>8} {n_str:>8} {t_str:>9}")

    db.close()
    print(f"\n✓ Benchmark complete.  Run ID: {run_id}")


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        methods=args.methods,
        sample_limit=args.limit,
        do_generation=args.generate,
    )
