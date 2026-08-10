import time
import pickle

from benchmark.config import SAMPLE_LIMIT, TOP_K, LLM_MODEL
from benchmark.dataset_registry import DATASETS

from sentence_transformers import SentenceTransformer

from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import LLMGenerator

from database.database import BenchmarkDatabase
from evaluation.retrieval_metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

# ── Configuration ─────────────────────────────────────────────
GENERATE_ANSWERS = False   # Set True for Stage 2 (LLM generation)
GENERATION_LIMIT = 50      # Only used when GENERATE_ANSWERS = True
RETRIEVAL_METHODS = ["BM25", "FAISS", "Hybrid"]

# ── Load embedding model ONCE ─────────────────────────────────
print("Loading embedding model...")
shared_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("✅ Embedding model loaded\n")

# ── Initialise DB and Generator ───────────────────────────────
db = BenchmarkDatabase()

if GENERATE_ANSWERS:
    generator = LLMGenerator(model=LLM_MODEL)

# ── Main Loop ─────────────────────────────────────────────────
for dataset in DATASETS:

    print("\n" + "=" * 70)
    print(f"DATASET : {dataset['name']} | DOMAIN : {dataset['domain']}")
    print("=" * 70)

    samples = dataset["loader"](dataset["path"])
    samples = samples[:SAMPLE_LIMIT]

    bm25_path     = f"../results/indexes/{dataset['name'].lower()}_bm25.pkl"
    faiss_path    = f"../results/indexes/{dataset['name'].lower()}_faiss.index"
    metadata_path = f"../results/indexes/{dataset['name'].lower()}_metadata.pkl"

    # Load metadata once per dataset for relevant_ids lookup
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    for method in RETRIEVAL_METHODS:

        print(f"\n  → Running {method}...")

        # Initialise retriever — shared model passed to FAISS and Hybrid
        if method == "BM25":
            retriever = BM25Retriever(bm25_path)
        elif method == "FAISS":
            retriever = FAISSRetriever(
                faiss_path,
                metadata_path,
                model=shared_model
            )
        else:
            retriever = HybridRetriever(
                bm25_path,
                faiss_path,
                metadata_path,
                model=shared_model
            )

        run_samples = samples[:GENERATION_LIMIT] if GENERATE_ANSWERS else samples

        for i, sample in enumerate(run_samples):

            question = sample.get("question", "")
            if not question:
                continue

            # Build relevant_ids from metadata
            sample_id = sample.get("id", "")
            relevant_ids = [
                m["chunk_id"]
                for m in metadata
                if m.get("sample_id") == sample_id
            ]
            if not relevant_ids:
                relevant_ids = []

            start = time.time()

            retrieved = retriever.search(question, top_k=TOP_K)
            retrieved_ids = [r.chunk_id for r in retrieved]

            answer = generator.generate(question, retrieved) if GENERATE_ANSWERS else ""

            elapsed = time.time() - start

            recall = recall_at_k(retrieved_ids, relevant_ids, TOP_K)
            mrr    = mean_reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg   = ndcg_at_k(retrieved_ids, relevant_ids, TOP_K)

            db.insert_result(
                dataset=dataset["name"],
                domain=dataset["domain"],
                retrieval_method=method,
                question=question,
                generated_answer=answer,
                recall=recall,
                mrr=mrr,
                ndcg=ndcg,
                faithfulness=0.0,
                hallucination=0.0,
                response_time=elapsed,
            )

            if (i + 1) % 25 == 0:
                print(f"     {i+1}/{len(run_samples)} done")

        print(f"  ✅ {method} on {dataset['name']} complete")

db.close()
print("\n✅ Full benchmark complete.")