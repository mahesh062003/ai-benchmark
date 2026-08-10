import time

from benchmark.dataset_registry import DATASETS
from benchmark.config import SAMPLE_LIMIT, TOP_K, LLM_MODEL

from retrieval.retriever_factory import get_retriever
from generation.generator import LLMGenerator

from database.database import BenchmarkDatabase

from evaluation.retrieval_metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k
)


RETRIEVAL_METHODS = [
    "BM25",
    "FAISS",
    "Hybrid"
]


def main():

    db = BenchmarkDatabase()

    generator = LLMGenerator(model=LLM_MODEL)

    for dataset in DATASETS:

        print("\n" + "=" * 70)
        print(f"DATASET : {dataset['name']}")
        print("=" * 70)

        samples = dataset["loader"](dataset["path"])
        samples = samples[:SAMPLE_LIMIT]

        for method in RETRIEVAL_METHODS:

            print(f"\nRunning {method} Retrieval...")

            retriever = get_retriever(
                method,
                dataset["name"]
            )

            completed = 0

            for sample in samples:

                question = sample.get("question", "")

                if not question:
                    continue

                start = time.time()

                results = retriever.search(
                    question,
                    top_k=TOP_K
                )

                answer = generator.generate(
                    question,
                    results
                )

                elapsed = time.time() - start

                retrieved_ids = [
                    r.chunk_id
                    for r in results
                ]

                # Temporary placeholder
                relevant_ids = [
                    retrieved_ids[0]
                ]

                recall = recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    TOP_K
                )

                mrr = mean_reciprocal_rank(
                    retrieved_ids,
                    relevant_ids
                )

                ndcg = ndcg_at_k(
                    retrieved_ids,
                    relevant_ids,
                    TOP_K
                )

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

                    response_time=elapsed

                )

                completed += 1

                if completed % 25 == 0:

                    print(
                        f"{completed}/{len(samples)} completed..."
                    )

    db.close()

    print("\nBenchmark Finished Successfully.")


if __name__ == "__main__":
    main()