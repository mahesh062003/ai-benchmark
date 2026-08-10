from benchmark.dataset_registry import DATASETS
from benchmark.config import SAMPLE_LIMIT

from preprocessing.chunker import DocumentChunker
from preprocessing.save_chunks import save_chunks

from indexing.bm25_index import BM25Indexer
from indexing.faiss_index import FAISSIndexer


def main():

    chunker = DocumentChunker()

    print("\n" + "=" * 70)
    print("AI BENCHMARK - BUILDING ALL DATASET INDEXES")
    print("=" * 70)

    for dataset in DATASETS:

        print("\n" + "=" * 70)
        print(f"PROCESSING {dataset['name']}")
        print("=" * 70)

        samples = dataset["loader"](dataset["path"])

        samples = samples[:SAMPLE_LIMIT]

        chunks = chunker.chunk_documents(samples)

        print(f"\nSamples Loaded : {len(samples)}")
        print(f"Chunks Created : {len(chunks)}")

        save_chunks(
                    chunks,
                    dataset["name"].lower()
                    )

        bm25 = BM25Indexer()
        bm25.build(chunks)
        bm25.save(dataset["name"].lower())

        faiss = FAISSIndexer()
        faiss.build(chunks)
        faiss.save(dataset["name"].lower())

        print(f"\n{dataset['name']} completed successfully.")

    print("\n" + "=" * 70)
    print("ALL DATASETS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()