from retrieval.bm25_retriever import BM25Retriever


def main():

    retriever = BM25Retriever(
        "../results/indexes/cuad_bm25.pkl"
    )

    query = input("\nEnter your question:\n> ")

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("TOP 5 RETRIEVED CHUNKS")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\nRank {i}")
        print("-" * 60)

        print(f"Score   : {result['score']:.4f}")
        print(f"Dataset : {result['dataset']}")
        print(f"Domain  : {result['domain']}")

        print("\nText:\n")

        print(result["text"][:800])

        print("\n")


if __name__ == "__main__":
    main()