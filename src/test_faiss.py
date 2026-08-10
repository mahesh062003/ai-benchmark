from retrieval.faiss_retriever import FAISSRetriever


def main():

    retriever = FAISSRetriever(
        "../results/indexes/cuad_faiss.index",
        "../results/indexes/cuad_metadata.pkl"
    )

    query = input("\nEnter your question:\n> ")

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("TOP 5 FAISS RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\nRank {i}")
        print("-" * 60)

        print(f"Score   : {result['score']:.4f}")
        print(f"Dataset : {result['dataset']}")
        print(f"Domain  : {result['domain']}")

        print("\nText:\n")

        print(result["text"][:800])


if __name__ == "__main__":
    main()