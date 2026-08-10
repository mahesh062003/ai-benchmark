from retrieval.hybrid_retriever import HybridRetriever


retriever = HybridRetriever(
    "../results/indexes/cuad_bm25.pkl",
    "../results/indexes/cuad_faiss.index",
    "../results/indexes/cuad_metadata.pkl"
)

query = input("Question: ")

results = retriever.search(
    query,
    top_k=5
)

print("\n" + "=" * 80)
print("HYBRID RETRIEVAL RESULTS")
print("=" * 80)

for result in results:

    print(f"\nRank : {result.rank}")
    print(f"Score: {result.score:.5f}")
    print(f"Dataset: {result.dataset}")
    print(f"Domain : {result.domain}")

    print("\nText:\n")
    print(result.text[:800])