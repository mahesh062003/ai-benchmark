from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import LLMGenerator


retriever = HybridRetriever(
    "../results/indexes/cuad_bm25.pkl",
    "../results/indexes/cuad_faiss.index",
    "../results/indexes/cuad_metadata.pkl"
)

generator = LLMGenerator()

question = input("Question: ")

results = retriever.search(
    question,
    top_k=5
)

answer = generator.generate(
    question,
    results
)

print("\n" + "=" * 80)
print("GENERATED ANSWER")
print("=" * 80)

print(answer)