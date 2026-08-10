from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid_retriever import HybridRetriever


def get_retriever(method, dataset):

    bm25 = f"../results/indexes/{dataset.lower()}_bm25.pkl"

    faiss = f"../results/indexes/{dataset.lower()}_faiss.index"

    metadata = f"../results/indexes/{dataset.lower()}_metadata.pkl"

    if method == "BM25":
        return BM25Retriever(bm25)

    if method == "FAISS":
        return FAISSRetriever(
            faiss,
            metadata
        )

    if method == "Hybrid":
        return HybridRetriever(
            bm25,
            faiss,
            metadata
        )

    raise ValueError(f"Unknown retrieval method: {method}")