from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.rrf import ReciprocalRankFusion


class HybridRetriever:

    def __init__(
        self,
        bm25_path,
        faiss_index_path,
        metadata_path,
        model=None  # ← accept shared model
    ):

        self.bm25 = BM25Retriever(bm25_path)

        self.faiss = FAISSRetriever(
            faiss_index_path,
            metadata_path,
            model=model  # ← pass shared model in
        )

        self.rrf = ReciprocalRankFusion()

    def search(self, query, top_k=5):

        bm25_results = self.bm25.search(
            query,
            top_k=top_k
        )

        faiss_results = self.faiss.search(
            query,
            top_k=top_k
        )

        fused = self.rrf.fuse(
            bm25_results,
            faiss_results
        )

        return fused[:top_k]