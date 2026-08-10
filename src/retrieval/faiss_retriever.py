import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

from retrieval.base import RetrievalResult


class FAISSRetriever:

    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="sentence-transformers/all-mpnet-base-v2",
        model=None  # ← accept shared model
    ):

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.documents = pickle.load(f)

        # Use shared model if passed, otherwise load fresh
        self.model = model if model else SentenceTransformer(model_name)

    def search(self, query, top_k=5):

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            embedding.astype(np.float32),
            top_k
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            chunk = self.documents[idx]

            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    dataset=chunk["dataset"],
                    domain=chunk["domain"],
                    text=chunk["text"],
                    score=float(score),
                    rank=rank
                )
            )

        return results