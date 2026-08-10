import pickle
import re

from retrieval.base import RetrievalResult


class BM25Retriever:

    def __init__(self, index_path):

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]

    def tokenize(self, text):

        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, top_k=5):

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):

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